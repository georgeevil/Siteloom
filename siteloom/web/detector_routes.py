"""The detector tuning lab (CLD-101/102/106).

New cameras arrive needing settings nobody knows yet, and the only safe
way to find them is to try candidates against real footage *beside*
what runs today. This screen is that loop:

* **Trial**: pick a source (an NVR window, a cached clip, an uploaded
  file), pick settings (a named scene preset, the merge↔split axis, or
  explicit overrides), run — sandboxed. A trial writes its report and
  annotated evidence frames under `media_dir/tuning/<run>/` and touches
  nothing else: no Event/Detection rows, no identity work, no vector
  store (CLD-106's non-negotiable, structural by construction).
* **Evidence first** (CLD-102): the run's product is annotated frames —
  every track birth, plus the clip's ordinary rhythm — with the
  track_eval numbers behind them as the tie-breaker. Two runs over the
  same source get the harness's own verdict arithmetic.
* **Apply is explicit and minimal**: to the site, or to one camera as a
  `DetectionOverride` holding only the fields that differ from the site
  values. Copy carries a tuned camera's override to a sibling, explicit
  about what rides along (`sample_fps` is a property of the scene, so
  it is a separate checkbox). Every apply snapshots `site.yaml` into
  `config-history/` first, and revert restores a snapshot — CLD-106's
  history rung, in its simplest honest form.
* **Live preview**: an MJPEG overlay running candidate settings on the
  live feed through its own DetectionModule — fresh tracker state, so
  the pipeline's per-camera trackers are untouched. One preview at a
  time, expiring on its own.

Trials run in the serving process, one at a time, observed through
OperationRun — exactly the /backfill pattern, for the same reasons.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse

from siteloom.config import DetectionOverride, save_config
from siteloom.tuning import (
    AXIS,
    PRESETS,
    apply_overrides,
    compare_reports,
    friendly_error,
    plain_comparison,
    plain_summary,
    recommend,
    run_trial,
)
from siteloom.web import nav, params
from siteloom.web.backfill_routes import parse_range

log = logging.getLogger(__name__)

#: One trial at a time in the serve process (GPU + per-camera tracker
#: state, the /backfill rule); `preview` is the single live-overlay
#: slot; `last` keeps the most recent trial's outcome for the page.
_state: dict = {"thread": None, "last": None, "preview": None}

#: A preview left open stops annotating after this long: it costs a
#: detector pass per shown frame, and a forgotten tab must not keep a
#: GPU warm overnight.
PREVIEW_TTL_S = 600.0
PREVIEW_FPS = 2.0
#: Upload cap. A tuning clip is a minute or two of footage, not an
#: archive; the backfill console owns bulk ingestion.
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
SNAPSHOT_KEEP = 40

#: The DetectionOverride fields an apply may write — the parse gate for
#: every settings form on this screen.
SETTING_FIELDS = ("model", "confidence", "class_confidence", "tracker",
                  "track_buffer_s")


def minimal_override(site, effective: dict) -> DetectionOverride | None:
    """The smallest DetectionOverride that turns `site` into
    `effective` — applying a trial's settings to a camera must not bake
    in restatements of site values, or the next site-wide change
    silently stops reaching that camera."""
    fields: dict = {}
    for name in ("model", "confidence", "class_confidence", "track_buffer_s"):
        if effective.get(name) is not None and effective[name] != getattr(site, name):
            fields[name] = effective[name]
    tracker = {
        k: v for k, v in (effective.get("tracker") or {}).items()
        if site.tracker.get(k) != v
    }
    if tracker:
        fields["tracker"] = tracker
    return DetectionOverride(**fields) if fields else None


def _safe_name(raw: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(raw or "clip").name).strip("-.")
    return name or "clip"


def register(app, templates, Session, config, hub=None) -> None:  # noqa: C901 — route table
    # A tab of Jobs: trials are long-running operations watched the same
    # way, and the screen is the operations answer to "this new camera
    # needs different settings".
    nav.add("/detector", "Tuning", "TN", tab_of="/jobs")

    tuning_root = Path(config.storage.media_dir) / "tuning"
    clips_dir = tuning_root / "clips"
    corpus_cache = Path.home() / ".cache" / "siteloom" / "track-corpus"

    # -- helpers -----------------------------------------------------------

    def _runs() -> list[dict]:
        runs = []
        if tuning_root.is_dir():
            for report_path in tuning_root.glob("*/report.json"):
                try:
                    report = json.loads(report_path.read_text())
                except ValueError:
                    continue
                runs.append({
                    "id": report_path.parent.name,
                    "report": report,
                    "at": report_path.stat().st_mtime,
                })
        runs.sort(key=lambda r: r["at"], reverse=True)
        return runs[:50]

    def _run(run_id: str) -> dict:
        # Containment before reads, the /media rule (CLD-49).
        path = (tuning_root / run_id / "report.json").resolve()
        if not path.is_relative_to(tuning_root.resolve()) or not path.is_file():
            raise HTTPException(404)
        return json.loads(path.read_text())

    def _sources() -> list[str]:
        seen = []
        for folder in (clips_dir, corpus_cache):
            if folder.is_dir():
                seen += sorted(p.name for p in folder.glob("*.mp4"))
        return seen

    def _source_path(name: str) -> Path:
        wanted = _safe_name(name)
        for folder in (clips_dir, corpus_cache):
            candidate = (folder / wanted).resolve()
            if candidate.is_relative_to(folder.resolve()) and candidate.is_file():
                return candidate
        raise HTTPException(400, f"unknown clip {name!r}")

    def _effective_rows() -> list[dict]:
        """Per-camera effective detection values, with what differs
        from the site marked — CLD-101's 'a merged config nobody can
        read is how a wrong setting survives for months'."""
        site = config.detection
        rows = []
        for cam in config.cameras:
            eff = site.for_camera(cam)
            overridden = (
                set(
                    k for k, v in cam.detection.model_dump().items()
                    if v is not None
                )
                if cam.detection
                else set()
            )
            rows.append({
                "camera": cam, "effective": eff, "overridden": overridden,
                "profile": "day",
            })
            if cam.night is not None:
                rows.append({
                    "camera": cam,
                    "effective": site.for_camera(cam, "night"),
                    "overridden": {
                        k for k, v in cam.night.model_dump().items()
                        if v is not None
                    },
                    "profile": "night",
                })
        return rows

    #: The wizard's field vocabulary: label + plain meaning, in display
    #: order. One list, so every screen names things the same way.
    FIELD_WORDS = (
        ("confidence", "Detection floor",
         "how sure the detector must be before a box counts at all"),
        ("sample_fps", "Frames per second",
         "how often the stream is looked at — faster subjects need more"),
        ("track_buffer_s", "Lost-track patience (seconds)",
         "how long a subject may vanish before their track is given up"),
        ("model", "Detector model",
         "the network doing the looking; bigger sees smaller subjects"),
    )

    def field_provenance(cam) -> list[dict]:
        """Each tunable field with its current value and where that
        value comes from — the wizard's 'defaults to X from Y' labels.
        Camera-agnostic when cam is None (the site row)."""
        site = config.detection
        eff = site.for_camera(cam) if cam is not None else site
        overridden = (
            {k for k, v in cam.detection.model_dump().items() if v is not None}
            if cam is not None and cam.detection
            else set()
        )
        where = (
            f"override on {cam.name or cam.id}" if cam is not None else None
        )
        rows = []
        for field, label, meaning in FIELD_WORDS:
            if field == "sample_fps":
                value = cam.sample_fps if cam is not None else 5.0
                origin = (
                    f"set on {cam.name or cam.id}" if cam is not None
                    else "the usual default"
                )
            else:
                value = getattr(eff, field)
                origin = where if field in overridden else "site default"
            rows.append({
                "field": field, "label": label, "meaning": meaning,
                "value": value, "origin": origin,
            })
        # The deliberately untunable ones, greyed out with their reason.
        rows.append({
            "field": None, "label": "Crop margin / classes / device",
            "meaning": "not tunable here: crop margin changes the "
                       "recognition vector space (a re-enrol event), and "
                       "classes/device describe the site and the machine",
            "value": None, "origin": "fixed",
        })
        return rows

    def _settings_diff(settings: dict, cam, profile: str = "day") -> list[dict]:
        """What applying a trial would actually change, against the
        camera's (or site's) current effective values for the profile."""
        site = config.detection
        eff = site.for_camera(cam, profile) if cam is not None else site
        rows = []
        for field in ("model", "confidence", "class_confidence",
                      "track_buffer_s"):
            new = settings.get(field)
            cur = getattr(eff, field)
            if new is not None and new != cur:
                rows.append({"field": field, "current": cur, "new": new})
        new_tracker = settings.get("tracker") or {}
        for key, value in new_tracker.items():
            if eff.tracker.get(key) != value:
                rows.append({
                    "field": f"tracker.{key}",
                    "current": eff.tracker.get(key, "(library default)"),
                    "new": value,
                })
        return rows

    def parse_settings(form: dict, base) -> tuple:
        """(effective DetectionConfig, sample_fps, human summary).

        Whole form parsed before anything runs (the /classes/detection
        contract, CLD-61). Layering order: camera-effective base →
        named preset → merge/split axis → explicit fields; later layers
        win, and the summary says what ended up different.
        """
        overrides: dict = {}
        pieces: list[str] = []
        sample_fps = None

        preset = (form.get("preset") or "").strip()
        if preset:
            if preset not in PRESETS:
                raise HTTPException(400, f"unknown preset {preset!r}")
            chosen = dict(PRESETS[preset]["settings"])
            sample_fps = chosen.pop("sample_fps", None)
            overrides.update(chosen)
            pieces.append(f"preset {preset}")

        axis_raw = (form.get("axis") or "").strip()
        if axis_raw and axis_raw != "0":
            position = params.as_int(axis_raw, "axis", low=-2, high=2)
            for field, value in AXIS[position].items():
                if field == "tracker":
                    overrides["tracker"] = {
                        **overrides.get("tracker", {}), **value
                    }
                else:
                    overrides[field] = value
            side = "split" if position < 0 else "merge"
            pieces.append(f"axis {position:+d} ({side})")

        if (form.get("confidence") or "").strip():
            overrides["confidence"] = params.as_confidence(
                form["confidence"], "confidence"
            )
        if (form.get("track_buffer_s") or "").strip():
            overrides["track_buffer_s"] = params.as_float(
                form["track_buffer_s"], "track_buffer_s", low=0.2, high=30.0
            )
        if (form.get("model") or "").strip():
            overrides["model"] = params.as_name(form["model"], "model")
        raw_cc = (form.get("class_confidence") or "").strip()
        if raw_cc:
            try:
                parsed_cc = yaml.safe_load(raw_cc)
            except yaml.YAMLError as exc:
                raise HTTPException(400, f"class_confidence: {exc}") from None
            overrides["class_confidence"] = {
                params.as_name(k, "class_confidence key"):
                    params.as_confidence(v, f"class_confidence[{k}]")
                for k, v in params.as_object(
                    parsed_cc or {}, "class_confidence"
                ).items()
            }
        raw_tracker = (form.get("tracker") or "").strip()
        if raw_tracker:
            try:
                parsed_tracker = yaml.safe_load(raw_tracker)
            except yaml.YAMLError as exc:
                raise HTTPException(400, f"tracker: {exc}") from None
            overrides["tracker"] = {
                **overrides.get("tracker", {}),
                **params.as_object(parsed_tracker, "tracker"),
            }
        for field in ("confidence", "track_buffer_s", "model",
                      "class_confidence", "tracker"):
            if field in overrides:
                pieces.append(field)

        if (form.get("sample_fps") or "").strip():
            sample_fps = params.as_float(
                form["sample_fps"], "sample_fps", low=0.5, high=30.0
            )

        effective = apply_overrides(base, overrides)
        # Validation pass: a typo'd tracker key should fail here, in the
        # form error, not inside ultralytics mid-trial.
        effective = type(base).model_validate(effective.model_dump())
        return effective, sample_fps, (", ".join(pieces) or "current settings")

    def _actor(request: Request | None) -> str:
        """Who is acting — the audit trail's convention: the username,
        or "(open)" in open single-operator mode."""
        user = getattr(request.state, "user", None) if request else None
        return user.username if user else "(open)"

    def _snapshot(
        request: Request | None = None, reason: str = "", summary: str = ""
    ) -> str | None:
        """Copy site.yaml into config-history/ before a write, with a
        metadata sidecar — who, when, why, and what (CLD-106's history
        rung). None when the config has no file behind it (tests)."""
        source = getattr(config, "_source_path", None)
        if not source or not Path(source).is_file():
            return None
        history = Path(source).parent / "config-history"
        history.mkdir(exist_ok=True)
        # Microseconds in the name: two changes in the same second are
        # two snapshots — a revert snapshots the pre-revert state, and a
        # same-second collision would overwrite the very file being
        # restored.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        name = f"site-{stamp}.yaml"
        shutil.copy2(source, history / name)
        (history / f"site-{stamp}.meta.json").write_text(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(),
            "actor": _actor(request),
            "reason": (reason or "").strip(),
            "summary": summary,
        }, indent=2))
        # Trimmed in pairs: a snapshot without its meta says "someone
        # changed something", which is the state this sidecar removes.
        for stale in sorted(history.glob("site-*.yaml"))[:-SNAPSHOT_KEEP]:
            stale.unlink()
            (history / f"{stale.stem}.meta.json").unlink(missing_ok=True)
        return name

    def _snapshots() -> list[dict]:
        source = getattr(config, "_source_path", None)
        if not source:
            return []
        history = Path(source).parent / "config-history"
        if not history.is_dir():
            return []
        rows = []
        for path in sorted(history.glob("site-*.yaml"), reverse=True):
            meta = None
            meta_path = history / f"{path.stem}.meta.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text())
                except ValueError:
                    meta = None
            rows.append({"name": path.name, "meta": meta})
        return rows

    def _persist() -> str | None:
        try:
            return save_config(config)
        except ValueError:
            return None  # no file behind this config (tests)

    # -- pages -------------------------------------------------------------

    def page_context(**kw) -> dict:
        thread = _state["thread"]
        preview = _state["preview"]
        return {
            "site_name": config.site_name or config.site_id,
            "cameras": config.cameras,
            "presets": PRESETS,
            "sources": _sources(),
            "runs": _runs(),
            "effective": _effective_rows(),
            "site_detection": config.detection,
            "running": thread is not None and thread.is_alive(),
            "last": _state["last"],
            "proposals": _proposals(),
            "preview": preview if preview and not _preview_expired(preview) else None,
            "snapshots": _snapshots()[:10],
            "error": None,
            "form": {},
            **kw,
        }

    def page(request: Request, status_code: int = 200, **kw):
        return templates.TemplateResponse(
            request, "detector.html", page_context(**kw), status_code=status_code
        )

    @app.get("/detector")
    def detector_page(request: Request):
        return page(request)

    @app.get("/detector/tune")
    def tune_wizard(
        request: Request,
        camera: str = "",
        step: str = "",
        source_kind: str = "nvr",
        start: str = "",
        end: str = "",
        clip: str = "",
    ):
        """The guided path: camera → footage → settings, state in the
        URL (bookmarkable, refresh-safe — the import-wizard rule). All
        three steps submit to the existing POST /detector/run; the
        wizard adds no mutation surface of its own."""
        cam = next((c for c in config.cameras if c.id == camera), None)
        if cam is None:
            step = "camera"
        elif step != "settings":
            step = "footage"
        camera_clips = [
            name for name in _sources()
            if cam is not None and name.startswith(f"{cam.id}-")
        ]
        from siteloom.localtime import site_zone

        zone = site_zone(config)
        now = datetime.now(timezone.utc)
        now_local = now.astimezone(zone) if zone else now
        return templates.TemplateResponse(
            request, "detector_tune.html", {
                "site_name": config.site_name or config.site_id,
                "step": step,
                "camera": cam,
                "cameras": config.cameras,
                "effective": _effective_rows(),
                "presets": PRESETS,
                "provenance": field_provenance(cam),
                "camera_clips": camera_clips,
                # Pre-filled "ten minutes, ending two minutes ago", in
                # site wall time — the zone every datetime-local input on
                # this site speaks (CLD-100). Not up-to-now: Protect
                # often cannot export the newest minute or two.
                "default_start": (now_local - timedelta(minutes=12))
                .strftime("%Y-%m-%dT%H:%M:00"),
                "default_end": (now_local - timedelta(minutes=2))
                .strftime("%Y-%m-%dT%H:%M:00"),
                # The footage step's choices, carried into the settings
                # step as hidden fields for the final POST.
                "chosen": {
                    "source_kind": source_kind, "start": start,
                    "end": end, "clip": clip,
                },
                "running": _state["thread"] is not None
                and _state["thread"].is_alive(),
            },
        )

    @app.get("/detector/help")
    def workflow_guide(request: Request):
        """The workflow reference, rendered from docs/tuning-workflows.md
        — one source of truth for operators and the repo alike."""
        from markdown_it import MarkdownIt

        doc = Path(__file__).resolve().parents[2] / "docs" / "tuning-workflows.md"
        if doc.is_file():
            body = MarkdownIt().render(doc.read_text())
        else:  # pragma: no cover — packaging without docs/
            body = "<p>The workflow guide ships in docs/tuning-workflows.md.</p>"
        return templates.TemplateResponse(
            request, "detector_help.html", {
                "site_name": config.site_name or config.site_id,
                "body": body,
            },
        )

    @app.get("/detector/upload")
    def upload_wizard(request: Request, camera: str = ""):
        """The downloaded-clip path: pick the file, say which camera the
        footage came from (that choice is the settings base and the
        later apply target), pick a scene, run."""
        cam = next((c for c in config.cameras if c.id == camera), None)
        return templates.TemplateResponse(
            request, "detector_tune.html", {
                "site_name": config.site_name or config.site_id,
                "step": "upload",
                "camera": cam,
                "cameras": config.cameras,
                "effective": _effective_rows(),
                "presets": PRESETS,
                "provenance": field_provenance(cam),
                "camera_clips": [],
                "default_start": "", "default_end": "",
                "chosen": {"source_kind": "upload", "start": "", "end": "",
                           "clip": ""},
                "running": _state["thread"] is not None
                and _state["thread"].is_alive(),
            },
        )

    @app.get("/detector/runs/{run_id}")
    def run_detail(request: Request, run_id: str, versus: str = ""):
        report = _run(run_id)
        comparison = None
        if versus:
            other = _run(versus)
            if other.get("source") != report.get("source"):
                raise HTTPException(
                    400,
                    "runs can only be compared over the same source clip — "
                    "different footage answers a different question",
                )
            comparison = compare_reports(other, report)
        report_cam = next(
            (c for c in config.cameras if c.id == report.get("camera")), None
        )
        return templates.TemplateResponse(
            request, "detector_run.html", {
                "site_name": config.site_name or config.site_id,
                "run_id": run_id,
                "report": report,
                "summary": plain_summary(report),
                "comparison_words": (
                    plain_comparison(comparison) if comparison else None
                ),
                "would_change": _settings_diff(
                    report.get("settings") or {}, report_cam,
                    report.get("profile") or "day",
                ),
                "report_camera": report_cam,
                "recommendations": recommend(report, report["sample_fps"]),
                "cameras": config.cameras,
                "same_source": [
                    r["id"] for r in _runs()
                    if r["report"].get("source") == report.get("source")
                    and r["id"] != run_id
                ],
                "versus": versus,
                "comparison": comparison,
            },
        )

    # -- running a trial ---------------------------------------------------

    @app.post("/detector/run")
    async def start_trial(
        request: Request,
        source_kind: str = Form(...),
        camera: str = Form(""),
        start: str = Form(""),
        end: str = Form(""),
        clip: str = Form(""),
        upload: UploadFile | None = File(None),
    ):
        form = dict((await request.form()).items())
        form.pop("upload", None)
        thread = _state["thread"]
        if thread is not None and thread.is_alive():
            return page(request, 409, form=form,
                        error="A trial is already running. Wait for it to finish.")
        params.one_of(source_kind, "source_kind", ("nvr", "clip", "upload"))

        cam = next((c for c in config.cameras if c.id == camera), None)
        # Which of the camera's profiles the trial starts from (CLD-129)
        # — night bases on the day-effective + night override, the same
        # resolution live ingest uses when the footage reads as IR.
        profile = "night" if form.get("profile") == "night" else "day"
        base = (
            config.detection.for_camera(cam, profile) if cam is not None
            else config.detection
        )
        try:
            det_cfg, sample_fps, summary = parse_settings(form, base)
        except HTTPException as exc:
            return page(request, exc.status_code, form=form, error=exc.detail)
        if profile == "night":
            summary = f"night profile, {summary}"
        if sample_fps is None:
            sample_fps = cam.sample_fps if cam is not None else 5.0

        # -- resolve the source, before the thread starts ------------------
        if source_kind == "clip":
            source = _source_path(clip)
        elif source_kind == "upload":
            if upload is None or not upload.filename:
                return page(request, 400, form=form,
                            error="Choose a video file to upload.")
            uploads = tuning_root / "uploads"
            uploads.mkdir(parents=True, exist_ok=True)
            target = uploads / _safe_name(upload.filename)
            written = 0
            with target.open("wb") as fh:
                while chunk := await upload.read(1 << 20):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        fh.close()
                        target.unlink(missing_ok=True)
                        return page(
                            request, 413, form=form,
                            error="Upload too large — a tuning clip is a "
                                  "minute or two, not an archive.",
                        )
                    fh.write(chunk)
            source = target
        else:
            if cam is None or cam.adapter != "unifi":
                return page(request, 400, form=form,
                            error="Pick a UniFi camera to pull NVR footage from.")
            try:
                from siteloom.localtime import site_zone

                begins, finishes = parse_range(start, end, site_zone(config))
            except ValueError as exc:
                return page(request, 400, form=form, error=str(exc))
            if (finishes - begins).total_seconds() > 15 * 60:
                return page(request, 400, form=form,
                            error="Keep NVR trial windows under 15 minutes — "
                                  "a trial is a probe, not a backfill.")
            clips_dir.mkdir(parents=True, exist_ok=True)
            source = clips_dir / (
                f"{cam.id}-{begins.strftime('%Y%m%d-%H%M%S')}"
                f"-{finishes.strftime('%H%M%S')}.mp4"
            )

        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            + f"-{cam.id if cam else 'clip'}"
        )
        out_dir = tuning_root / run_id
        _state["last"] = {
            "run_id": run_id, "status": "running", "summary": summary,
            "source": source.name,
        }
        nvr_window = (begins, finishes) if source_kind == "nvr" else None

        def work():
            from siteloom.progress import ProgressReporter

            last = _state["last"]
            try:
                with ProgressReporter(
                    Session, "detector-trial", target=source.name,
                    resume_command="(re-run from /detector)", bar=False,
                ) as reporter:
                    if nvr_window is not None and not source.is_file():
                        with reporter.phase("Downloading NVR footage"):
                            from siteloom.adapters.unifi import UniFiProtectAdapter

                            adapter = UniFiProtectAdapter(unifi=config.unifi)
                            adapter.connect()
                            try:
                                adapter.download_clip(
                                    cam.source, *nvr_window, source
                                )
                            finally:
                                adapter.close()
                    with reporter.phase("Detecting"):
                        def tick(n, total):
                            reporter.advance(frames=1)

                        report = run_trial(
                            source, det_cfg, sample_fps, out_dir,
                            group_for=config.events.group_for,
                            progress=tick,
                            label={"camera": cam.id if cam else None,
                                   "profile": profile},
                        )
                last.update(status="complete", frames=report["frames"],
                            groups=len(report["groups"]))
            except Exception as exc:  # pragma: no cover — via OperationRun
                log.exception("detector trial failed")
                raw = f"{type(exc).__name__}: {exc}"
                # The operator-facing sentence first; the raw text rides
                # along behind a disclosure. A tuning-lab user is exactly
                # who a bare uiprotect BadRequest means nothing to.
                last.update(
                    status="failed", error=raw, friendly=friendly_error(raw)
                )
            finally:
                if last.get("status") == "running":
                    last["status"] = "stopped"
                _state["thread"] = None

        thread = threading.Thread(target=work, name="detector-trial", daemon=True)
        _state["thread"] = thread
        thread.start()
        return RedirectResponse("/detector", status_code=303)

    # -- applying settings -------------------------------------------------

    @app.post("/detector/apply")
    def apply_settings(
        request: Request,
        run_id: str = Form(...),
        target: str = Form(...),
        reason: str = Form(""),
    ):
        """Adopt a trial's settings — to the site, or as one camera's
        minimal override. Snapshot first, so it is revertable."""
        report = _run(run_id)
        settings = report.get("settings") or {}
        # "cam-id:night" targets the camera's night profile (CLD-129).
        target_cam, _, target_profile = target.partition(":")
        target_profile = target_profile or "day"
        cam_for_diff = next(
            (c for c in config.cameras if c.id == target_cam), None
        )
        changed = [
            r["field"]
            for r in _settings_diff(settings, cam_for_diff, target_profile)
        ]
        _snapshot(
            request, reason,
            f"applied trial {run_id} to {target}"
            + (f" (changed: {', '.join(changed)})" if changed else " (no diff)"),
        )
        if target == "site":
            det = config.detection
            det.model = settings.get("model", det.model)
            det.confidence = settings.get("confidence", det.confidence)
            det.class_confidence = settings.get(
                "class_confidence", det.class_confidence
            )
            det.tracker = settings.get("tracker", det.tracker)
            det.track_buffer_s = settings.get(
                "track_buffer_s", det.track_buffer_s
            )
        else:
            cam = cam_for_diff
            if cam is None:
                raise HTTPException(400, f"{target!r} is not a configured camera")
            if target_profile == "night":
                # The diff base for night is the camera's DAY-effective
                # config, not the site: restating a day override into
                # the night layer would detach it from future day
                # changes. sample_fps stays a camera property — the
                # scene's speed does not change with the light.
                cam.night = minimal_override(
                    config.detection.for_camera(cam), settings
                )
            else:
                cam.detection = minimal_override(config.detection, settings)
                if report.get("sample_fps"):
                    cam.sample_fps = float(report["sample_fps"])
        _persist()
        return RedirectResponse("/detector", status_code=303)

    @app.post("/detector/copy")
    def copy_settings(
        request: Request,
        from_camera: str = Form(...),
        to_camera: str = Form(...),
        include_sample_fps: str = Form("0"),
        reason: str = Form(""),
    ):
        """Carry one tuned camera's override to a sibling (CLD-106).

        Explicit about what rides along: the detection override always,
        `sample_fps` only when asked — it is a property of the scene,
        and 'the other back one is like this' may not extend to how
        fast subjects cross it.
        """
        src = next((c for c in config.cameras if c.id == from_camera), None)
        dst = next((c for c in config.cameras if c.id == to_camera), None)
        if src is None or dst is None or src is dst:
            raise HTTPException(400, "pick two different configured cameras")
        _snapshot(
            request, reason,
            f"copied {from_camera}'s detection override to {to_camera}"
            + (" with sample_fps" if include_sample_fps == "1" else ""),
        )
        dst.detection = (
            src.detection.model_copy(deep=True) if src.detection else None
        )
        if include_sample_fps == "1":
            dst.sample_fps = src.sample_fps
        _persist()
        return RedirectResponse("/detector", status_code=303)

    @app.post("/detector/reset-camera")
    def reset_camera(
        request: Request,
        camera: str = Form(...),
        profile: str = Form("day"),
        reason: str = Form(""),
    ):
        cam = next((c for c in config.cameras if c.id == camera), None)
        if cam is None:
            raise HTTPException(400, f"{camera!r} is not a configured camera")
        if profile == "night":
            _snapshot(request, reason, f"removed {camera}'s night profile")
            cam.night = None
        else:
            _snapshot(request, reason, f"reset {camera} to the site defaults")
            cam.detection = None
        _persist()
        return RedirectResponse("/detector", status_code=303)

    @app.post("/detector/revert")
    def revert(
        request: Request, snapshot: str = Form(...), reason: str = Form("")
    ):
        """Restore a config-history snapshot.

        The file is restored whole; the live process re-applies the
        fields this lab manages (site detection, per-camera detection
        overrides and sample_fps). Anything else the snapshot differs
        in takes effect on the next restart — said on the page rather
        than discovered.
        """
        source = getattr(config, "_source_path", None)
        if not source:
            raise HTTPException(400, "this config has no file to revert")
        history = Path(source).parent / "config-history"
        target = (history / _safe_name(snapshot)).resolve()
        if not target.is_relative_to(history.resolve()) or not target.is_file():
            raise HTTPException(404, "no such snapshot")
        from siteloom.config import load_config

        restored = load_config(target)  # validated before anything applies
        # Reverting is itself a change worth reverting — snapshot the
        # pre-revert state before overwriting it.
        _snapshot(request, reason, f"reverted to {target.name}")
        shutil.copy2(target, source)
        config.detection = restored.detection
        by_id = {c.id: c for c in restored.cameras}
        for cam in config.cameras:
            if cam.id in by_id:
                cam.detection = by_id[cam.id].detection
                cam.sample_fps = by_id[cam.id].sample_fps
        return RedirectResponse("/detector", status_code=303)

    # -- the overnight search (CLD-102) ------------------------------------

    def _proposals() -> list[dict]:
        root = tuning_root / "proposals"
        rows = []
        if root.is_dir():
            for path in root.glob("*/*/proposal.json"):
                try:
                    data = json.loads(path.read_text())
                except ValueError:
                    continue
                rows.append({
                    "camera": path.parent.parent.name,
                    "stamp": path.parent.name,
                    "winner": (data.get("winner") or {}).get("name"),
                    "verdict": data.get("verdict"),
                })
        rows.sort(key=lambda r: r["stamp"], reverse=True)
        return rows[:20]

    @app.get("/detector/search")
    def search_confirm(request: Request, camera: str = ""):
        """The budget statement before the button: which clips, how many
        trials, roughly how long — a search is hours of GPU an operator
        should spend knowingly, not discover."""
        cam = next((c for c in config.cameras if c.id == camera), None)
        clips = [
            name for name in _sources()
            if cam is not None and name.startswith(f"{cam.id}-")
        ]
        cand_count = 0
        if cam is not None:
            from siteloom.tuning_search import candidates as _candidates

            cand_count = len(_candidates(config.detection.for_camera(cam)))
        # baselines + round1(all) + round2(half × 2 clips) + finals.
        est_trials = (
            len(clips) + cand_count + (cand_count // 2) * min(2, max(0, len(clips) - 1))
            + max(1, cand_count // 4) * max(0, len(clips) - 3)
        )
        return templates.TemplateResponse(
            request, "detector_search.html", {
                "site_name": config.site_name or config.site_id,
                "camera": cam,
                "cameras": config.cameras,
                "clips": clips,
                "cand_count": cand_count,
                "est_trials": est_trials,
                "est_minutes": max(1, est_trials // 2),
                "running": _state["thread"] is not None
                and _state["thread"].is_alive(),
            },
        )

    @app.post("/detector/search")
    async def start_search(request: Request, camera: str = Form(...)):
        """Run the propose-only search over this camera's clips.

        Shares the trial slot — a search IS trials, and the GPU has one
        consumer. Cancellable from /jobs; an interrupted search leaves
        its trials (reusable evidence) and no proposal file — never a
        half-claim."""
        form = await request.form()
        thread = _state["thread"]
        if thread is not None and thread.is_alive():
            return page(request, 409,
                        error="A trial is already running. Wait for it to finish.")
        cam = next((c for c in config.cameras if c.id == camera), None)
        if cam is None:
            raise HTTPException(400, f"{camera!r} is not a configured camera")
        chosen = form.getlist("clips")
        clip_paths = [_source_path(name) for name in chosen]
        if not clip_paths:
            raise HTTPException(
                400,
                "pick at least one clip — pull an NVR window or upload "
                "footage from this camera first",
            )
        from siteloom.tuning_search import candidates as _candidates
        from siteloom.tuning_search import successive_halving

        base = config.detection.for_camera(cam)
        cands = _candidates(base)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_dir = tuning_root / "proposals" / cam.id / stamp
        _state["last"] = {
            "run_id": None, "status": "running",
            "summary": f"settings search, {len(cands)} candidates",
            "source": f"{len(clip_paths)} clip(s) from {cam.id}",
        }

        def work():
            from siteloom.progress import ProgressReporter

            last = _state["last"]
            seq = {"n": 0}
            try:
                with ProgressReporter(
                    Session, "detector-search", target=cam.id,
                    resume_command="(re-run from /detector/search)", bar=False,
                ) as reporter:
                    with reporter.phase("Searching"):
                        def runner(clip, overrides, tag):
                            seq["n"] += 1
                            rid = f"{stamp}-search-{seq['n']:02d}"
                            report = run_trial(
                                clip, apply_overrides(base, overrides),
                                cam.sample_fps, tuning_root / rid,
                                group_for=config.events.group_for,
                                label={"camera": cam.id, "profile": "day",
                                       "candidate": tag, "search": stamp},
                            )
                            reporter.advance(trials=1)
                            return rid, report

                        proposal = successive_halving(
                            clip_paths, cands, runner,
                            check_interrupt=reporter.check_interrupt,
                            log_round=lambda msg: reporter.bump(),
                        )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "proposal.json").write_text(json.dumps({
                        "camera": cam.id,
                        "at": datetime.now(timezone.utc).isoformat(),
                        "candidates": [c["name"] for c in cands],
                        **proposal,
                    }, indent=2))
                last.update(status="complete")
            except Exception as exc:  # pragma: no cover — via OperationRun
                log.exception("settings search failed")
                raw = f"{type(exc).__name__}: {exc}"
                last.update(status="failed", error=raw,
                            friendly=friendly_error(raw))
            finally:
                if last.get("status") == "running":
                    last["status"] = "stopped"
                _state["thread"] = None

        thread = threading.Thread(
            target=work, name="detector-search", daemon=True
        )
        _state["thread"] = thread
        thread.start()
        return RedirectResponse("/detector", status_code=303)

    @app.get("/detector/proposals/{camera_id}/{stamp}")
    def proposal_detail(request: Request, camera_id: str, stamp: str):
        path = (
            tuning_root / "proposals" / _safe_name(camera_id)
            / _safe_name(stamp) / "proposal.json"
        ).resolve()
        if not path.is_relative_to(tuning_root.resolve()) or not path.is_file():
            raise HTTPException(404)
        proposal = json.loads(path.read_text())
        winner = proposal.get("winner")
        winner_run = None
        if winner and winner.get("trials"):
            winner_run = sorted(winner["trials"].values())[0]
        cam = next((c for c in config.cameras if c.id == camera_id), None)
        return templates.TemplateResponse(
            request, "detector_proposal.html", {
                "site_name": config.site_name or config.site_id,
                "proposal": proposal,
                "camera": cam,
                "camera_id": camera_id,
                "stamp": stamp,
                "winner_run": winner_run,
            },
        )

    # -- live preview ------------------------------------------------------

    def _preview_expired(slot: dict) -> bool:
        return time.monotonic() - slot["started"] > PREVIEW_TTL_S

    @app.post("/detector/preview")
    async def start_preview(request: Request, camera: str = Form(...)):
        if hub is None:
            raise HTTPException(503, "live view is not available")
        cam = next((c for c in config.cameras if c.id == camera), None)
        if cam is None:
            raise HTTPException(400, f"{camera!r} is not a configured camera")
        form = dict((await request.form()).items())
        profile = "night" if form.get("profile") == "night" else "day"
        det_cfg, sample_fps, summary = parse_settings(
            form, config.detection.for_camera(cam, profile)
        )
        if profile == "night":
            summary = f"night profile, {summary}"
        _state["preview"] = {
            "camera": cam.id,
            "cfg": det_cfg,
            "sample_fps": sample_fps or cam.sample_fps,
            "summary": summary,
            "started": time.monotonic(),
        }
        return RedirectResponse("/detector#preview", status_code=303)

    @app.post("/detector/preview/stop")
    def stop_preview():
        _state["preview"] = None
        return RedirectResponse("/detector", status_code=303)

    @app.get("/detector/preview.mjpeg")
    def preview_stream():
        """The candidate settings, drawn on the live feed.

        A fresh DetectionModule per stream: its tracker state is the
        preview's own, so the pipeline's per-camera trackers never see
        these frames. Bounded three ways — the slot's TTL, the hub's
        own shutdown checks (CLD-132), and the slot being replaced.
        """
        slot = _state["preview"]
        if slot is None or _preview_expired(slot) or hub is None:
            raise HTTPException(404, "no preview is armed")

        def gen():
            import cv2
            import numpy as np

            from siteloom.dispatch.base import Job
            from siteloom.modules.detection import DetectionModule
            from siteloom.tuning import _draw

            module = DetectionModule(slot["cfg"])
            last = 0.0
            for jpeg in hub.frames(slot["camera"]):
                if _state["preview"] is not slot or _preview_expired(slot):
                    break
                now = time.monotonic()
                if now - last < 1.0 / PREVIEW_FPS:
                    continue
                last = now
                frame = cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None:
                    continue
                result = module.process(Job(module="detection", payload={
                    "image_jpeg": jpeg,
                    "camera_id": f"preview-{slot['camera']}",
                    "sample_fps": slot["sample_fps"],
                }))
                _draw(frame, result.get("detections", []))
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if not ok:
                    continue
                out = buf.tobytes()
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(out)}\r\n\r\n".encode()
                    + out + b"\r\n"
                )

        return StreamingResponse(
            gen(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )
