"""The backfill console (CLD-93).

`siteloom backfill-unifi` was terminal-only, which left the console able
to tell an operator that a gap exists — `siteloom run` heartbeats an
OperationRun and /jobs shows it going stale (CLD-15) — and unable to help
them close it. It is also the only camera-derived path that produces
`NoiseEvent` rows at all: audio never reaches the detector on a live
stream, so /noise can only ever fill from downloaded NVR clips.

Two halves, and the read is the more valuable one:

* **What is already covered.** The resumability model is "the remaining
  clips are the ones still pending" (`BackfillClip`), and nothing
  surfaced it. Pending / done / failed and the covered range, per camera,
  is the whole state of a resumable sweep.
* **Starting one.** Camera, start, optional end, and the motion-window vs
  full-sweep choice. Full sweep is the non-default because it downloads
  every minute of the range rather than the minutes the NVR thinks
  something moved, and the form says so.

Deliberately scoped to `backfill-unifi`. Plain `siteloom backfill` over a
media archive has no per-file checkpointing (CLD-12) — an interrupted run
restarts from the top — so putting it behind the same button as the
resumable path would have the screen imply a guarantee it does not have.
The page says which one is missing and why, rather than quietly omitting
it.

The run happens in the *serving* process, observed through OperationRun
like every other long job. Several cameras may run side by side — one
thread and one row each, over one shared pipeline, the shape live ingest
has (CLD-317) — but never the *same* camera twice: the pipeline keeps
per-camera tracker state, so a second run on a running camera is a
failure rather than a slowdown, and is refused with the camera named.
Known non-guard: /jobs/reindex keeps its own single-flight state, so a
reindex and a backfill of one camera can overlap; the reindex purges
first, so they would interleave. Decided out of scope for CLD-317.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from fastapi import Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select

from siteloom.backfill import ACTIVE_STATUSES
from siteloom.store import BackfillClip
from siteloom.web import nav

log = logging.getLogger(__name__)

#: Per-camera run state, in the serve process. `runs` keeps each camera's
#: most recent scan/process outcome (the dict `backfill.new_state`
#: builds) so a re-run over an already-covered range reads as the no-op
#: it is, and so a running camera can refuse a second start. The lock
#: makes "is it busy" and "now it is" one step, or two submits racing
#: through the check would both start it — and every *reader* takes it
#: too, because a page rendering while a submit inserts is "dictionary
#: changed size during iteration". Re-entrant: the submit reads under
#: the lock it already holds.
_state: dict = {"runs": {}, "lock": threading.RLock()}


def run_states() -> list[dict]:
    """A snapshot of every camera's run state, safe to iterate."""
    with _state["lock"]:
        return list(_state["runs"].values())


def running_cameras() -> list[str]:
    """Camera ids with a backfill in flight (waiting, scanning or processing)."""
    return [
        state["camera"]
        for state in run_states()
        if state.get("status") in ACTIVE_STATUSES
    ]

#: Chunk sizes offered for a full sweep. Deliberately a short list: the
#: number trades clip count against clip length, and every option here
#: downloads the entire range either way.
CHUNK_CHOICES = (5.0, 15.0, 30.0, 60.0)


def parse_range(start: str, end: str, zone=None) -> tuple[datetime, datetime]:
    """The window to scan, or a ValueError naming what is wrong.

    A `datetime-local` input has no offset, and operators think in wall
    time — the **site's** wall time (CLD-100), which is what `zone`
    carries; None is the unset-zone rung and reads as UTC. Interpreting
    in the server process's zone (`.astimezone()`, the old behaviour) is
    wrong the day the box serving the site sits in a different zone than
    the cameras. Returned aware; `UnifiBackfill.scan` converts. An
    open-ended range means "up to now", which is the common case after a
    crash: you know when ingest stopped, not when it should stop catching
    up.
    """
    from siteloom.localtime import as_aware

    text = (start or "").strip()
    if not text:
        raise ValueError("Enter a start time.")
    try:
        begins = as_aware(datetime.fromisoformat(text), zone)
    except ValueError:
        raise ValueError(f"{text!r} is not a date and time.") from None
    if (end or "").strip():
        try:
            finishes = as_aware(datetime.fromisoformat(end.strip()), zone)
        except ValueError:
            raise ValueError(f"{end.strip()!r} is not a date and time.") from None
    else:
        finishes = datetime.now(timezone.utc)
    if finishes <= begins:
        raise ValueError("The end of the range must be after its start.")
    return begins, finishes


def clip_rows(session, cameras) -> list[dict]:
    """Per-camera BackfillClip state: the read this screen exists for.

    Counts are scoped per camera and `failed` is reported separately from
    `pending`, following the library indexer's rule — nothing picks a
    failed clip up again without an explicit retry, so folding the two
    together would show a sweep as nearly done when part of it will never
    run.

    Cameras with clips but no configuration left (renamed, removed) still
    get a row, flagged. Their footage is in the database either way, and
    dropping them would make the counts on this page quietly disagree
    with the table.
    """
    grouped = session.execute(
        select(
            BackfillClip.camera_id,
            BackfillClip.status,
            func.count(),
            func.coalesce(func.sum(BackfillClip.frames), 0),
            func.min(BackfillClip.start),
            func.max(BackfillClip.end),
            func.max(BackfillClip.processed_at),
        ).group_by(BackfillClip.camera_id, BackfillClip.status)
    ).all()

    per_camera: dict[str, dict] = {}
    for camera_id, status, count, frames, first, last, processed in grouped:
        row = per_camera.setdefault(
            camera_id,
            {
                "id": camera_id,
                "name": camera_id,
                "configured": False,
                "pending": 0,
                "done": 0,
                "failed": 0,
                "total": 0,
                "frames": 0,
                "covered_from": None,
                "covered_to": None,
                "done_from": None,
                "done_to": None,
                "last_processed": None,
            },
        )
        row[status] = row.get(status, 0) + count
        row["total"] += count
        row["frames"] += frames or 0
        row["covered_from"] = min(filter(None, (row["covered_from"], first)))
        row["covered_to"] = max(filter(None, (row["covered_to"], last)))
        if processed is not None:
            row["last_processed"] = max(
                filter(None, (row["last_processed"], processed))
            )
        if status == "done":
            row["done_from"] = first
            row["done_to"] = last

    rows = []
    for cam in cameras:
        row = per_camera.pop(cam.id, None) or {
            "id": cam.id,
            "pending": 0,
            "done": 0,
            "failed": 0,
            "total": 0,
            "frames": 0,
            "covered_from": None,
            "covered_to": None,
            "done_from": None,
            "done_to": None,
            "last_processed": None,
        }
        row["name"] = cam.name or cam.id
        row["id"] = cam.id
        row["configured"] = True
        rows.append(row)
    # Whatever is left has clips but no camera in the config any more.
    rows.extend(sorted(per_camera.values(), key=lambda r: r["id"]))
    return rows


def register(app, templates, Session, config) -> None:
    # A tab of Jobs: watching runs and starting archive sweeps are one
    # operations surface, on the same restricted-readable floor (neither
    # is in RESTRICTED_DENIED_PREFIXES; mutations are admin either way).
    nav.add("/backfill", "Backfill", "BF", tab_of="/jobs")

    def unifi_cameras():
        return [c for c in config.cameras if c.adapter == "unifi"]

    def page_context(**kw) -> dict:
        cameras = unifi_cameras()
        with Session() as session:
            rows = clip_rows(session, cameras)
        outstanding = sum(r["pending"] for r in rows)
        failed = sum(r["failed"] for r in rows)
        clips = sum(r["total"] for r in rows)
        # Three different nothings, and an operator acts differently on
        # each: no camera can be backfilled at all / nothing has been
        # asked for yet / everything asked for is finished.
        if not cameras and not clips:
            empty_reason = "no-cameras"
        elif not clips:
            empty_reason = "no-clips"
        elif not outstanding and not failed:
            empty_reason = "all-done"
        else:
            empty_reason = None
        running = running_cameras()
        order = {c.id: i for i, c in enumerate(cameras)}
        runs = sorted(run_states(), key=lambda s: order.get(s["camera"], len(order)))
        return {
            "site_name": config.site_name or config.site_id,
            "cameras": cameras,
            "rows": rows,
            "clips": clips,
            "outstanding": outstanding,
            "failed": failed,
            "empty_reason": empty_reason,
            "chunk_choices": CHUNK_CHOICES,
            "running": running,
            "all_running": bool(cameras) and all(c.id in running for c in cameras),
            "runs": runs,
            "parallel": config.backfill.parallel,
            "form": {},
            "error": None,
            **kw,
        }

    def page(request: Request, status_code: int = 200, **kw):
        return templates.TemplateResponse(
            request, "backfill.html", page_context(**kw), status_code=status_code
        )

    @app.get("/backfill")
    def backfill_page(request: Request):
        return page(request)

    @app.post("/backfill/start")
    def start_backfill(
        request: Request,
        cameras: list[str] = Form([]),
        camera: str = Form(""),
        start: str = Form(""),
        end: str = Form(""),
        mode: str = Form("events"),
        chunk_minutes: float = Form(15.0),
        retry_failed: str = Form("0"),
    ):
        """Scan the NVR for the range, then ingest what it registered —
        for every camera picked, side by side (CLD-317).

        All or nothing: if any picked camera is already running, nothing
        starts and the answer names it. A queued start would surprise
        the operator later; a partial one would leave the form lying
        about which cameras it began. `camera` (singular) is the form's
        old name and still counts — a script posting it must keep
        working.
        """
        wanted_ids: list[str] = []
        for cam_id in [*cameras, camera]:
            cam_id = (cam_id or "").strip()
            if cam_id and cam_id not in wanted_ids:
                wanted_ids.append(cam_id)
        form = {
            "cameras": wanted_ids,
            "start": start,
            "end": end,
            "mode": mode,
            "chunk_minutes": chunk_minutes,
            "retry_failed": retry_failed == "1",
        }
        if not wanted_ids:
            return page(request, 400, error="Choose at least one camera.", form=form)
        known = {c.id: c for c in unifi_cameras()}
        unknown = [cam_id for cam_id in wanted_ids if cam_id not in known]
        if unknown:
            return page(
                request,
                400,
                error=f"{unknown[0]!r} is not a configured UniFi camera.",
                form=form,
            )
        cams = [known[cam_id] for cam_id in wanted_ids]
        try:
            from siteloom.localtime import site_zone

            begins, finishes = parse_range(start, end, site_zone(config))
        except ValueError as exc:
            return page(request, 400, error=str(exc), form=form)

        from siteloom.backfill import BackfillRequest, new_state

        sweep = mode == "sweep"
        chunk = max(1.0, min(float(chunk_minutes), 24 * 60)) if sweep else None
        wants_retry = retry_failed == "1"
        req = BackfillRequest(
            start=begins, end=finishes, chunk_minutes=chunk, retry_failed=wants_retry
        )

        def resume_for(cam) -> str:
            # Rebuilt from the whole invocation, not a couple of
            # interesting fields: an operator continuing this run in a
            # shell must get the sweep and the retry back, or the shell
            # run means something else. One camera per line — each row
            # on /jobs is one camera's run.
            return " ".join(
                [
                    "siteloom backfill-unifi",
                    cam.id,
                    f"--start {begins.isoformat()}",
                    f"--end {finishes.isoformat()}",
                ]
                + ([f"--chunk-minutes {chunk:g}"] if chunk else [])
                + (["--retry-failed"] if wants_retry else [])
            )

        with _state["lock"]:
            busy = [c for c in cams if c.id in running_cameras()]
            if busy:
                names = ", ".join(c.name or c.id for c in busy)
                return page(
                    request,
                    409,
                    error=(
                        f"A backfill of {names} is already running. "
                        "Wait for it to finish, or pick other cameras."
                    ),
                    form=form,
                )
            states = {c.id: new_state(c, req, resume_for(c)) for c in cams}
            _state["runs"].update(states)

        def work():
            from siteloom.backfill import run_backfills
            from siteloom.ingest import IngestService
            from siteloom.progress import ProgressReporter

            try:
                service = IngestService(config)
            except Exception as exc:  # pragma: no cover — via the banner
                log.exception("backfill could not start")
                for state in states.values():
                    state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                return
            run_backfills(
                service,
                cams,
                req,
                parallel=config.backfill.parallel,
                # uvicorn owns the signals in this process; a stop reaches
                # these rows through `jobs cancel`, not Ctrl-C.
                make_reporter=lambda cam: ProgressReporter(
                    Session,
                    "backfill-unifi",
                    target=cam.id,
                    resume_command=states[cam.id]["resume"],
                    bar=False,
                    signals=False,
                ),
                states=states,
            )

        threading.Thread(target=work, name="siteloom-backfill", daemon=True).start()
        return RedirectResponse("/backfill", status_code=303)
