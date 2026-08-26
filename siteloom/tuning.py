"""Detector tuning trials — CLD-106's sandbox, CLD-102's scene analysis.

A trial answers "what would this camera see under those settings?"
without touching anything live: it reads a video, runs the real
DetectionModule over it exactly as ingest samples a stream, and writes
a report plus annotated evidence frames into its own directory. No
Event or Detection rows, no identity jobs, no vector-store access —
CLD-106's non-negotiable ("a trial must never enrol into the identity
store") is structural here: the identity pipeline is simply never
invoked, so there is nothing to leak.

Evidence over metrics (CLD-102): the operator-facing output is the
annotated frames — every track *birth*, plus a steady rhythm of
ordinary frames — with the track_eval numbers as the tie-breaker behind
them. A non-technical operator cannot evaluate "0 implausible bridges";
they can evaluate "these two boxes are the same person, and the trial
says so".

Two CLD-102 translations live here rather than in the web layer so the
CLI and tests share them:

* **Named scene presets** (`PRESETS`): an operator can accurately pick
  a *description of their scene*; they cannot pick a `match_thresh`.
  Each preset says whether it is corpus-tested or a reasoned starting
  point — the honesty matters, because only one camera has a corpus.
* **The merge↔split axis** (`axis_overrides`): one slider between two
  named failures — "more likely to merge two people" and "more likely
  to split one person" — moving `track_buffer_s`, `match_thresh` and
  `new_track_thresh` together. Exposing those three separately
  guarantees nonsensical combinations, and `match_thresh` is a
  cost-space value with no intuitive meaning at all.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from siteloom.config import DetectionConfig
from siteloom.dispatch.base import Job
from siteloom.track_eval import Observation, TrackingReport, summarize

log = logging.getLogger(__name__)

#: Cap on evidence images per trial: births first (each is a claim the
#: operator can check), ordinary rhythm frames fill the rest.
MAX_MOMENTS = 48
#: Ordinary annotated frames kept, spread evenly across the clip.
RHYTHM_FRAMES = 12
#: Mean HSV saturation below this reads as IR/grayscale footage — the
#: signal for "appearance embeddings are weak here" (CLD-129's cheap
#: half; profile *switching* stays with that ticket).
IR_SATURATION = 14.0

#: Named scene presets (CLD-102 rung 1). `tested: True` means measured
#: on the tracker corpus; everything else is a reasoned starting point
#: for a camera kind the corpus does not cover yet, and says so in the
#: UI. Settings are overrides on top of the camera's effective config.
PRESETS: dict[str, dict[str, Any]] = {
    "site-defaults": {
        "label": "Site defaults",
        "description": "What this camera runs today — the corpus-tested "
                       "BoT-SORT + ReID configuration.",
        "tested": True,
        "settings": {},
    },
    "bytetrack-2s": {
        "label": "Plain ByteTrack (pre-2026-08-25 shipped)",
        "description": "Motion-only matching, 2 s buffer. The fallback "
                       "if ReID misbehaves on a camera the corpus does "
                       "not cover — IR is the suspect case.",
        "tested": True,
        "settings": {
            "track_buffer_s": 2.0,
            "tracker": {
                "tracker_type": "bytetrack",
                "with_reid": False,
                "new_track_thresh": 0.25,
            },
        },
    },
    "close-doorway": {
        "label": "Close doorway",
        "description": "Subjects fill the frame and cross it in a "
                       "second: sample fast, drop lost tracks fast.",
        "tested": False,
        "settings": {"sample_fps": 8.0, "track_buffer_s": 2.0},
    },
    "wide-driveway": {
        "label": "Wide driveway / street",
        "description": "Small, distant subjects: a lower floor finds "
                       "them, and the identify gates upstream still "
                       "refuse crops too small to embed.",
        "tested": False,
        "settings": {"confidence": 0.35, "sample_fps": 5.0},
    },
    "garden-foliage": {
        "label": "Garden with moving foliage",
        "description": "Wind-driven clutter: demand more confidence "
                       "before anything becomes a detection at all.",
        "tested": False,
        "settings": {"confidence": 0.5,
                     "class_confidence": {"person": 0.65}},
    },
    "carpark-night": {
        "label": "Car park at night (IR)",
        "description": "IR removes the colour appearance re-ID leans "
                       "on; plain ByteTrack with a short buffer avoids "
                       "trusting appearance that is not there.",
        "tested": False,
        "settings": {
            "track_buffer_s": 2.0,
            "tracker": {
                "tracker_type": "bytetrack",
                "with_reid": False,
                "new_track_thresh": 0.25,
            },
        },
    },
}

#: The merge↔split axis (CLD-102 rung 2). Position 0 is the camera's
#: effective config untouched; negative splits more (short buffer,
#: strict matching), positive merges more (long buffer, lenient
#: matching). The three knobs move together on purpose.
AXIS: dict[int, dict[str, Any]] = {
    -2: {"track_buffer_s": 1.0,
         "tracker": {"match_thresh": 0.9, "new_track_thresh": 0.6}},
    -1: {"track_buffer_s": 2.0,
         "tracker": {"match_thresh": 0.85, "new_track_thresh": 0.55}},
    0: {},
    1: {"track_buffer_s": 6.0,
        "tracker": {"match_thresh": 0.75, "new_track_thresh": 0.4}},
    2: {"track_buffer_s": 8.0,
        "tracker": {"match_thresh": 0.7, "new_track_thresh": 0.35}},
}


def axis_overrides(position: int) -> dict[str, Any]:
    """Tracker-side overrides for one axis position; KeyError on a
    position the axis does not define — the caller shows the form
    error, nothing guesses."""
    return AXIS[position]


def apply_overrides(base: DetectionConfig, overrides: dict[str, Any]) -> DetectionConfig:
    """A DetectionConfig with `overrides` layered on: `tracker` merges
    (the DetectionOverride rule), everything else replaces. `sample_fps`
    is not a detection field and must be handled by the caller."""
    merged = base.model_copy(deep=True)
    for field, value in overrides.items():
        if field == "sample_fps" or value is None:
            continue
        if field == "tracker":
            merged.tracker = {**merged.tracker, **value}
        else:
            setattr(merged, field, value)
    return merged


def _report_dict(r: TrackingReport) -> dict[str, Any]:
    return {
        "tracks": r.tracks,
        "observations": r.observations,
        "detection_rate": round(r.detection_rate, 3),
        "median_step_iou": (
            round(r.median_step_iou, 3) if r.median_step_iou is not None else None
        ),
        "median_box_px": round(r.median_box_px, 1),
        "bridges": len(r.bridges),
        "implausible_bridges": len(r.implausible_bridges),
        "crossings": r.crossings,
        "mid_occlusion_births": len(r.mid_occlusion_births),
        "post_occlusion_births": len(r.post_occlusion_births),
    }


_COLORS = [(80, 200, 255), (120, 255, 120), (255, 160, 100),
           (255, 120, 220), (140, 140, 255), (200, 255, 60)]


def _draw(frame, dets: list[dict]) -> None:
    import cv2

    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        tid = d.get("track_id")
        color = _COLORS[tid % len(_COLORS)] if tid is not None else (160, 160, 160)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class_name']}" + (f" #{tid}" if tid is not None else "")
        label += f" {d['confidence']:.2f}"
        cv2.putText(frame, label, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _open_writer(path: Path, frame, sample_fps: float):
    """An annotated-video writer, or None. avc1 plays in a <video> tag;
    mp4v is the fallback where the build lacks it — best-effort either
    way, the evidence JPEGs are the product that must exist."""
    import cv2

    h, w = frame.shape[:2]
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), sample_fps, (w, h)
        )
        if writer.isOpened():
            return writer
        writer.release()
    return None


def run_trial(
    video_path: str | Path,
    det_cfg: DetectionConfig,
    sample_fps: float,
    out_dir: str | Path,
    *,
    module: Any | None = None,
    group_for: Callable[[str], list[str]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run one detector configuration over one video, sandboxed.

    Writes `report.json`, birth/rhythm evidence JPEGs and (best-effort)
    an annotated mp4 into `out_dir`, and returns the report. `module`
    is injectable so tests run without model weights; `group_for` maps
    a class to its event class-group (defaults to each class alone) so
    metrics are per group, never pooled across e.g. people and cars.
    """
    import cv2

    from siteloom.modules.detection import DetectionModule

    video_path = Path(video_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    module = module or DetectionModule(det_cfg)
    group_for = group_for or (lambda cls: [cls])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(native / sample_fps))
    expected = max(1, total // step) if total else 0
    rhythm_every = max(1, (expected or RHYTHM_FRAMES) // RHYTHM_FRAMES)

    by_group: dict[str, list[Observation]] = {}
    group_frames: dict[str, int] = {}
    classes: dict[str, dict[str, list[float]]] = {}
    seen_tracks: set[tuple[str, int]] = set()
    moments: list[dict[str, Any]] = []
    births = 0
    saturation: list[float] = []
    luma: list[float] = []

    writer = None
    video_out: str | None = None
    frames = index = 0
    started = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step:
            index += 1
            continue
        t = index / native
        index += 1
        frames += 1
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        result = module.process(Job(module="detection", payload={
            "image_jpeg": buf.tobytes(),
            "camera_id": "detector-trial",
            "timestamp": t,
            "sample_fps": sample_fps,
        }))
        dets = result.get("detections", [])

        if frames % 10 == 1:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation.append(float(hsv[:, :, 1].mean()))
            luma.append(float(hsv[:, :, 2].mean()))

        frame_groups: set[str] = set()
        new_track = False
        for d in dets:
            cls = d["class_name"]
            stats = classes.setdefault(cls, {"widths": [], "confs": []})
            x1, y1, x2, y2 = d["bbox"]
            stats["widths"].append(float(x2 - x1))
            stats["confs"].append(float(d["confidence"]))
            if d.get("track_id") is None:
                continue
            group = "|".join(group_for(cls))
            frame_groups.add(group)
            by_group.setdefault(group, []).append(
                Observation(int(d["track_id"]), t, (x1, y1, x2, y2))
            )
            if (group, int(d["track_id"])) not in seen_tracks:
                seen_tracks.add((group, int(d["track_id"])))
                new_track = True
        for group in frame_groups:
            group_frames[group] = group_frames.get(group, 0) + 1

        annotated = None
        if dets:
            annotated = frame.copy()
            _draw(annotated, dets)
        # Births are the claims worth checking; rhythm frames give the
        # clip's ordinary texture around them.
        keep = None
        if new_track and annotated is not None:
            keep, births = "birth", births + 1
        elif frames % rhythm_every == 1 and annotated is not None:
            keep = "rhythm"
        if keep and len(moments) < MAX_MOMENTS:
            name = f"moment-{frames:05d}.jpg"
            cv2.imwrite(str(out / name), annotated)
            moments.append({"file": name, "t": round(t, 2), "kind": keep,
                            "detections": len(dets)})

        if writer is None:
            writer = _open_writer(out / "annotated.mp4", frame, sample_fps)
            video_out = "annotated.mp4" if writer is not None else None
        if writer is not None:
            writer.write(annotated if annotated is not None else frame)

        if progress is not None:
            progress(frames, expected)
    cap.release()
    if writer is not None:
        writer.release()

    interval = step / native
    groups = {
        key: _report_dict(summarize(
            obs,
            frames=frames,
            frames_with_detection=group_frames.get(key, 0),
            sample_interval_s=interval,
        ))
        for key, obs in by_group.items()
    }
    scene_classes = {
        cls: {
            "count": len(v["confs"]),
            "median_width_px": round(statistics.median(v["widths"]), 1),
            "median_confidence": round(statistics.median(v["confs"]), 3),
            "p25_confidence": round(
                statistics.quantiles(v["confs"], n=4)[0], 3
            ) if len(v["confs"]) >= 4 else round(min(v["confs"]), 3),
        }
        for cls, v in classes.items()
    }
    sat = statistics.mean(saturation) if saturation else 0.0
    report = {
        "source": video_path.name,
        "sample_fps": sample_fps,
        "settings": {
            "model": det_cfg.model,
            "confidence": det_cfg.confidence,
            "class_confidence": det_cfg.class_confidence,
            "tracker": det_cfg.tracker,
            "track_buffer_s": det_cfg.track_buffer_s,
        },
        "frames": frames,
        "native_fps": round(native, 2),
        "seconds": round(time.time() - started, 1),
        "groups": groups,
        "scene": {
            "classes": scene_classes,
            "saturation_mean": round(sat, 1),
            "luma_mean": round(statistics.mean(luma), 1) if luma else 0.0,
            "ir": sat < IR_SATURATION,
        },
        "moments": moments,
        "video": video_out,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def compare_reports(baseline: dict, candidate: dict) -> dict[str, Any]:
    """Per-group verdicts between two persisted trial reports of the
    same source, using the harness's own verdict arithmetic
    (`track_eval.verdict_from_counts`) so the lab cannot rule
    differently than `track_ab.py` on the same numbers."""
    from siteloom.track_eval import verdict_from_counts

    def switch_like(g: dict) -> int:
        return (g["implausible_bridges"] + g["mid_occlusion_births"]
                + g["post_occlusion_births"])

    out: dict[str, Any] = {}
    base_groups = baseline.get("groups", {})
    cand_groups = candidate.get("groups", {})
    for key in sorted(set(base_groups) | set(cand_groups)):
        b, c = base_groups.get(key), cand_groups.get(key)
        if b is None or c is None:
            out[key] = {"verdict": "only in one run"}
            continue
        out[key] = {
            "verdict": verdict_from_counts(
                switch_like(b), b["tracks"], switch_like(c), c["tracks"]
            ),
            "switch_like": (switch_like(b), switch_like(c)),
            "tracks": (b["tracks"], c["tracks"]),
        }
    return out


def recommend(
    report: dict[str, Any], sample_fps: float
) -> list[dict[str, Any]]:
    """Setting suggestions read off one trial's report, each with its
    reason and its basis — `measured` (a corpus-backed rule) or
    `heuristic` (a reasoned starting point). Suggestions are proposals:
    nothing here applies anything (CLD-102 — auto-apply is the wrong
    default for a security system).
    """
    out: list[dict[str, Any]] = []
    groups = report.get("groups", {})
    scene = report.get("scene", {})

    person = groups.get("person")
    if person and person["median_step_iou"] is not None:
        if person["median_step_iou"] < 0.5 and sample_fps < 5.0:
            out.append({
                "field": "sample_fps", "current": sample_fps, "suggested": 5.0,
                "basis": "measured",
                "reason": (
                    f"Consecutive person boxes overlap only "
                    f"{person['median_step_iou']:.2f} between samples — the "
                    "sampling is too coarse for walking pace, which is the "
                    "CLD-5 fragmentation mode; no tracker setting fixes it."
                ),
            })

    if scene.get("ir"):
        out.append({
            "field": None, "current": None, "suggested": None,
            "basis": "heuristic",
            "reason": (
                f"This footage reads as IR/grayscale (mean saturation "
                f"{scene['saturation_mean']}) — appearance re-ID leans on "
                "colour that is not there. Consider the plain-ByteTrack "
                "preset for night, and contribute a clip from this camera "
                "to the tracker corpus before trusting ReID settings on it."
            ),
        })

    for key, g in groups.items():
        if g["tracks"] >= 3 and g["observations"] / g["tracks"] < 3:
            cls = key.split("|")[0]
            info = scene.get("classes", {}).get(cls)
            if info and info["count"] >= 5:
                out.append({
                    "field": f"class_confidence.{cls}",
                    "current": None,
                    "suggested": info["median_confidence"],
                    "basis": "heuristic",
                    "reason": (
                        f"{g['tracks']} {key} tracks averaged under 3 "
                        "frames each — marginal detections churning into "
                        "trackless fragments; a floor near this class's "
                        f"median confidence ({info['median_confidence']}) "
                        "keeps the solid ones."
                    ),
                })

    for key, g in groups.items():
        if g["mid_occlusion_births"] + g["post_occlusion_births"] > 0:
            out.append({
                "field": None, "current": None, "suggested": None,
                "basis": "measured",
                "reason": (
                    f"{key}: {g['mid_occlusion_births']} mid + "
                    f"{g['post_occlusion_births']} post occlusion births — "
                    "subjects overlap on this camera. This window is worth "
                    "adding to the tracker corpus (the occlusion metrics "
                    "were built from exactly such a clip)."
                ),
            })

    info = scene.get("classes", {}).get("person")
    if info and info["median_width_px"] < 64:
        out.append({
            "field": None, "current": None, "suggested": None,
            "basis": "heuristic",
            "reason": (
                f"People are small here (median {info['median_width_px']} px "
                "wide) — near the identify_min_crop_px gate (48 px), so many "
                "frames will be detected but refused identification. "
                "Expected, not a fault; a camera position question, not a "
                "settings one."
            ),
        })

    if not groups:
        out.append({
            "field": None, "current": None, "suggested": None,
            "basis": "heuristic",
            "reason": (
                "Nothing was tracked in this clip. Either the scene was "
                "empty, the confidence floor is above everything in it, or "
                "the classes present are not in detection.classes — re-run "
                "with a lower floor to tell which."
            ),
        })
    return out
