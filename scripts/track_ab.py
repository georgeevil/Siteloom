#!/usr/bin/env python3
"""A/B tracker configurations over real footage (CLD-98).

Tracker settings trade ID switches against fragmentation, and every knob
improves one of them. Argument cannot settle that; this can.

    scripts/track_ab.py fetch                    # pull clips from Protect
    scripts/track_ab.py run                      # every config, every clip
    scripts/track_ab.py run --config botsort-reid
    scripts/track_ab.py check                    # regression gate, exits 1
    scripts/track_ab.py suggest                  # clips operators have flagged

The corpus is `config/track_corpus.yaml`: which minute of which camera,
and what failure it is a case of. Footage is cached under
~/.cache/siteloom/track-corpus and never enters the repository.

`check` is the one worth wiring into a habit — it runs only the shipped
configuration and fails if any clip exceeds the expectations recorded
beside it, so a tracker change that quietly reintroduces CLD-97 says so.

Needs model weights and (for `fetch`) NVR access, which is why this is a
script and not a test. The arithmetic it depends on lives in
`siteloom.track_eval` and is unit-tested there.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from siteloom.track_eval import (  # noqa: E402
    Observation,
    TrackingReport,
    compare,
    summarize,
)

CACHE = Path.home() / ".cache" / "siteloom" / "track-corpus"
DEFAULT_CORPUS = REPO / "config" / "track_corpus.yaml"


def load_corpus(path: Path) -> dict:
    corpus = yaml.safe_load(path.read_text())
    if not corpus.get("clips"):
        sys.exit(f"{path} defines no clips")
    return corpus


def clip_path(clip: dict) -> Path:
    return CACHE / f"{clip['id']}.mp4"


# -- fetch ----------------------------------------------------------------


def fetch(corpus: dict, config_path: str) -> int:
    """Pull any clip not already cached. Idempotent by design — the
    corpus grows, and re-fetching a 16 MB minute to add one case is
    pointless."""
    from siteloom.adapters.unifi import UniFiProtectAdapter
    from siteloom.config import load_config

    cfg = load_config(config_path)
    wanted = [c for c in corpus["clips"] if not clip_path(c).is_file()]
    if not wanted:
        print(f"all {len(corpus['clips'])} clips already cached in {CACHE}")
        return 0

    CACHE.mkdir(parents=True, exist_ok=True)
    adapter = UniFiProtectAdapter(unifi=cfg.unifi)
    adapter.connect()
    try:
        for clip in wanted:
            camera = next(
                (c for c in cfg.cameras if c.id == clip["camera"]), None
            )
            if camera is None:
                print(f"  {clip['id']}: no camera {clip['camera']!r} in config — skipped")
                continue
            out = clip_path(clip)
            start = clip["start"].replace(tzinfo=timezone.utc)
            end = clip["end"].replace(tzinfo=timezone.utc)
            print(f"  {clip['id']}: {start} .. {end}")
            adapter.download_clip(camera.source, start, end, out)
            print(f"    -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    finally:
        adapter.close()
    return 0


# -- run ------------------------------------------------------------------


def detection_config(base_config: str, tracker: dict):
    from siteloom.config import load_config

    cfg = load_config(base_config).detection
    # fuse_score must stay off at our sampling rate (CLD-5) regardless of
    # what a variant sets, unless the variant is explicitly testing it.
    cfg.tracker = {"fuse_score": False, **cfg.tracker, **tracker}
    return cfg


def run_clip(clip: dict, det_cfg, sample_fps: float, bridge_gap_s: float) -> TrackingReport:
    import cv2

    from siteloom.dispatch.base import Job
    from siteloom.modules.detection import DetectionModule

    path = clip_path(clip)
    if not path.is_file():
        raise FileNotFoundError(f"{path} — run `track_ab.py fetch` first")

    module = DetectionModule(det_cfg)
    cap = cv2.VideoCapture(str(path))
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / sample_fps))

    observations: list[Observation] = []
    frames = with_det = index = 0
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
        result = module.process(Job(
            module="detection",
            payload={"image_jpeg": buf.tobytes(), "camera_id": clip["id"],
                     "timestamp": t},
        ))
        people = [
            d for d in result.get("detections", [])
            if d.get("class_name") == "person" and d.get("track_id") is not None
        ]
        if people:
            with_det += 1
        for d in people:
            x1, y1, x2, y2 = d["bbox"]
            observations.append(
                Observation(int(d["track_id"]), t, (x1, y1, x2, y2))
            )
    cap.release()

    return summarize(
        observations,
        frames=frames,
        frames_with_detection=with_det,
        # The real spacing between the frames handed to the detector,
        # which is `step` source frames at the clip's native rate — not
        # the requested fps, since step is rounded to a whole frame.
        sample_interval_s=step / native,
        bridge_gap_s=bridge_gap_s,
        seconds=time.time() - started,
    )


def line(name: str, r: TrackingReport) -> str:
    worst = r.worst_bridge
    worst_s = (
        f"{worst.gap_s:.1f}s/{worst.jump_widths:.1f}w" if worst else "—"
    )
    step = f"{r.median_step_iou:.2f}" if r.median_step_iou is not None else "—"
    return (
        f"  {name:<20} tracks {r.tracks:>3}  det {r.detection_rate:>4.0%}  "
        f"step-IoU {step:>4}  bridges {len(r.bridges):>2}"
        f" ({len(r.implausible_bridges)} implausible)  worst {worst_s:>11}"
    )


def run(corpus: dict, config_path: str, only: str | None, out_path: str | None) -> int:
    configs = corpus.get("configs") or {"shipped": {}}
    if only:
        if only not in configs:
            sys.exit(f"no config {only!r}; have {', '.join(configs)}")
        configs = {only: configs[only], **({"shipped": configs["shipped"]}
                                           if only != "shipped" and "shipped" in configs
                                           else {})}
    results: dict[str, dict[str, TrackingReport]] = {}
    for clip in corpus["clips"]:
        print(f"\n{clip['id']}  ({clip['camera']})")
        if clip.get("case"):
            print(f"  case: {' '.join(clip['case'].split())}")
        for name, tracker in configs.items():
            try:
                report = run_clip(
                    clip, detection_config(config_path, tracker),
                    corpus.get("sample_fps", 5.0),
                    corpus.get("bridge_gap_s", 2.0),
                )
            except Exception as exc:
                print(f"  {name:<20} FAILED: {type(exc).__name__}: {exc}")
                continue
            results.setdefault(clip["id"], {})[name] = report
            print(line(name, report))

        # A verdict, not just rows — reading one improved number and
        # shipping is the failure this whole script exists to prevent.
        per_clip = results.get(clip["id"], {})
        baseline = per_clip.get("shipped")
        if baseline:
            for name, report in per_clip.items():
                if name == "shipped":
                    continue
                v = compare(baseline, report)
                print(f"    vs shipped — {name}: {v['verdict']}"
                      f"  (implausible {v['implausible_bridges'][0]}→{v['implausible_bridges'][1]},"
                      f" tracks {v['tracks'][0]}→{v['tracks'][1]})")

    if out_path:
        payload = {
            clip_id: {
                name: {
                    "tracks": r.tracks,
                    "detection_rate": r.detection_rate,
                    "bridges": len(r.bridges),
                    "implausible_bridges": len(r.implausible_bridges),
                    "median_step_iou": r.median_step_iou,
                }
                for name, r in per.items()
            }
            for clip_id, per in results.items()
        }
        Path(out_path).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out_path}")
    return 0


# -- check ----------------------------------------------------------------


def check(corpus: dict, config_path: str) -> int:
    """Regression gate: shipped config against each clip's expectations."""
    failures = []
    for clip in corpus["clips"]:
        expect = clip.get("expect") or {}
        if not expect:
            continue
        report = run_clip(
            clip, detection_config(config_path, corpus["configs"]["shipped"]),
            corpus.get("sample_fps", 5.0), corpus.get("bridge_gap_s", 2.0),
        )
        print(line(clip["id"], report))
        got_bridges = len(report.implausible_bridges)
        if got_bridges > expect.get("max_implausible_bridges", 10**9):
            failures.append(
                f"{clip['id']}: {got_bridges} implausible bridges, "
                f"expected at most {expect['max_implausible_bridges']}"
            )
        if report.tracks > expect.get("max_tracks", 10**9):
            failures.append(
                f"{clip['id']}: {report.tracks} tracks, "
                f"expected at most {expect['max_tracks']}"
            )
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nall clips within expectations")
    return 0


# -- suggest --------------------------------------------------------------


def suggest(corpus: dict, config_path: str) -> int:
    """Print corpus entries for events operators marked multi-subject.

    Closes the loop the marker exists for (CLD-103). An operator flags
    "more than one person here" while reviewing; that mark already
    carries a camera and a time window, which is a clip definition. This
    turns them into YAML to paste, rather than asking anyone to translate
    a database row into a config by hand.

    Prints rather than writes: which flagged events are worth keeping as
    permanent regression cases is a judgement, and a corpus that grows
    itself would fill with duplicates of the same easy failure.
    """
    from datetime import timedelta

    from siteloom.config import load_config
    from siteloom.store import Event, get_session, make_engine

    cfg = load_config(config_path)
    have = {c["id"] for c in corpus["clips"]}
    Session = get_session(make_engine(cfg.storage.db_url))
    with Session() as session:
        flagged = (
            session.query(Event)
            .filter(Event.multi_subject.is_(True))
            .order_by(Event.multi_subject_at.desc())
            .limit(50)
            .all()
        )
        rows = [
            (e.id, e.camera_id, e.first_seen, e.last_seen, e.detection_count)
            for e in flagged
        ]

    if not rows:
        print("No events flagged as multi-subject yet.")
        print("Mark one with \"more than one person here\" on an event page.")
        return 0

    print(f"# {len(rows)} flagged event(s). Paste what is worth keeping into")
    print(f"# {DEFAULT_CORPUS.name}, then run `track_ab.py fetch`.\n")
    for event_id, camera, first, last, dets in rows:
        clip_id = f"event-{event_id}-multi-subject"
        if clip_id in have:
            print(f"# {clip_id}: already in the corpus\n")
            continue
        # Pad, because the interesting part is usually the approach and
        # the hole rather than the detections themselves.
        start = (first - timedelta(seconds=15)).replace(microsecond=0)
        end = (last + timedelta(seconds=15)).replace(microsecond=0)
        print(f"""  - id: {clip_id}
    camera: {camera}
    start: {start}
    end: {end}
    case: >
      Operator marked event {event_id} as containing more than one
      subject: {dets} detections under a single track. Describe what was
      actually in it before relying on this as a regression case.
    expect:
      max_implausible_bridges: 0
""")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("fetch", "run", "check", "suggest"))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--config", dest="site_config", default="site.yaml",
                        help="site config, for camera/NVR details and the "
                             "detection settings variants are layered over")
    parser.add_argument("--only", dest="only",
                        help="run a single named tracker config")
    parser.add_argument("--json", dest="out", help="write results here")
    args = parser.parse_args()

    corpus = load_corpus(Path(args.corpus))
    if args.command == "fetch":
        return fetch(corpus, args.site_config)
    if args.command == "check":
        return check(corpus, args.site_config)
    if args.command == "suggest":
        return suggest(corpus, args.site_config)
    return run(corpus, args.site_config, args.only, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
