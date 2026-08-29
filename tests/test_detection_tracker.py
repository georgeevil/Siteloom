"""Tracker config materialization (no YOLO involved — the file is what
ultralytics consumes; its content is the contract)."""

from __future__ import annotations

import yaml

from siteloom.config import DetectionConfig
from siteloom.modules.detection import TRACKER_DEFAULTS, tracker_config_path


def test_default_disables_fuse_score():
    path = tracker_config_path(DetectionConfig())
    data = yaml.safe_load(path.read_text())
    # The one non-library default: fused cost breaks track confirmation
    # at our few-fps sampling rate (CLD-5).
    assert data["fuse_score"] is False
    # BoT-SORT with appearance-verified re-acquisition (2026-08-25,
    # corpus-measured): motion-only matching is what minted occlusion
    # phantoms, and the 4 s buffer below is only safe alongside it.
    assert data["tracker_type"] == "botsort"
    assert data["with_reid"] is True
    for key, value in TRACKER_DEFAULTS.items():
        if key != "fuse_score":
            assert data[key] == value


def test_lost_tracks_do_not_coast_for_seconds():
    """`track_buffer` counts sampled frames, and ultralytics' 30 assumes
    30 fps video. At the few fps Siteloom samples, that is six seconds of
    a dead track waiting to adopt the next person who walks near its
    Kalman prediction — which is exactly how one event came to hold two
    people (CLD-96). The number is only meaningful as a duration, so the
    configured quantity is `track_buffer_s` and the materialized frame
    count must give the same coast duration at every sampling rate.
    """
    cfg = DetectionConfig()
    for sample_fps in (2.0, 5.0, 10.0):
        data = yaml.safe_load(
            tracker_config_path(cfg, sample_fps).read_text()
        )
        coast_s = data["track_buffer"] / sample_fps
        assert coast_s == cfg.track_buffer_s, (
            f"at {sample_fps} fps a lost track survives {coast_s:.1f}s "
            f"(track_buffer={data['track_buffer']})"
        )
    # A caller with no meaningful sampling rate gets the frames-based
    # fallback, which is the same 4 s at the live cameras' 5 fps.
    fallback = yaml.safe_load(tracker_config_path(cfg).read_text())
    assert fallback["track_buffer"] == TRACKER_DEFAULTS["track_buffer"] == 20


def test_track_buffer_is_still_overridable_per_deployment():
    """An explicit frame count is a deliberate departure and wins over
    the seconds-based derivation."""
    cfg = DetectionConfig(tracker={"track_buffer": 45})
    path = tracker_config_path(cfg, sample_fps=5.0)
    assert yaml.safe_load(path.read_text())["track_buffer"] == 45


def test_overrides_merge_and_hash_to_distinct_files():
    default = tracker_config_path(DetectionConfig())
    tuned = tracker_config_path(
        DetectionConfig(tracker={"fuse_score": False, "match_thresh": 0.9})
    )
    assert tuned != default
    assert yaml.safe_load(tuned.read_text())["match_thresh"] == 0.9
    # Same config maps back to the same file.
    assert tracker_config_path(DetectionConfig()) == default


# -- per-class confidence ---------------------------------------------------


class _FakeBoxes:
    """Just enough of ultralytics' Boxes for DetectionModule.process."""

    def __init__(self, rows):
        import numpy as np

        self.cls = np.array([r[0] for r in rows])
        self.conf = np.array([r[1] for r in rows])
        self.xyxy = np.array([r[2] for r in rows], dtype=float)
        self.id = np.array([r[3] for r in rows])

    def __len__(self):
        return len(self.cls)


def _detect_with(cfg, rows):
    """Run DetectionModule.process with a stubbed model over `rows` of
    (class_idx, confidence, xyxy, track_id)."""
    import cv2
    import numpy as np
    from types import SimpleNamespace

    from siteloom.dispatch import Job
    from siteloom.modules.detection import DetectionModule

    module = DetectionModule(cfg)
    seen = {}

    class FakeModel:
        def track(self, image, **kwargs):
            seen.update(kwargs)
            return [
                SimpleNamespace(
                    names={0: "person", 1: "dog"}, boxes=_FakeBoxes(rows)
                )
            ]

    module._models["cam1"] = FakeModel()
    ok, jpeg = cv2.imencode(".jpg", np.zeros((60, 80, 3), dtype=np.uint8))
    assert ok
    job = Job(
        module="detection",
        payload={"image_jpeg": jpeg.tobytes(), "camera_id": "cam1"},
    )
    return module.process(job)["detections"], seen


def test_per_class_confidence_filters_output():
    from siteloom.config import DetectionConfig

    cfg = DetectionConfig(
        confidence=0.4, class_confidence={"dog": 0.8}, device="cpu"
    )
    rows = [
        (0, 0.45, (1, 1, 20, 40), 1),  # person above global floor -> kept
        (0, 0.35, (1, 1, 20, 40), 2),  # person below global floor -> dropped
        (1, 0.60, (1, 1, 20, 40), 3),  # dog below its per-class 0.8 -> dropped
        (1, 0.85, (1, 1, 20, 40), 4),  # dog above 0.8 -> kept
    ]
    detections, seen = _detect_with(cfg, rows)
    assert [(d["class_name"], d["confidence"]) for d in detections] == [
        ("person", 0.45),
        ("dog", 0.85),
    ]
    # The model itself ran at the lowest applicable threshold, so the
    # per-class filter had everything to work with.
    assert seen["conf"] == 0.4


def test_class_confidence_can_lower_below_global():
    from siteloom.config import DetectionConfig

    cfg = DetectionConfig(
        confidence=0.5, class_confidence={"person": 0.3}, device="cpu"
    )
    rows = [(0, 0.35, (1, 1, 20, 40), 1), (1, 0.45, (1, 1, 20, 40), 2)]
    detections, seen = _detect_with(cfg, rows)
    assert [d["class_name"] for d in detections] == ["person"]
    assert seen["conf"] == 0.3  # detector floor follows the lowest override
