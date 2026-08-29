"""Day/night detection profiles (CLD-129).

The monitor's hysteresis is unit-tested on synthetic readings; the
ingest wiring is tested by driving _process_frame with synthetic frames
and a stub detector that records which profile each payload carried;
the module's composite key is tested like the per-camera overrides —
all weightless.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from siteloom.config import (
    CameraConfig,
    DetectionConfig,
    DetectionOverride,
    IdentityConfig,
    SiteConfig,
    StorageConfig,
    load_config,
    save_config,
)
from siteloom.scene import IR_SATURATION, ProfileMonitor, mean_saturation

DT = 0.2  # 5 fps


def feed(monitor, readings, t0=0.0):
    return [
        monitor.observe(t0 + i * DT, sat) for i, sat in enumerate(readings)
    ]


def test_a_flip_needs_a_sustained_streak_not_a_glint():
    monitor = ProfileMonitor(confirm_frames=5)
    # Four dark frames — a headlight sweep — then colour again: no flip.
    assert feed(monitor, [5.0] * 4 + [60.0] * 3) == ["day"] * 7
    # Five sustained dark frames flip it on the fifth.
    out = feed(monitor, [5.0] * 6, t0=10.0)
    assert out == ["day"] * 4 + ["night"] * 2


def test_dusk_cannot_flap_the_tracker():
    """Every flip restarts the camera's track ids — the dwell floor is
    what keeps a threshold-straddling dusk from doing that per minute."""
    monitor = ProfileMonitor(confirm_frames=2, min_dwell_s=60.0)
    feed(monitor, [5.0] * 3)                      # flips to night at ~0.4s
    assert monitor.profile == "night"
    out = feed(monitor, [60.0] * 10, t0=1.0)      # bright again, too soon
    assert set(out) == {"night"}                  # held by the dwell
    out = feed(monitor, [60.0] * 3, t0=120.0)     # well past the dwell
    assert out[-1] == "day"


def test_a_backfill_seam_resets_the_streak_not_the_profile():
    monitor = ProfileMonitor(confirm_frames=3)
    feed(monitor, [5.0] * 2, t0=100.0)            # streak building
    out = feed(monitor, [5.0] * 2, t0=0.0)        # time jumps backwards
    assert out == ["day", "day"]                  # streak restarted
    assert feed(monitor, [5.0], t0=0.4) == ["night"]  # its own evidence flips


def test_gray_frames_read_as_ir_and_colored_ones_do_not():
    gray = np.full((40, 40, 3), 90, dtype=np.uint8)
    assert mean_saturation(gray) < IR_SATURATION
    colored = np.zeros((40, 40, 3), dtype=np.uint8)
    colored[:, :, 2] = 200  # strong red
    assert mean_saturation(colored) > IR_SATURATION


# -- config layering -------------------------------------------------------


def cam(**kw) -> CameraConfig:
    return CameraConfig(id="c1", adapter="file", source="x", **kw)


def test_night_layers_over_the_day_effective_settings():
    site = DetectionConfig(confidence=0.4)
    camera = cam(
        detection=DetectionOverride(confidence=0.6,
                                    tracker={"match_thresh": 0.85}),
        night=DetectionOverride(tracker={"with_reid": False}),
    )
    day = site.for_camera(camera)
    night = site.for_camera(camera, "night")
    assert day.confidence == 0.6
    assert night.confidence == 0.6           # day tuning carries into night
    assert night.tracker["match_thresh"] == 0.85
    assert night.tracker["with_reid"] is False
    assert "with_reid" not in day.tracker


def test_the_default_profile_keeps_every_existing_caller_unchanged():
    site = DetectionConfig()
    camera = cam(night=DetectionOverride(confidence=0.9))
    assert site.for_camera(camera) is site   # day, no day override


def test_the_night_profile_round_trips_through_the_config_file(tmp_path):
    config = SiteConfig(
        site_id="t",
        cameras=[cam(night=DetectionOverride(confidence=0.7))],
        storage=StorageConfig(db_url="sqlite:///x.db", media_dir="m"),
    )
    path = tmp_path / "site.yaml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.cameras[0].night.confidence == 0.7


# -- module + ingest wiring ------------------------------------------------


def test_the_module_keys_night_on_a_composite_camera_id():
    import cv2
    from types import SimpleNamespace

    from siteloom.dispatch import Job
    from siteloom.modules.detection import DetectionModule

    site = DetectionConfig(confidence=0.4, device="cpu")
    night_cfg = site.for_camera(cam(night=DetectionOverride(confidence=0.7)),
                                "night")
    module = DetectionModule(site, per_camera={"c1#night": night_cfg})

    class Boxes:
        cls = np.array([0])
        conf = np.array([0.5])
        xyxy = np.array([[1.0, 1.0, 20.0, 40.0]])
        id = np.array([1])

        def __len__(self):
            return 1

    class FakeModel:
        def track(self, image, **kw):
            return [SimpleNamespace(names={0: "person"}, boxes=Boxes())]

    module._models["c1"] = FakeModel()
    module._models["c1#night"] = FakeModel()
    ok, jpeg = cv2.imencode(".jpg", np.zeros((60, 80, 3), dtype=np.uint8))
    assert ok

    def detect(profile):
        payload = {"image_jpeg": jpeg.tobytes(), "camera_id": "c1"}
        if profile:
            payload["profile"] = profile
        return module.process(Job(module="detection", payload=payload))[
            "detections"
        ]

    assert len(detect(None)) == 1   # 0.5 clears the day floor of 0.4
    assert detect("night") == []    # night demands 0.7


def test_ingest_switches_profiles_from_the_footage(tmp_path):
    """Sixteen gray frames through a night-configured camera: the first
    ride the day profile while the streak builds, the rest are night —
    and a camera with no night profile never pays for the probe."""
    from siteloom.adapters.base import Frame
    from siteloom.dispatch import LocalBackend
    from siteloom.ingest import IngestService

    seen: list[str | None] = []

    class RecordingDetector:
        def process(self, job):
            seen.append(job.payload.get("profile"))
            return {"detections": []}

    config = SiteConfig(
        site_id="t",
        cameras=[
            CameraConfig(
                id="dark", adapter="file", source="x",
                modules=["detection"],
                night=DetectionOverride(confidence=0.7),
            ),
        ],
        identity=IdentityConfig(enabled=False),
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/t.db", media_dir=str(tmp_path / "m")
        ),
    )
    dispatcher = LocalBackend()
    dispatcher.register("detection", RecordingDetector())
    service = IngestService(config, dispatcher=dispatcher)

    from siteloom.scene import CONFIRM_FRAMES

    gray = np.full((40, 40, 3), 90, dtype=np.uint8)
    t0 = datetime(2026, 8, 20, 3, 0, 0, tzinfo=timezone.utc)
    for i in range(CONFIRM_FRAMES + 3):
        service._process_frame(
            config.cameras[0],
            Frame(image=gray, timestamp=t0 + timedelta(seconds=i * DT),
                  source_id="dark"),
        )
    assert seen[: CONFIRM_FRAMES - 1] == ["day"] * (CONFIRM_FRAMES - 1)
    assert set(seen[CONFIRM_FRAMES:]) == {"night"}

    # A camera without a night profile carries no profile at all.
    seen.clear()
    config.cameras[0].night = None
    service._profiles.clear()
    service._process_frame(
        config.cameras[0], Frame(image=gray, timestamp=t0, source_id="dark")
    )
    assert seen == [None]
