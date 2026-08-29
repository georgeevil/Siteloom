"""Per-camera detection settings (CLD-101).

The override follows EventRulesOverride's rules — only non-None fields
apply — with one deliberate exception: `tracker` merges over the site
dict instead of replacing it, because a camera overriding one knob must
not silently drop the site's fuse_score.
"""

from __future__ import annotations

import yaml

from siteloom.config import CameraConfig, DetectionConfig, DetectionOverride


def cam(**override) -> CameraConfig:
    return CameraConfig(
        id="c1", adapter="file", source="x",
        detection=DetectionOverride(**override) if override else None,
    )


def test_no_override_returns_the_site_config_itself():
    site = DetectionConfig()
    assert site.for_camera(cam()) is site


def test_only_non_none_fields_apply():
    site = DetectionConfig(confidence=0.4, model="yolo11s.pt")
    eff = site.for_camera(cam(confidence=0.6))
    assert eff.confidence == 0.6
    assert eff.model == "yolo11s.pt"      # untouched
    assert site.confidence == 0.4          # the site object is never edited


def test_tracker_merges_instead_of_replacing():
    """A camera turning one knob keeps the site's other decisions —
    fuse_score off is the CLD-5 rule and must survive any override."""
    site = DetectionConfig(tracker={"fuse_score": False, "match_thresh": 0.8})
    eff = site.for_camera(cam(tracker={"match_thresh": 0.9}))
    assert eff.tracker == {"fuse_score": False, "match_thresh": 0.9}


def test_track_buffer_derives_from_the_camera_effective_seconds():
    from siteloom.modules.detection import tracker_config_path

    site = DetectionConfig()
    eff = site.for_camera(cam(track_buffer_s=2.0))
    data = yaml.safe_load(tracker_config_path(eff, 5.0).read_text())
    assert data["track_buffer"] == 10


def test_poisoning_and_structural_fields_are_not_overridable():
    """crop_margin changes the embedding space (a migration, CLD-106);
    classes and device describe the site and the machine. None of them
    may vary per camera."""
    fields = set(DetectionOverride.model_fields)
    assert "crop_margin" not in fields
    assert "classes" not in fields
    assert "device" not in fields


def test_module_honors_the_per_camera_config():
    """The same module instance applies each camera's own floors."""
    import cv2
    import numpy as np
    from types import SimpleNamespace

    from siteloom.dispatch import Job
    from siteloom.modules.detection import DetectionModule

    site = DetectionConfig(confidence=0.4, device="cpu")
    strict = site.for_camera(cam(confidence=0.7))
    module = DetectionModule(site, per_camera={"c1": strict})

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
    module._models["other"] = FakeModel()
    ok, jpeg = cv2.imencode(".jpg", np.zeros((60, 80, 3), dtype=np.uint8))
    assert ok

    def detect(camera_id):
        return module.process(Job(module="detection", payload={
            "image_jpeg": jpeg.tobytes(), "camera_id": camera_id,
        }))["detections"]

    assert detect("c1") == []          # 0.5 < the camera's 0.7 floor
    assert len(detect("other")) == 1   # 0.5 clears the site's 0.4


def test_tracker_paths_are_per_camera():
    from siteloom.modules.detection import DetectionModule

    site = DetectionConfig()
    tuned = site.for_camera(cam(tracker={"match_thresh": 0.9}))
    module = DetectionModule(site, per_camera={"c1": tuned})
    assert module._tracker_for("c1", 5.0) != module._tracker_for("other", 5.0)
    assert yaml.safe_load(
        module._tracker_for("c1", 5.0).read_text()
    )["match_thresh"] == 0.9
