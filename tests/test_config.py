from pathlib import Path

import pytest

from siteloom.config import CameraConfig, SiteConfig, ZoneConfig, load_config

EXAMPLE = Path(__file__).parent.parent / "config" / "site.example.yaml"


def test_example_config_loads():
    cfg = load_config(EXAMPLE)
    assert cfg.site_id == "kai-apartments"
    assert cfg.backend.kind == "local"
    assert cfg.cameras[0].adapter == "file"
    assert "detection" in cfg.cameras[0].modules


def test_zone_points_must_be_normalized():
    with pytest.raises(ValueError):
        ZoneConfig(name="bad", points=[(0, 0), (2.0, 0.5), (1, 1)])


def test_zone_needs_three_points():
    with pytest.raises(ValueError):
        ZoneConfig(name="line", points=[(0, 0), (1, 1)])


def test_camera_defaults():
    cam = CameraConfig(id="c1", source="rtsp://x")
    assert cam.modules == ["detection", "identity"]
    assert cam.sample_fps == 2.0
    assert cam.require_zone is False


def test_site_config_defaults():
    cfg = SiteConfig(site_id="s", cameras=[CameraConfig(id="c", source="x")])
    assert cfg.detection.device == "mps"
    assert cfg.storage.db_url.startswith("sqlite")


def test_recognition_api_defaults_closed():
    # Biometric surface (NFR5, CLD-47): disabled by default, keyless
    # serving is an explicit opt-in, rate limiting on by default.
    api = SiteConfig(site_id="s").integrations.recognition_api
    assert api.enabled is False
    assert api.api_key == ""
    assert api.allow_open is False
    assert api.rate_limit_per_minute == 60


def test_event_rules_defaults():
    cfg = SiteConfig(site_id="s")
    assert cfg.events.min_detections == 3
    assert cfg.events.min_confidence == 0.5
    assert cfg.events.stitch_gap_s == 15.0
    assert cfg.events.identify_only_significant is True
    assert cfg.events.group_for("truck") == ["car", "truck", "bus"]
    assert cfg.events.group_for("person") == ["person"]


def test_event_rules_per_camera_override_merges_only_set_fields():
    from siteloom.config import EventRulesOverride

    cfg = SiteConfig(
        site_id="s",
        cameras=[
            CameraConfig(
                id="c",
                source="x",
                events=EventRulesOverride(min_detections=1, stitch_gap_s=5.0),
            )
        ],
    )
    eff = cfg.events.for_camera(cfg.cameras[0])
    assert eff.min_detections == 1
    assert eff.stitch_gap_s == 5.0
    assert eff.min_confidence == cfg.events.min_confidence  # untouched default
    # No override -> the site object itself (no copy churn).
    plain = CameraConfig(id="p", source="x")
    assert cfg.events.for_camera(plain) is cfg.events


def test_event_rules_override_reaches_the_identify_gates():
    """CLD-39: the gates that actually control identity churn are
    per-camera, not just the significance four."""
    from siteloom.config import EventRulesOverride

    cfg = SiteConfig(
        site_id="s",
        cameras=[
            CameraConfig(
                id="doorway",
                source="x",
                events=EventRulesOverride(
                    stitch_min_iou=0.2,
                    stitch_candidates=9,
                    identify_min_confidence=0.75,
                    identify_min_crop_px=96,
                    identify_only_significant=False,
                ),
            )
        ],
    )
    eff = cfg.events.for_camera(cfg.cameras[0])
    assert eff.stitch_min_iou == 0.2
    assert eff.stitch_candidates == 9
    assert eff.identify_min_confidence == 0.75
    assert eff.identify_min_crop_px == 96
    # False is a value, not "unset" — the merge must not skip it.
    assert eff.identify_only_significant is False
    assert cfg.events.identify_only_significant is True  # site untouched


def test_identity_threshold_prefers_the_camera_then_the_identifier():
    from siteloom.config import CameraIdentityOverride

    cfg = SiteConfig(
        site_id="s",
        cameras=[
            CameraConfig(
                id="doorway",
                source="x",
                identity=CameraIdentityOverride(thresholds={"face": 0.44}),
            ),
            CameraConfig(id="quiet", source="x"),
        ],
    )
    doorway, quiet = cfg.cameras
    assert cfg.identity.threshold_for("face", doorway) == 0.44
    # Only the named identifier moves; the others keep their own scale.
    assert cfg.identity.threshold_for("person", doorway) == 0.80
    assert cfg.identity.threshold_for("face", quiet) == 0.36
    assert cfg.identity.threshold_for("face") == 0.36
    # An identifier the YAML never named (auto-added class) has no
    # site-wide value here — the registry's default applies — but a
    # camera may still pin one.
    assert cfg.identity.threshold_for("deer", quiet) is None
    doorway.identity.thresholds["deer"] = 0.9
    assert cfg.identity.threshold_for("deer", doorway) == 0.9


def test_camera_identity_thresholds_must_be_similarities():
    from siteloom.config import CameraIdentityOverride

    with pytest.raises(ValueError):
        CameraIdentityOverride(thresholds={"face": 42})


def test_per_camera_overrides_round_trip_through_yaml(tmp_path):
    from siteloom.config import (
        CameraIdentityOverride,
        EventRulesOverride,
        save_config,
    )

    cfg = SiteConfig(
        site_id="s",
        cameras=[
            CameraConfig(
                id="doorway",
                source="x",
                events=EventRulesOverride(identify_min_crop_px=96),
                identity=CameraIdentityOverride(thresholds={"face": 0.44}),
            )
        ],
    )
    path = tmp_path / "site.yaml"
    save_config(cfg, path)
    cam = load_config(path).cameras[0]
    assert cam.events.identify_min_crop_px == 96
    assert cam.events.min_detections is None  # unset stays unset
    assert cam.identity.thresholds == {"face": 0.44}


def test_event_rules_round_trip_through_yaml(tmp_path):
    from siteloom.config import save_config

    cfg = SiteConfig(site_id="s")
    cfg.events.min_detections = 7
    path = tmp_path / "site.yaml"
    save_config(cfg, path)
    again = load_config(path)
    assert again.events.min_detections == 7
    assert again.events.class_groups == [["car", "truck", "bus"]]
