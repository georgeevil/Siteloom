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


# -- identifier defaults survive being named (CLD-125) ----------------------
#
# `identifiers` has a default_factory, so a config that spells out its
# identifiers used to replace the built-ins wholesale and every unstated
# field fell back to the bare field default — 0.0 margin, 1 sighting, the
# pre-CLD-41 behavior. Naming `face:` to change its threshold therefore
# switched off consistency gating without saying so.


def _identifiers(yaml_text: str, tmp_path: Path):
    path = tmp_path / "site.yaml"
    path.write_text(yaml_text)
    return load_config(path).identity.identifiers


def test_naming_an_identifier_keeps_the_gates_it_did_not_mention(tmp_path):
    idents = _identifiers(
        """
site_id: s
identity:
  identifiers:
    face:
      algo: face
      applies_to: [person]
      threshold: 0.42
""",
        tmp_path,
    )
    face = idents["face"]
    assert face.threshold == 0.42  # what the operator said wins
    assert face.min_margin == 0.05  # ... and what they did not say is kept
    assert face.min_sightings == 2


def test_an_explicit_value_still_beats_the_default(tmp_path):
    """The whole reason the merge runs on the raw mapping: after
    validation, absent and `1` are the same value, and an operator who
    wants first-sighting minting must still be able to ask for it."""
    idents = _identifiers(
        """
site_id: s
identity:
  identifiers:
    face:
      algo: face
      applies_to: [person]
      min_sightings: 1
      min_margin: 0.0
""",
        tmp_path,
    )
    assert idents["face"].min_sightings == 1
    assert idents["face"].min_margin == 0.0


def test_naming_vehicle_keeps_its_plate_settings(tmp_path):
    """The CLD-125 overlay covers the plate fields too: tuning the
    vehicle threshold must not silently turn plate OCR off or un-ration
    the CLD-130 cadence cap."""
    idents = _identifiers(
        """
site_id: s
identity:
  identifiers:
    vehicle:
      algo: generic
      applies_to: [car, truck]
      threshold: 0.9
""",
        tmp_path,
    )
    vehicle = idents["vehicle"]
    assert vehicle.threshold == 0.9
    assert vehicle.plate_ocr is True
    assert vehicle.plate_ocr_interval_s == 1.0


def test_per_camera_plate_floors_load_from_yaml(tmp_path):
    """The CLD-128 shape, straight from the issue: a camera says what its
    pixels can carry, everything unstated inherits the identifier."""
    path = tmp_path / "site.yaml"
    path.write_text(
        """
site_id: s
cameras:
  - id: backyard-puerta
    adapter: file
    source: x
    identity:
      plate_floors: {min_width_px: 30, min_char_confidence: 0.85}
"""
    )
    config = load_config(path)
    floors = config.identity.plate_floors_for("vehicle", config.cameras[0])
    assert floors.min_width_px == 30
    assert floors.min_char_confidence == 0.85
    assert floors.min_chars == 4


def test_omitting_an_identifier_still_removes_it(tmp_path):
    """Merging is per named key. Wholesale replacement got one thing
    right — a config that lists only `face` runs only `face` — and that
    has to survive."""
    idents = _identifiers(
        """
site_id: s
identity:
  identifiers:
    face:
      algo: face
      applies_to: [person]
""",
        tmp_path,
    )
    assert set(idents) == {"face"}


def test_an_identifier_with_no_built_in_is_left_alone(tmp_path):
    idents = _identifiers(
        """
site_id: s
identity:
  identifiers:
    dog:
      algo: generic
      applies_to: [dog]
      threshold: 0.7
""",
        tmp_path,
    )
    assert idents["dog"].threshold == 0.7
    # Not silently gated: a new identifier is the operator's own, and
    # auto-added classes mint on the first sighting by design.
    assert idents["dog"].min_sightings == 1


def test_a_config_naming_nothing_keeps_every_built_in(tmp_path):
    idents = _identifiers("site_id: s\n", tmp_path)
    assert set(idents) == {"face", "person", "vehicle"}
    assert idents["person"].min_sightings == 2
    assert idents["vehicle"].min_sightings == 1  # deliberate, PRD §6.4


def test_naming_an_identifier_keeps_the_learning_gates_too(tmp_path):
    """The learn gates (CLD-139) join the same merge, and for the same
    reason: they are the fields a site will never restate, and losing
    them silently restores the ungated accretion they exist to stop."""
    idents = _identifiers(
        """
site_id: s
identity:
  identifiers:
    person:
      algo: generic
      applies_to: [person]
      threshold: 0.75
""",
        tmp_path,
    )
    person = idents["person"]
    assert person.threshold == 0.75
    assert person.learn_min_quality == 0.6
    assert person.learn_max_per_event == 3


def test_switching_one_learning_gate_off_keeps_the_others(tmp_path):
    """The documented rollback: `learn_max_per_event: 0` restores
    pre-CLD-139 accretion for that identifier without a redeploy — and
    must not take the margin, the sightings or the quality floor with
    it."""
    idents = _identifiers(
        """
site_id: s
identity:
  identifiers:
    person:
      algo: generic
      applies_to: [person]
      learn_max_per_event: 0
""",
        tmp_path,
    )
    person = idents["person"]
    assert person.learn_max_per_event == 0  # asked for, and honoured
    assert person.learn_min_quality == 0.6
    assert person.min_margin == 0.02
    assert person.min_sightings == 2


def test_an_auto_added_class_is_gated_by_the_field_defaults(tmp_path):
    """A class nobody configured gets the conservative values rather than
    nothing: adding a class to `detection.classes` is meant to be the
    only step, so its identifier cannot arrive with learning wide open."""
    from siteloom.config import IdentityConfig
    from siteloom.identity.registry import IdentifierRegistry

    registry = IdentifierRegistry(IdentityConfig(auto_add_classes=True))
    _, deer = registry.identifiers_for("deer")[0]
    assert deer.learn_min_quality == 0.6
    assert deer.learn_max_per_event == 3


def test_saved_config_round_trips_its_gates(tmp_path):
    """The console writes a full dump, so a saved file states every
    field — the merge must be a no-op over it, not a re-defaulting."""
    from siteloom.config import save_config

    cfg = SiteConfig(site_id="s")
    cfg.identity.identifiers["face"].min_sightings = 3
    path = tmp_path / "site.yaml"
    save_config(cfg, path)
    assert load_config(path).identity.identifiers["face"].min_sightings == 3


def test_backfill_parallel_is_zero_or_more():
    from siteloom.config import BackfillConfig

    assert BackfillConfig().parallel == 2
    assert BackfillConfig(parallel=0).parallel == 0
    with pytest.raises(ValueError):
        BackfillConfig(parallel=-1)
