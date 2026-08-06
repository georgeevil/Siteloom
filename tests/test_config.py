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
    assert cam.modules == ["detection"]
    assert cam.sample_fps == 2.0
    assert cam.require_zone is False


def test_site_config_defaults():
    cfg = SiteConfig(site_id="s", cameras=[CameraConfig(id="c", source="x")])
    assert cfg.detection.device == "mps"
    assert cfg.storage.db_url.startswith("sqlite")
