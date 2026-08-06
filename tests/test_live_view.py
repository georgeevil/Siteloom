"""Live view: the shared frame hub and its web routes.

The hub is exercised with an "rtsp" camera whose source is a local file —
OpenCV opens both the same way, so the reader/fan-out logic runs without
a camera on the network.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.web.app import create_app
from siteloom.web.live import LiveHub


@pytest.fixture
def config(sample_video, tmp_path):
    return SiteConfig(
        site_id="test-site",
        site_name="Test Site",
        cameras=[
            CameraConfig(id="cam1", name="Front", adapter="rtsp", source=str(sample_video)),
            CameraConfig(id="filecam", adapter="file", source=str(sample_video)),
        ],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/live.db", media_dir=str(tmp_path / "media")
        ),
    )


def test_hub_lists_only_live_cameras(config):
    hub = LiveHub(config)
    assert [c.id for c in hub.cameras()] == ["cam1"]


def test_hub_snapshot_and_shared_frames(config):
    hub = LiveHub(config)
    try:
        jpeg = hub.snapshot("cam1")
        assert jpeg is not None and jpeg.startswith(b"\xff\xd8")

        gen = hub.frames("cam1")
        frames = [next(gen), next(gen)]
        gen.close()
        assert all(f.startswith(b"\xff\xd8") for f in frames)
        # One reader serves everyone: a second viewer must not have
        # spawned a second feed thread for the same camera.
        assert len(hub._feeds) <= 1
    finally:
        hub.stop()


def test_hub_unknown_camera(config):
    hub = LiveHub(config)
    with pytest.raises(KeyError):
        next(hub.frames("nope"))
    with pytest.raises(KeyError):
        hub.snapshot("filecam")  # file cameras have no live stream


def test_live_routes(config):
    app = create_app(config)
    with TestClient(app) as client:
        page = client.get("/live")
        assert page.status_code == 200
        assert "Front" in page.text
        assert "/live/cam1/stream.mjpeg" in page.text

        snap = client.get("/live/cam1/snapshot.jpg")
        assert snap.status_code == 200
        assert snap.headers["content-type"] == "image/jpeg"
        assert snap.content.startswith(b"\xff\xd8")

        assert client.get("/live/nope/stream.mjpeg").status_code == 404
        assert client.get("/live/filecam/snapshot.jpg").status_code == 404
