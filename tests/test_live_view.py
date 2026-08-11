"""Live view: the shared frame hub and its web routes.

The hub is exercised with an "rtsp" camera whose source is a local file —
OpenCV opens both the same way, so the reader/fan-out logic runs without
a camera on the network.
"""

from __future__ import annotations

import threading

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


class _RecordingCond:
    """A feed's Condition, minus the thread: keeps every predicate handed to it."""

    def __init__(self):
        self.predicates = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def wait_for(self, predicate, timeout=None):
        self.predicates.append(predicate)
        return predicate()

    def wait(self, timeout=None):
        return False

    def notify_all(self):
        pass


class _StubFeed:
    """Only the state `LiveHub.frames` reads — no RTSP, no reader thread."""

    def __init__(self, seq, frame, alive=True):
        self.cond = _RecordingCond()
        self.seq = seq
        self.frame = frame
        self.clients = 0
        self.last_client = 0.0
        self.stopping = threading.Event()
        self.alive = alive

    def is_alive(self):
        return self.alive


def test_frames_predicate_binds_the_feed_it_was_made_for(config, monkeypatch):
    """Each wait_for predicate asks *its own* feed about *its own* seq (CLD-99).

    `frames()` rebinds `feed` (a dead reader is replaced mid-stream) and
    `seen` (every frame advances it), so a closure over those names would
    report on whatever the loop holds when it finally runs — on a
    multi-camera site, another camera's reader. Two feeds are the minimum
    that can tell the two behaviours apart: with one, the last iteration
    is the only iteration and a late-binding closure looks correct.
    """
    hub = LiveHub(config)
    first = _StubFeed(seq=1, frame=b"\xff\xd8first")
    second = _StubFeed(seq=5, frame=b"\xff\xd8second")
    queue = [first, second]

    def fake_ensure(camera_id):
        return queue.pop(0) if queue else second

    monkeypatch.setattr(hub, "_ensure", fake_ensure)

    gen = hub.frames("cam1")
    assert next(gen) == first.frame  # served off the first feed...
    first.alive = False  # ...which then dies under the viewer
    assert next(gen) == second.frame  # ...and is rebuilt onto the second
    gen.close()

    # The loop's three waits, in order, and the (feed, seen) each was
    # created with: first at seq 0, first again after the frame at seq 1,
    # then the replacement feed starting over at seq 0.
    predicates = first.cond.predicates + second.cond.predicates
    bindings = [(first, 0), (first, 1), (second, 0)]
    assert len(predicates) == len(bindings)

    for predicate, (feed, seen) in zip(predicates, bindings, strict=True):
        other = second if feed is first else first
        # Its own feed at its own seq means "nothing new" — no matter what
        # the other feed is doing. Reverse the two and the answer flips.
        feed.seq, other.seq = seen, seen + 1
        assert predicate() is False
        feed.seq, other.seq = seen + 1, seen
        assert predicate() is True


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
