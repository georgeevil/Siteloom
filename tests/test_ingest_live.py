"""Live-camera ingest: per-camera workers and reconnect-on-drop (CLD-14).

Uses a fake live adapter registered under the "rtsp" key — no network,
no OpenCV capture — so the loop logic is tested deterministically.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pytest

import siteloom.ingest as ingest_mod
from siteloom.adapters import ADAPTERS
from siteloom.adapters.base import CameraAdapter, Frame, StreamInfo
from siteloom.config import CameraConfig, IdentityConfig, SiteConfig, StorageConfig
from siteloom.dispatch import LocalBackend
from siteloom.ingest import IngestService
from siteloom.store import Detection


class FakeLiveSource:
    """Yields `per_connection` frames ~10ms apart, then ends (a "drop")."""

    def __init__(self, source_id: str, per_connection: int | None):
        self.source_id = source_id
        self._n = per_connection

    def frames(self, sample_fps: float):
        yielded = 0
        while self._n is None or yielded < self._n:
            yield Frame(
                image=np.zeros((24, 32, 3), dtype=np.uint8),
                timestamp=datetime.now(timezone.utc),
                source_id=self.source_id,
            )
            yielded += 1
            time.sleep(0.01)


class FakeLiveAdapter(CameraAdapter):
    """Counts connects per source so tests can assert reconnection."""

    connects: dict[str, int] = defaultdict(int)
    frames_per_connection: int | None = 3

    def __init__(self, source: str = "", **_kw):
        self._source = source

    def connect(self) -> None:
        type(self).connects[self._source] += 1

    def list_streams(self) -> list[StreamInfo]:
        return [StreamInfo(id=self._source, name=self._source, kind="live")]

    def get_live_stream(self, stream_id: str) -> FakeLiveSource:
        return FakeLiveSource(stream_id, self.frames_per_connection)


class CountingDetector:
    """One detection per frame, tagged with the source camera id."""

    def process(self, job):
        return {
            "detections": [
                {
                    "class_name": "person",
                    "confidence": 0.9,
                    "bbox": [10.0, 10.0, 50.0, 90.0],
                    "track_id": 1,
                    "zones": [],
                    "crop_jpeg": b"\xff\xd8fakejpg",
                }
            ]
        }


def make_service(tmp_path, camera_ids: list[str]) -> IngestService:
    config = SiteConfig(
        site_id="test-site",
        cameras=[
            CameraConfig(
                id=cam_id,
                adapter="rtsp",
                source=f"src-{cam_id}",
                sample_fps=5.0,
                modules=["detection"],
            )
            for cam_id in camera_ids
        ],
        identity=IdentityConfig(enabled=False),
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/test.db", media_dir=str(tmp_path / "media")
        ),
    )
    dispatcher = LocalBackend()
    dispatcher.register("detection", CountingDetector())
    return IngestService(config, dispatcher=dispatcher)


@pytest.fixture(autouse=True)
def fake_live(monkeypatch):
    FakeLiveAdapter.connects = defaultdict(int)
    FakeLiveAdapter.frames_per_connection = 3
    monkeypatch.setitem(ADAPTERS, "rtsp", FakeLiveAdapter)
    monkeypatch.setattr(ingest_mod, "LIVE_BACKOFF_S", 0.01)
    monkeypatch.setattr(ingest_mod, "LIVE_BACKOFF_MAX_S", 0.05)


def detections_by_camera(service):
    from siteloom.store import Event

    with service.Session() as session:
        rows = (
            session.query(Event.camera_id, Detection.id)
            .join(Detection, Detection.event_id == Event.id)
            .all()
        )
    counts: dict[str, int] = defaultdict(int)
    for camera_id, _ in rows:
        counts[camera_id] += 1
    return counts


def test_reconnects_until_max_frames(tmp_path):
    """A drop every 3 frames must not end the run — the worker reconnects
    until the per-camera frame budget is met."""
    service = make_service(tmp_path, ["cam1"])
    service.run(max_frames=10)

    assert detections_by_camera(service)["cam1"] == 10
    # 10 frames at 3 per connection = 4 connects (fresh adapter each time).
    assert FakeLiveAdapter.connects["src-cam1"] == 4


def test_two_live_cameras_run_concurrently(tmp_path):
    """With never-ending streams, both cameras only produce frames if they
    genuinely run in parallel — sequentially, cam2 would never start."""
    FakeLiveAdapter.frames_per_connection = None  # never drop
    service = make_service(tmp_path, ["cam1", "cam2"])

    stopper = threading.Timer(0.5, service.stop)
    stopper.start()
    try:
        service.run()
    finally:
        stopper.cancel()

    counts = detections_by_camera(service)
    assert counts["cam1"] > 0
    assert counts["cam2"] > 0


def test_stop_ends_promptly_between_reconnects(tmp_path, monkeypatch):
    """stop() during a backoff wait must end the run immediately, not
    after the full backoff interval."""
    monkeypatch.setattr(ingest_mod, "LIVE_BACKOFF_S", 30.0)
    monkeypatch.setattr(ingest_mod, "LIVE_BACKOFF_MAX_S", 30.0)
    service = make_service(tmp_path, ["cam1"])

    runner = threading.Thread(target=service.run)
    runner.start()
    # Wait until the worker has consumed its first connection's frames
    # and is sitting in the 30s backoff wait.
    deadline = time.monotonic() + 5.0
    while not FakeLiveAdapter.connects["src-cam1"] and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)

    stopped_at = time.monotonic()
    service.stop()
    runner.join(timeout=3.0)
    assert not runner.is_alive()
    assert time.monotonic() - stopped_at < 3.0


def test_local_backend_serializes_per_module():
    """Warm module instances are not thread-safe; concurrent submits to
    the same module must never overlap."""
    from siteloom.dispatch import Job

    class Overlappy:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self._lock = threading.Lock()

        def process(self, job):
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self._lock:
                self.active -= 1
            return {}

    backend = LocalBackend()
    module = Overlappy()
    backend.register("m", module)

    threads = [
        threading.Thread(
            target=lambda: backend.submit_and_wait(Job(module="m", payload={}))
        )
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert module.max_active == 1
