"""NVR backfill: two-phase scan/process, dedupe, resumability (PRD §6.6).

A fake NVR adapter stands in for uiprotect — it serves canned recording
events and "exports" clips by copying the synthetic sample video — so the
whole path (scan -> clip download -> live pipeline -> store) runs without
a console or model weights.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone

import pytest

from siteloom.adapters.unifi import RecordingEvent
from siteloom.backfill import UnifiBackfill
from siteloom.config import CameraConfig, IdentityConfig, SiteConfig, StorageConfig
from siteloom.dispatch import LocalBackend
from siteloom.ingest import IngestService
from siteloom.store import BackfillClip, Event

T0 = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)


class StubDetector:
    def process(self, job):
        return {
            "detections": [
                {
                    "class_name": "person",
                    "confidence": 0.9,
                    "bbox": [10.0, 10.0, 50.0, 90.0],
                    "track_id": 7,
                    "zones": [],
                    "crop_jpeg": b"\xff\xd8fakejpg",
                }
            ]
        }


class FakeNvrAdapter:
    """Serves canned RecordingEvents; exports clips from the sample video."""

    def __init__(self, sample_video, events, fail_ids=()):
        self.sample_video = sample_video
        self.events = events
        self.fail_ids = set(fail_ids)
        self.downloads = []

    def connect(self):
        pass

    def close(self):
        pass

    def list_recording_events(self, start, end, stream_id=None):
        return [
            ev
            for ev in self.events
            if (stream_id is None or ev.camera_id == stream_id)
            and ev.start >= start
            and ev.end <= end
        ]

    def download_clip(self, stream_id, start, end, output):
        for ev in self.events:
            if ev.id in self.fail_ids and abs(
                (ev.start - start).total_seconds()
            ) <= 10:
                raise IOError("export failed")
        self.downloads.append((stream_id, start, end))
        shutil.copy(self.sample_video, output)
        return output


def make_events():
    # Two visits an hour apart, plus a smart-detect window nested inside
    # the first motion window (Protect fires both for the same activity).
    return [
        RecordingEvent("ev-1", "nvr-cam-1", "motion", T0, T0 + timedelta(seconds=20)),
        RecordingEvent(
            "ev-1s",
            "nvr-cam-1",
            "smartDetectZone",
            T0 + timedelta(seconds=2),
            T0 + timedelta(seconds=10),
        ),
        RecordingEvent(
            "ev-2",
            "nvr-cam-1",
            "motion",
            T0 + timedelta(hours=1),
            T0 + timedelta(hours=1, seconds=20),
        ),
    ]


@pytest.fixture
def env(sample_video, tmp_path, monkeypatch):
    config = SiteConfig(
        site_id="test-site",
        cameras=[
            CameraConfig(
                id="front",
                adapter="unifi",
                source="nvr-cam-1",
                sample_fps=5.0,
                modules=["detection"],
            )
        ],
        identity=IdentityConfig(enabled=False),
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/bf.db", media_dir=str(tmp_path / "media")
        ),
    )
    dispatcher = LocalBackend()
    dispatcher.register("detection", StubDetector())
    service = IngestService(config, dispatcher=dispatcher)
    adapter = FakeNvrAdapter(sample_video, make_events())
    monkeypatch.setattr(
        "siteloom.backfill.build_adapter", lambda cam, cfg: adapter
    )
    return service, adapter


def scan_range(backfill, **kwargs):
    return backfill.scan(T0 - timedelta(minutes=5), T0 + timedelta(hours=2), **kwargs)


def test_scan_registers_and_dedupes(env):
    service, adapter = env
    backfill = UnifiBackfill(service, service.config.cameras[0])

    first = scan_range(backfill)
    # ev-1s is fully inside ev-1's padded window — no new footage.
    assert first.added == 2
    assert first.skipped == 1
    assert first.pending == 2

    again = scan_range(backfill)
    assert again.added == 0
    assert again.pending == 2


def test_process_runs_live_pipeline_with_historical_time(env):
    service, adapter = env
    backfill = UnifiBackfill(service, service.config.cameras[0])
    scan_range(backfill)

    result = backfill.process()
    assert result.processed == 2
    assert result.failed == 0
    assert result.remaining == 0
    assert result.frames > 0

    with service.Session() as session:
        clips = session.query(BackfillClip).order_by(BackfillClip.start).all()
        assert [c.status for c in clips] == ["done", "done"]
        events = session.query(Event).order_by(Event.first_seen).all()
        # Same track id in both clips, an hour apart: the link gap must
        # split them into two visits instead of stapling them together.
        assert len(events) == 2
        # Frames carry clip time, not wall time (5s pad before T0).
        assert abs((events[0].first_seen - T0.replace(tzinfo=None)).total_seconds()) < 30
        assert events[1].first_seen - events[0].first_seen >= timedelta(minutes=50)


def test_process_is_resumable(env):
    service, adapter = env
    backfill = UnifiBackfill(service, service.config.cameras[0])
    scan_range(backfill)

    first = backfill.process(limit=1)
    assert first.processed == 1
    assert first.remaining == 1

    downloads_after_first = len(adapter.downloads)
    rest = backfill.process()
    assert rest.processed == 1
    assert rest.remaining == 0
    # The finished clip is not downloaded or ingested again.
    assert len(adapter.downloads) == downloads_after_first + 1


def test_failed_is_not_pending(env):
    service, adapter = env
    adapter.fail_ids = {"ev-1"}
    backfill = UnifiBackfill(service, service.config.cameras[0])
    scan_range(backfill)

    result = backfill.process()
    assert result.processed == 1
    assert result.failed == 1
    assert result.failed_total == 1
    assert result.remaining == 0

    # A plain rerun leaves the failure alone...
    rerun = backfill.process()
    assert rerun.processed == 0
    assert rerun.retried == 0

    # ...retry_failed re-queues it, and attempts records the history.
    adapter.fail_ids = set()
    retry = backfill.process(retry_failed=True)
    assert retry.retried == 1
    assert retry.processed == 1
    with service.Session() as session:
        clip = session.query(BackfillClip).filter_by(external_id="ev-1").one()
        assert clip.status == "done"
        assert clip.attempts == 2


def test_requires_unifi_camera(env):
    service, _ = env
    cam = CameraConfig(id="f", adapter="file", source="x")
    with pytest.raises(ValueError):
        UnifiBackfill(service, cam)
