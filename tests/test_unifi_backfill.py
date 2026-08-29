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


# -- several cameras side by side (CLD-317) -------------------------------


def _two_camera_env(sample_video, tmp_path, monkeypatch, name, fail_ids=()):
    """Two UniFi cameras over one shared pipeline, one fake NVR adapter
    *each* — the runner builds an adapter per camera and must keep it
    that way, so the fake does too."""
    root = tmp_path / name
    root.mkdir()
    config = SiteConfig(
        site_id="test-site",
        cameras=[
            CameraConfig(id="front", adapter="unifi", source="nvr-cam-1", sample_fps=5.0, modules=["detection"]),
            CameraConfig(id="gate", adapter="unifi", source="nvr-cam-2", sample_fps=5.0, modules=["detection"]),
        ],
        identity=IdentityConfig(enabled=False),
        storage=StorageConfig(db_url=f"sqlite:///{root}/bf.db", media_dir=str(root / "media")),
    )
    dispatcher = LocalBackend()
    dispatcher.register("detection", StubDetector())
    service = IngestService(config, dispatcher=dispatcher)
    gate_events = [
        RecordingEvent("g-1", "nvr-cam-2", "motion", T0 + timedelta(minutes=5), T0 + timedelta(minutes=5, seconds=20)),
        RecordingEvent("g-2", "nvr-cam-2", "motion", T0 + timedelta(minutes=40), T0 + timedelta(minutes=40, seconds=20)),
    ]
    adapters = {
        "nvr-cam-1": FakeNvrAdapter(sample_video, make_events(), fail_ids=fail_ids),
        "nvr-cam-2": FakeNvrAdapter(sample_video, gate_events),
    }
    monkeypatch.setattr("siteloom.backfill.build_adapter", lambda cam, cfg: adapters[cam.source])
    return service, adapters


def _event_snapshot(Session):
    """Events by content — the same shape the resume-equivalence harness
    compares, `confidence_sum` rounded because merge order can differ
    float-associatively between two runs."""
    with Session() as session:
        return sorted(
            (
                e.camera_id,
                e.class_name,
                e.first_seen,
                e.last_seen,
                e.detection_count,
                round(e.confidence_sum, 6),
                e.significant,
            )
            for e in session.query(Event).all()
        )


def _reporter(service):
    from siteloom.progress import ProgressReporter

    return lambda cam: ProgressReporter(
        service.Session, "backfill-unifi", target=cam.id, bar=False, signals=False
    )


def _request():
    from siteloom.backfill import BackfillRequest

    return BackfillRequest(start=T0 - timedelta(minutes=5), end=T0 + timedelta(hours=2))


def test_two_cameras_in_parallel_land_where_one_after_the_other_lands(
    sample_video, tmp_path, monkeypatch
):
    """The differential that protects the parallel path: the same corpus
    through two threads must produce the same events as through one."""
    from siteloom.backfill import run_backfills
    from siteloom.store import OperationRun

    seq, _ = _two_camera_env(sample_video, tmp_path, monkeypatch, "sequential")
    for cam in seq.config.cameras:
        runner = UnifiBackfill(seq, cam)
        runner.scan(_request().start, _request().end)
        runner.process()

    par, _ = _two_camera_env(sample_video, tmp_path, monkeypatch, "parallel")
    states = run_backfills(
        par, par.config.cameras, _request(), parallel=2, make_reporter=_reporter(par)
    )
    assert {cam_id: s["status"] for cam_id, s in states.items()} == {
        "front": "complete",
        "gate": "complete",
    }
    assert states["front"]["processed"] == 2 and states["gate"]["processed"] == 2
    assert _event_snapshot(par.Session) == _event_snapshot(seq.Session)
    assert len(_event_snapshot(par.Session)) == 4
    with par.Session() as session:
        runs = session.query(OperationRun).all()
    # One row per camera, each closed out complete.
    assert sorted((r.target, r.status) for r in runs) == [
        ("front", "complete"),
        ("gate", "complete"),
    ]


def test_a_slot_cap_of_one_still_finishes_every_camera(sample_video, tmp_path, monkeypatch):
    """The waiting camera heartbeats on the slot instead of deadlocking on it."""
    from siteloom.backfill import run_backfills

    service, _ = _two_camera_env(sample_video, tmp_path, monkeypatch, "capped")
    states = run_backfills(
        service, service.config.cameras, _request(), parallel=1, make_reporter=_reporter(service)
    )
    assert [s["status"] for s in states.values()] == ["complete", "complete"]
    assert len(_event_snapshot(service.Session)) == 4


def test_a_failing_camera_does_not_sink_its_siblings(sample_video, tmp_path, monkeypatch):
    from siteloom.backfill import run_backfills

    service, adapters = _two_camera_env(sample_video, tmp_path, monkeypatch, "onebad")

    def broken(*args, **kwargs):
        raise IOError("NVR refused the export list")

    adapters["nvr-cam-1"].list_recording_events = broken
    states = run_backfills(
        service, service.config.cameras, _request(), parallel=2, make_reporter=_reporter(service)
    )
    assert states["front"]["status"] == "failed"
    assert "NVR refused" in states["front"]["error"]
    assert states["gate"]["status"] == "complete"
    assert states["gate"]["processed"] == 2


def test_the_same_camera_twice_is_one_run(sample_video, tmp_path, monkeypatch):
    from siteloom.backfill import run_backfills

    service, _ = _two_camera_env(sample_video, tmp_path, monkeypatch, "dupe")
    front = service.config.cameras[0]
    states = run_backfills(
        service, [front, front], _request(), parallel=2, make_reporter=_reporter(service)
    )
    assert list(states) == ["front"]
    assert states["front"]["processed"] == 2
