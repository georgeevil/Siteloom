"""`siteloom run` heartbeats an OperationRun row (CLD-15).

A day-long soak that dies at hour 3 must not keep looking healthy, so live
ingest goes through ProgressReporter like every other long operation. What
is worth testing here is what the reporter cannot test for itself: that the
per-camera worker threads (CLD-14) tick *one* reporter without losing
counts, and that a stop reaches them through it.

Reuses the fake live adapter from test_ingest_live.py — no network, no
OpenCV, so the loop stays deterministic.
"""

from __future__ import annotations

import json
import threading
import time

from siteloom.progress import ProgressReporter
from siteloom.store import OperationRun

from test_ingest_live import (  # noqa: F401 — fake_live is an autouse fixture
    FakeLiveAdapter,
    fake_live,
    make_service,
)


def latest_run(service) -> OperationRun:
    with service.Session() as session:
        return (
            session.query(OperationRun).order_by(OperationRun.id.desc()).first()
        )


def counters_of(run: OperationRun) -> dict[str, int]:
    return json.loads(run.counters)


def reporter(service, **kw) -> ProgressReporter:
    return ProgressReporter(service.Session, "run", target="test-site", bar=False, **kw)


def test_run_records_a_job_row_with_per_camera_counters(tmp_path):
    """The row is the whole point: it is what `siteloom jobs` reads."""
    service = make_service(tmp_path, ["cam1"])
    with reporter(service, resume_command="siteloom run -c site.yaml") as progress:
        service.run(max_frames=6, progress=progress)

    run = latest_run(service)
    assert run.kind == "run"
    assert run.status == "complete"
    assert run.resume_command == "siteloom run -c site.yaml"
    # pid/host are what OperationRun.is_stale checks to tell a dead run
    # from a slow one; without them a kill -9 leaves a healthy row.
    assert run.pid > 0 and run.host

    counters = counters_of(run)
    assert counters["frames"] == 6
    assert counters["detections"] == 6  # CountingDetector: one per frame
    assert counters["cam1.frames"] == 6
    # 3 frames per connection, so the stream drops once before the budget
    # is met — a soak's blind spots have to be countable.
    assert counters["reconnects"] == 1


def test_aggregate_counters_survive_concurrent_cameras(tmp_path):
    """Two workers ticking one reporter must not lose updates.

    `counters[k] += v` is read-modify-write; unsynchronized, the totals
    silently drift below the per-camera parts. Asserting the identity
    rather than a fixed number keeps this a race test, not a timing test.
    """
    FakeLiveAdapter.frames_per_connection = None  # never drop
    service = make_service(tmp_path, ["cam1", "cam2"])

    with reporter(service) as progress:
        stopper = threading.Timer(0.5, service.stop)
        stopper.start()
        try:
            service.run(progress=progress)
        finally:
            stopper.cancel()

    counters = counters_of(latest_run(service))
    assert counters["cam1.frames"] > 0 and counters["cam2.frames"] > 0
    assert counters["frames"] == counters["cam1.frames"] + counters["cam2.frames"]
    assert (
        counters["detections"]
        == counters["cam1.detections"] + counters["cam2.detections"]
    )


def test_interrupt_through_the_reporter_stops_the_workers(tmp_path):
    """The reporter owns the signal handlers, so the workers learn about
    Ctrl-C by polling it — the run must end and record `interrupted`."""
    FakeLiveAdapter.frames_per_connection = None  # never drop
    service = make_service(tmp_path, ["cam1"])

    with reporter(service, resume_command="siteloom run") as progress:
        # What the reporter's SIGINT handler does, without racing a real
        # signal (the test_resume_equivalence.py convention).
        flag = threading.Timer(0.3, lambda: setattr(progress, "interrupt_requested", True))
        flag.start()
        started = time.monotonic()
        try:
            service.run(progress=progress)
        finally:
            flag.cancel()

    assert service.stopped
    assert time.monotonic() - started < 5.0
    run = latest_run(service)
    assert run.status == "interrupted"
    assert run.finished_at is not None


def test_run_without_a_reporter_records_nothing(tmp_path):
    """Backfill and the tests drive run() reporter-free; that path must
    stay silent rather than minting rows nobody closes."""
    service = make_service(tmp_path, ["cam1"])
    service.run(max_frames=3)

    with service.Session() as session:
        assert session.query(OperationRun).count() == 0
