"""Progress reporting: DB heartbeat, ETA maths, interrupt handling."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from rich.console import Console

from siteloom.progress import Interrupted, ProgressReporter, humanize
from siteloom.store import OperationRun, get_session, init_db, make_engine


@pytest.fixture
def Session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/prog.db")
    init_db(engine)
    return get_session(engine)


@pytest.fixture
def reporter(Session):
    # force_terminal=False keeps the bar off, so tests exercise the
    # non-TTY path deterministically.
    return ProgressReporter(
        Session,
        "test-op",
        target="/tmp/x",
        resume_command="siteloom test resume",
        console=Console(force_terminal=False, quiet=True),
    )


def latest(Session) -> OperationRun:
    with Session() as session:
        return session.query(OperationRun).order_by(OperationRun.id.desc()).first()


def test_run_recorded_and_completed(Session, reporter):
    with reporter as p:
        with p.phase("Working", total=10):
            for _ in range(10):
                p.advance(widgets=2)
    run = latest(Session)
    assert run.kind == "test-op"
    assert run.status == "complete"
    assert run.current == 10
    assert run.total == 10
    assert json.loads(run.counters)["widgets"] == 20
    assert "Working" in json.loads(run.phase_timings)
    assert run.finished_at is not None


def test_progress_visible_mid_run(Session, reporter):
    """The heartbeat is the point: another process can read progress
    while the run is still going."""
    with reporter as p:
        with p.phase("Working", total=100):
            p.advance(5)
            p._beat(force=True)
            mid = latest(Session)
            assert mid.status == "running"
            assert mid.current == 5
            assert mid.phase == "Working"
            assert mid.percent == 5.0


def test_interrupt_marks_run_and_keeps_progress(Session, reporter):
    with reporter as p:
        with p.phase("Working", total=100):
            p.advance(7)
            p.interrupt_requested = True
            with pytest.raises(Interrupted):
                p.check_interrupt()
    run = latest(Session)
    assert run.status == "interrupted"
    assert run.current == 7  # work done before the stop is preserved
    assert run.resume_command == "siteloom test resume"


def test_failure_recorded(Session, reporter):
    with pytest.raises(ValueError):
        with reporter as p:
            with p.phase("Working", total=5):
                p.advance()
                raise ValueError("boom")
    run = latest(Session)
    assert run.status == "failed"
    assert "ValueError: boom" in run.message


def test_multiple_phases_timed(Session, reporter):
    with reporter as p:
        with p.phase("First", total=2):
            p.advance(2)
        with p.phase("Second", total=3):
            p.advance(3)
    timings = json.loads(latest(Session).phase_timings)
    assert set(timings) == {"First", "Second"}


def test_disabled_reporter_is_inert(Session):
    p = ProgressReporter(Session, "test-op", enabled=False)
    with p as prog:
        with prog.phase("Working", total=3):
            prog.advance(3)
    with Session() as session:
        assert session.query(OperationRun).count() == 0


# -- derived metrics --------------------------------------------------------


def make_run(**kw) -> OperationRun:
    base = datetime(2026, 8, 5, 12, 0, 0)
    defaults = dict(
        kind="k", status="running", started_at=base,
        updated_at=base + timedelta(seconds=100), current=25, total=100,
    )
    defaults.update(kw)
    return OperationRun(**defaults)


def test_eta_from_observed_rate():
    # 25 items in 100s -> 0.25/s; 75 left -> 300s
    assert make_run().eta_s == pytest.approx(300.0)


def test_no_eta_for_finished_run():
    run = make_run(status="complete")
    assert run.eta_s is None


def test_no_eta_without_total():
    assert make_run(total=0).eta_s is None


def test_stale_detection():
    """A 'running' row whose heartbeat stopped means the process died —
    it must not be reported as healthy."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert make_run(updated_at=now - timedelta(minutes=10)).is_stale
    assert not make_run(updated_at=now).is_stale
    assert not make_run(status="complete", updated_at=now - timedelta(days=1)).is_stale


@pytest.mark.parametrize(
    "seconds,expected",
    [(None, "—"), (5, "5s"), (65, "1m 05s"), (3725, "1h 02m")],
)
def test_humanize(seconds, expected):
    assert humanize(seconds) == expected
