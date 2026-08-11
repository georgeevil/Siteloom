"""A pid is not a process: cancelling after pid reuse (CLD-57).

`OperationRun` records the pid that is doing the work, and `jobs cancel`
signals it. Pids are recycled, so a row left behind by a job that died
names a number the OS has since handed to something else — and the
signal lands there instead. CLD-91 put cancel on a page that lists every
run including the dead ones, which makes that one click by someone who
cannot see a process table.

The fix is to record *which* process, not just which pid: the OS-reported
start time, which no later process on the same pid can repeat. Both the
question "may I signal this?" (`request_cancel`) and the question "is
anything still working on this?" (`OperationRun.is_stale`) are answered
by one function, `health.process_verdict`, so they cannot disagree.

What is asserted below, in the order it matters:

* a pid whose process identity no longer matches is refused, and nothing
  is signalled;
* a matching one is signalled exactly as before;
* a row from before the column existed cannot prove anything, and is
  therefore **refused** — deliberately, and it says why;
* a platform that cannot read start times refuses honestly rather than
  signalling optimistically. "Probably fine" is the bug.

No real process is spawned and raced: every state is constructed.
"""

from __future__ import annotations

import os
import shutil
import signal
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import text
from typer.testing import CliRunner

from siteloom import cli, cli_library, health, progress
from siteloom.config import SiteConfig, StorageConfig
from siteloom.health import (
    MATCH,
    MISMATCH,
    NO_PROCESS,
    UNREADABLE,
    UNRECORDED,
    hostname,
    process_identity,
    process_verdict,
)
from siteloom.progress import ProgressReporter, request_cancel, stale_runs
from siteloom.store import OperationRun, get_session, init_db, make_engine

runner = CliRunner()

#: A token no live process can be wearing: the recorded run started at a
#: point in time this one demonstrably did not.
OTHER_PROCESS = "start:1"


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _dead_pid() -> int:
    """A pid that has certainly exited: spawn and reap a trivial child."""
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


@pytest.fixture
def env(tmp_path):
    config = SiteConfig(
        site_id="t",
        site_name="T",
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/jobs.db", media_dir=str(tmp_path / "m")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    return config, get_session(engine), engine


def add_run(Session, **kwargs) -> int:
    """A row as another process would have left it: fresh heartbeat, so
    only the process identity can settle whether it is still alive."""
    fields = dict(
        kind="library-index",
        status="running",
        started_at=_now() - timedelta(minutes=5),
        updated_at=_now(),
        current=120,
        total=500,
        resume_command="siteloom library index --source 3 --all",
        pid=os.getpid(),
        host=hostname(),
        process_start=process_identity(os.getpid()) or "",
    )
    fields.update(kwargs)
    with Session() as session:
        run = OperationRun(**fields)
        session.add(run)
        session.commit()
        return run.id


@pytest.fixture
def no_signals(monkeypatch):
    """Records every signal that would have been delivered, delivering none.

    Signal 0 is passed through: it delivers nothing and is how liveness
    is probed, and `os.kill` is one object shared by every module — a
    blanket stub would quietly answer "alive" for a pid that is gone.
    """
    sent: list[tuple[int, int]] = []
    real_kill = os.kill

    def fake_kill(pid: int, sig: int):
        if sig == 0:
            return real_kill(pid, sig)
        sent.append((pid, sig))
        return None

    monkeypatch.setattr(progress.os, "kill", fake_kill)
    return sent


def unreadable_platform(monkeypatch):
    """A host that will not say when a process started — no /proc, no
    usable `ps`. The macOS-without-`ps` / hardened-container case."""
    monkeypatch.setattr(
        health, "process_identity", lambda pid: "" if pid > 0 else None
    )


# -- the identity itself -----------------------------------------------------


def test_identity_of_this_process_is_stable_and_specific():
    mine = process_identity(os.getpid())
    assert mine  # this platform can answer, in CI and on the target
    assert mine == process_identity(os.getpid())  # stable across calls
    assert mine != OTHER_PROCESS


def test_identity_of_a_departed_process_is_none_not_empty():
    """None means gone; "" means unknown. Conflating them would make an
    unknown process reapable, or a dead one look cancellable."""
    assert process_identity(_dead_pid()) is None
    assert process_identity(0) is None
    assert process_identity(-1) is None


def test_verdict_answers_five_ways():
    mine = process_identity(os.getpid())
    assert process_verdict(os.getpid(), mine) == MATCH
    assert process_verdict(os.getpid(), OTHER_PROCESS) == MISMATCH
    assert process_verdict(os.getpid(), "") == UNRECORDED
    assert process_verdict(_dead_pid(), OTHER_PROCESS) == NO_PROCESS
    assert process_verdict(0, mine) == NO_PROCESS


def test_the_no_proc_path_reads_a_start_time_too(monkeypatch):
    """The primary target is macOS, which has no /proc — there the token
    comes from `ps -o lstart=`. Exercised on CI by hiding /proc, so the
    path the target platform takes is not the untested one."""
    if shutil.which("ps") is None:
        pytest.skip("no ps on this box")
    monkeypatch.setattr(health, "_PROC", Path("/nonexistent-proc"))

    mine = process_identity(os.getpid())
    assert mine and mine.startswith("lstart:")
    assert process_verdict(os.getpid(), mine) == MATCH
    assert process_verdict(os.getpid(), "lstart:Thu Jan  1 00:00:00 1970") == MISMATCH
    assert process_identity(_dead_pid()) is None


def test_an_unknown_identity_never_matches_another(monkeypatch):
    """"" == "" is the trap: two processes nobody can identify would
    otherwise be declared the same process."""
    unreadable_platform(monkeypatch)
    assert process_verdict(os.getpid(), "") == UNREADABLE
    assert process_verdict(os.getpid(), OTHER_PROCESS) == UNREADABLE


# -- cancelling --------------------------------------------------------------


def test_cancel_refuses_a_recycled_pid(env, no_signals):
    """The bug: the row's pid is alive, but it is not this run's process.

    Signalling it would reach whatever inherited the number — on a
    single-operator box, quite likely the operator's own shell.
    """
    _config, Session, _engine = env
    run_id = add_run(Session, process_start=OTHER_PROCESS)

    with Session() as session:
        result = request_cancel(session, run_id)

    assert not result.ok
    assert result.reason == "pid_reused"
    assert "reap" in result.detail
    assert no_signals == []
    with Session() as session:
        assert session.get(OperationRun, run_id).status == "running"


def test_force_does_not_bypass_the_check(env, no_signals):
    """--force chooses SIGKILL over SIGINT; it does not choose a target."""
    _config, Session, _engine = env
    run_id = add_run(Session, process_start=OTHER_PROCESS)

    with Session() as session:
        assert not request_cancel(session, run_id, force=True).ok
    assert no_signals == []


def test_cancel_signals_a_pid_that_is_still_its_process(env, no_signals):
    """The check refuses what it cannot prove, and nothing else."""
    _config, Session, _engine = env
    run_id = add_run(Session)  # records this process's real identity

    with Session() as session:
        result = request_cancel(session, run_id)

    assert result.ok and result.reason == "requested"
    assert no_signals == [(os.getpid(), signal.SIGINT)]


def test_cancel_refuses_a_row_written_before_the_column_existed(env, no_signals):
    """A pre-migration row holds '' and can prove nothing.

    The decision, stated: it is refused. Such a row is indistinguishable
    from one whose pid was recycled, and the cost of being wrong is
    asymmetric — refusing costs an operator one `reap` or a Ctrl-C at the
    job's own terminal, guessing costs whatever else holds that pid.
    Runs started after the upgrade cancel normally, and the message says
    so.
    """
    _config, Session, _engine = env
    run_id = add_run(Session, process_start="")

    with Session() as session:
        result = request_cancel(session, run_id)

    assert not result.ok
    assert result.reason == "unverified"
    assert "predates" in result.detail
    assert no_signals == []


def test_cancel_refuses_where_the_platform_cannot_tell(env, no_signals, monkeypatch):
    """Not "probably fine": a host that cannot identify a process cannot
    be allowed to signal one on the strength of a pid."""
    _config, Session, _engine = env
    run_id = add_run(Session)
    unreadable_platform(monkeypatch)

    with Session() as session:
        result = request_cancel(session, run_id)

    assert not result.ok
    assert result.reason == "unverified"
    assert "start times" in result.detail
    assert no_signals == []


def test_a_dead_pid_still_reads_as_no_process(env, no_signals):
    """The pre-existing refusal keeps its own reason — 'reused' and
    'gone' are different things to be told."""
    _config, Session, _engine = env
    run_id = add_run(Session, pid=_dead_pid())

    with Session() as session:
        result = request_cancel(session, run_id)

    assert result.reason == "no_process"
    assert no_signals == []


def test_an_in_process_run_is_still_cancellable_without_a_token(
    env, no_signals, monkeypatch
):
    """The registry path (CLD-91) proves ownership by construction — the
    reporter is an object this interpreter made for this row — so it must
    not start depending on a token the platform may not have."""
    _config, Session, _engine = env
    unreadable_platform(monkeypatch)
    reporter = ProgressReporter(
        Session, "library-index", bar=False, signals=False
    )
    with reporter:
        with Session() as session:
            result = request_cancel(session, reporter.run_id)
        assert result.ok and result.reason == "requested"
        assert reporter.interrupt_requested
    assert no_signals == []  # asked to stop, never signalled


def test_cli_cancel_of_a_recycled_pid_exits_nonzero(env, monkeypatch, no_signals):
    config, Session, _engine = env
    monkeypatch.setattr(
        cli_library, "_light_setup", lambda *a, **kw: (config, Session)
    )
    run_id = add_run(Session, process_start=OTHER_PROCESS)

    result = runner.invoke(
        cli.app, ["jobs", "cancel", str(run_id), "--config", "unused.yaml"]
    )
    assert result.exit_code == 1
    assert "jobs reap" in result.output
    assert no_signals == []


# -- staleness, which must agree with cancel ---------------------------------


def test_a_recycled_pid_is_stale_immediately(env):
    """One question, one answer: a row `request_cancel` refuses to signal
    because its process is gone must not read as healthy on /jobs."""
    _config, Session, _engine = env
    reused = add_run(Session, process_start=OTHER_PROCESS)
    mine = add_run(Session)

    with Session() as session:
        assert session.get(OperationRun, reused).is_stale
        assert not session.get(OperationRun, mine).is_stale
        assert [r.id for r in stale_runs(session)] == [reused]


def test_an_unprovable_row_falls_back_to_the_heartbeat(env, monkeypatch):
    """Unknown is not dead. A row that cannot prove its identity is
    judged by its heartbeat, exactly as a run on another host is —
    reaping it on suspicion would record a stop that never happened."""
    _config, Session, _engine = env
    warm = add_run(Session, process_start="")
    cold = add_run(Session, process_start="", updated_at=_now() - timedelta(minutes=10))

    with Session() as session:
        assert not session.get(OperationRun, warm).is_stale
        assert session.get(OperationRun, cold).is_stale

    unreadable_platform(monkeypatch)
    with Session() as session:
        assert not session.get(OperationRun, warm).is_stale
        assert session.get(OperationRun, cold).is_stale


# -- writing and migrating the column ----------------------------------------


def test_the_reporter_records_its_own_identity(env):
    _config, Session, _engine = env
    with ProgressReporter(Session, "library-index", bar=False, signals=False) as p:
        with Session() as session:
            run = session.get(OperationRun, p.run_id)
            assert run.pid == os.getpid()
            assert run.process_start == process_identity(os.getpid())


def test_migration_fills_old_rows_with_an_empty_identity(env):
    """`_ensure_columns` emits a DDL DEFAULT for a non-nullable scalar
    default, so existing rows land on '' — the value that means "cannot
    prove it" — rather than NULL."""
    if sqlite3.sqlite_version_info < (3, 35):
        pytest.skip("ALTER TABLE .. DROP COLUMN needs sqlite >= 3.35")
    _config, Session, engine = env

    # Reconstruct a database from before the column existed.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE operation_runs DROP COLUMN process_start"))
        conn.execute(
            text(
                "INSERT INTO operation_runs "
                "(kind, target, phase, status, started_at, updated_at, current, "
                " total, counters, phase_timings, message, resume_command, pid, host) "
                "VALUES ('library-index', '', '', 'running', :t, :t, 1, 2, '{}', "
                "'{}', '', '', :pid, :host)"
            ),
            {"t": _now(), "pid": os.getpid(), "host": hostname()},
        )

    init_db(engine)  # the additive migration, as any command would run it

    with Session() as session:
        run = session.query(OperationRun).one()
        assert run.process_start == ""
        # And it behaves as decided: unprovable, so refused, not signalled.
        assert request_cancel(session, run.id).reason == "unverified"
