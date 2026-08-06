"""Preflight checks, health endpoints, and job supervision.

These are the parts an operator relies on when something is already
wrong, so they have to work in exactly that situation: a held vector
store, a dead process, an unwritable directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from siteloom import health
from siteloom.cli_library import jobs_app
from siteloom.config import IdentityConfig, SiteConfig, StorageConfig
from siteloom.health import FAIL, OK, WARN, Report, run_checks
from siteloom.store import OperationRun, get_session, init_db, make_engine

runner = CliRunner()


@pytest.fixture
def config(tmp_path):
    return SiteConfig(
        site_id="test",
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/health.db",
            media_dir=str(tmp_path / "media"),
        ),
        identity=IdentityConfig(enabled=False),
    )


@pytest.fixture
def config_file(tmp_path, config):
    import yaml

    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    return path


def status_of(report: Report, name: str) -> str:
    return next(c.status for c in report.checks if c.name == name)


# -- checks -----------------------------------------------------------------


def test_healthy_deployment_passes(config):
    report = run_checks(config)
    assert report.ok, [c for c in report.checks if c.status == FAIL]
    assert status_of(report, "database") == OK
    assert status_of(report, "media dir") == OK


def test_unwritable_media_dir_fails(config, tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("I am a file, not a directory")
    config.storage.media_dir = str(blocker / "media")
    report = run_checks(config, [health.check_media_dir])
    assert status_of(report, "media dir") == FAIL
    assert not report.ok


def test_held_vector_store_is_reported_with_a_remedy(config, tmp_path):
    """The failure that actually happens in practice: serve is running,
    so a job cannot open the store. The operator needs to be told that,
    not a Qdrant stack trace."""
    from siteloom.identity import VectorStore

    config.identity.enabled = True
    config.identity.vector_db_path = str(tmp_path / "vectors")
    holder = VectorStore(config.identity.vector_db_path)
    try:
        report = run_checks(config, [health.check_vector_store])
    finally:
        holder.close()
    check = report.checks[0]
    assert check.status == FAIL
    assert "held by another process" in check.detail
    assert "vector_db_path" in check.remedy


def test_a_crashing_check_does_not_hide_the_others(config):
    def explodes(report, cfg):
        raise RuntimeError("boom")

    report = run_checks(config, [explodes, health.check_media_dir])
    assert len(report.checks) == 2
    assert report.checks[0].status == FAIL and "boom" in report.checks[0].detail
    assert status_of(report, "media dir") == OK


def test_abandoned_runs_are_flagged(config):
    Session = _session(config)
    with Session() as session:
        session.add(_run(pid=_dead_pid(), host=health.hostname()))
        session.commit()
    report = run_checks(config, [health.check_jobs])
    check = report.checks[0]
    assert check.status == WARN
    assert "abandoned" in check.detail
    assert "jobs reap" in check.remedy


def test_process_alive_on_self_and_on_the_departed():
    assert health.process_alive(os.getpid())
    assert not health.process_alive(_dead_pid())
    assert not health.process_alive(0)


# -- endpoints --------------------------------------------------------------


def test_healthz_does_not_touch_the_database(config):
    from siteloom.web.app import create_app

    client = TestClient(create_app(config))
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["site"] == "test"
    assert body["pid"] == os.getpid()


def test_readyz_reports_unready_with_503(config, tmp_path):
    from siteloom.web.app import create_app

    app = create_app(config)  # created while the config is still sound
    config.storage.media_dir = str(tmp_path / "gone" / "media")
    (tmp_path / "gone").write_text("a file where a directory should be")
    client = TestClient(app)
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["ok"] is False


def test_readyz_ok(config):
    from siteloom.web.app import create_app

    client = TestClient(create_app(config))
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["ok"] is True


# -- jobs supervision -------------------------------------------------------


def test_reap_closes_out_dead_runs_but_keeps_the_resume_command(config, config_file):
    Session = _session(config)
    with Session() as session:
        session.add(_run(pid=_dead_pid(), host=health.hostname()))
        session.commit()

    result = runner.invoke(jobs_app, ["reap", "--config", str(config_file), "--yes"])
    assert result.exit_code == 0, result.output
    with Session() as session:
        run = session.query(OperationRun).one()
        assert run.status == "abandoned"
        assert run.finished_at is not None
        assert run.resume_command == "siteloom library index --all"
        assert run.current == 40  # position preserved


def test_reap_leaves_live_runs_alone(config, config_file):
    Session = _session(config)
    with Session() as session:
        session.add(_run(pid=os.getpid(), host=health.hostname()))
        session.commit()

    result = runner.invoke(jobs_app, ["reap", "--config", str(config_file), "--yes"])
    assert result.exit_code == 0
    assert "nothing to reap" in result.output
    with Session() as session:
        assert session.query(OperationRun).one().status == "running"


def test_cancel_refuses_a_run_from_another_host(config, config_file):
    Session = _session(config)
    with Session() as session:
        session.add(_run(pid=os.getpid(), host="some-other-machine"))
        session.commit()

    result = runner.invoke(jobs_app, ["cancel", "1", "--config", str(config_file)])
    assert result.exit_code == 1
    assert "some-other-machine" in result.output


def test_cancel_signals_a_live_process(config, config_file):
    """The point of `jobs cancel`: stop a job from a terminal that did
    not start it, gracefully enough that it can resume."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import signal, time\n"
         "signal.signal(signal.SIGINT, lambda *a: None)\n"
         "time.sleep(30)"]
    )
    try:
        Session = _session(config)
        with Session() as session:
            session.add(_run(pid=child.pid, host=health.hostname()))
            session.commit()
        result = runner.invoke(jobs_app, ["cancel", "1", "--config", str(config_file)])
        assert result.exit_code == 0, result.output
        assert "finish the current batch" in result.output
        assert child.poll() is None  # asked to stop, not killed
    finally:
        child.kill()
        child.wait()


# -- helpers ----------------------------------------------------------------


def _session(config):
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    return get_session(engine)


def _run(**kw) -> OperationRun:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return OperationRun(
        kind="library-index",
        status="running",
        started_at=now - timedelta(seconds=60),
        updated_at=now,  # fresh heartbeat: only the pid can prove it dead
        current=40,
        total=100,
        resume_command="siteloom library index --all",
        **kw,
    )


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid
