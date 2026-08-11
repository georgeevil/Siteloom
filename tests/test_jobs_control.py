"""Stopping and closing out jobs from the console (CLD-91).

`/jobs` could watch a run and not act on one, which was survivable while
every long job started in a terminal — whoever started it had the
terminal. The import wizard removed that: a multi-hour index run starts
in the browser, in a background thread of the serving process, with no
terminal anywhere to Ctrl-C.

The properties worth holding, in order of how badly they bite:

* Cancel means one thing. The console and `siteloom jobs cancel` go
  through `progress.request_cancel`, so a run cancelled from either ends
  the same way and hands back the same resume command.
* A cancel that cannot be delivered says so. A run on another host, or a
  row with nothing behind it, must not redirect to a page that looks
  like it worked — an operator told "cancelled" who sees no change
  reaches for `kill -9` and loses the batch.
* A cancel is not an outcome. The row stays `running` until the work
  loop reaches its next safe point; the UI says as much.
* Reap is for dead rows only, and preserves what they knew.
"""

from __future__ import annotations

import os
import signal
import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from rich.console import Console
from typer.testing import CliRunner

from siteloom import cli, cli_library, progress
from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.health import hostname, process_identity
from siteloom.progress import (
    REAP_MESSAGE,
    CancelResult,
    Interrupted,
    ProgressReporter,
)
from siteloom.store import (
    OperationRun,
    User,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web.app import create_app
from siteloom.web.auth import hash_password, required_role

runner = CliRunner()


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.fixture
def env(tmp_path):
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/jobs.db", media_dir=str(tmp_path / "m")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    client = TestClient(create_app(config), follow_redirects=False)
    return client, Session, config


def add_run(Session, **kwargs) -> int:
    """A row as some other process would have left it."""
    fields = dict(
        kind="library-index",
        target="/srv/media/archive",
        status="running",
        started_at=_now() - timedelta(minutes=30),
        updated_at=_now(),
        current=120,
        total=500,
        resume_command="siteloom library index --source 3 --all",
        pid=0,
        host=hostname(),
    )
    fields.update(kwargs)
    with Session() as session:
        run = OperationRun(**fields)
        session.add(run)
        session.commit()
        return run.id


def dead_run(Session, **kwargs) -> int:
    """A row nothing is working on: no pid, and a cold heartbeat."""
    kwargs.setdefault("pid", 0)
    kwargs.setdefault("updated_at", _now() - timedelta(minutes=10))
    return add_run(Session, **kwargs)


def live_run(Session, **kwargs) -> int:
    """A row this very process is behind, so `is_stale` says False.

    It records the process identity a reporter would have written, which
    is what makes the pid on it mean *this* process rather than whatever
    wears that number next (CLD-57).
    """
    kwargs.setdefault("pid", os.getpid())
    kwargs.setdefault("host", hostname())
    kwargs.setdefault("updated_at", _now())
    kwargs.setdefault("process_start", process_identity(os.getpid()) or "")
    return add_run(Session, **kwargs)


def fetch(Session, run_id: int) -> OperationRun:
    with Session() as session:
        return session.get(OperationRun, run_id)


def start_reporter(Session, **kwargs) -> ProgressReporter:
    """A reporter entered off the main thread, as the console's jobs are.

    That is the case the whole feature exists for: no signal handler
    could be installed, and the pid on the row is the web server's.
    """
    reporter = ProgressReporter(
        Session,
        kwargs.pop("kind", "library-index"),
        target="/srv/media/archive",
        resume_command=kwargs.pop(
            "resume_command", "siteloom library index --source 3 --all"
        ),
        console=Console(force_terminal=False, quiet=True),
        bar=False,
        **kwargs,
    )
    thread = threading.Thread(target=reporter.__enter__)
    thread.start()
    thread.join()
    return reporter


# -- cancel ------------------------------------------------------------------


def test_cancel_asks_an_in_process_run_to_stop(env):
    """The wizard's own case: no signal handler, and the pid is ours.

    Signalling it would stop the web server, so the console sets the very
    flag Ctrl-C sets and the work loop honours it at its next safe point.
    """
    client, Session, _config = env
    reporter = start_reporter(Session)
    run_id = reporter.run_id
    try:
        assert client.post(f"/jobs/{run_id}/cancel").status_code == 303
        assert reporter.interrupt_requested is True
        # Not an outcome yet: the batch in flight still gets to commit.
        assert fetch(Session, run_id).status == "running"
        with pytest.raises(Interrupted):
            reporter.check_interrupt()
    finally:
        reporter.__exit__(Interrupted, Interrupted(), None)

    run = fetch(Session, run_id)
    assert run.status == "interrupted"
    assert run.resume_command == "siteloom library index --source 3 --all"


def test_console_and_cli_cancel_through_one_function(env, monkeypatch):
    """Two mechanisms would be two definitions of a cancelled run."""
    client, Session, config = env
    seen: list[tuple[int, bool]] = []

    def fake_cancel(session, run_id, force=False):
        seen.append((run_id, force))
        return CancelResult(True, "requested", f"asked #{run_id} to stop", run_id)

    monkeypatch.setattr(progress, "request_cancel", fake_cancel)
    monkeypatch.setattr(
        cli_library, "_light_setup", lambda *a, **kw: (config, Session)
    )

    assert client.post("/jobs/7/cancel").status_code == 303
    result = runner.invoke(cli.app, ["jobs", "cancel", "7", "--config", "unused.yaml"])
    assert result.exit_code == 0

    assert seen == [(7, False), (7, False)]


def test_cancel_signals_the_owning_process(env, monkeypatch):
    client, Session, _config = env
    run_id = live_run(Session)
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(progress.os, "kill", lambda pid, sig: sent.append((pid, sig)))

    assert client.post(f"/jobs/{run_id}/cancel").status_code == 303
    # Signal 0 is the liveness probe; the stop itself is SIGINT — the
    # console never offers SIGKILL, which would lose the batch in flight.
    assert sent[-1] == (os.getpid(), signal.SIGINT)
    assert all(sig != signal.SIGKILL for _pid, sig in sent)


def test_cancel_on_another_host_fails_honestly(env):
    client, Session, _config = env
    run_id = add_run(Session, host="nvr-two", pid=4242)

    response = client.post(f"/jobs/{run_id}/cancel")
    assert response.status_code == 409
    body = response.json()
    assert body["reason"] == "other_host"
    assert "nvr-two" in body["error"]
    # And nothing was pretended: the row is untouched.
    assert fetch(Session, run_id).status == "running"


def test_cancel_on_a_dead_row_points_at_reap(env):
    client, Session, _config = env
    run_id = dead_run(Session)

    response = client.post(f"/jobs/{run_id}/cancel")
    assert response.status_code == 409
    assert response.json()["reason"] == "no_process"
    assert "reap" in response.json()["error"]
    assert fetch(Session, run_id).status == "running"


def test_cancel_of_a_finished_or_missing_run(env):
    client, Session, _config = env
    done = add_run(Session, status="complete", finished_at=_now())

    already = client.post(f"/jobs/{done}/cancel")
    assert already.status_code == 409
    assert already.json()["reason"] == "not_running"
    assert client.post("/jobs/999/cancel").status_code == 404


# -- reap --------------------------------------------------------------------


def test_reap_closes_a_stale_row_keeping_its_position(env):
    client, Session, _config = env
    run_id = dead_run(Session)

    assert client.post(f"/jobs/{run_id}/reap").status_code == 303

    run = fetch(Session, run_id)
    assert run.status == "abandoned"
    assert run.message == REAP_MESSAGE
    assert run.finished_at is not None
    # The point of reaping over deleting: what it knew survives.
    assert (run.current, run.total) == (120, 500)
    assert run.resume_command == "siteloom library index --source 3 --all"


def test_bulk_reap_leaves_a_live_run_alone(env):
    client, Session, _config = env
    first, second = dead_run(Session), dead_run(Session, kind="takeout-import")
    working = live_run(Session)

    response = client.post("/jobs/reap")
    assert response.status_code == 303

    assert fetch(Session, first).status == "abandoned"
    assert fetch(Session, second).status == "abandoned"
    assert fetch(Session, working).status == "running"


def test_a_live_row_is_not_reapable(env):
    """Reaping a running job would record a stop that never happened."""
    client, Session, _config = env
    run_id = live_run(Session)

    response = client.post(f"/jobs/{run_id}/reap")
    assert response.status_code == 409
    assert response.json()["reason"] == "not_stale"
    assert fetch(Session, run_id).status == "running"

    nothing = client.post("/jobs/reap")
    assert nothing.status_code == 409


def test_cli_reap_writes_the_same_row_as_the_console(env, monkeypatch):
    client, Session, config = env
    monkeypatch.setattr(
        cli_library, "_light_setup", lambda *a, **kw: (config, Session)
    )
    from_console = dead_run(Session)
    from_cli = dead_run(Session)

    assert client.post(f"/jobs/{from_console}/reap").status_code == 303
    result = runner.invoke(
        cli.app, ["jobs", "reap", "--config", "unused.yaml", "--yes"]
    )
    assert result.exit_code == 0

    console_row, cli_row = fetch(Session, from_console), fetch(Session, from_cli)
    assert console_row.status == cli_row.status == "abandoned"
    assert console_row.message == cli_row.message == REAP_MESSAGE


# -- the page ----------------------------------------------------------------


def test_page_offers_cancel_and_says_it_is_a_request(env):
    client, Session, _config = env
    run_id = live_run(Session)

    html = client.get("/jobs").text
    assert f'action="/jobs/{run_id}/cancel"' in html
    assert "confirm(" in html  # not undoable, and one line from eight others
    assert "finishes the batch it is in" in html


def test_page_offers_reap_and_shows_the_resume_command(env):
    client, Session, _config = env
    run_id = dead_run(Session)

    html = client.get("/jobs").text
    assert f'action="/jobs/{run_id}/reap"' in html
    assert 'action="/jobs/reap"' in html  # bulk, for a board full of them
    assert "siteloom library index --source 3 --all" in html

    client.post(f"/jobs/{run_id}/reap")
    after = client.get("/jobs").text
    # A closed-out row is still resumable, and says how.
    assert "siteloom library index --source 3 --all" in after


def test_notice_is_echoed_back_bounded(env):
    client, Session, _config = env
    run_id = dead_run(Session)
    redirect = client.post(f"/jobs/{run_id}/reap").headers["location"]
    assert redirect.startswith("/jobs?notice=")
    assert "resume command" in client.get(redirect).text
    assert "x" * 300 in client.get("/jobs?notice=" + "x" * 900).text
    assert "x" * 400 not in client.get("/jobs?notice=" + "x" * 900).text


# -- roles -------------------------------------------------------------------


def test_stopping_a_job_is_an_admin_action(env):
    client, Session, _config = env
    assert required_role("POST", "/jobs/12/cancel") == "admin"
    assert required_role("POST", "/jobs/12/reap") == "admin"
    assert required_role("POST", "/jobs/reap") == "admin"
    assert required_role("POST", "/jobs/reindex") == "admin"
    # The trailing slash in the `/jobs/` prefix still keeps the console
    # itself readable; its floor is now the bottom rung (CLD-31).
    assert required_role("GET", "/jobs") == "restricted"

    run_id = add_run(Session, host="nvr-two", pid=4242)
    with Session() as session:
        for name, role in (("ed", "edit"), ("ada", "admin")):
            session.add(
                User(
                    username=name,
                    password_hash=hash_password("hunter2!"),
                    role=role,
                    created_at=_now(),
                )
            )
        session.commit()

    def login(username):
        assert (
            client.post(
                "/login",
                data={"username": username, "password": "hunter2!", "next": "/"},
            ).status_code
            == 303
        )

    login("ed")
    assert client.post(f"/jobs/{run_id}/cancel").status_code == 403
    assert client.post("/jobs/reap").status_code == 403
    client.post("/logout")

    login("ada")
    # 409 rather than 403: past the gate, refused on its own honest terms.
    assert client.post(f"/jobs/{run_id}/cancel").status_code == 409
