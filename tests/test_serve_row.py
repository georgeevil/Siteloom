"""`serve` as a watchable, steerable run (CLD-105).

`run` got an OperationRun row in CLD-15 so a 24 h soak is visible from
another terminal and a crash at hour 3 reads as stale rather than
healthy. A web server that is up for weeks has the same problem and was
the last long-lived command without one.

Two hazards, and the tests that hold them:

* **Handler ownership.** uvicorn installs its own SIGINT/SIGTERM
  handlers and drives shutdown through them. Two owners racing for
  SIGINT is how a Ctrl-C gets swallowed — `IngestService._stop_signals`
  documents the same collision from the other side. So the reporter is
  built with `signals=False` and must leave the process's handlers
  exactly as it found them, while keeping everything else.
* **Cancel has to mean something.** `siteloom jobs cancel` reports
  success as soon as the flag is set. If nothing in this process turns
  that flag into a shutdown, the operator is told the server stopped and
  it keeps serving — the exact failure `test_jobs_control.py` exists to
  prevent.

No port is bound anywhere here: `run_server` is injected.
"""

from __future__ import annotations

import signal
import threading

import pytest

from siteloom import serve as serve_mod
from siteloom.config import IdentityConfig, SiteConfig, StorageConfig
from siteloom.progress import ProgressReporter, request_cancel
from siteloom.store import OperationRun, get_session, init_db, make_engine


@pytest.fixture
def Session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/serve.db")
    init_db(engine)
    return get_session(engine)


@pytest.fixture
def config(tmp_path):
    return SiteConfig(
        site_id="kai",
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/serve.db", media_dir=str(tmp_path / "media")
        ),
        identity=IdentityConfig(enabled=False),
    )


class FakeServer:
    """Stands in for uvicorn.Server: started, should_exit, run()."""

    def __init__(self, block=False):
        self.started = False
        self.should_exit = False
        self.ran = False
        self._block = block
        self.running = threading.Event()

    def run(self):
        self.ran = True
        self.started = True
        self.running.set()
        if self._block:
            while not self.should_exit:
                if not self.running.wait(0.05):
                    break


def _serve(config, Session, server, **kwargs):
    serve_mod.serve_supervised(
        config,
        host="127.0.0.1",
        port=8000,
        resume_command="siteloom serve --config site.yaml",
        app=object(),
        run_server=lambda s: server.run(),
        session_factory=Session,
        **kwargs,
    )


def _server_factory(server, monkeypatch):
    monkeypatch.setattr(
        serve_mod, "_build_server", lambda app, **kw: server
    )


def test_serve_records_a_run_that_completes(config, Session, monkeypatch):
    server = FakeServer()
    _server_factory(server, monkeypatch)
    _serve(config, Session, server)

    with Session() as session:
        run = session.query(OperationRun).one()
    assert run.kind == "serve"
    assert run.status == "complete"
    assert "http://127.0.0.1:8000" in run.target
    # A server has no unit of done-ness, exactly like `run`: "resume"
    # means start it again.
    assert run.resume_command == "siteloom serve --config site.yaml"
    assert server.ran


def test_serve_records_a_crash_as_failed(config, Session, monkeypatch):
    server = FakeServer()
    _server_factory(server, monkeypatch)

    def boom(_):
        raise RuntimeError("port in use")

    with pytest.raises(RuntimeError):
        serve_mod.serve_supervised(
            config,
            host="127.0.0.1",
            port=8000,
            app=object(),
            run_server=boom,
            session_factory=Session,
        )
    with Session() as session:
        run = session.query(OperationRun).one()
    assert run.status == "failed"
    assert "port in use" in (run.message or "")


def test_a_stop_is_not_a_crash(config, Session, monkeypatch):
    """uvicorn restores the original handlers on the way out and then
    re-raises the signal it captured, so an ordinary Ctrl-C arrives here
    as a KeyboardInterrupt *after* a clean shutdown. Recording that as
    `failed` would make every `systemctl stop` look like a crash."""
    server = FakeServer()
    _server_factory(server, monkeypatch)

    def shutdown_then_reraise(_):
        server.started = True
        raise KeyboardInterrupt

    stopped = serve_mod.serve_supervised(
        config,
        host="127.0.0.1",
        port=8000,
        app=object(),
        run_server=shutdown_then_reraise,
        session_factory=Session,
    )
    assert stopped
    with Session() as session:
        assert session.query(OperationRun).one().status == "interrupted"


def test_should_exit_alone_records_a_stop(config, Session, monkeypatch):
    """uvicorn handled the signal itself and returned normally. The run
    still ended because it was asked to, not because it finished."""
    server = FakeServer()
    _server_factory(server, monkeypatch)

    def handled_by_uvicorn(_):
        server.should_exit = True

    stopped = serve_mod.serve_supervised(
        config,
        host="127.0.0.1",
        port=8000,
        app=object(),
        run_server=handled_by_uvicorn,
        session_factory=Session,
    )
    assert stopped
    with Session() as session:
        assert session.query(OperationRun).one().status == "interrupted"


def test_sigterm_under_the_capture_does_not_kill_the_process(
    config, Session, monkeypatch
):
    """The handler installed underneath uvicorn's is what makes SIGTERM
    survivable at the moment uvicorn re-raises it: the default action is
    immediate death, which would leave the row saying `running` forever.
    """
    server = FakeServer()
    _server_factory(server, monkeypatch)

    def reraise_sigterm_the_way_uvicorn_does(_):
        server.started = True
        # uvicorn restores the previous handler and then raises the
        # captured signal into it. Ours must absorb it.
        signal.raise_signal(signal.SIGTERM)

    stopped = serve_mod.serve_supervised(
        config,
        host="127.0.0.1",
        port=8000,
        app=object(),
        run_server=reraise_sigterm_the_way_uvicorn_does,
        session_factory=Session,
    )
    assert stopped
    with Session() as session:
        assert session.query(OperationRun).one().status == "interrupted"


def test_handlers_are_restored_afterwards(config, Session, monkeypatch):
    server = FakeServer()
    _server_factory(server, monkeypatch)
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    _serve(config, Session, server)
    assert {s: signal.getsignal(s) for s in before} == before


def test_the_reporter_does_not_own_the_signals(config, Session, monkeypatch):
    """The handler in place while the server runs is serve's own
    recorder — one that sets the flag and returns — not the reporter's.

    The distinction matters: the reporter's handler prints a notice and,
    on a second signal, restores SIG_DFL and re-raises. uvicorn is
    driving shutdown here, so the layer underneath it must do nothing but
    remember that a stop was asked for.
    """
    seen = {}

    def capture(_server):
        seen["sigint"] = signal.getsignal(signal.SIGINT)

    server = FakeServer()
    _server_factory(server, monkeypatch)
    serve_mod.serve_supervised(
        config,
        host="127.0.0.1",
        port=8000,
        app=object(),
        run_server=capture,
        session_factory=Session,
    )
    handler = seen["sigint"]
    assert callable(handler)
    assert getattr(handler, "__self__", None) is None  # not ProgressReporter._on_stop
    assert handler.__qualname__.startswith("_record_stop_intent")


def test_cancel_from_another_terminal_stops_the_server(config, Session, monkeypatch):
    """A cancel that is reported and not delivered is the worst outcome:
    the operator reaches for kill -9."""
    server = FakeServer(block=True)
    _server_factory(server, monkeypatch)
    monkeypatch.setattr(serve_mod, "SERVE_HEARTBEAT_S", 0.02)

    thread = threading.Thread(
        target=_serve, args=(config, Session, server), daemon=True
    )
    thread.start()
    assert server.running.wait(5)

    # Wait for the row, then cancel it the way `siteloom jobs cancel` and
    # the /jobs page both do.
    for _ in range(200):
        with Session() as session:
            run = session.query(OperationRun).first()
        if run is not None:
            break
        threading.Event().wait(0.02)
    assert run is not None

    with Session() as session:
        result = request_cancel(session, run.id)
    assert result.ok, result.detail

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert server.should_exit

    with Session() as session:
        row = session.query(OperationRun).one()
    assert row.status == "interrupted"


def test_signals_false_keeps_the_row_and_the_registry(Session):
    """`signals=False` is a third switch, not a shade of `bar` or
    `enabled`: it gives up signal ownership and nothing else."""
    before = signal.getsignal(signal.SIGTERM)
    with ProgressReporter(Session, "serve", signals=False) as progress:
        assert signal.getsignal(signal.SIGTERM) is before
        assert progress.run_id is not None
        with Session() as session:
            # Still cancellable — the registry entry is what carries that.
            assert request_cancel(session, progress.run_id).ok
        assert progress.interrupt_requested
    assert signal.getsignal(signal.SIGTERM) is before


def test_signals_true_still_installs(Session):
    before = signal.getsignal(signal.SIGTERM)
    with ProgressReporter(Session, "run") as progress:
        assert signal.getsignal(signal.SIGTERM) is not before
        assert progress.run_id is not None
    assert signal.getsignal(signal.SIGTERM) is before


def test_heartbeat_writes_without_claiming_work(Session):
    with ProgressReporter(Session, "serve", signals=False, bar=False) as progress:
        with Session() as session:
            first = session.get(OperationRun, progress.run_id).updated_at
        progress.heartbeat()
        with Session() as session:
            row = session.get(OperationRun, progress.run_id)
            assert row.updated_at >= first
            assert row.current == 0  # nothing was claimed
