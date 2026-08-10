"""Running the web UI as a supervised process.

`serve` was the last long-lived command invisible to `siteloom jobs`:
`run` got an OperationRun row in CLD-15 precisely so a 24 h soak is
watchable from another terminal and a crash at hour 3 reads as stale
rather than healthy, and a web server that is up for weeks has exactly
the same problem. So it gets a row too.

Two things make that awkward, and both are handled here rather than in
`cli.py`, so the CLI command stays a signature and a call:

* **uvicorn owns the signals.** `ProgressReporter` installs handlers for
  every member of STOP_SIGNALS; uvicorn installs its own and drives the
  shutdown through them. Two owners racing for SIGINT is how a Ctrl-C
  gets swallowed, so the reporter is constructed with `signals=False`
  and uvicorn keeps ownership. The reporter is still registered in
  `_LIVE_REPORTERS`, so `siteloom jobs cancel` still reaches this
  process — we poll `interrupt_requested` and set `should_exit`.
* **A server has no unit of work.** There is nothing to `advance()`, so
  the row is kept alive by `heartbeat()` on a timer. `current`/`total`
  stay 0, which `jobs list` already renders sensibly, and there are no
  counters: this row's job is liveness, not analytics.

`run_server` is injectable so the lifecycle above can be tested without
binding a port — the same seam that makes `FrigateConsumer.handle_message`
testable.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator

log = logging.getLogger(__name__)

#: How often the row is written. Well inside `OperationRun.is_stale`'s
#: 120 s cold-heartbeat backstop, so a server that dies without closing
#: its row is reported stale within a couple of minutes rather than
#: looking healthy forever.
SERVE_HEARTBEAT_S = 5.0


def serve_supervised(
    config: Any,
    *,
    host: str,
    port: int,
    log_level: str = "INFO",
    resume_command: str = "",
    app: Any = None,
    run_server: Callable[[Any], None] | None = None,
    session_factory: Any = None,
) -> bool:
    """Serve `config`'s web UI, heartbeating an OperationRun row.

    Returns whether it was stopped rather than ending on its own, so the
    CLI can exit 130 the way every other interruptible command does.
    """
    from siteloom.progress import ProgressReporter
    from siteloom.store import get_session, make_engine

    if session_factory is None:
        session_factory = get_session(make_engine(config.storage.db_url))
    if app is None:
        from siteloom.web.app import create_app

        app = create_app(config)
    server = _build_server(app, host=host, port=port, log_level=log_level)
    if run_server is None:
        run_server = _run
    log.info(
        "serving %s on http://%s:%d (pid %d)", config.site_id, host, port, os.getpid()
    )
    with ProgressReporter(
        session_factory,
        "serve",
        target=f"{config.site_id} http://{host}:{port}",
        resume_command=resume_command,
        # No bar: a server is not a job with an end, and a Rich live
        # display would fight uvicorn's own console output.
        bar=False,
        signals=False,
    ) as progress:
        with progress.phase("serving"):
            stop = threading.Event()
            beat = threading.Thread(
                target=_heartbeat_loop,
                args=(progress, server, stop),
                name="siteloom-serve-heartbeat",
                daemon=True,
            )
            beat.start()
            try:
                with _record_stop_intent(progress):
                    run_server(server)
            except KeyboardInterrupt:
                # uvicorn re-raises the signal it captured, and on an
                # uncaptured SIGINT the default handler turns that into a
                # KeyboardInterrupt. An operator stopping a server is not
                # a crash.
                progress.request_stop("stopped by signal")
            finally:
                stop.set()
            if getattr(server, "should_exit", False):
                # Whatever asked — a signal uvicorn handled, or a cancel
                # relayed by the heartbeat thread — this was a stop, and
                # the row has to say so rather than `complete`.
                progress.request_stop("stopped by signal")
    return progress.interrupt_requested


@contextmanager
def _record_stop_intent(progress: Any) -> Iterator[None]:
    """Own the stop signals just deeply enough to survive uvicorn's.

    uvicorn installs its own SIGINT/SIGTERM handlers while serving,
    saves whatever was there before, restores them on the way out, and
    then **re-raises the signal it captured** so the process ends the way
    it would have without the capture. That last step is the problem: on
    SIGTERM, "the way it would have" is immediate death, so `__exit__`
    never runs and the row is left saying `running` until something reaps
    it — a clean `systemctl stop` reading as an abandoned process.

    So a handler that only records intent is installed *underneath*
    uvicorn's. uvicorn overwrites it for the duration, restores it, and
    re-raises into it; it sets the flag and returns, and the reporter
    gets to record `interrupted` and print the resume command.

    If uvicorn does not capture signals at all (it declines off the main
    thread), this handler is simply the live one, and the heartbeat
    thread turns the flag into `should_exit`. Either way the reporter
    itself stays out of it — two owners racing for SIGINT is how a
    Ctrl-C gets swallowed.
    """
    from siteloom.progress import STOP_SIGNALS

    def on_stop(signum: int, frame: Any) -> None:
        progress.request_stop(f"{signal.Signals(signum).name} received")

    previous: dict[int, Any] = {}
    for signum in STOP_SIGNALS:
        try:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, on_stop)
        except ValueError:  # not the main thread; nothing to own
            previous.pop(signum, None)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _build_server(app: Any, *, host: str, port: int, log_level: str) -> Any:
    """A uvicorn Server, not `uvicorn.run`.

    `uvicorn.run` builds and runs in one call, which leaves nothing to
    hold — and `should_exit` on the server object is the only way a
    cancel arriving from another terminal can end this process without a
    signal.
    """
    import uvicorn

    return uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level=log_level.lower())
    )


def _run(server: Any) -> None:
    server.run()


def _heartbeat_loop(progress: Any, server: Any, stop: threading.Event) -> None:
    """Keep the row warm, and relay a cancel into uvicorn's shutdown.

    `siteloom jobs cancel` on a `serve` row calls `request_stop`, which
    sets the same flag a first Ctrl-C would. Nothing else is watching it
    in this process, so this loop is what turns it into a shutdown —
    otherwise `cancel` would report success against a server that keeps
    serving, which is the failure mode `test_jobs_control.py` exists to
    prevent.
    """
    from siteloom.service import notify

    announced = False
    while not stop.wait(SERVE_HEARTBEAT_S):
        try:
            progress.heartbeat()
        except Exception as exc:  # never let telemetry take the server down
            log.debug("serve heartbeat failed: %s", exc)
        # READY=1 only once the sockets are actually bound. Under
        # Type=notify this is what `systemctl start` is waiting on, so
        # sending it early would be a lie with consequences; under
        # anything else NOTIFY_SOCKET is unset and this is a no-op.
        if not announced and getattr(server, "started", False):
            announced = True
            notify.ready()
        if progress.interrupt_requested and not getattr(server, "should_exit", False):
            log.info("stop requested — shutting the server down")
            notify.stopping()
            server.should_exit = True
