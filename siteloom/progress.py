"""Progress reporting for long-running operations.

Importing a 30,000-photo archive takes tens of minutes. Three things make
that humane rather than opaque, and this module provides all three from
one object:

1. **A live view where you launched it** — a Rich progress bar with rate
   and ETA on a terminal, or periodic log lines when output is redirected
   (background runs, cron, CI), so neither case is silent.
2. **A live view from anywhere else** — every phase tick heartbeats an
   OperationRun row, so `siteloom jobs`, the /jobs page, or another shell
   can watch a run they did not start. A process that dies leaves its
   last position behind instead of vanishing.
3. **A safe exit** — Ctrl-C finishes the current batch, commits, records
   the run as interrupted, and prints the exact command to resume.

Usage:

    with ProgressReporter(Session, "takeout-import", target=str(root),
                          resume_command="siteloom takeout import ...") as p:
        with p.phase("Scanning files", total=n):
            for f in files:
                ...
                p.advance(faces=len(faces))
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from siteloom.health import (
    MATCH,
    MISMATCH,
    NO_PROCESS,
    UNREADABLE,
    hostname,
    process_identity,
    process_verdict,
)
from siteloom.store import OperationRun

log = logging.getLogger(__name__)

HEARTBEAT_S = 2.0  # DB write interval
LOG_INTERVAL_S = 15.0  # non-TTY log line interval

# Every signal that means "stop": Ctrl-C, a service manager stopping the
# unit, and the terminal going away. A job that only honours SIGINT
# survives the operator and dies to the machine, which is backwards.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def humanize(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


class Interrupted(Exception):
    """Raised inside the work loop after Ctrl-C so callers can commit."""


#: Reporters heartbeating inside *this* process, keyed by run id.
#:
#: A job started from the console runs in a background thread of the
#: serving process, where the signal handler below could not be
#: installed (see `__enter__`) and where a signal would hit the web
#: server rather than the job. `request_cancel` therefore looks here
#: first and sets the same flag the handler sets, so a cancel means one
#: thing whether it comes from a terminal, another host's terminal, or
#: the browser.
_LIVE_REPORTERS: dict[int, "ProgressReporter"] = {}
_REGISTRY_LOCK = threading.Lock()


class ProgressReporter:
    def __init__(
        self,
        session_factory,
        kind: str,
        target: str = "",
        resume_command: str = "",
        console: Console | None = None,
        enabled: bool = True,
        bar: bool = True,
        signals: bool = True,
    ):
        self.Session = session_factory
        self.kind = kind
        self.target = target
        self.resume_command = resume_command
        self.enabled = enabled
        self.console = console or Console(stderr=True)
        # `bar` hides the live bar; `enabled` switches the whole reporter
        # off. They are separate because the heartbeat and the Ctrl-C
        # handler are what make a run observable and resumable — a user
        # who only wants a quieter terminal must not silently lose them.
        self.interactive = self.console.is_terminal and enabled and bar
        # `signals=False` leaves the process's stop handlers alone, for a
        # caller whose runtime already owns them — uvicorn installs its
        # own SIGINT/SIGTERM handling and drives the shutdown. Two owners
        # racing for SIGINT is how a Ctrl-C gets swallowed, which is the
        # same hazard `IngestService._stop_signals(active=…)` exists for,
        # with ownership inverted. It is a third switch, not a shade of
        # the other two: the row and the heartbeat stay on, and the
        # reporter still registers in _LIVE_REPORTERS, so `jobs cancel`
        # still reaches it. The caller owns polling `interrupt_requested`.
        self.signals = signals

        self.counters: dict[str, int] = defaultdict(int)
        self.phase_timings: dict[str, float] = {}
        self.run_id: int | None = None
        self.interrupt_requested = False

        self._progress: Progress | None = None
        self._task_id = None
        self._phase_name = ""
        self._phase_started = 0.0
        self._current = 0
        self._total = 0
        self._last_beat = 0.0
        self._last_log = 0.0
        self._started = time.monotonic()
        self._prev_handlers: dict[int, object] = {}
        # One reporter, many tickers: live ingest runs a worker thread per
        # camera (CLD-15) and they all advance the same counters. Reentrant
        # because advance() -> _beat() re-enters, and held across the
        # heartbeat write so a run's row never shows a half-applied tick.
        self._lock = threading.RLock()

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> ProgressReporter:
        if not self.enabled:
            return self
        with self.Session() as session:
            run = OperationRun(
                kind=self.kind,
                target=self.target,
                status="running",
                started_at=_now(),
                updated_at=_now(),
                resume_command=self.resume_command,
                pid=os.getpid(),
                host=hostname(),
                # Which process, not just which pid: without it, anyone
                # acting on this row later can only hope the pid still
                # means us (CLD-57). "" where the platform will not say,
                # which `request_cancel` reads as "cannot prove it".
                process_start=process_identity(os.getpid()) or "",
            )
            session.add(run)
            session.commit()
            self.run_id = run.id
        with _REGISTRY_LOCK:
            _LIVE_REPORTERS[self.run_id] = self
        for signum in STOP_SIGNALS if self.signals else ():
            try:
                self._prev_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._on_stop)
            except ValueError:
                # Not the main thread (embedded callers); the run is
                # still tracked, it just cannot be stopped by signal.
                self._prev_handlers.pop(signum, None)
        if self.interactive:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TextColumn("{task.fields[extra]}"),
                TimeElapsedColumn(),
                TextColumn("eta"),
                TimeRemainingColumn(),
                console=self.console,
                transient=False,
            )
            self._progress.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self._close_phase()
            if self._progress is not None:
                self._progress.stop()
            for signum, handler in self._prev_handlers.items():
                signal.signal(signum, handler)

            if exc_type is Interrupted or self.interrupt_requested:
                self.finish("interrupted", "stopped by user")
                self._print_resume()
                return exc_type is Interrupted  # swallow only our own signal
            if exc_type is not None:
                self.finish("failed", f"{exc_type.__name__}: {exc}")
                return False
            self.finish("complete")
            return False
        finally:
            # The row outlives the process; the registry entry must not —
            # a finished run is not cancellable and must never look it.
            with _REGISTRY_LOCK:
                _LIVE_REPORTERS.pop(self.run_id, None)

    def _on_stop(self, signum, frame) -> None:
        # The first signal asks the loop to stop cleanly; a second one is
        # the sender insisting, so restore default handling and let it
        # through. SIGTERM's default is death, which is the right answer
        # to a service manager that has already asked once.
        if self.interrupt_requested:
            signal.signal(signum, signal.SIG_DFL)
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            os.kill(os.getpid(), signum)
            return
        self.interrupt_requested = True
        name = signal.Signals(signum).name
        self.console.print(
            f"\n[yellow]{name} received — finishing the current batch and "
            f"saving progress. Send it again to abort immediately.[/]"
        )

    def request_stop(self, reason: str = "cancel requested") -> None:
        """Ask this run to stop cleanly, without a signal.

        Exactly what the first Ctrl-C does — set the flag the work loop
        checks after each commit — reachable from another thread of the
        same process. A console-started job has no signal handler of its
        own (it is not on the main thread) and shares its pid with the
        web server, so this is the only stop it can be given that does
        not take the server down with it.
        """
        with self._lock:
            if self.interrupt_requested:
                return
            self.interrupt_requested = True
        log.info(
            "run %s (%s): %s — finishing the current batch and saving progress",
            self.run_id,
            self.kind,
            reason,
        )

    def check_interrupt(self) -> None:
        """Call at a safe point (after a commit) to honour Ctrl-C."""
        if self.interrupt_requested:
            raise Interrupted

    # -- phases ------------------------------------------------------------

    @contextmanager
    def phase(self, name: str, total: int = 0):
        with self._lock:
            self._close_phase_locked()
            self._phase_name = name
            self._phase_started = time.monotonic()
            self._current, self._total = 0, total
            if self._progress is not None:
                self._task_id = self._progress.add_task(
                    name, total=total or None, extra=""
                )
            else:
                log.info("%s: starting (%s items)", name, total or "unknown")
            self._beat(force=True)
        try:
            yield self
        finally:
            self._close_phase()

    def _close_phase(self) -> None:
        with self._lock:
            self._close_phase_locked()

    def _close_phase_locked(self) -> None:
        if not self._phase_name:
            return
        elapsed = time.monotonic() - self._phase_started
        self.phase_timings[self._phase_name] = round(elapsed, 1)
        rate = self._current / elapsed if elapsed > 0 else 0.0
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=self._current)
        else:
            log.info(
                "%s: done — %d items in %s (%.1f/s)",
                self._phase_name,
                self._current,
                humanize(elapsed),
                rate,
            )
        self._beat(force=True)
        self._phase_name = ""
        self._task_id = None

    def set_total(self, total: int) -> None:
        with self._lock:
            self._total = total
            if self._progress is not None and self._task_id is not None:
                self._progress.update(self._task_id, total=total or None)

    # -- ticking -----------------------------------------------------------

    def advance(self, n: int = 1, **counters: int) -> None:
        with self._lock:
            self._current += n
            for key, value in counters.items():
                self.counters[key] += value
            extra = self._extra_text()
            if self._progress is not None and self._task_id is not None:
                self._progress.update(
                    self._task_id, completed=self._current, extra=extra
                )
            self._beat()
            self._maybe_log()

    def heartbeat(self) -> None:
        """Write the row without claiming any work happened.

        A daemon has no items to advance — `serve` least of all — but it
        still has to prove it is alive, because a row whose `updated_at`
        goes cold is exactly what `OperationRun.is_stale` reads. Forced,
        because the caller is a timer that already decided it is time.
        """
        self._beat(force=True)

    def bump(self, **counters: int) -> None:
        """Update counters without advancing the item count."""
        with self._lock:
            for key, value in counters.items():
                self.counters[key] += value

    def _extra_text(self) -> str:
        if not self.counters:
            return ""
        parts = [f"{k} {v}" for k, v in list(self.counters.items())[:4]]
        return "· " + " · ".join(parts) + " "

    def _maybe_log(self) -> None:
        if self.interactive:
            return
        now = time.monotonic()
        if now - self._last_log < LOG_INTERVAL_S:
            return
        self._last_log = now
        elapsed = now - self._phase_started
        rate = self._current / elapsed if elapsed > 0 else 0.0
        eta = (self._total - self._current) / rate if rate > 0 and self._total else None
        log.info(
            "%s: %d/%s (%.0f%%) %.1f/s eta %s %s",
            self._phase_name,
            self._current,
            self._total or "?",
            (self._current / self._total * 100) if self._total else 0,
            rate,
            humanize(eta),
            self._extra_text().strip("· "),
        )

    # -- persistence -------------------------------------------------------

    def _beat(self, force: bool = False) -> None:
        if self.run_id is None:
            return
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_beat < HEARTBEAT_S:
                return
            self._last_beat = now
            try:
                with self.Session() as session:
                    run = session.get(OperationRun, self.run_id)
                    if run is None:
                        return
                    run.phase = self._phase_name
                    run.current = self._current
                    run.total = self._total
                    run.counters = json.dumps(dict(self.counters))
                    run.phase_timings = json.dumps(self.phase_timings)
                    run.updated_at = _now()
                    session.commit()
            except Exception as exc:  # never let telemetry break the job
                log.debug("progress heartbeat failed: %s", exc)

    def finish(self, status: str, message: str = "") -> None:
        if self.run_id is None:
            return
        with self._lock:
            self._finish_locked(status, message)

    def _finish_locked(self, status: str, message: str) -> None:
        try:
            with self.Session() as session:
                run = session.get(OperationRun, self.run_id)
                if run is None:
                    return
                run.status = status
                run.message = message
                run.finished_at = _now()
                run.updated_at = _now()
                run.counters = json.dumps(dict(self.counters))
                run.phase_timings = json.dumps(self.phase_timings)
                session.commit()
        except Exception as exc:
            log.debug("progress finish failed: %s", exc)

    def _print_resume(self) -> None:
        if self.resume_command:
            self.console.print("\n[green]Progress saved.[/] Resume with:")
            # soft_wrap: a resume command broken across lines by the
            # console width is a command nobody can paste.
            self.console.print(f"  {self.resume_command}", soft_wrap=True, highlight=False)

    # -- summary -----------------------------------------------------------

    def summary_lines(self) -> list[str]:
        total_elapsed = time.monotonic() - self._started
        lines = [f"elapsed {humanize(total_elapsed)}"]
        for name, seconds in self.phase_timings.items():
            lines.append(f"  {name}: {humanize(seconds)}")
        return lines


# -- stopping and closing out runs ------------------------------------------
#
# `siteloom jobs cancel|reap` and the /jobs console both act on runs they
# did not start. Both go through the functions below rather than each
# reaching for a signal of its own: two mechanisms that stop a job
# differently are two definitions of what a cancelled run means, and the
# resume command an operator is handed afterwards depends on which one
# ran.

#: What a reaped row records. It is not a failure — the work that was
#: committed is still there, and the resume command still resumes it.
REAP_MESSAGE = "process gone; closed out by reap"


@dataclass
class CancelResult:
    """The outcome of asking a run to stop, honestly.

    `ok` is False whenever the run was *not* asked — a run on another
    host, a row whose process is already gone or whose pid has since
    been handed to something else, a run that already finished. Callers
    must render the reason rather than reporting a cancellation that
    never happened: an operator who is told "cancelled" and sees no
    change reaches for `kill -9` and loses the batch.
    """

    ok: bool
    #: Machine-readable: requested | killed | not_found | not_running |
    #: other_host | no_process | pid_reused | unverified
    reason: str
    detail: str
    run_id: int
    kind: str = ""
    pid: int = 0
    host: str = ""


def request_cancel(session, run_id: int, force: bool = False) -> CancelResult:
    """Ask a run to stop, from anywhere that can see the database.

    A cancel is a request, not a kill: the run finishes its current
    batch, commits, records itself `interrupted` and keeps its resume
    command. `force` is the escape hatch — SIGKILL, losing the work since
    the last commit — and is deliberately not offered in-process, where
    it would take down whatever else that process is doing.

    Nothing is signalled until the row's pid is *proven* to still be the
    process that wrote it (CLD-57). A pid outlives its process, so the
    row's recorded process identity is checked first and a signal is
    refused on anything short of a match — including a row too old to
    carry one. Refusing costs an operator one `reap`; guessing costs
    whatever else happened to be wearing that pid, which on a
    single-operator box is likely their own shell.
    """
    run = session.get(OperationRun, run_id)
    if run is None:
        return CancelResult(False, "not_found", f"no run #{run_id}", run_id)
    if run.status != "running":
        return CancelResult(
            False,
            "not_running",
            f"run #{run_id} is already {run.status}",
            run_id,
            run.kind,
            run.pid,
            run.host,
        )
    pid, host, kind = run.pid, run.host, run.kind

    with _REGISTRY_LOCK:
        reporter = _LIVE_REPORTERS.get(run_id)
    # The row must also claim this process. Run ids are per-database, so
    # a registry hit alone would let a row restored from elsewhere, or
    # written by another deployment against another database, be
    # "cancelled" by stopping an unrelated local job.
    #
    # This path needs no process-identity check and must not grow one:
    # the reporter is a live object in *this* interpreter, registered
    # under the id of the row it itself created, so "is that still the
    # process that wrote this row" is answered by construction. Requiring
    # a token here would only break in-process cancel on a platform that
    # cannot produce one — and nothing is signalled either way.
    if reporter is not None and pid == os.getpid() and host in ("", hostname()):
        # Ours: a background thread of this very process. Signalling the
        # pid would hit the process, not the job.
        reporter.request_stop()
        note = (
            " (a hard kill would have taken this process down with it, "
            "so it was asked to stop instead)"
            if force
            else ""
        )
        return CancelResult(
            True,
            "requested",
            f"asked #{run_id} ({kind}) to stop{note}; it will finish the "
            f"current batch, save progress and keep its resume command",
            run_id,
            kind,
            pid,
            host,
        )

    if host and host != hostname():
        return CancelResult(
            False,
            "other_host",
            f"run #{run_id} belongs to {host}; cancel it there "
            f"(this is {hostname()})",
            run_id,
            kind,
            pid,
            host,
        )
    verdict = process_verdict(pid, run.process_start or "")
    if verdict == NO_PROCESS:
        return CancelResult(
            False,
            "no_process",
            f"run #{run_id} ({kind}) has no live process — it died without "
            f"recording an outcome; clear it with reap",
            run_id,
            kind,
            pid,
            host,
        )
    if verdict == MISMATCH:
        return CancelResult(
            False,
            "pid_reused",
            f"run #{run_id} ({kind}) recorded pid {pid}, but that pid now "
            f"belongs to a different process — the run is gone and the "
            f"pid was reused; signalling it would hit a stranger. Clear "
            f"the row with reap",
            run_id,
            kind,
            pid,
            host,
        )
    if verdict == UNREADABLE:
        return CancelResult(
            False,
            "unverified",
            f"run #{run_id} ({kind}): {hostname()} will not report process "
            f"start times, so pid {pid} cannot be shown to still be this "
            f"run — refusing rather than risk signalling an unrelated "
            f"process. Stop it where it runs (Ctrl-C), or reap the row "
            f"once its heartbeat goes cold",
            run_id,
            kind,
            pid,
            host,
        )
    if verdict != MATCH:  # UNRECORDED: written before pids were qualified
        return CancelResult(
            False,
            "unverified",
            f"run #{run_id} ({kind}) recorded no process identity — it "
            f"predates that column, so pid {pid} cannot be shown to still "
            f"be this run — refusing rather than risk signalling an "
            f"unrelated process. Stop it where it runs (Ctrl-C), or reap "
            f"the row once its heartbeat goes cold; runs started since "
            f"the upgrade cancel normally",
            run_id,
            kind,
            pid,
            host,
        )

    os.kill(pid, signal.SIGKILL if force else signal.SIGINT)
    if force:
        return CancelResult(
            True,
            "killed",
            f"killed #{run_id} ({kind}, pid {pid}) — work since the last "
            f"commit is lost",
            run_id,
            kind,
            pid,
            host,
        )
    return CancelResult(
        True,
        "requested",
        f"asked #{run_id} ({kind}, pid {pid}) to stop; it will finish the "
        f"current batch, save progress and keep its resume command",
        run_id,
        kind,
        pid,
        host,
    )


def stale_runs(session, run_ids: list[int] | None = None) -> list[OperationRun]:
    """Rows still saying `running` that nothing is working on any more.

    `is_stale` is a property, not a column (it asks the operating system
    about a pid), so the filtering happens in Python over the small set
    of rows that still claim to be running.
    """
    from sqlalchemy import select

    query = select(OperationRun).filter(OperationRun.status == "running")
    if run_ids is not None:
        if not run_ids:
            return []
        query = query.filter(OperationRun.id.in_(run_ids))
    return [run for run in session.scalars(query).all() if run.is_stale]


def reap_runs(session, runs) -> int:
    """Close dead rows out as `abandoned`, keeping what they knew.

    Position, counters and resume command are left exactly as the dead
    process last wrote them — reaping ends the ambiguity of a `running`
    row with nobody behind it, it does not discard the run.
    """
    runs = list(runs)
    for run in runs:
        run.status = "abandoned"
        run.message = REAP_MESSAGE
        run.finished_at = _now()
    session.commit()
    return len(runs)


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Console logging plus an optional rotating file.

    The console handler goes to stderr so a Rich progress bar (also
    stderr, and redraw-aware) and log lines do not corrupt each other,
    and so piping stdout stays clean.
    """
    from logging.handlers import RotatingFileHandler

    from rich.logging import RichHandler

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    if sys.stderr.isatty():
        console_handler: logging.Handler = RichHandler(
            console=Console(stderr=True), show_path=False, rich_tracebacks=True
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    root.addHandler(console_handler)

    if log_file:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10_000_000, backupCount=3
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root.addHandler(file_handler)

    # Third-party chatter that would otherwise drown the progress view.
    for noisy in ("PIL", "matplotlib", "urllib3", "httpx", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
