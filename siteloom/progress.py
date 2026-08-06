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
import time
from collections import defaultdict
from contextlib import contextmanager
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

from siteloom.store import OperationRun

log = logging.getLogger(__name__)

HEARTBEAT_S = 2.0  # DB write interval
LOG_INTERVAL_S = 15.0  # non-TTY log line interval


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
        self._prev_sigint = None

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
            )
            session.add(run)
            session.commit()
            self.run_id = run.id
        self._prev_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._on_sigint)
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
        self._close_phase()
        if self._progress is not None:
            self._progress.stop()
        if self._prev_sigint is not None:
            signal.signal(signal.SIGINT, self._prev_sigint)

        if exc_type is Interrupted or self.interrupt_requested:
            self.finish("interrupted", "stopped by user")
            self._print_resume()
            return exc_type is Interrupted  # swallow only our own signal
        if exc_type is not None:
            self.finish("failed", f"{exc_type.__name__}: {exc}")
            return False
        self.finish("complete")
        return False

    def _on_sigint(self, signum, frame) -> None:
        # First Ctrl-C asks the loop to stop cleanly; a second one is the
        # user insisting, so restore default handling and let it through.
        if self.interrupt_requested:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        self.interrupt_requested = True
        self.console.print(
            "\n[yellow]Interrupt received — finishing the current batch and "
            "saving progress. Press Ctrl-C again to abort immediately.[/]"
        )

    def check_interrupt(self) -> None:
        """Call at a safe point (after a commit) to honour Ctrl-C."""
        if self.interrupt_requested:
            raise Interrupted

    # -- phases ------------------------------------------------------------

    @contextmanager
    def phase(self, name: str, total: int = 0):
        self._close_phase()
        self._phase_name = name
        self._phase_started = time.monotonic()
        self._current, self._total = 0, total
        if self._progress is not None:
            self._task_id = self._progress.add_task(name, total=total or None, extra="")
        else:
            log.info("%s: starting (%s items)", name, total or "unknown")
        self._beat(force=True)
        try:
            yield self
        finally:
            self._close_phase()

    def _close_phase(self) -> None:
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
        self._total = total
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, total=total or None)

    # -- ticking -----------------------------------------------------------

    def advance(self, n: int = 1, **counters: int) -> None:
        self._current += n
        for key, value in counters.items():
            self.counters[key] += value
        extra = self._extra_text()
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=self._current, extra=extra)
        self._beat()
        self._maybe_log()

    def bump(self, **counters: int) -> None:
        """Update counters without advancing the item count."""
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
            self.console.print(
                f"\n[green]Progress saved.[/] Resume with:\n  "
                f"[bold]{self.resume_command}[/]"
            )

    # -- summary -----------------------------------------------------------

    def summary_lines(self) -> list[str]:
        total_elapsed = time.monotonic() - self._started
        lines = [f"elapsed {humanize(total_elapsed)}"]
        for name, seconds in self.phase_timings.items():
            lines.append(f"  {name}: {humanize(seconds)}")
        return lines


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
