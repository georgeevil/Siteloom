"""Preflight and health checks (`siteloom doctor`, `/readyz`).

Most Siteloom failures are environmental rather than logical: a vector
store already held by another process, model weights that never
downloaded, a media directory that is not writable, a database from
before a column existed. Each one surfaces deep inside a long run, as a
stack trace, minutes after the operator walked away.

This module answers "is this deployment fit to run?" up front, from one
place, so the CLI and the web app agree on the answer. Every check
returns a `Check` rather than raising: a diagnostic that dies on the
first problem hides the other four.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

OK = "ok"
WARN = "warn"
FAIL = "fail"

# Below this, an archive index or a training export will fail partway.
MIN_FREE_GB = 2.0


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    # What to do about it. Empty when there is nothing to do.
    remedy: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warned(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARN]

    @property
    def ok(self) -> bool:
        return not self.failed

    def add(self, name: str, status: str, detail: str = "", remedy: str = "") -> Check:
        check = Check(name, status, detail, remedy)
        self.checks.append(check)
        return check

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail}
                for c in self.checks
            ],
        }


def hostname() -> str:
    return socket.gethostname()


# -- process identity (CLD-57) ----------------------------------------------
#
# A pid alone does not name a process: the OS recycles pids, so the pid an
# `OperationRun` recorded can be worn by something entirely unrelated by
# the time anyone acts on the row. `jobs cancel` sends a signal to that
# pid, which on a single-operator box quite plausibly means the
# operator's own shell. What distinguishes this process from a future one
# wearing its pid is when it started, which the OS knows and never
# reuses within a pid's lifetime.
#
# Both `OperationRun.is_stale` and `progress.request_cancel` ask the same
# question — "is that still the process that wrote this row?" — and both
# get it from `process_verdict`, so they cannot come to disagree.

#: `pid` is the process that recorded the token.
MATCH = "match"
#: The pid now belongs to something else: recycled. Provably not the run.
MISMATCH = "mismatch"
#: No such process (or never a valid pid).
NO_PROCESS = "no_process"
#: The process is there, but the row never recorded an identity — every
#: row written before this column existed, and any row written where the
#: platform could not answer. Nothing can be proven either way.
UNRECORDED = "unrecorded"
#: This host cannot report process start times, so no live pid can be
#: matched against anything. Also unprovable, but for a reason the
#: operator can act on differently.
UNREADABLE = "unreadable"

#: The verdicts that mean "provably nothing is working on this any more".
#: The other two are *unprovable*, not dead: a run that cannot show its
#: identity is judged by its heartbeat instead, never reaped on suspicion.
PROVEN_GONE = (NO_PROCESS, MISMATCH)

_PROC = Path("/proc")


def process_alive(pid: int) -> bool:
    """Is this pid a live process on this host?

    Signal 0 checks for existence without delivering anything. A pid
    owned by another user raises PermissionError — it exists, which is
    the question being asked. Pid reuse can make a dead process look
    alive, so this answers "is *something* there", never "is it the
    process I mean" — `process_verdict` answers that one.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def process_identity(pid: int) -> str | None:
    """A token for the process *instance* wearing `pid` right now.

    Three answers, and the difference between the last two is the whole
    point of this function:

    * a token — stable for the life of the process, never reused by a
      later process on the same pid;
    * ``None`` — there is no such process;
    * ``""`` — there is a process, but this platform (or these
      permissions) will not say when it started. Not a token, and must
      never be compared as one: "" == "" would make every unknown
      process match every other.
    """
    if pid <= 0:
        return None
    if (_PROC / "self" / "stat").exists():
        return _proc_identity(pid)
    return _ps_identity(pid)


def _proc_identity(pid: int) -> str | None:
    """Linux: field 22 of /proc/<pid>/stat, the start time in clock ticks
    since boot. Cheap, exact, and no subprocess."""
    try:
        raw = (_PROC / str(pid) / "stat").read_text()
    except OSError:
        # Gone is the common case. But /proc entries can also be hidden
        # (hidepid) from a process that can still see the pid at all, and
        # "hidden" must not be reported as "dead" — that would make a
        # live run look reapable.
        return "" if process_alive(pid) else None
    # comm (field 2) is parenthesised and may itself contain spaces and
    # ')', so the fields are counted from the *last* ')'.
    _, _, rest = raw.rpartition(")")
    fields = rest.split()
    if len(fields) < 20:  # rest starts at field 3, so field 22 is index 19
        return ""
    return f"start:{fields[19]}"


def _ps_identity(pid: int) -> str | None:
    """macOS/BSD: `ps -o lstart=`, the one start-time source available
    without adding a dependency (psutil) to read one integer.

    Second granularity, which is ample: recycling a pid inside the same
    second requires exhausting the whole pid space in that second. It
    forks, which is affordable because only a row that still claims to be
    `running` ever asks — `is_stale` returns before this on every other.
    """
    try:
        proc = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "" if process_alive(pid) else None
    started = " ".join(proc.stdout.split())
    if proc.returncode != 0 or not started:
        return "" if process_alive(pid) else None
    return f"lstart:{started}"


def process_verdict(pid: int, recorded: str) -> str:
    """Is `pid` still the process that recorded `recorded`?

    Returns one of MATCH / MISMATCH / NO_PROCESS / UNRECORDED /
    UNREADABLE. Only MATCH means yes; the other four are all reasons not
    to signal it. In particular an absent token is never treated as a
    match — "probably fine" is exactly what signals the wrong process.
    """
    if pid <= 0:
        return NO_PROCESS
    current = process_identity(pid)
    if current is None:
        return NO_PROCESS
    if not current:
        return UNREADABLE
    if not recorded:
        return UNRECORDED
    return MATCH if current == recorded else MISMATCH


# -- individual checks ------------------------------------------------------


def check_database(report: Report, config) -> None:
    from sqlalchemy import inspect, select

    from siteloom.store import Base, get_session, init_db, make_engine

    url = config.storage.db_url
    try:
        engine = make_engine(url)
        init_db(engine)  # additive migrations; safe and idempotent
        expected = {t.name for t in Base.metadata.sorted_tables}
        present = set(inspect(engine).get_table_names())
        missing = sorted(expected - present)
        if missing:
            report.add(
                "database",
                FAIL,
                f"{url} is missing tables: {', '.join(missing)}",
                "siteloom init-db --config <cfg>",
            )
            return
        with get_session(engine)() as session:
            session.execute(select(1))
        report.add("database", OK, f"{url} ({len(expected)} tables)")
    except Exception as exc:
        report.add(
            "database",
            FAIL,
            f"{url}: {type(exc).__name__}: {exc}",
            "check storage.db_url and that the directory exists",
        )


def check_media_dir(report: Report, config) -> None:
    path = Path(config.storage.media_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".siteloom-write-probe"
        probe.write_text("")
        probe.unlink()
    except Exception as exc:
        report.add(
            "media dir",
            FAIL,
            f"{path} is not writable: {exc}",
            "fix permissions or repoint storage.media_dir",
        )
        return

    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < MIN_FREE_GB:
        report.add(
            "media dir",
            WARN,
            f"{path} writable, {free_gb:.1f} GB free",
            f"below {MIN_FREE_GB:.0f} GB — a long index or export may fail partway",
        )
    else:
        report.add("media dir", OK, f"{path}, {free_gb:.1f} GB free")


def check_vector_store(report: Report, config) -> None:
    """Embedded Qdrant is one client per path per machine — so this
    check is also the answer to "why won't my job start?"."""
    if not config.identity.enabled:
        report.add("vector store", OK, "identity disabled")
        return
    path = config.identity.vector_db_path
    try:
        from siteloom.identity import VectorStore

        store = VectorStore(path)
        store.close()
        report.add("vector store", OK, f"{path} (embedded qdrant, openable)")
    except RuntimeError:
        # The expected contention case: serve, frigate, or another CLI
        # job has it open.
        report.add(
            "vector store",
            FAIL,
            f"{path} is held by another process",
            "stop the other siteloom process (pgrep -fl siteloom) "
            "or point it at a different identity.vector_db_path",
        )
    except Exception as exc:
        report.add("vector store", FAIL, f"{path}: {type(exc).__name__}: {exc}")


def check_detection_model(report: Report, config) -> None:
    model = Path(config.detection.model)
    if model.exists():
        report.add("detector weights", OK, f"{model} ({model.stat().st_size / 1e6:.1f} MB)")
    else:
        # Ultralytics downloads a bare name like "yolo11n.pt" on first
        # use; a path that was meant to be local and isn't is the error.
        report.add(
            "detector weights",
            WARN if model.name == str(model) else FAIL,
            f"{model} not present in {Path.cwd()}",
            "ultralytics will download it on first use (needs network), "
            "or set detection.model to a local path",
        )


def check_face_models(report: Report, config) -> None:
    """Cached weights are verified by digest, not just counted.

    The cache is a directory any local process can write and these files
    decide identity, so "present" is not the question — "is it the file
    we pinned?" is. Read-only by design: this reports, `ensure_model`
    deletes. An unverifiable file is a FAIL, not a WARN; the face path
    will refuse to load it, so the deployment is not fit to run.
    """
    from siteloom.identity import embedders

    cache = embedders.models_dir()
    try:
        statuses = [embedders.check_cached_model(s, cache) for s in embedders.FACE_MODELS]
    except Exception as exc:  # unreadable directory, exotic filesystem
        report.add(
            "face models",
            FAIL,
            f"{cache} could not be inspected: {type(exc).__name__}: {exc}",
            f"check permissions on {cache}, or set SITELOOM_MODELS_DIR to a "
            "directory holding verified weights",
        )
        return

    bad = [s for s in statuses if s.state in ("corrupt", "unreadable")]
    if bad:
        report.add(
            "face models",
            FAIL,
            "; ".join(f"{s.path.name}: {s.detail}" for s in bad),
            "delete "
            + " ".join(str(s.path) for s in bad)
            + " and re-run to fetch a clean copy — a digest mismatch is a "
            "corrupted download, a tampered cache, or an upstream that "
            "republished the file; siteloom will not load it either way",
        )
        return

    missing = [s for s in statuses if s.state == "missing"]
    if missing:
        report.add(
            "face models",
            WARN,
            f"{len(statuses) - len(missing)}/{len(statuses)} cached in {cache}",
            "downloaded and verified automatically on first face operation "
            "(needs network); for an offline install pre-seed that directory "
            "or set SITELOOM_MODELS_DIR",
        )
        return

    report.add("face models", OK, f"{len(statuses)}/{len(statuses)} verified in {cache}")


def check_plate_ocr(report: Report, config) -> None:
    identifiers = config.identity.identifiers
    wanted = any(getattr(i, "plate_ocr", False) for i in identifiers.values())
    if not wanted:
        report.add("plate OCR", OK, "not requested by any identifier")
        return
    try:
        import fast_plate_ocr  # noqa: F401
        import open_image_models  # noqa: F401

        report.add("plate OCR", OK, "dependencies present")
    except ImportError:
        report.add(
            "plate OCR",
            WARN,
            "requested by config but not installed — vehicles fall back to visual re-ID",
            "pip install -r requirements-plates.txt",
        )


def check_jobs(report: Report, config) -> None:
    """Runs still marked `running` whose process is gone. They are what
    makes `siteloom jobs` untrustworthy until they are reaped."""
    from sqlalchemy import select

    from siteloom.store import OperationRun, get_session, make_engine

    try:
        Session = get_session(make_engine(config.storage.db_url))
        with Session() as session:
            runs = session.scalars(
                select(OperationRun).filter(OperationRun.status == "running")
            ).all()
            stale = [r for r in runs if r.is_stale]
            live = len(runs) - len(stale)
    except Exception as exc:
        report.add("jobs", WARN, f"could not read operation_runs: {exc}")
        return

    if stale:
        report.add(
            "jobs",
            WARN,
            f"{len(stale)} abandoned run(s) still marked running "
            f"(ids: {', '.join(str(r.id) for r in stale)})",
            "siteloom jobs reap --config <cfg>",
        )
    else:
        report.add("jobs", OK, f"{live} running, none abandoned")


def check_integrations(report: Report, config) -> None:
    integrations = config.integrations
    enabled = []
    if integrations.mqtt.enabled:
        enabled.append(f"mqtt://{integrations.mqtt.host}:{integrations.mqtt.port}")
    if integrations.frigate.enabled:
        enabled.append(f"frigate {integrations.frigate.api_url}")
    if integrations.recognition_api.enabled:
        enabled.append(
            "recognition-api"
            + ("" if integrations.recognition_api.api_key else " (no api key set)")
        )
    if integrations.webhooks:
        enabled.append(f"{len(integrations.webhooks)} webhook(s)")

    if integrations.frigate.enabled and not integrations.mqtt.enabled:
        report.add(
            "integrations",
            WARN,
            "frigate is enabled but mqtt is not — results will not be republished",
            "set integrations.mqtt.enabled: true",
        )
        return
    report.add("integrations", OK, ", ".join(enabled) or "none enabled")


def check_services(report: Report, config) -> None:
    """Installed service units, and whether two of them would collide.

    Reads unit *files* only — no `systemctl`, no `launchctl`. `doctor`
    runs as this unit's own `ExecStartPre`, so asking the manager about
    the service it is in the middle of starting is a question with no
    good answer and a plausible hang. Live state is
    `siteloom service status`, which is never on a boot path.
    """
    import platform as _platform

    from siteloom.config import load_config
    from siteloom.service import ServiceError, backend_for_platform

    try:
        backend = backend_for_platform(system=_platform.system().lower())
    except ServiceError:
        report.add("services", OK, "no supported service manager on this platform")
        return
    units = backend.installed("user") + backend.installed("system")
    if not units:
        report.add("services", OK, "no units installed")
        return

    # The same rule check_vector_store enforces at runtime, applied to
    # what is configured to start rather than to what is running: two
    # units sharing an embedded Qdrant directory can never both run, and
    # finding that out at boot is finding it out too late.
    by_store: dict[str, list[str]] = {}
    for unit in units:
        try:
            other = load_config(unit.config_path)
        except Exception:
            continue
        if other.identity.enabled and other.identity.vector_db_path:
            by_store.setdefault(other.identity.vector_db_path, []).append(unit.label)
    clashes = {path: labels for path, labels in by_store.items() if len(labels) > 1}
    names = ", ".join(u.label for u in units)
    if clashes:
        path, labels = next(iter(clashes.items()))
        report.add(
            "services",
            FAIL,
            f"{' and '.join(labels)} both use identity.vector_db_path {path}",
            "embedded qdrant is one client per path per machine — point one at a "
            "different identity.vector_db_path, disable identity on one, or move "
            "to a qdrant server",
        )
        return
    report.add("services", OK, f"{len(units)} installed: {names}")


# -- entry points -----------------------------------------------------------

# Ordered cheapest-and-most-fundamental first, so the output reads as a
# dependency chain: nothing below matters if the database is unreachable.
CHECKS = [
    check_database,
    check_media_dir,
    check_vector_store,
    check_detection_model,
    check_face_models,
    check_plate_ocr,
    check_jobs,
    check_integrations,
    check_services,
]

# What a readiness probe can answer in milliseconds, without opening the
# vector store the serving process already holds.
LIVE_CHECKS = [check_database, check_media_dir, check_jobs]


def run_checks(config, checks=None) -> Report:
    report = Report()
    for check in checks or CHECKS:
        try:
            check(report, config)
        except Exception as exc:  # a broken check must not mask the rest
            report.add(check.__name__, FAIL, f"check crashed: {type(exc).__name__}: {exc}")
    return report
