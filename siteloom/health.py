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


def process_alive(pid: int) -> bool:
    """Is this pid a live process on this host?

    Signal 0 checks for existence without delivering anything. A pid
    owned by another user raises PermissionError — it exists, which is
    the question being asked. Pid reuse can make a dead process look
    alive; callers therefore treat True as "not proven dead" and keep
    their timeout as the backstop.
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


def hostname() -> str:
    return socket.gethostname()


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
