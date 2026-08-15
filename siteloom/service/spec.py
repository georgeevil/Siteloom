"""What a Siteloom service is, independent of who supervises it.

`ServiceSpec` is to the renderers what `health.Check` is to `doctor`'s
Rich table: a set of facts with no idea how they will be written down.
Both backends render one, `service print-unit` prints the result, and the
tests bind against a spec built with no I/O — which is the only reason
any of this is checkable on a machine with neither `launchctl` nor
`systemctl` installed.
"""

from __future__ import annotations

import getpass
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The long-lived commands worth supervising. Everything else in the CLI
#: either finishes or is resumable by hand.
UNITS = ("serve", "run", "frigate")

#: Stop timeouts, per unit, as a floor over `service.stop_timeout_s`.
#: `run` and `frigate` drain per-camera / per-event work and commit in
#: batches; `serve` only has to close connections. docs/operations.md
#: makes the same argument in prose: a batch's worth of time, so 30 s is
#: generous for a request and 5 s is not enough for an index batch.
#: "Only has to close connections" is not free, though — `serve` sizes
#: uvicorn's graceful-shutdown deadline off this value (`serve.py`), so
#: lowering it shortens the window a `/live` stream has to be released
#: in, and the two must stay ordered: graceful deadline < SIGKILL.
STOP_TIMEOUT_FLOOR_S = {"serve": 0, "run": 60, "frigate": 60}

#: 128 + SIGINT. Every interruptible command exits this when it was
#: stopped rather than finished — including `serve`, which records the
#: stop and then says so. A supervisor that reads it as a crash restarts
#: the thing the operator just cancelled, so systemd is told; launchd has
#: no way to express it, which the docs say plainly.
INTERRUPTED_EXIT = 130


@dataclass(frozen=True)
class ServiceSpec:
    unit: str
    label: str
    description: str
    #: Absolute argv, ready to exec. Never a shell string — neither
    #: backend runs a shell, and pretending otherwise is how a config
    #: path with a space becomes two arguments.
    program: list[str]
    #: `siteloom doctor` argv, or empty to skip the preflight.
    preflight: list[str]
    working_dir: Path
    environment: dict[str, str]
    config_path: Path
    site_id: str
    scope: str = "user"
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    restart: str = "on-failure"
    restart_delay_s: int = 10
    start_limit_burst: int = 5
    start_limit_interval_s: int = 300
    stop_timeout_s: int = 30
    #: Exit codes the supervisor must treat as a clean stop.
    success_exit_status: tuple[int, ...] = ()
    start_at_boot: bool = True
    notify: bool = False
    #: Account a system-scope unit runs as. Empty at user scope, where
    #: the manager already is the user.
    run_as: str = ""
    read_write_paths: tuple[Path, ...] = ()
    #: Stamped into the unit as the generated-by marker: `install`
    #: refuses to replace a file that does not carry one, and
    #: `uninstall` refuses to remove it.
    generator: str = "siteloom"
    vector_db_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def generator_id() -> str:
    """`siteloom <version>`, for the generated-by marker."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"siteloom {version('siteloom')}"
    except PackageNotFoundError:  # a checkout that was never installed
        return "siteloom"


def siteloom_program() -> list[str]:
    """How to invoke this Siteloom, absolutely.

    A unit that says `siteloom` and hopes resolves it against systemd's
    minimal PATH, or launchd's, which is the classic "works in my shell"
    failure. The console script next to the running interpreter is the
    normal answer; `-m siteloom.cli` covers a checkout run without one
    (`cli.py` has the `__main__` hook for exactly this).
    """
    candidate = Path(sys.executable).with_name("siteloom")
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return [str(candidate)]
    return [sys.executable, "-m", "siteloom.cli"]


def _command_params(unit: str) -> list[Any]:
    """The click parameters of the `siteloom <unit>` command.

    Read off the real command rather than listed here, for the reason
    `_resume_command` exists: a hand-written flag list is a copy that
    silently stops matching the day someone adds an option.
    """
    import typer.main

    from siteloom.cli import app

    command = typer.main.get_command(app)
    sub = command.commands[unit.replace("_", "-")]  # type: ignore[attr-defined]
    return list(sub.params)


def unit_argv(unit: str, values: dict[str, Any]) -> list[str]:
    """`siteloom <unit>` with `values`, as absolute argv."""
    from siteloom.cli_library import invocation_tokens

    return [*siteloom_program(), unit, *invocation_tokens(_command_params(unit), values)]


def _sqlite_dir(db_url: str, base: Path) -> Path | None:
    """The directory holding a sqlite database, if that is what this is.

    A directory, not the file: SQLite writes `-wal` and `-shm` siblings,
    so a sandbox granted the file alone fails on the first write. A
    non-sqlite URL contributes nothing, which is correct — a Postgres
    deployment needs no local write path for it.
    """
    if not db_url.startswith("sqlite"):
        return None
    _, _, tail = db_url.partition("///")
    if not tail or tail == ":memory:":
        return None
    path = Path(tail)
    # `storage.db_url` is deliberately not anchored (config.py: "it is a
    # URL, not a path"), so a relative one follows the working directory
    # — which the unit sets to the config's own directory.
    return (path if path.is_absolute() else base / path).parent


def _read_write_paths(config: Any, base: Path, log_dir: Path) -> tuple[Path, ...]:
    """Every directory this deployment writes to.

    Emitted even though the default `ProtectSystem=full` does not need
    it: it costs nothing there, and it is the list an operator who edits
    `print-unit` output up to `ProtectSystem=strict` would otherwise have
    to assemble by hand and get wrong.
    """
    paths: list[Path] = [Path(config.storage.media_dir), log_dir]
    if config.identity.enabled and config.identity.vector_db_path:
        paths.append(Path(config.identity.vector_db_path))
    if getattr(config.training, "output_dir", ""):
        paths.append(Path(config.training.output_dir))
    db_dir = _sqlite_dir(config.storage.db_url, base)
    if db_dir is not None:
        paths.append(db_dir)
    models = os.environ.get("SITELOOM_MODELS_DIR")
    if models:
        paths.append(Path(models))
    seen: dict[str, Path] = {}
    for path in paths:
        resolved = path if path.is_absolute() else base / path
        seen.setdefault(str(resolved), resolved)
    return tuple(seen.values())


def _label(unit: str, site_id: str, platform: str) -> str:
    """A stable name for this unit, in the platform's own idiom.

    The site id is in it because the PRD's direction is multi-site: two
    deployments on one machine is a supported shape, and two units
    fighting over one label is not.
    """
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in site_id).strip("-")
    if platform == "darwin":
        return f"dev.siteloom.{slug}.{unit}" if slug else f"dev.siteloom.{unit}"
    return f"siteloom-{slug}-{unit}" if slug else f"siteloom-{unit}"


def spec_from_config(
    config: Any,
    unit: str,
    *,
    platform: str,
    scope: str = "user",
    host: str | None = None,
    port: int | None = None,
    start_at_boot: bool | None = None,
    notify: bool = False,
    preflight: bool = True,
) -> ServiceSpec:
    """Everything the renderers need, resolved from one `SiteConfig`.

    Pure apart from reading `sys.executable` and `SITELOOM_MODELS_DIR`;
    given a config it does no filesystem or subprocess work at all.
    """
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}; expected one of {', '.join(UNITS)}")
    config_path = Path(getattr(config, "_source_path", "site.yaml")).resolve()
    # Not os.getcwd(): the operator ran `service install` from wherever
    # they happened to be, and _ANCHORED_PATHS (CLD-64) makes the config
    # file's own directory the one place every relative path in the YAML
    # resolves correctly. It is load-bearing rather than cosmetic because
    # `storage.db_url` is explicitly not anchored, so a relative
    # `sqlite:///siteloom.db` follows this directory and nothing else.
    working_dir = config_path.parent
    svc = config.service
    log_dir = Path(svc.log_dir)
    if not log_dir.is_absolute():
        log_dir = working_dir / log_dir

    # launchd's StandardOutPath is not rotated by launchd, and journald
    # is not everywhere. The project's own rotating handler (10 MB x 3)
    # is, so every unit names a file for it and the platform's own
    # streams stay a crash channel.
    values: dict[str, Any] = {
        "config": str(config_path),
        "log_file": str(log_dir / f"{unit}.log"),
    }
    if unit == "serve":
        # Only an install-time override travels in the unit; otherwise
        # the unit reads `service.host`/`service.port` from the config,
        # so there is one place to change the port.
        if host is not None and host != svc.host:
            values["host"] = host
        if port is not None and port != svc.port:
            values["port"] = port
    else:
        values["quiet"] = True  # no Rich bar into a log file

    environment = {
        "SITELOOM_CONFIG": str(config_path),
        # Functional as well as a marker: a process can tell it was
        # started by a unit rather than by hand.
        "SITELOOM_SERVICE_UNIT": unit,
    }
    models = os.environ.get("SITELOOM_MODELS_DIR")
    if models:
        # A service that falls back to the home cache tries to download
        # weights the operator already pre-seeded (docs/operations.md).
        environment["SITELOOM_MODELS_DIR"] = models

    label = _label(unit, config.site_id, platform)

    return ServiceSpec(
        unit=unit,
        label=label,
        description=f"Siteloom {_DESCRIPTIONS[unit]} ({config.site_id})",
        program=unit_argv(unit, values),
        preflight=(
            [*siteloom_program(), "doctor", "--config", str(config_path)]
            if preflight
            else []
        ),
        working_dir=working_dir,
        environment=environment,
        config_path=config_path,
        site_id=config.site_id,
        scope=scope,
        stdout_path=log_dir / f"{label}.out",
        stderr_path=log_dir / f"{label}.err",
        restart=svc.restart,
        restart_delay_s=svc.restart_delay_s,
        stop_timeout_s=max(svc.stop_timeout_s, STOP_TIMEOUT_FLOOR_S[unit]),
        success_exit_status=(INTERRUPTED_EXIT,),
        start_at_boot=svc.start_at_boot if start_at_boot is None else start_at_boot,
        notify=notify and unit == "serve",
        # The installing operator, not root: a system unit running
        # Siteloom as root would write crops and a database that the
        # operator who ran every other command cannot then open.
        run_as=getpass.getuser() if scope == "system" else "",
        generator=generator_id(),
        read_write_paths=_read_write_paths(config, working_dir, log_dir),
        vector_db_path=(
            config.identity.vector_db_path if config.identity.enabled else ""
        ),
    )


_DESCRIPTIONS = {
    "serve": "web UI",
    "run": "camera ingestion",
    "frigate": "Frigate event consumer",
}
