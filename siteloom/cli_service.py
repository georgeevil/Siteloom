"""`siteloom service` — install and control Siteloom under launchd/systemd.

One verb set over two supervisors. Nothing here decides *what* a unit
says — that is `siteloom/service/spec.py` — and nothing here talks to a
supervisor directly; `manager.Backend` owns the subprocess boundary so
these commands can be tested without one.

Read-only with respect to the vector store, deliberately: `service
status` is the command you reach for while the server is holding
embedded Qdrant, which is exactly when a full bootstrap would fail
(cli_library._light_setup says the same thing about `jobs`).
"""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path

import typer

from siteloom.cli_library import CONFIG_OPT

service_app = typer.Typer(
    help="Install and control Siteloom under the OS service manager."
)

UNIT_OPT = typer.Option("serve", "--unit", "-u", help="serve | run | frigate")
SCOPE_OPT = typer.Option(
    "user", "--scope", help="user (default) | system (needs root; prints instructions)"
)
HOST_OPT = typer.Option(None, "--host", help="Override service.host for this unit")
PORT_OPT = typer.Option(None, "--port", help="Override service.port for this unit")


def _spec(config, unit, scope, host=None, port=None, notify=False, start_at_boot=None):
    """Load the config and resolve a spec, without opening anything else."""
    from siteloom.config import load_config
    from siteloom.service import spec_from_config

    try:
        cfg = load_config(config)
    except Exception as exc:
        typer.echo(f"could not load {config}: {exc}", err=True)
        raise typer.Exit(2) from exc
    try:
        spec = spec_from_config(
            cfg,
            unit,
            # The same answer the backend was chosen with. Read
            # `platform.system()` a second time here and a test (or a
            # cross-platform render) can end up with a systemd label on a
            # launchd job, which loads under a name nothing can address.
            platform="darwin" if _is_darwin() else "linux",
            scope=scope,
            host=host,
            port=port,
            notify=notify,
            start_at_boot=start_at_boot,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    return cfg, spec


def _backend(runner=None):
    from siteloom.service import ServiceError, backend_for_platform

    try:
        return backend_for_platform(runner)
    except ServiceError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


@service_app.command("print-unit")
def service_print_unit(
    config: Path = CONFIG_OPT,
    unit: str = UNIT_OPT,
    scope: str = SCOPE_OPT,
    host: str = HOST_OPT,
    port: int = PORT_OPT,
    notify: bool = typer.Option(
        False, "--notify", help="systemd Type=notify (serve only; see notify.py)"
    ),
):
    """Render the unit this config would install, and write nothing.

    The review step, not a convenience: an operator who wants to read the
    directives before trusting them, or to hand it to config management,
    gets the exact text with no side effect. Works on every platform,
    including ones with neither supervisor, because rendering is pure.
    """
    from siteloom.service import launchd, systemd

    _, spec = _spec(config, unit, scope, host, port, notify)
    renderer = launchd if _is_darwin() else systemd
    typer.echo(renderer.render(spec), nl=False)


def _is_darwin() -> bool:
    import platform as _platform

    return _platform.system().lower() == "darwin"


def _vector_store_conflict(backend, spec, cfg):
    """Another installed unit that would fight this one for Qdrant.

    `health.check_vector_store` already diagnoses this — but at runtime,
    minutes into whatever the operator started. Catching it when the unit
    is written costs one config load per installed unit and turns a 3am
    restart failure into a refusal at install time.
    """
    from siteloom.config import load_config

    if not spec.vector_db_path:
        return None
    for other in backend.installed(spec.scope):
        if other.label == spec.label:
            continue
        try:
            other_cfg = load_config(other.config_path)
        except Exception:
            continue
        if not other_cfg.identity.enabled:
            continue
        if other_cfg.identity.vector_db_path == spec.vector_db_path:
            return other
    return None


@service_app.command("install")
def service_install(
    config: Path = CONFIG_OPT,
    unit: str = UNIT_OPT,
    scope: str = SCOPE_OPT,
    host: str = HOST_OPT,
    port: int = PORT_OPT,
    start_at_boot: bool = typer.Option(
        None,
        "--start-at-boot/--no-start-at-boot",
        help="Start with the machine (systemd) or at login (launchd agent)",
    ),
    notify: bool = typer.Option(False, "--notify", help="systemd Type=notify (serve)"),
    start: bool = typer.Option(True, "--start/--no-start", help="Start it now"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before replacing"),
    force: bool = typer.Option(
        False, "--force", help="Replace a unit siteloom did not write; ignore a clash"
    ),
):
    """Write the unit for `--unit` and hand it to the service manager."""
    cfg, spec = _spec(config, unit, scope, host, port, notify, start_at_boot)
    backend = _backend()
    if scope == "system" and _is_darwin():
        # Writing to /Library/LaunchDaemons and bootstrapping the system
        # domain both need root, and this CLI does not invoke sudo on an
        # operator's behalf. Render and tell them what to run.
        typer.echo(backend.render(spec), nl=False)
        typer.echo(
            f"\n# Not installed: --scope system needs root on macOS. To install:\n"
            f"#   sudo cp <the above> /Library/LaunchDaemons/{spec.label}.plist\n"
            f"#   sudo launchctl bootstrap system "
            f"/Library/LaunchDaemons/{spec.label}.plist",
            err=True,
        )
        raise typer.Exit(1)

    path = backend.unit_path(spec)
    body = backend.render(spec)
    if path.exists():
        existing = path.read_text()
        if existing == body:
            typer.echo(f"unchanged: {path}")
        else:
            marker = backend.read_marker(path)
            if marker is None and not force:
                typer.echo(
                    f"{path} exists and was not written by siteloom — refusing to "
                    f"replace it. Move it aside, or pass --force.",
                    err=True,
                )
                raise typer.Exit(1)
            if not yes:
                diff = difflib.unified_diff(
                    existing.splitlines(),
                    body.splitlines(),
                    fromfile=f"{path} (installed)",
                    tofile="generated",
                    lineterm="",
                )
                for line in diff:
                    typer.echo(line)
                if not typer.confirm(f"replace {path}?"):
                    raise typer.Exit(1)
            _write(path, body)
            typer.echo(f"updated: {path}")
    else:
        clash = _vector_store_conflict(backend, spec, cfg)
        if clash is not None and not force:
            typer.echo(
                f"{clash.label} already uses identity.vector_db_path "
                f"{spec.vector_db_path!r}.\n"
                f"Embedded Qdrant allows one client per path per machine, so the "
                f"two units cannot run together. Point one at a different "
                f"identity.vector_db_path, disable identity on one of them, or "
                f"move to a Qdrant server. --force installs anyway.",
                err=True,
            )
            raise typer.Exit(1)
        _write(path, body)
        typer.echo(f"installed: {path}  (fingerprint {_fingerprint(body)})")

    for directory in {p for p in (spec.stdout_path, spec.stderr_path) if p}:
        directory.parent.mkdir(parents=True, exist_ok=True)

    backend.reload(spec.scope)
    if spec.start_at_boot:
        backend.enable(spec)
    if start:
        result = backend.start(spec)
        if result.returncode != 0:
            typer.echo(
                f"unit written, but starting it failed: "
                f"{(result.stderr or result.stdout).strip()}",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(f"started: {spec.label}")
    _linger_hint(spec)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _linger_hint(spec) -> None:
    """A user unit does not survive logout without lingering.

    Reported, never fixed: `loginctl enable-linger` is a change to the
    login manager's behaviour for the whole account, and quietly making
    it on an operator's behalf — with sudo — is not this command's
    business.
    """
    import getpass
    import platform as _platform
    import shutil
    import subprocess

    if spec.scope != "user" or not spec.start_at_boot:
        return
    if _platform.system().lower() != "linux" or not shutil.which("loginctl"):
        return
    user = getpass.getuser()
    try:
        out = subprocess.run(
            ["loginctl", "show-user", user, "--property=Linger"],
            capture_output=True,
            text=True,
        ).stdout
    except OSError:
        return
    if "Linger=no" in out:
        typer.echo(
            f"note: this user unit will stop at logout and not start at boot.\n"
            f"      sudo loginctl enable-linger {user}",
            err=True,
        )


@service_app.command("uninstall")
def service_uninstall(
    config: Path = CONFIG_OPT,
    unit: str = UNIT_OPT,
    scope: str = SCOPE_OPT,
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask"),
    force: bool = typer.Option(False, "--force", help="Remove an unmarked unit too"),
):
    """Stop the service and remove its unit."""
    _, spec = _spec(config, unit, scope)
    backend = _backend()
    path = backend.unit_path(spec)
    if not path.exists():
        typer.echo(f"not installed: {path}")
        return
    if backend.read_marker(path) is None and not force:
        typer.echo(
            f"{path} was not written by siteloom — refusing to remove it. "
            f"Pass --force if you are sure.",
            err=True,
        )
        raise typer.Exit(1)
    if not yes and not typer.confirm(f"stop {spec.label} and remove {path}?"):
        raise typer.Exit(1)
    backend.stop(spec)
    backend.disable(spec)
    path.unlink()
    backend.reload(spec.scope)
    typer.echo(f"removed: {path}")


def _verb(config, unit, scope, action: str):
    _, spec = _spec(config, unit, scope)
    backend = _backend()
    if not backend.unit_path(spec).exists():
        typer.echo(
            f"{spec.label} is not installed — `siteloom service install "
            f"--unit {unit}` first",
            err=True,
        )
        raise typer.Exit(1)
    result = getattr(backend, action)(spec)
    if result.returncode != 0:
        typer.echo((result.stderr or result.stdout).strip(), err=True)
        raise typer.Exit(1)
    typer.echo(f"{action}: {spec.label}")


@service_app.command("start")
def service_start(config: Path = CONFIG_OPT, unit: str = UNIT_OPT, scope: str = SCOPE_OPT):
    """Start the service now."""
    _verb(config, unit, scope, "start")


@service_app.command("stop")
def service_stop(config: Path = CONFIG_OPT, unit: str = UNIT_OPT, scope: str = SCOPE_OPT):
    """Stop the service. SIGTERM, so an in-flight batch commits first."""
    _verb(config, unit, scope, "stop")


@service_app.command("restart")
def service_restart(
    config: Path = CONFIG_OPT, unit: str = UNIT_OPT, scope: str = SCOPE_OPT
):
    """Stop and start the service."""
    _verb(config, unit, scope, "restart")


@service_app.command("status")
def service_status(
    config: Path = CONFIG_OPT,
    unit: str = UNIT_OPT,
    scope: str = SCOPE_OPT,
    all_units: bool = typer.Option(False, "--all", help="Every installed Siteloom unit"),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable"),
):
    """Report whether the service is installed and running.

    Exits 0 running, 3 stopped, 4 not installed — the LSB init-script
    convention, so this is usable from a monitoring script and not only
    by eye.
    """
    import json as _json

    from siteloom.service import STATUS_NOT_INSTALLED, STATUS_RUNNING

    _, spec = _spec(config, unit, scope)
    backend = _backend()
    rows = []
    if all_units:
        for installed in backend.installed(scope):
            probe = _spec(installed.config_path, installed.unit, scope)[1]
            code, detail = backend.status(probe)
            rows.append((probe, code, detail))
        if not rows:
            typer.echo("no siteloom units installed")
            raise typer.Exit(STATUS_NOT_INSTALLED)
    else:
        code, detail = backend.status(spec)
        rows.append((spec, code, detail))

    if json_out:
        typer.echo(
            _json.dumps(
                [
                    {
                        "unit": s.unit,
                        "label": s.label,
                        "config": str(s.config_path),
                        "status": detail,
                        "code": code,
                        "path": str(backend.unit_path(s)),
                    }
                    for s, code, detail in rows
                ],
                indent=2,
            )
        )
    else:
        for s, code, detail in rows:
            typer.echo(f"{s.label:<40} {detail:<16} {backend.unit_path(s)}")
        _live_rows(config, [s.unit for s, _, _ in rows])

    # With --all the exit code is the worst state seen: a monitoring
    # script asking "is this deployment up?" wants one answer.
    raise typer.Exit(max(code for _, code, _ in rows) if rows else STATUS_RUNNING)


def _live_rows(config, units) -> None:
    """The OperationRun rows behind these units, if the DB is readable.

    The manager knows whether a process exists; the row knows whether it
    is doing anything and whether it went stale. Best-effort — a status
    command must not fail because the database moved.
    """
    try:
        from sqlalchemy import select

        from siteloom.cli_library import _light_setup
        from siteloom.store import OperationRun

        _, Session = _light_setup(config, level="WARNING")
        with Session() as session:
            runs = session.scalars(
                select(OperationRun)
                .filter(OperationRun.status == "running")
                .filter(OperationRun.kind.in_(list(units)))
            ).all()
            for run in runs:
                state = "stale" if run.is_stale else "running"
                typer.echo(f"  job {run.id} {run.kind} {state} pid {run.pid} {run.host}")
    except Exception:
        return


@service_app.command("logs")
def service_logs(
    config: Path = CONFIG_OPT,
    unit: str = UNIT_OPT,
    scope: str = SCOPE_OPT,
    lines: int = typer.Option(50, "--lines", "-n"),
    follow: bool = typer.Option(False, "--follow", "-f"),
):
    """Tail this service's logs (journalctl, or the plist's log files)."""
    import subprocess

    _, spec = _spec(config, unit, scope)
    backend = _backend()
    argv = backend.logs_argv(spec, lines, follow)
    if backend.name == "launchd":
        typer.echo(
            "# launchd has no journal; showing the crash streams and the "
            "rotating log.\n"
            "# For launchd's own messages: "
            "log show --predicate 'process == \"siteloom\"' --last 1h",
            err=True,
        )
    raise typer.Exit(subprocess.run(argv).returncode)
