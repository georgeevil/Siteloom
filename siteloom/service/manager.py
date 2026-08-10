"""Talking to the supervisor, and to the unit files on disk.

Everything with a side effect lives here so that `spec.py` and the two
renderers stay pure. The subprocess boundary is a single injectable
`Runner`, which is what lets the CLI be tested without a service manager
— the same seam as `FrigateConsumer`'s injected snapshot fetcher.
"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from siteloom.service import launchd, systemd
from siteloom.service.spec import ServiceSpec

#: LSB init-script convention, so `siteloom service status` is usable in
#: a monitoring script rather than only by eye.
STATUS_RUNNING = 0
STATUS_STOPPED = 3
#: LSB calls 4 "status unknown", which covers both "no unit here" and
#: "the manager itself could not be reached". Both are the operator's
#: problem to look at; neither may be reported as "stopped", because a
#: monitoring script reads stopped as a thing to restart.
STATUS_NOT_INSTALLED = 4

#: What `real_runner` returns when the manager binary is not there at all.
NO_MANAGER = 127


class ServiceError(RuntimeError):
    """Something the operator has to decide about, not a stack trace."""


@dataclass
class Completed:
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[Sequence[str]], Completed]


def real_runner(argv: Sequence[str]) -> Completed:
    try:
        proc = subprocess.run(list(argv), capture_output=True, text=True)
    except OSError as exc:
        # A Linux container without systemd, or a PATH that has lost
        # launchctl. The platform said this manager should be here, so
        # this is worth a sentence rather than a traceback — and it must
        # not be mistaken for "the service is stopped".
        return Completed(127, "", f"{argv[0]}: {exc}")
    return Completed(proc.returncode, proc.stdout, proc.stderr)


@dataclass
class InstalledUnit:
    """What a marked unit file on disk says about itself."""

    path: Path
    label: str
    unit: str
    config_path: Path
    site_id: str = ""


class Backend:
    name = ""

    def __init__(self, runner: Runner | None = None) -> None:
        self.run = runner or real_runner

    # -- files -------------------------------------------------------------

    def unit_path(self, spec: ServiceSpec) -> Path:
        raise NotImplementedError

    def render(self, spec: ServiceSpec) -> str:
        raise NotImplementedError

    def read_marker(self, path: Path) -> InstalledUnit | None:
        """The unit's own record of who wrote it, or None if we did not.

        `uninstall` refuses what it cannot recognise, and `install`
        refuses to clobber it — a hand-written unit at the same path is
        somebody's deliberate work.
        """
        raise NotImplementedError

    def installed(self, scope: str) -> list[InstalledUnit]:
        directory = self.unit_dir(scope)
        if not directory.is_dir():
            return []
        found = []
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            marker = self.read_marker(path)
            if marker is not None:
                found.append(marker)
        return found

    def unit_dir(self, scope: str) -> Path:
        raise NotImplementedError

    # -- verbs -------------------------------------------------------------

    def enable(self, spec: ServiceSpec) -> Completed:
        raise NotImplementedError

    def disable(self, spec: ServiceSpec) -> Completed:
        raise NotImplementedError

    def start(self, spec: ServiceSpec) -> Completed:
        raise NotImplementedError

    def stop(self, spec: ServiceSpec) -> Completed:
        raise NotImplementedError

    def restart(self, spec: ServiceSpec) -> Completed:
        raise NotImplementedError

    def status(self, spec: ServiceSpec) -> tuple[int, str]:
        """(LSB code, one-line detail)."""
        raise NotImplementedError

    def logs_argv(self, spec: ServiceSpec, lines: int, follow: bool) -> list[str]:
        raise NotImplementedError

    def reload(self, scope: str = "user") -> None:
        """Tell the manager a unit file changed. A no-op where it is."""


class SystemdBackend(Backend):
    name = "systemd"

    def unit_dir(self, scope: str) -> Path:
        return systemd.unit_dir(scope)

    def unit_path(self, spec: ServiceSpec) -> Path:
        return systemd.unit_path(spec)

    def render(self, spec: ServiceSpec) -> str:
        return systemd.render(spec)

    def read_marker(self, path: Path) -> InstalledUnit | None:
        if path.suffix != ".service":
            return None
        try:
            text = path.read_text()
        except OSError:
            return None
        if systemd.MARKER_KEY not in text:
            return None
        fields = dict(
            re.findall(r"^X-Siteloom-(Unit|Config|Site)=(.*)$", text, re.MULTILINE)
        )
        if "Unit" not in fields or "Config" not in fields:
            return None
        return InstalledUnit(
            path=path,
            label=path.stem,
            unit=fields["Unit"].strip(),
            config_path=Path(fields["Config"].strip()),
            site_id=fields.get("Site", "").strip(),
        )

    def _ctl(self, spec: ServiceSpec, *args: str) -> list[str]:
        return self._ctl_scope(spec.scope, *args)

    def _ctl_scope(self, scope: str, *args: str) -> list[str]:
        return ["systemctl", *(["--user"] if scope == "user" else []), *args]

    def reload(self, scope: str = "user") -> None:
        self.run(self._ctl_scope(scope, "daemon-reload"))

    def enable(self, spec: ServiceSpec) -> Completed:
        return self.run(self._ctl(spec, "enable", spec.label))

    def disable(self, spec: ServiceSpec) -> Completed:
        return self.run(self._ctl(spec, "disable", spec.label))

    def start(self, spec: ServiceSpec) -> Completed:
        return self.run(self._ctl(spec, "start", spec.label))

    def stop(self, spec: ServiceSpec) -> Completed:
        return self.run(self._ctl(spec, "stop", spec.label))

    def restart(self, spec: ServiceSpec) -> Completed:
        return self.run(self._ctl(spec, "restart", spec.label))

    def status(self, spec: ServiceSpec) -> tuple[int, str]:
        if not self.unit_path(spec).exists():
            return STATUS_NOT_INSTALLED, "not installed"
        result = self.run(self._ctl(spec, "is-active", spec.label))
        # `is-active` prints the state word on stdout even when it exits
        # non-zero, so an empty stdout means it never got as far as
        # having an opinion — no systemd binary, or no bus to ask (a
        # container, a login session without one). That is "unknown",
        # not "stopped": a monitoring script reads stopped as something
        # to restart, and there is nothing here that could.
        state = result.stdout.strip()
        if not state:
            detail = result.stderr.strip() or "no answer from systemctl"
            return STATUS_NOT_INSTALLED, detail
        return (STATUS_RUNNING if state == "active" else STATUS_STOPPED), state

    def logs_argv(self, spec: ServiceSpec, lines: int, follow: bool) -> list[str]:
        scope = ["--user"] if spec.scope == "user" else []
        argv = ["journalctl", *scope, "-u", spec.label, "-n", str(lines)]
        if follow:
            argv.append("-f")
        return argv


class LaunchdBackend(Backend):
    name = "launchd"

    def __init__(self, runner: Runner | None = None, uid: int | None = None) -> None:
        super().__init__(runner)
        import os

        self.uid = os.getuid() if uid is None else uid

    def unit_dir(self, scope: str) -> Path:
        return launchd.agent_dir(scope)

    def unit_path(self, spec: ServiceSpec) -> Path:
        return launchd.unit_path(spec)

    def render(self, spec: ServiceSpec) -> str:
        return launchd.render(spec)

    def read_marker(self, path: Path) -> InstalledUnit | None:
        if path.suffix != ".plist":
            return None
        try:
            data = launchd.parse(path.read_text())
        except Exception:
            return None
        env = data.get("EnvironmentVariables") or {}
        unit = env.get(launchd.MARKER_ENV)
        config = env.get("SITELOOM_CONFIG")
        if not unit or not config:
            return None
        return InstalledUnit(
            path=path,
            label=str(data.get("Label", path.stem)),
            unit=str(unit),
            config_path=Path(str(config)),
        )

    def _target(self, spec: ServiceSpec) -> str:
        domain = "system" if spec.scope == "system" else f"gui/{self.uid}"
        return f"{domain}/{spec.label}"

    def _domain(self, spec: ServiceSpec) -> str:
        return "system" if spec.scope == "system" else f"gui/{self.uid}"

    def enable(self, spec: ServiceSpec) -> Completed:
        # `bootstrap` is the modern load: it registers the job with the
        # domain. `load`/`unload` still exist but are deprecated and
        # silently do nothing in some states, which is worse than an
        # error because the operator thinks the service is running.
        return self.run(
            ["launchctl", "bootstrap", self._domain(spec), str(self.unit_path(spec))]
        )

    def disable(self, spec: ServiceSpec) -> Completed:
        return self.run(["launchctl", "bootout", self._target(spec)])

    def start(self, spec: ServiceSpec) -> Completed:
        return self.run(["launchctl", "kickstart", self._target(spec)])

    def stop(self, spec: ServiceSpec) -> Completed:
        return self.run(["launchctl", "kill", "SIGTERM", self._target(spec)])

    def restart(self, spec: ServiceSpec) -> Completed:
        return self.run(["launchctl", "kickstart", "-k", self._target(spec)])

    def status(self, spec: ServiceSpec) -> tuple[int, str]:
        if not self.unit_path(spec).exists():
            return STATUS_NOT_INSTALLED, "not installed"
        result = self.run(["launchctl", "print", self._target(spec)])
        if result.returncode == NO_MANAGER:
            return STATUS_NOT_INSTALLED, result.stderr.strip() or "launchctl not found"
        if result.returncode != 0:
            return STATUS_STOPPED, "not loaded"
        match = re.search(r"^\s*state\s*=\s*(\S+)", result.stdout, re.MULTILINE)
        state = match.group(1) if match else "loaded"
        return (
            (STATUS_RUNNING if state == "running" else STATUS_STOPPED),
            state,
        )

    def logs_argv(self, spec: ServiceSpec, lines: int, follow: bool) -> list[str]:
        # launchd has no journal; the crash channel is the file the plist
        # names, and the real log is the project's own rotating one.
        argv = ["tail", "-n", str(lines)]
        if follow:
            argv.append("-f")
        paths = [p for p in (spec.stderr_path, spec.stdout_path) if p is not None]
        return [*argv, *(str(p) for p in paths)]


def backend_for_platform(
    runner: Runner | None = None, system: str | None = None
) -> Backend:
    name = (system or platform.system()).lower()
    if name == "darwin":
        return LaunchdBackend(runner)
    if name == "linux":
        return SystemdBackend(runner)
    raise ServiceError(
        f"siteloom service supports launchd (macOS) and systemd (Linux); "
        f"on {name or 'this platform'}, use `siteloom service print-unit` as a "
        f"starting point, or run `siteloom serve` under your own supervisor"
    )
