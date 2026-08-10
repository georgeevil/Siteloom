"""`siteloom service` against a fake supervisor (CLD-105).

The subprocess boundary is one injectable runner, so everything here
runs on a machine with neither `launchctl` nor `systemctl` — and, more
importantly, without installing anything into the developer's real
`~/.config/systemd/user`. Every test redirects the unit directory.

What is being held:

* **Nothing gets clobbered.** A unit siteloom did not write is somebody's
  deliberate work; install and uninstall both refuse it.
* **A collision is caught when the unit is written, not at 4am.**
  `health.check_vector_store` already diagnoses two processes fighting
  over embedded Qdrant, but only once one of them has failed to start.
* **`status` is scriptable.** LSB exit codes, so a monitoring script can
  ask without parsing prose.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from siteloom import cli_service
from siteloom.config import (
    IdentityConfig,
    ServiceConfig,
    SiteConfig,
    StorageConfig,
)
from siteloom.service import (
    STATUS_NOT_INSTALLED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    ServiceError,
    backend_for_platform,
)
from siteloom.service.manager import Completed, LaunchdBackend, SystemdBackend

runner = CliRunner()


class FakeRunner:
    """Records argv, answers with whatever the test set up."""

    def __init__(self, replies=None):
        self.calls: list[list[str]] = []
        self.replies = replies or {}

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        for needle, reply in self.replies.items():
            if needle in argv:
                return reply
        return Completed(0, "", "")

    def argv_containing(self, token):
        return [c for c in self.calls if token in c]


@pytest.fixture
def site(tmp_path):
    def _make(name="site.yaml", site_id="kai", vector_path="vectors", **service):
        cfg = SiteConfig(
            site_id=site_id,
            storage=StorageConfig(
                db_url=f"sqlite:///{tmp_path}/{site_id}.db", media_dir="media"
            ),
            identity=IdentityConfig(enabled=True, vector_db_path=vector_path),
            service=ServiceConfig(**service),
        )
        path = tmp_path / name
        path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
        return path

    return _make


@pytest.fixture
def units(tmp_path, monkeypatch):
    """A private unit directory, for both backends."""
    directory = tmp_path / "units"
    directory.mkdir()
    monkeypatch.setattr(SystemdBackend, "unit_dir", lambda self, scope: directory)
    monkeypatch.setattr(LaunchdBackend, "unit_dir", lambda self, scope: directory)
    monkeypatch.setattr(
        SystemdBackend,
        "unit_path",
        lambda self, spec: directory / f"{spec.label}.service",
    )
    monkeypatch.setattr(
        LaunchdBackend, "unit_path", lambda self, spec: directory / f"{spec.label}.plist"
    )
    return directory


@pytest.fixture
def fake(monkeypatch):
    """Install the fake runner and pin the platform to Linux."""
    fake = FakeRunner()
    monkeypatch.setattr(cli_service, "_backend", lambda runner=None: SystemdBackend(fake))
    monkeypatch.setattr(cli_service, "_is_darwin", lambda: False)
    return fake


# -- print-unit -------------------------------------------------------------


def test_print_unit_writes_nothing(site, units, fake):
    before = sorted(p.name for p in units.iterdir())
    result = runner.invoke(
        cli_service.service_app, ["print-unit", "--config", str(site())]
    )
    assert result.exit_code == 0
    assert "ExecStart=" in result.output
    assert sorted(p.name for p in units.iterdir()) == before
    assert not fake.calls  # and talks to no supervisor


def test_print_unit_works_where_nothing_is_supported(site, monkeypatch):
    """Rendering is pure, so the escape hatch for an unsupported platform
    is a real one rather than an apology."""
    monkeypatch.setattr(cli_service, "_is_darwin", lambda: False)
    result = runner.invoke(
        cli_service.service_app, ["print-unit", "--config", str(site())]
    )
    assert result.exit_code == 0


def test_unsupported_platform_names_both_managers():
    with pytest.raises(ServiceError) as exc:
        backend_for_platform(system="windows")
    assert "launchd" in str(exc.value) and "systemd" in str(exc.value)


# -- install ----------------------------------------------------------------


def test_install_writes_enables_and_starts(site, units, fake):
    config = site()
    result = runner.invoke(
        cli_service.service_app, ["install", "--config", str(config)]
    )
    assert result.exit_code == 0, result.output
    written = units / "siteloom-kai-serve.service"
    assert written.exists()
    assert "daemon-reload" in [c[-1] for c in fake.calls]
    assert fake.argv_containing("enable")
    assert fake.argv_containing("start")


def test_install_no_start_leaves_it_stopped(site, units, fake):
    result = runner.invoke(
        cli_service.service_app, ["install", "--config", str(site()), "--no-start"]
    )
    assert result.exit_code == 0
    assert not fake.argv_containing("start")


def test_install_refuses_a_unit_it_did_not_write(site, units, fake):
    config = site()
    path = units / "siteloom-kai-serve.service"
    handwritten = "[Service]\nExecStart=/usr/bin/true\n"
    path.write_text(handwritten)

    result = runner.invoke(
        cli_service.service_app, ["install", "--config", str(config)]
    )
    assert result.exit_code == 1
    assert "not written by siteloom" in result.output
    assert path.read_text() == handwritten  # untouched


def test_install_force_replaces_a_foreign_unit(site, units, fake):
    config = site()
    path = units / "siteloom-kai-serve.service"
    path.write_text("[Service]\nExecStart=/usr/bin/true\n")

    result = runner.invoke(
        cli_service.service_app,
        ["install", "--config", str(config), "--force", "--yes"],
    )
    assert result.exit_code == 0
    assert "ExecStart=/usr/bin/true" not in path.read_text()


def test_reinstalling_the_same_config_is_a_no_op(site, units, fake):
    config = site()
    runner.invoke(cli_service.service_app, ["install", "--config", str(config)])
    again = runner.invoke(cli_service.service_app, ["install", "--config", str(config)])
    assert again.exit_code == 0
    assert "unchanged" in again.output


def test_a_port_change_alone_does_not_change_the_unit(site, units, fake):
    """The unit reads the config, so there is one place to change a port.

    Editing `service.port` and reinstalling is a no-op on the unit file
    precisely because the port was never copied into it — which is the
    property that keeps a generated unit from going stale.
    """
    config = site()
    runner.invoke(cli_service.service_app, ["install", "--config", str(config)])
    site(port=9001)  # same path, rewritten
    again = runner.invoke(cli_service.service_app, ["install", "--config", str(config)])
    assert "unchanged" in again.output


def test_changed_unit_shows_a_diff_before_replacing(site, units, fake):
    config = site()
    runner.invoke(cli_service.service_app, ["install", "--config", str(config)])
    site(restart="always")  # a setting that *is* a directive
    result = runner.invoke(
        cli_service.service_app, ["install", "--config", str(config)], input="y\n"
    )
    assert result.exit_code == 0
    assert "--- " in result.output and "+++ " in result.output
    assert "updated" in result.output
    assert "Restart=always" in (units / "siteloom-kai-serve.service").read_text()


def test_declining_the_diff_leaves_the_unit_alone(site, units, fake):
    config = site()
    runner.invoke(cli_service.service_app, ["install", "--config", str(config)])
    before = (units / "siteloom-kai-serve.service").read_text()
    site(restart="always")
    result = runner.invoke(
        cli_service.service_app, ["install", "--config", str(config)], input="n\n"
    )
    assert result.exit_code == 1
    assert (units / "siteloom-kai-serve.service").read_text() == before


def test_install_refuses_a_second_unit_sharing_the_vector_store(
    site, units, fake, tmp_path
):
    """`health.check_vector_store` finds this at runtime, minutes into
    whatever the operator started. Finding it when the unit is written is
    strictly better and costs one config load."""
    serve_cfg = site(name="a.yaml", site_id="kai", vector_path=str(tmp_path / "shared"))
    runner.invoke(cli_service.service_app, ["install", "--config", str(serve_cfg)])

    ingest_cfg = site(
        name="b.yaml", site_id="other", vector_path=str(tmp_path / "shared")
    )
    result = runner.invoke(
        cli_service.service_app,
        ["install", "--config", str(ingest_cfg), "--unit", "run"],
    )
    assert result.exit_code == 1
    assert "one client per path per machine" in result.output
    assert not (units / "siteloom-other-run.service").exists()


def test_install_allows_distinct_vector_stores(site, units, fake, tmp_path):
    a = site(name="a.yaml", site_id="kai", vector_path=str(tmp_path / "one"))
    runner.invoke(cli_service.service_app, ["install", "--config", str(a)])
    b = site(name="b.yaml", site_id="other", vector_path=str(tmp_path / "two"))
    result = runner.invoke(
        cli_service.service_app, ["install", "--config", str(b), "--unit", "run"]
    )
    assert result.exit_code == 0
    assert (units / "siteloom-other-run.service").exists()


def test_install_force_overrides_the_collision(site, units, fake, tmp_path):
    a = site(name="a.yaml", site_id="kai", vector_path=str(tmp_path / "shared"))
    runner.invoke(cli_service.service_app, ["install", "--config", str(a)])
    b = site(name="b.yaml", site_id="other", vector_path=str(tmp_path / "shared"))
    result = runner.invoke(
        cli_service.service_app,
        ["install", "--config", str(b), "--unit", "run", "--force"],
    )
    assert result.exit_code == 0


# -- uninstall --------------------------------------------------------------


def test_uninstall_stops_disables_removes(site, units, fake):
    config = site()
    runner.invoke(cli_service.service_app, ["install", "--config", str(config)])
    fake.calls.clear()

    result = runner.invoke(
        cli_service.service_app, ["uninstall", "--config", str(config), "--yes"]
    )
    assert result.exit_code == 0
    assert not (units / "siteloom-kai-serve.service").exists()
    verbs = [c[2] for c in fake.calls if len(c) > 2]
    assert verbs.index("stop") < verbs.index("disable")


def test_uninstall_refuses_an_unmarked_unit(site, units, fake):
    config = site()
    path = units / "siteloom-kai-serve.service"
    path.write_text("[Service]\nExecStart=/usr/bin/true\n")
    result = runner.invoke(
        cli_service.service_app, ["uninstall", "--config", str(config), "--yes"]
    )
    assert result.exit_code == 1
    assert path.exists()


def test_uninstall_of_nothing_is_not_an_error(site, units, fake):
    result = runner.invoke(
        cli_service.service_app, ["uninstall", "--config", str(site()), "--yes"]
    )
    assert result.exit_code == 0
    assert "not installed" in result.output


# -- status -----------------------------------------------------------------


def _status_backend(monkeypatch, units, reply):
    fake = FakeRunner({"is-active": reply})
    monkeypatch.setattr(cli_service, "_backend", lambda runner=None: SystemdBackend(fake))
    monkeypatch.setattr(cli_service, "_is_darwin", lambda: False)
    return fake


def test_status_exit_codes_follow_lsb(site, units, monkeypatch):
    """0 running, 3 stopped, 4 not installed — so `siteloom service
    status` is usable from a monitoring script, not only by eye."""
    config = site()

    _status_backend(monkeypatch, units, Completed(0, "active\n", ""))
    assert (
        runner.invoke(
            cli_service.service_app, ["status", "--config", str(config)]
        ).exit_code
        == STATUS_NOT_INSTALLED
    )

    runner.invoke(cli_service.service_app, ["install", "--config", str(config)])
    assert (
        runner.invoke(
            cli_service.service_app, ["status", "--config", str(config)]
        ).exit_code
        == STATUS_RUNNING
    )

    _status_backend(monkeypatch, units, Completed(3, "inactive\n", ""))
    assert (
        runner.invoke(
            cli_service.service_app, ["status", "--config", str(config)]
        ).exit_code
        == STATUS_STOPPED
    )


def test_status_json(site, units, monkeypatch):
    import json

    config = site()
    _status_backend(monkeypatch, units, Completed(0, "active\n", ""))
    runner.invoke(cli_service.service_app, ["install", "--config", str(config)])
    result = runner.invoke(
        cli_service.service_app, ["status", "--config", str(config), "--json"]
    )
    payload = json.loads(result.output)
    assert payload[0]["unit"] == "serve"
    assert payload[0]["status"] == "active"


def test_status_all_reports_the_worst_state(site, units, monkeypatch, tmp_path):
    """A monitoring script asking "is this deployment up?" wants one
    answer, and the answer is the unhealthiest unit."""
    a = site(name="a.yaml", site_id="kai", vector_path=str(tmp_path / "one"))
    b = site(name="b.yaml", site_id="other", vector_path=str(tmp_path / "two"))
    _status_backend(monkeypatch, units, Completed(3, "inactive\n", ""))
    runner.invoke(cli_service.service_app, ["install", "--config", str(a)])
    runner.invoke(cli_service.service_app, ["install", "--config", str(b), "--unit", "run"])

    result = runner.invoke(
        cli_service.service_app, ["status", "--config", str(a), "--all"]
    )
    assert result.exit_code == STATUS_STOPPED
    assert "siteloom-kai-serve" in result.output
    assert "siteloom-other-run" in result.output


# -- verbs ------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["start", "stop", "restart"])
def test_verbs_reach_systemctl(site, units, fake, verb):
    config = site()
    runner.invoke(cli_service.service_app, ["install", "--config", str(config)])
    fake.calls.clear()
    result = runner.invoke(cli_service.service_app, [verb, "--config", str(config)])
    assert result.exit_code == 0
    assert fake.calls[-1] == ["systemctl", "--user", verb, "siteloom-kai-serve"]


def test_verbs_refuse_when_not_installed(site, units, fake):
    result = runner.invoke(
        cli_service.service_app, ["start", "--config", str(site())]
    )
    assert result.exit_code == 1
    assert "not installed" in result.output


def test_launchd_uses_the_modern_api(site, units, monkeypatch):
    """`load`/`unload` are deprecated and silently no-op in some states,
    which is worse than an error: the operator believes it worked."""
    fake = FakeRunner()
    backend = LaunchdBackend(fake, uid=501)
    monkeypatch.setattr(cli_service, "_backend", lambda runner=None: backend)
    monkeypatch.setattr(cli_service, "_is_darwin", lambda: True)

    config = site()
    runner.invoke(cli_service.service_app, ["install", "--config", str(config)])
    assert fake.argv_containing("bootstrap")
    fake.calls.clear()
    runner.invoke(cli_service.service_app, ["restart", "--config", str(config)])
    assert fake.calls[-1] == [
        "launchctl",
        "kickstart",
        "-k",
        "gui/501/dev.siteloom.kai.serve",
    ]
    assert not any("load" in c or "unload" in c for c in fake.calls)


def test_macos_system_scope_prints_instead_of_sudo(site, units, monkeypatch):
    """Installing into /Library/LaunchDaemons needs root, and this CLI
    does not invoke sudo on an operator's behalf."""
    fake = FakeRunner()
    monkeypatch.setattr(
        cli_service, "_backend", lambda runner=None: LaunchdBackend(fake, uid=501)
    )
    monkeypatch.setattr(cli_service, "_is_darwin", lambda: True)
    result = runner.invoke(
        cli_service.service_app,
        ["install", "--config", str(site()), "--scope", "system"],
    )
    assert result.exit_code == 1
    assert "sudo launchctl bootstrap system" in result.output
    assert not fake.calls


def test_bad_config_exits_two(tmp_path, units, fake):
    """Same code `doctor` uses for a config it cannot load."""
    broken = tmp_path / "broken.yaml"
    broken.write_text("site_id:\n  - not a string\n")
    result = runner.invoke(
        cli_service.service_app, ["print-unit", "--config", str(broken)]
    )
    assert result.exit_code == 2


def test_unknown_unit_exits_two(site, units, fake):
    result = runner.invoke(
        cli_service.service_app,
        ["print-unit", "--config", str(site()), "--unit", "backfill"],
    )
    assert result.exit_code == 2
    assert "unknown unit" in result.output


def test_launchd_status_reads_the_state_line(site, units, tmp_path):
    from siteloom.service import spec_from_config
    from siteloom.config import load_config

    spec = spec_from_config(load_config(site()), "serve", platform="darwin")
    (units / f"{spec.label}.plist").write_text("<plist/>")

    running = LaunchdBackend(
        FakeRunner({"print": Completed(0, "\tstate = running\n", "")}), uid=501
    )
    assert running.status(spec) == (STATUS_RUNNING, "running")

    # `launchctl print` failing means the job is not loaded in the
    # domain, which is stopped — not "unknown".
    unloaded = LaunchdBackend(
        FakeRunner({"print": Completed(113, "", "Could not find service")}), uid=501
    )
    assert unloaded.status(spec) == (STATUS_STOPPED, "not loaded")


def test_a_missing_manager_is_unknown_not_stopped(site, units, monkeypatch):
    """A Linux container with no systemd must not report the service as
    stopped — a monitoring script reads stopped as a thing to restart,
    and there is nothing here to restart it with."""
    from siteloom.service.manager import NO_MANAGER

    config = site()
    _status_backend(monkeypatch, units, Completed(0, "active\n", ""))
    runner.invoke(cli_service.service_app, ["install", "--config", str(config)])

    _status_backend(
        monkeypatch, units, Completed(NO_MANAGER, "", "systemctl: not found")
    )
    result = runner.invoke(cli_service.service_app, ["status", "--config", str(config)])
    assert result.exit_code == STATUS_NOT_INSTALLED
    assert "not found" in result.output


def test_a_manager_with_no_bus_is_unknown_too(site, units, monkeypatch):
    """The binary exists and cannot answer — a container, or a login
    session without a user bus. `is-active` prints the state word on
    stdout even when it exits non-zero, so an empty stdout means it never
    formed an opinion."""
    config = site()
    _status_backend(monkeypatch, units, Completed(0, "active\n", ""))
    runner.invoke(cli_service.service_app, ["install", "--config", str(config)])

    _status_backend(
        monkeypatch, units, Completed(1, "", "Failed to connect to bus: No medium found")
    )
    result = runner.invoke(cli_service.service_app, ["status", "--config", str(config)])
    assert result.exit_code == STATUS_NOT_INSTALLED
    assert "Failed to connect to bus" in result.output


def test_inactive_is_stopped_not_unknown(site, units, monkeypatch):
    """The ordinary stopped case still has to read as stopped."""
    config = site()
    _status_backend(monkeypatch, units, Completed(0, "active\n", ""))
    runner.invoke(cli_service.service_app, ["install", "--config", str(config)])

    _status_backend(monkeypatch, units, Completed(3, "inactive\n", ""))
    result = runner.invoke(cli_service.service_app, ["status", "--config", str(config)])
    assert result.exit_code == STATUS_STOPPED


def test_launchd_logs_tail_both_streams(site, units):
    from siteloom.service import spec_from_config
    from siteloom.config import load_config

    spec = spec_from_config(load_config(site()), "serve", platform="darwin")
    argv = LaunchdBackend(FakeRunner(), uid=501).logs_argv(spec, 20, follow=True)
    assert argv[:4] == ["tail", "-n", "20", "-f"]
    assert str(spec.stderr_path) in argv and str(spec.stdout_path) in argv


def test_systemd_logs_use_journalctl_at_the_right_scope(site, units):
    from siteloom.service import spec_from_config
    from siteloom.config import load_config

    cfg = load_config(site())
    user = spec_from_config(cfg, "serve", platform="linux")
    system = spec_from_config(cfg, "serve", platform="linux", scope="system")
    backend = SystemdBackend(FakeRunner())
    assert backend.logs_argv(user, 10, False) == [
        "journalctl",
        "--user",
        "-u",
        "siteloom-kai-serve",
        "-n",
        "10",
    ]
    assert "--user" not in backend.logs_argv(system, 10, False)


def test_installed_lists_only_marked_units(site, units, fake):
    config = site()
    runner.invoke(cli_service.service_app, ["install", "--config", str(config)])
    (units / "someone-elses.service").write_text("[Service]\nExecStart=/bin/true\n")

    backend = SystemdBackend(fake)
    found = backend.installed("user")
    assert [u.label for u in found] == ["siteloom-kai-serve"]
    assert found[0].config_path == Path(config).resolve()
