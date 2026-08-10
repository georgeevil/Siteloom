"""Turning a config into a service unit, correctly (CLD-105).

A generated unit is a promise that starting it means the same thing as
running the command by hand. Three ways that promise breaks, all of them
silent until a machine reboots at 4am:

* **A relative anything.** `siteloom` resolved against systemd's minimal
  PATH, a `--config` relative to a working directory the operator no
  longer remembers, a `media_dir` anchored to `/`. The unit has to carry
  absolute answers to all three.
* **A dropped flag.** The same bug `_resume_command` was written for: a
  hand-maintained flag list stops matching the command the day someone
  adds an option to it. So the argv is reflected off the real command's
  parameters, and this file has a test that fails if anyone replaces
  that with a list.
* **A wrong stop.** `siteloom jobs cancel` exits 130 by design; a
  supervisor told 130 is a crash restarts the job the operator just
  cancelled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import typer
import typer.main
import yaml

from siteloom.cli_library import invocation_tokens
from siteloom.config import (
    IdentityConfig,
    ServiceConfig,
    SiteConfig,
    StorageConfig,
    load_config,
)
from siteloom.service import spec_from_config
from siteloom.service.spec import (
    INTERRUPTED_EXIT,
    _sqlite_dir,
    siteloom_program,
    unit_argv,
)


@pytest.fixture
def config_file(tmp_path):
    cfg = SiteConfig(
        site_id="kai",
        storage=StorageConfig(
            db_url="sqlite:///siteloom.db", media_dir="media"
        ),
        identity=IdentityConfig(enabled=True, vector_db_path="vectors"),
        service=ServiceConfig(host="127.0.0.1", port=8000),
    )
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return path


@pytest.fixture
def config(config_file):
    return load_config(config_file)


def test_program_is_absolute(config):
    spec = spec_from_config(config, "serve", platform="linux")
    assert Path(spec.program[0]).is_absolute()
    # Bare "siteloom" resolved against a service manager's minimal PATH
    # is the classic "works in my shell" failure.
    assert spec.program[0] != "siteloom"


def test_program_falls_back_to_module_when_no_console_script(monkeypatch, tmp_path):
    """A checkout installed without the console script still gets a unit."""
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    assert siteloom_program() == [str(tmp_path / "python"), "-m", "siteloom.cli"]


def test_config_is_absolute_and_always_present(config, config_file):
    spec = spec_from_config(config, "serve", platform="linux")
    assert "--config" in spec.program
    passed = spec.program[spec.program.index("--config") + 1]
    assert Path(passed).is_absolute()
    assert Path(passed) == config_file.resolve()


def test_working_dir_is_the_config_dir_not_the_cwd(config, config_file, monkeypatch):
    """CLD-64, expressed as a unit directive.

    _ANCHORED_PATHS makes the config file's directory the one place every
    relative path in the YAML resolves correctly, and `storage.db_url` is
    explicitly *not* anchored — so a relative sqlite:/// follows this
    directory and nothing else. The directory the operator happened to
    run `service install` from is irrelevant.
    """
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    spec = spec_from_config(config, "serve", platform="linux")
    assert spec.working_dir == config_file.parent


def test_host_and_port_omitted_unless_overridden(config):
    """The unit reads the config, so there is one place to change a port."""
    same = spec_from_config(config, "serve", platform="linux", host="127.0.0.1", port=8000)
    assert "--port" not in same.program

    changed = spec_from_config(config, "serve", platform="linux", port=9000)
    assert changed.program[changed.program.index("--port") + 1] == "9000"


def test_models_dir_propagates_only_when_set(config, monkeypatch):
    monkeypatch.delenv("SITELOOM_MODELS_DIR", raising=False)
    assert "SITELOOM_MODELS_DIR" not in spec_from_config(
        config, "serve", platform="linux"
    ).environment

    # A service that falls back to the home cache tries to re-download
    # weights the operator pre-seeded (docs/operations.md).
    monkeypatch.setenv("SITELOOM_MODELS_DIR", "/opt/models")
    spec = spec_from_config(config, "serve", platform="linux")
    assert spec.environment["SITELOOM_MODELS_DIR"] == "/opt/models"
    assert Path("/opt/models") in spec.read_write_paths


def test_new_options_appear_without_touching_spec_py():
    """The `_resume_command` bug class, caught by construction.

    A hand-written flag list in spec.py would pass every other test in
    this file and silently stop carrying flags added later. Reflection
    over the command's own parameters cannot drift, and this asserts the
    mechanism rather than one of its outputs.
    """
    app = typer.Typer()

    @app.command()
    def demo(
        config: str = typer.Option("site.yaml", "--config"),
        invented_later: str = typer.Option(None, "--invented-later"),
    ):
        pass

    params = typer.main.get_command(app).params
    tokens = invocation_tokens(params, {"config": "c.yaml", "invented_later": "x"})
    assert "--invented-later" in tokens and "x" in tokens


def test_invocation_tokens_are_raw_so_both_callers_can_serialize(tmp_path):
    """Quoting is the caller's business.

    A resume line is read by a shell and must be quoted; an `ExecStart=`
    argv is not, and quoting it there would put literal quotes inside a
    filename. So the shared helper hands back the raw value and each
    caller does its own escaping.
    """
    import shlex

    app = typer.Typer()

    @app.command()
    def demo(config: str = typer.Option("site.yaml", "--config")):
        pass

    spacey = str(tmp_path / "my site.yaml")
    tokens = invocation_tokens(typer.main.get_command(app).params, {"config": spacey})
    assert tokens == ["--config", spacey]
    assert shlex.quote(tokens[1]) != tokens[1]


@pytest.mark.parametrize("unit", ["serve", "run", "frigate"])
def test_cancel_exit_code_is_a_clean_stop_everywhere(config, unit):
    """`jobs cancel` exits 130, `serve` included. A supervisor that reads
    that as a crash restarts the thing the operator just stopped."""
    assert spec_from_config(config, unit, platform="linux").success_exit_status == (
        INTERRUPTED_EXIT,
    )


def test_stop_timeout_floor_per_unit(config):
    """A batch's worth of time — docs/operations.md's argument as a number."""
    assert spec_from_config(config, "serve", platform="linux").stop_timeout_s == 30
    assert spec_from_config(config, "run", platform="linux").stop_timeout_s == 60


def test_read_write_paths_cover_every_write(config, config_file):
    spec = spec_from_config(config, "serve", platform="linux")
    base = config_file.parent
    for expected in (base / "media", base / "vectors", base / "logs"):
        assert expected in spec.read_write_paths
    # The sqlite *directory*, not the file: SQLite writes -wal and -shm
    # siblings, so a sandbox granted the file alone fails on first write.
    assert base in spec.read_write_paths


def test_non_sqlite_db_contributes_no_local_path(tmp_path):
    assert _sqlite_dir("postgresql://host/db", tmp_path) is None
    assert _sqlite_dir("sqlite:///:memory:", tmp_path) is None
    assert _sqlite_dir("sqlite:///data/s.db", tmp_path) == tmp_path / "data"


def test_labels_are_per_site_and_per_platform(config):
    assert spec_from_config(config, "serve", platform="darwin").label == (
        "dev.siteloom.kai.serve"
    )
    assert spec_from_config(config, "serve", platform="linux").label == (
        "siteloom-kai-serve"
    )


def test_unknown_unit_is_refused(config):
    with pytest.raises(ValueError, match="unknown unit"):
        spec_from_config(config, "backfill", platform="linux")


def test_units_log_to_the_rotating_handler(config, config_file):
    """launchd does not rotate StandardOutPath, so the unit always names
    a file for the project's own rotating handler and the platform
    streams stay a crash channel."""
    for unit in ("serve", "run"):
        spec = spec_from_config(config, unit, platform="darwin")
        assert "--log-file" in spec.program
        assert spec.program[spec.program.index("--log-file") + 1] == str(
            config_file.parent / "logs" / f"{unit}.log"
        )


def test_ingest_units_are_quiet(config):
    """A Rich progress bar rendered into a log file is noise; the
    heartbeat and the signal handler survive --quiet by design."""
    assert "--quiet" in spec_from_config(config, "run", platform="linux").program


def test_unit_argv_uses_the_real_command(config_file):
    argv = unit_argv("serve", {"config": str(config_file), "port": 9999})
    assert argv[len(siteloom_program())] == "serve"
    assert "--port" in argv and "9999" in argv
