"""The directives a generated unit does and does not carry (CLD-105).

Rendering is pure — spec in, text out — which is the only reason this is
checkable on a machine with neither `launchctl` nor `systemctl`. It is
the same seam that makes `detect_episodes` a pure function.

The absences matter as much as the presences here, so several of these
assert that a plausible-looking directive is *not* emitted. Each one is a
setting somebody will eventually be tempted to add, with a reason it is
wrong for this project written next to it in the renderer.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest
import yaml

from siteloom.config import (
    IdentityConfig,
    ServiceConfig,
    SiteConfig,
    StorageConfig,
    load_config,
)
from siteloom.service import spec_from_config
from siteloom.service import launchd, systemd


def _config(tmp_path, **service):
    cfg = SiteConfig(
        site_id="kai",
        storage=StorageConfig(db_url="sqlite:///siteloom.db", media_dir="media"),
        identity=IdentityConfig(enabled=True, vector_db_path="vectors"),
        service=ServiceConfig(**service),
    )
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return load_config(path)


def _ini(text: str) -> configparser.RawConfigParser:
    # strict=False: systemd allows a key to repeat (Environment=, and
    # ExecStartPre= in general), which configparser would otherwise
    # reject outright.
    parser = configparser.RawConfigParser(strict=False)
    parser.optionxform = str  # systemd keys are case-sensitive
    parser.read_string(text)
    return parser


# -- systemd ----------------------------------------------------------------


def test_systemd_type_is_exec_not_simple(tmp_path):
    """`simple` reports a start as successful before the exec is even
    attempted, so a moved venv looks like a healthy service that quietly
    is not there."""
    spec = spec_from_config(_config(tmp_path), "serve", platform="linux")
    assert _ini(systemd.render(spec))["Service"]["Type"] == "exec"


def test_systemd_notify_is_opt_in(tmp_path):
    cfg = _config(tmp_path)
    plain = _ini(systemd.render(spec_from_config(cfg, "serve", platform="linux")))
    assert "NotifyAccess" not in plain["Service"]

    notified = _ini(
        systemd.render(spec_from_config(cfg, "serve", platform="linux", notify=True))
    )
    assert notified["Service"]["Type"] == "notify"
    assert notified["Service"]["NotifyAccess"] == "main"


def test_systemd_restart_has_a_crashloop_brake(tmp_path):
    unit = _ini(
        systemd.render(spec_from_config(_config(tmp_path), "serve", platform="linux"))
    )
    assert unit["Service"]["Restart"] == "on-failure"
    assert unit["Service"]["RestartSec"] == "10"
    # StartLimit* belong to [Unit] — they have since systemd 229, and a
    # unit that puts them under [Service] gets a warning and no limit.
    assert unit["Unit"]["StartLimitBurst"] == "5"
    assert unit["Unit"]["StartLimitIntervalSec"] == "300"


def test_systemd_restart_never_emits_nothing(tmp_path):
    unit = _ini(
        systemd.render(
            spec_from_config(
                _config(tmp_path, restart="never"), "serve", platform="linux"
            )
        )
    )
    assert "Restart" not in unit["Service"]
    assert "StartLimitBurst" not in unit["Unit"]


@pytest.mark.parametrize("unit", ["serve", "run", "frigate"])
def test_systemd_tells_the_supervisor_that_130_is_a_clean_stop(tmp_path, unit):
    """Without this, `siteloom jobs cancel` on a service-managed process
    exits 130, Restart=on-failure reads it as a crash, and the thing the
    operator just cancelled comes straight back."""
    spec = spec_from_config(_config(tmp_path), unit, platform="linux")
    assert _ini(systemd.render(spec))["Service"]["SuccessExitStatus"] == "130"


def test_systemd_stop_timeout_is_a_batch_worth_of_time(tmp_path):
    cfg = _config(tmp_path)
    serve = _ini(systemd.render(spec_from_config(cfg, "serve", platform="linux")))
    run = _ini(systemd.render(spec_from_config(cfg, "run", platform="linux")))
    assert serve["Service"]["TimeoutStopSec"] == "30"
    assert run["Service"]["TimeoutStopSec"] == "60"


def test_systemd_leaves_kill_semantics_at_their_defaults(tmp_path):
    """SIGTERM is already handled exactly like Ctrl-C, and ingest's
    workers are threads rather than child processes — so KillMode=mixed
    (the tempting wrong answer) buys nothing and KillMode=process would
    orphan dispatcher subprocesses."""
    unit = _ini(
        systemd.render(spec_from_config(_config(tmp_path), "run", platform="linux"))
    )
    assert "KillMode" not in unit["Service"]
    assert "KillSignal" not in unit["Service"]


def test_systemd_does_not_decide_where_state_lives(tmp_path):
    """RuntimeDirectory/StateDirectory/LogsDirectory would put state
    under /var/lib or $XDG_STATE_HOME, in direct conflict with CLD-64:
    the config file decides where media_dir and the vector store are, and
    two mechanisms answering that question is the bug."""
    unit = _ini(
        systemd.render(spec_from_config(_config(tmp_path), "serve", platform="linux"))
    )
    for key in ("RuntimeDirectory", "StateDirectory", "LogsDirectory"):
        assert key not in unit["Service"]


def test_systemd_sandbox_is_full_not_strict_and_leaves_home_alone(tmp_path):
    """`strict` fails minutes into a run when one write path is missing;
    ProtectHome would hide ~/.cache/siteloom/models, where the face
    weights live."""
    unit = _ini(
        systemd.render(spec_from_config(_config(tmp_path), "serve", platform="linux"))
    )
    assert unit["Service"]["ProtectSystem"] == "full"
    assert unit["Service"]["NoNewPrivileges"] == "yes"
    assert "ProtectHome" not in unit["Service"]


def test_systemd_read_write_paths_are_emitted_for_the_hand_edit(tmp_path):
    cfg = _config(tmp_path)
    unit = _ini(systemd.render(spec_from_config(cfg, "serve", platform="linux")))
    paths = unit["Service"]["ReadWritePaths"]
    assert str(tmp_path / "media") in paths
    assert str(tmp_path / "vectors") in paths


def test_systemd_wanted_by_matches_the_scope(tmp_path):
    """A user unit under multi-user.target never starts."""
    cfg = _config(tmp_path)
    user = _ini(systemd.render(spec_from_config(cfg, "serve", platform="linux")))
    system = _ini(
        systemd.render(spec_from_config(cfg, "serve", platform="linux", scope="system"))
    )
    assert user["Install"]["WantedBy"] == "default.target"
    assert system["Install"]["WantedBy"] == "multi-user.target"


def test_systemd_network_ordering_is_system_scope_only(tmp_path):
    """A user manager has no network-online.target to wait on."""
    cfg = _config(tmp_path)
    assert "network-online.target" not in systemd.render(
        spec_from_config(cfg, "run", platform="linux")
    )
    assert "network-online.target" in systemd.render(
        spec_from_config(cfg, "run", platform="linux", scope="system")
    )


def test_systemd_preflight_gates_on_doctor(tmp_path):
    unit = _ini(
        systemd.render(spec_from_config(_config(tmp_path), "serve", platform="linux"))
    )
    assert "doctor" in unit["Service"]["ExecStartPre"]


def test_systemd_quotes_a_path_with_a_space(tmp_path):
    """`ExecStart=` is not a shell line, but it is not a raw string
    either: an unquoted space is an argument separator, and a config
    under "~/My Site" is an ordinary thing on the primary target."""
    spacey = tmp_path / "My Site"
    spacey.mkdir()
    cfg = _config(spacey)
    text = systemd.render(spec_from_config(cfg, "serve", platform="linux"))
    exec_start = _ini(text)["Service"]["ExecStart"]
    assert f'"{spacey / "site.yaml"}"' in exec_start


def test_systemd_marker_round_trips(tmp_path):
    cfg = _config(tmp_path)
    spec = spec_from_config(cfg, "serve", platform="linux")
    unit = _ini(systemd.render(spec))["Unit"]
    assert unit["X-Siteloom-Unit"] == "serve"
    assert Path(unit["X-Siteloom-Config"]) == spec.config_path
    assert unit[systemd.MARKER_KEY].startswith("siteloom")


def test_systemd_warns_against_workers_in_the_file_itself(tmp_path):
    """One Qdrant client per path per machine, so a second uvicorn worker
    cannot open the vector store. Say so where somebody editing the unit
    will read it."""
    text = systemd.render(
        spec_from_config(_config(tmp_path), "serve", platform="linux")
    )
    assert "--workers" in text


# -- launchd ----------------------------------------------------------------


def test_launchd_keepalive_is_a_dict_for_on_failure(tmp_path):
    """A bare `true` — what the hand-written template used to say — also
    resurrects a server the operator stopped cleanly."""
    plist = launchd.parse(
        launchd.render(spec_from_config(_config(tmp_path), "serve", platform="darwin"))
    )
    assert plist["KeepAlive"] == {"SuccessfulExit": False}


def test_launchd_keepalive_variants(tmp_path):
    always = launchd.parse(
        launchd.render(
            spec_from_config(
                _config(tmp_path, restart="always"), "serve", platform="darwin"
            )
        )
    )
    assert always["KeepAlive"] is True

    never = launchd.parse(
        launchd.render(
            spec_from_config(
                _config(tmp_path, restart="never"), "serve", platform="darwin"
            )
        )
    )
    assert "KeepAlive" not in never
    assert "ThrottleInterval" not in never


def test_launchd_exit_timeout_beats_the_20s_default(tmp_path):
    cfg = _config(tmp_path)
    assert (
        launchd.plist_dict(spec_from_config(cfg, "run", platform="darwin"))["ExitTimeOut"]
        == 60
    )


def test_launchd_has_no_process_type(tmp_path):
    """`Background` is the plausible wrong answer: it puts the job in a
    throttled task-policy band that would starve YOLO inference on the
    Apple Silicon target this project is built for."""
    plist = launchd.parse(
        launchd.render(spec_from_config(_config(tmp_path), "run", platform="darwin"))
    )
    assert "ProcessType" not in plist


def test_launchd_program_arguments_survive_a_space(tmp_path):
    spacey = tmp_path / "My Site"
    spacey.mkdir()
    spec = spec_from_config(_config(spacey), "serve", platform="darwin")
    plist = launchd.parse(launchd.render(spec))
    assert plist["ProgramArguments"] == spec.program
    assert str(spacey / "site.yaml") in plist["ProgramArguments"]


def test_launchd_marker_is_also_functional(tmp_path):
    """The env var tells a running process it was started by a unit, as
    well as telling `uninstall` that siteloom wrote this file."""
    plist = launchd.parse(
        launchd.render(spec_from_config(_config(tmp_path), "serve", platform="darwin"))
    )
    assert plist["EnvironmentVariables"][launchd.MARKER_ENV] == "serve"
    assert plist["EnvironmentVariables"]["SITELOOM_CONFIG"].endswith("site.yaml")


def test_launchd_throttle_interval_floor(tmp_path):
    """launchd complains below 10 and throttles anyway."""
    plist = launchd.parse(
        launchd.render(
            spec_from_config(
                _config(tmp_path, restart_delay_s=1), "serve", platform="darwin"
            )
        )
    )
    assert plist["ThrottleInterval"] == 10


@pytest.mark.parametrize("unit", ["serve", "run", "frigate"])
def test_both_renderers_produce_something_parseable(tmp_path, unit):
    cfg = _config(tmp_path)
    _ini(systemd.render(spec_from_config(cfg, unit, platform="linux")))
    launchd.parse(launchd.render(spec_from_config(cfg, unit, platform="darwin")))
