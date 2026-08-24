"""Preflight checks, health endpoints, and job supervision.

These are the parts an operator relies on when something is already
wrong, so they have to work in exactly that situation: a held vector
store, a dead process, an unwritable directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from siteloom import health
from siteloom.cli_library import jobs_app
from siteloom.config import IdentityConfig, SiteConfig, StorageConfig
from siteloom.health import FAIL, OK, WARN, Report, run_checks
from siteloom.store import OperationRun, get_session, init_db, make_engine

runner = CliRunner()


@pytest.fixture
def config(tmp_path):
    return SiteConfig(
        site_id="test",
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/health.db",
            media_dir=str(tmp_path / "media"),
        ),
        identity=IdentityConfig(enabled=False),
    )


@pytest.fixture
def config_file(tmp_path, config):
    import yaml

    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    return path


def status_of(report: Report, name: str) -> str:
    return next(c.status for c in report.checks if c.name == name)


# -- checks -----------------------------------------------------------------


def test_healthy_deployment_passes(config):
    _session(config)  # init-db has run, as it has on any deployment that served
    report = run_checks(config)
    assert report.ok, [c for c in report.checks if c.status == FAIL]
    assert status_of(report, "database") == OK
    assert status_of(report, "media dir") == OK


def test_a_fresh_database_warns_and_does_not_block_boot(config):
    """`doctor` is every generated unit's ExecStartPre, and the
    serve/run it gates creates the schema at startup — so an
    uninitialized database is a warning with a remedy, never a FAIL
    that blocks the boot that was about to heal it (CLD-54)."""
    report = run_checks(config, [health.check_database])
    check = report.checks[0]
    assert check.status == WARN
    assert "missing tables" in check.detail
    assert "init-db" in check.remedy
    assert report.ok  # exit 0: the unit may start


def test_check_database_inspects_and_never_migrates(config):
    """Schema drift is *reported*, not healed: the check runs from the
    public /readyz, and an unauthenticated caller must not be able to
    trigger DDL. The old check called init_db, which would have added
    the dropped column straight back (CLD-54)."""
    from sqlalchemy import inspect, text

    engine = make_engine(config.storage.db_url)
    init_db(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE operation_runs DROP COLUMN process_start"))
    report = run_checks(config, [health.check_database])
    check = report.checks[0]
    assert check.status == WARN
    assert "operation_runs.process_start" in check.detail
    assert "init-db" in check.remedy
    columns = {c["name"] for c in inspect(engine).get_columns("operation_runs")}
    assert "process_start" not in columns  # reported, not repaired


def test_unwritable_media_dir_fails(config, tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("I am a file, not a directory")
    config.storage.media_dir = str(blocker / "media")
    report = run_checks(config, [health.check_media_dir])
    assert status_of(report, "media dir") == FAIL
    assert not report.ok


def test_held_vector_store_is_reported_with_a_remedy(config, tmp_path):
    """The failure that actually happens in practice: serve is running,
    so a job cannot open the store. The operator needs to be told that,
    not a Qdrant stack trace."""
    from siteloom.identity import VectorStore

    config.identity.enabled = True
    config.identity.vector_db_path = str(tmp_path / "vectors")
    holder = VectorStore(config.identity.vector_db_path)
    try:
        report = run_checks(config, [health.check_vector_store])
    finally:
        holder.close()
    check = report.checks[0]
    assert check.status == FAIL
    assert "held by another process" in check.detail
    assert "vector_db_path" in check.remedy


def test_a_crashing_check_does_not_hide_the_others(config):
    def explodes(report, cfg):
        raise RuntimeError("boom")

    report = run_checks(config, [explodes, health.check_media_dir])
    assert len(report.checks) == 2
    assert report.checks[0].status == FAIL and "boom" in report.checks[0].detail
    assert status_of(report, "media dir") == OK


def test_abandoned_runs_are_flagged(config):
    Session = _session(config)
    with Session() as session:
        session.add(_run(pid=_dead_pid(), host=health.hostname()))
        session.commit()
    report = run_checks(config, [health.check_jobs])
    check = report.checks[0]
    assert check.status == WARN
    assert "abandoned" in check.detail
    assert "jobs reap" in check.remedy


# -- identity gating is reported, not inferred (CLD-125) --------------------


def _gating_report(**identifiers):
    from siteloom.config import IdentifierConfig

    cfg = SiteConfig(site_id="t")
    cfg.identity.identifiers = {
        key: IdentifierConfig(**kwargs) for key, kwargs in identifiers.items()
    }
    return run_checks(cfg, [health.check_identity_gating]).checks[0]


def test_the_built_in_identifiers_are_gated():
    check = run_checks(SiteConfig(site_id="t"), [health.check_identity_gating])
    assert check.checks[0].status == OK
    # The effective values are printed, because the file they came from
    # may not mention them at all.
    assert "face margin 0.05/2 sightings" in check.checks[0].detail


def test_an_ungated_person_identifier_warns_with_the_fix():
    check = _gating_report(
        face=dict(algo="face", applies_to=["person"], threshold=0.36)
    )
    assert check.status == WARN
    assert "face" in check.detail and "no margin" in check.detail
    assert "mints on one sighting" in check.detail
    assert "min_sightings: 2" in check.remedy


def test_a_vehicle_minting_on_one_sighting_is_not_a_complaint():
    """Deliberate for vehicles: few of them, a strict threshold, and the
    plate-learning flow expects first-sighting rows."""
    check = _gating_report(
        vehicle=dict(
            algo="generic", applies_to=["car"], min_margin=0.02, min_sightings=1
        )
    )
    assert check.status == OK


def test_a_zero_margin_is_reported_for_any_class():
    check = _gating_report(
        vehicle=dict(algo="generic", applies_to=["car"], min_sightings=1)
    )
    assert check.status == WARN
    assert "no margin" in check.detail


def test_the_learning_gates_are_printed_with_the_others(tmp_path):
    """Same argument as the margin: the values in force are printed
    because the file they came from may not mention them, and the whole
    point of the CLD-139 knobs is that an operator can turn them off."""
    check = run_checks(SiteConfig(site_id="t"), [health.check_identity_gating])
    assert check.checks[0].status == OK
    assert "learn ≥0.60 ×3/event" in check.checks[0].detail


def test_learning_switched_off_entirely_is_reported():
    """Both knobs at 0 is the pre-CLD-139 behaviour — every matching
    frame accretes, so one visit can fill a gallery and a wrong match
    recruits more wrong vectors. Worth saying out loud, exactly as
    `min_margin: 0` is."""
    check = _gating_report(
        vehicle=dict(
            algo="generic",
            applies_to=["car"],
            min_margin=0.02,
            min_sightings=1,
            learn_min_quality=0,
            learn_max_per_event=0,
        )
    )
    assert check.status == WARN
    assert "learning ungated" in check.detail
    assert "learn_min_quality" in check.remedy and "learn_max_per_event" in check.remedy


def test_one_learning_gate_off_is_a_choice_not_a_complaint():
    """A site may want an unlimited count above a strict floor, or the
    reverse. Only losing both is the ungated case."""
    for knobs in ({"learn_min_quality": 0}, {"learn_max_per_event": 0}):
        check = _gating_report(
            vehicle=dict(
                algo="generic",
                applies_to=["car"],
                min_margin=0.02,
                min_sightings=1,
                **knobs,
            )
        )
        assert check.status == OK


def test_process_alive_on_self_and_on_the_departed():
    assert health.process_alive(os.getpid())
    assert not health.process_alive(_dead_pid())
    assert not health.process_alive(0)


# -- endpoints --------------------------------------------------------------


def test_healthz_says_ok_and_nothing_else(config):
    """Liveness is one bit. The body used to name the site and the
    server pid; both probes are public and the console is treated as
    internet-exposed, so the answer maps nothing (CLD-54)."""
    from siteloom.web.app import create_app

    client = TestClient(create_app(config))
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz_reports_unready_with_503(config, tmp_path):
    from siteloom.web.app import create_app

    app = create_app(config)  # created while the config is still sound
    config.storage.media_dir = str(tmp_path / "gone" / "media")
    (tmp_path / "gone").write_text("a file where a directory should be")
    client = TestClient(app)
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["ok"] is False


def test_readyz_ok(config):
    from siteloom.web.app import create_app

    client = TestClient(create_app(config))
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["ok"] is True


# -- /readyz is public, so it must not mutate or map (CLD-54) ---------------


def test_readyz_runs_no_migration(config):
    """Drop a table out from under a running app: the old endpoint's
    database check called init_db, so the next probe would have quietly
    re-created it. A probe answers; it does not repair."""
    from sqlalchemy import inspect, text

    from siteloom.web.app import create_app

    client = TestClient(create_app(config))
    engine = make_engine(config.storage.db_url)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE noise_events"))
    response = client.get("/readyz")
    assert response.status_code in (200, 503)
    assert response.json()  # answered
    assert "noise_events" not in inspect(engine).get_table_names()


def test_readyz_writes_nothing_to_the_database_file(config, tmp_path):
    """The same property from the filesystem's side: with the database
    file read-only, the probe still answers and the file is untouched
    byte for byte."""
    from siteloom.web.app import create_app

    client = TestClient(create_app(config))  # init_db ran while writable
    db_file = tmp_path / "health.db"
    before = db_file.read_bytes()
    db_file.chmod(0o444)
    try:
        response = client.get("/readyz")
        assert response.status_code in (200, 503)
        assert response.json()  # a real answer, not a stack trace
        assert db_file.read_bytes() == before
    finally:
        db_file.chmod(0o644)


def test_readyz_body_maps_nothing_for_an_unauthenticated_caller(config, tmp_path):
    """Check names and statuses are actionable to a service manager; the
    detail and remedy strings name the db url, the media path and the
    fix, which belong to `siteloom doctor` at a terminal — never to a
    public endpoint. Redacted here, not by weakening doctor."""
    from siteloom.web.app import create_app

    client = TestClient(create_app(config))
    response = client.get("/readyz")
    text = response.text
    assert str(tmp_path) not in text  # covers both the db path and media path
    assert config.storage.media_dir not in text
    assert config.storage.db_url not in text
    host = health.hostname()
    if len(host) > 3:  # a tiny hostname would substring-match JSON keys
        assert host not in text
    payload = response.json()
    assert set(payload) == {"ok", "checks"}
    assert all(set(c) == {"name", "status"} for c in payload["checks"])


def test_readyz_work_is_bounded_under_hammering(config, monkeypatch):
    """The endpoint is public: hammering it must cost one check run per
    cache window, not one per request."""
    from siteloom.web.app import create_app

    client = TestClient(create_app(config))
    calls = {"n": 0}
    real = health.run_checks

    def counting(cfg, checks=None):
        calls["n"] += 1
        return real(cfg, checks)

    monkeypatch.setattr(health, "run_checks", counting)
    for _ in range(5):
        assert client.get("/readyz").status_code == 200
    assert calls["n"] == 1


# -- jobs supervision -------------------------------------------------------


def test_reap_closes_out_dead_runs_but_keeps_the_resume_command(config, config_file):
    Session = _session(config)
    with Session() as session:
        session.add(_run(pid=_dead_pid(), host=health.hostname()))
        session.commit()

    result = runner.invoke(jobs_app, ["reap", "--config", str(config_file), "--yes"])
    assert result.exit_code == 0, result.output
    with Session() as session:
        run = session.query(OperationRun).one()
        assert run.status == "abandoned"
        assert run.finished_at is not None
        assert run.resume_command == "siteloom library index --all"
        assert run.current == 40  # position preserved


def test_reap_leaves_live_runs_alone(config, config_file):
    Session = _session(config)
    with Session() as session:
        session.add(_run(pid=os.getpid(), host=health.hostname()))
        session.commit()

    result = runner.invoke(jobs_app, ["reap", "--config", str(config_file), "--yes"])
    assert result.exit_code == 0
    assert "nothing to reap" in result.output
    with Session() as session:
        assert session.query(OperationRun).one().status == "running"


def test_cancel_refuses_a_run_from_another_host(config, config_file):
    Session = _session(config)
    with Session() as session:
        session.add(_run(pid=os.getpid(), host="some-other-machine"))
        session.commit()

    result = runner.invoke(jobs_app, ["cancel", "1", "--config", str(config_file)])
    assert result.exit_code == 1
    assert "some-other-machine" in result.output


def test_cancel_signals_a_live_process(config, config_file):
    """The point of `jobs cancel`: stop a job from a terminal that did
    not start it, gracefully enough that it can resume."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import signal, time\n"
         "signal.signal(signal.SIGINT, lambda *a: None)\n"
         "time.sleep(30)"]
    )
    try:
        Session = _session(config)
        with Session() as session:
            session.add(
                _run(
                    pid=child.pid,
                    host=health.hostname(),
                    # As the child's own reporter would have recorded it:
                    # an unqualified pid is refused, by design (CLD-57).
                    process_start=health.process_identity(child.pid) or "",
                )
            )
            session.commit()
        result = runner.invoke(jobs_app, ["cancel", "1", "--config", str(config_file)])
        assert result.exit_code == 0, result.output
        assert "finish the current batch" in result.output
        assert child.poll() is None  # asked to stop, not killed
    finally:
        child.kill()
        child.wait()


# -- installed service units ------------------------------------------------


@pytest.fixture
def linux_units(tmp_path, monkeypatch):
    """A unit directory `check_services` will actually look in.

    The check picks its backend off the live platform, so on macOS these
    tests read ~/Library/LaunchAgents — the developer's real machine —
    and never see the fixture they just installed. Pinning the platform
    rather than the factory keeps the real construction path, runner and
    all, and `platform` is imported inside the check.
    """
    import platform as _platform

    from siteloom.service.manager import SystemdBackend

    units = tmp_path / "units"
    monkeypatch.setattr(_platform, "system", lambda: "Linux")
    # User scope only. Handing both scopes the same directory counts every
    # unit twice, which makes a single installed unit look like a
    # vector-store collision with itself.
    monkeypatch.setattr(
        SystemdBackend,
        "unit_dir",
        lambda self, scope: units if scope == "user" else tmp_path / "no-system-units",
    )
    return units


def _install_unit(directory, label, config_path, unit="serve"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{label}.service").write_text(
        "[Unit]\n"
        f"X-Siteloom-Unit={unit}\n"
        f"X-Siteloom-Config={config_path}\n"
        "X-Siteloom-Generator=siteloom 0.1.0\n"
        "[Service]\nExecStart=/bin/true\n"
    )


def test_check_services_reports_installed_units(config_file, linux_units):
    _install_unit(linux_units, "siteloom-test-serve", config_file)

    report = run_checks(_load(config_file), [health.check_services])
    check = report.checks[0]
    assert check.status == OK
    # Exact, not a substring: a unit counted once per scope would read
    # "2 installed" and still satisfy an `in`.
    assert check.detail == "1 installed: siteloom-test-serve"


def test_check_services_catches_a_vector_store_collision(
    tmp_path, config_file, linux_units
):
    """The rule check_vector_store enforces at runtime, applied to what
    is configured to start. Two units sharing an embedded Qdrant
    directory can never both run, and a boot is a late time to find out.
    """
    import yaml

    shared = str(tmp_path / "shared-vectors")
    paths = []
    for name in ("a", "b"):
        cfg = SiteConfig(
            site_id=name,
            storage=StorageConfig(db_url=f"sqlite:///{tmp_path}/{name}.db"),
            identity=IdentityConfig(enabled=True, vector_db_path=shared),
        )
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
        paths.append(path)
        _install_unit(linux_units, f"siteloom-{name}-serve", path)

    report = run_checks(_load(config_file), [health.check_services])
    check = report.checks[0]
    assert check.status == FAIL
    assert shared in check.detail
    assert "one client per path per machine" in check.remedy


def test_check_services_never_shells_out(config_file, linux_units, monkeypatch):
    """`doctor` runs as this unit's own ExecStartPre. Asking the service
    manager about the service it is in the middle of starting is a
    question with no good answer and a plausible hang, so the check reads
    unit files and nothing else."""
    import subprocess as _subprocess

    _install_unit(linux_units, "siteloom-test-serve", config_file)

    def forbidden(*args, **kwargs):
        raise AssertionError("check_services must not run a subprocess")

    monkeypatch.setattr(_subprocess, "run", forbidden)
    monkeypatch.setattr(_subprocess, "Popen", forbidden)
    report = run_checks(_load(config_file), [health.check_services])
    assert report.checks[0].status == OK
    # Naming the fixture's own unit is what stops this passing vacuously:
    # "no units installed" is also OK, so an assertion on the status alone
    # holds just as well when the check never saw the unit at all.
    assert "siteloom-test-serve" in report.checks[0].detail


def test_check_services_is_not_a_readiness_check():
    """/readyz runs in the serving process and must stay milliseconds;
    walking unit directories and loading other configs is neither."""
    assert health.check_services in health.CHECKS
    assert health.check_services not in health.LIVE_CHECKS


def _load(path):
    from siteloom.config import load_config

    return load_config(path)


# -- helpers ----------------------------------------------------------------


def _session(config):
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    return get_session(engine)


def _run(**kw) -> OperationRun:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return OperationRun(
        kind="library-index",
        status="running",
        started_at=now - timedelta(seconds=60),
        updated_at=now,  # fresh heartbeat: only the pid can prove it dead
        current=40,
        total=100,
        resume_command="siteloom library index --all",
        **kw,
    )


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid
