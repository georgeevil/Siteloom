"""`siteloom events retag` recomputes significance from stored counts.

The gate normally flips at ingest; retag is the retroactive twin — it
must agree with ingest's rules (same EventConfig.for_camera), respect
per-camera overrides, leave externally curated (Frigate) events alone,
and be idempotent.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import yaml
from typer.testing import CliRunner

from siteloom import cli
from siteloom.config import SiteConfig, save_config
from siteloom.store import Camera, Event, get_session, init_db, make_engine

runner = CliRunner()


@pytest.fixture
def env(tmp_path):
    config = SiteConfig(
        site_id="test",
        storage={"db_url": f"sqlite:///{tmp_path}/retag.db", "media_dir": str(tmp_path / "m")},
    )
    config_path = tmp_path / "site.yaml"
    save_config(config, config_path)
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    t = datetime(2026, 8, 6, 12, 0, 0)
    with Session() as s:
        s.add(Camera(id="cam1", site_id="test", name="Cam"))
        # A real visit, a 1-detection fragment (both grandfathered as
        # significant by the column default), and a Frigate event.
        s.add(
            Event(
                camera_id="cam1", track_id=1, class_name="person",
                first_seen=t, last_seen=t + timedelta(seconds=30),
                detection_count=20, best_confidence=0.9,
            )
        )
        s.add(
            Event(
                camera_id="cam1", track_id=None, class_name="person",
                first_seen=t, last_seen=t,
                detection_count=1, best_confidence=0.4,
            )
        )
        s.add(
            Event(
                camera_id="frigate-cam", external_id="fr-1", class_name="person",
                first_seen=t, last_seen=t,
                detection_count=1, best_confidence=0.0,
            )
        )
        s.commit()
    return config_path, Session


def _flags(Session) -> dict:
    with Session() as s:
        return {
            (e.track_id, e.external_id): e.significant
            for e in s.query(Event).all()
        }


def test_retag_applies_rules_and_spares_external_events(env):
    config_path, Session = env
    result = runner.invoke(cli.app, ["events", "retag", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    flags = _flags(Session)
    assert flags[(1, None)] is True  # real visit stays
    assert flags[(None, None)] is False  # fragment demoted
    assert flags[(None, "fr-1")] is True  # Frigate event untouched
    assert "retagged 1 of 3 events: 3 -> 2 significant (1 ephemeral)" in result.output


def test_retag_is_idempotent(env):
    config_path, Session = env
    runner.invoke(cli.app, ["events", "retag", "--config", str(config_path)])
    first = _flags(Session)
    result = runner.invoke(cli.app, ["events", "retag", "--config", str(config_path)])
    assert result.exit_code == 0
    assert "retagged 0 of" in result.output
    assert _flags(Session) == first


def test_retag_honours_per_camera_overrides(env, tmp_path):
    config_path, Session = env
    data = yaml.safe_load(config_path.read_text())
    data["cameras"] = [
        {
            "id": "cam1",
            "adapter": "file",
            "source": "x",
            "events": {"min_detections": 1, "min_confidence": 0.1},
        }
    ]
    config_path.write_text(yaml.safe_dump(data, sort_keys=False))
    runner.invoke(cli.app, ["events", "retag", "--config", str(config_path)])
    flags = _flags(Session)
    # With the relaxed per-camera gate even the fragment qualifies.
    assert flags[(None, None)] is True
