"""MQTT payloads, webhook routing, and the additive schema migration."""

from __future__ import annotations

import json
import threading
from datetime import datetime

import pytest

from siteloom.config import MqttConfig, WebhookConfig
from siteloom.integrations.mqtt import MqttPublisher, event_payload, identity_payload
from siteloom.integrations.webhooks import WebhookNotifier, classify_resolution
from siteloom.store import Event, Identity

TS = datetime(2026, 8, 5, 12, 0, 0)


def make_event(**kw):
    defaults = dict(
        id=7, camera_id="driveway", track_id=3, external_id=None,
        class_name="person", first_seen=TS, last_seen=TS,
        detection_count=4, best_confidence=0.91, guest_window=False,
    )
    defaults.update(kw)
    return Event(**defaults)


def make_identity(**kw):
    defaults = dict(
        id=12, identifier_key="face", class_name="person", label="Ann",
        first_seen=TS, last_seen=TS,
    )
    defaults.update(kw)
    return Identity(**defaults)


def test_event_payload_shape():
    payload = event_payload(make_event(), camera_name="Driveway")
    assert payload["source"] == "siteloom"
    assert payload["camera"] == "Driveway"
    assert payload["label"] == "person"
    assert payload["detections"] == 4
    json.dumps(payload)  # must be serializable as-is


def test_identity_payload_known_vs_unknown():
    known = identity_payload(make_event(), make_identity(), 0.87, is_new=False)
    assert known["known"] is True
    assert known["name"] == "Ann"
    unknown = identity_payload(
        make_event(), make_identity(label=None), 0.2, is_new=True
    )
    assert unknown["known"] is False
    assert unknown["name"] is None
    assert unknown["display_name"].startswith("unknown-")
    assert unknown["new_identity"] is True


def test_publisher_disabled_is_noop():
    publisher = MqttPublisher(MqttConfig(enabled=False))
    assert publisher.publish("events", {"a": 1}) is False


def test_publisher_broker_down_degrades_quietly():
    """NFR1: a dead broker is a log line, not an exception."""
    publisher = MqttPublisher(
        MqttConfig(enabled=True, host="127.0.0.1", port=1)  # nothing listens
    )
    assert publisher.publish("events", {"a": 1}) is False
    assert publisher.publish("events", {"a": 2}) is False  # still quiet


def test_classify_resolution():
    assert classify_resolution(make_identity(), False, False) == "identity.match"
    assert (
        classify_resolution(make_identity(label=None), True, False)
        == "identity.unknown"
    )
    assert classify_resolution(make_identity(), False, True) == "identity.new_plate"


def test_webhook_fires_only_subscribed_events(monkeypatch):
    delivered = []
    done = threading.Event()

    def fake_post(self, hook, body):
        delivered.append((hook.url, body))
        done.set()

    monkeypatch.setattr(WebhookNotifier, "_post", fake_post)
    notifier = WebhookNotifier(
        [
            WebhookConfig(url="http://a", events=["identity.match"]),
            WebhookConfig(url="http://b", events=["identity.unknown"]),
        ]
    )
    assert notifier.fire("identity.match", {"x": 1}) == 1
    done.wait(2)
    assert delivered == [("http://a", {"x": 1, "event_type": "identity.match"})]
    assert notifier.fire("noise.episode", {}) == 0


def test_additive_migration_adds_new_column(tmp_path):
    """A DB created before Event.external_id existed must keep working
    after upgrade — init_db adds the missing column."""
    import sqlite3

    from siteloom.store import get_session, init_db, make_engine

    db = tmp_path / "old.db"
    engine = make_engine(f"sqlite:///{db}")
    init_db(engine)
    # Simulate the pre-upgrade schema by dropping the column (its index
    # first — SQLite refuses to drop an indexed column).
    with sqlite3.connect(db) as conn:
        conn.execute("DROP INDEX IF EXISTS ix_events_external_id")
        conn.execute("ALTER TABLE events DROP COLUMN external_id")
    engine.dispose()

    engine2 = make_engine(f"sqlite:///{db}")
    init_db(engine2)  # migration runs here
    with get_session(engine2)() as session:
        session.add(make_event(id=None, external_id="frigate-1"))
        session.commit()
        assert session.query(Event).one().external_id == "frigate-1"
