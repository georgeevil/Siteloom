"""Frigate consumer: event parsing, filtering, dedupe, identity flow.

No broker or Frigate needed — messages are fed directly to
handle_message and the snapshot fetcher is a stub.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from siteloom.config import IdentityConfig, SiteConfig, StorageConfig
from siteloom.dispatch import LocalBackend
from siteloom.identity import IdentityResolver, VectorStore
from siteloom.integrations.frigate import FrigateConsumer
from siteloom.store import Camera, Event, EventIdentity, Identity, get_session, init_db, make_engine


class StubIdentity:
    """Constant person embedding -> one stable identity."""

    def process(self, job):
        return {
            "embeddings": [
                {
                    "identifier": "person",
                    "algo": "generic",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "plate": None,
                }
            ]
        }


def jpeg_bytes() -> bytes:
    image = np.full((60, 60, 3), 100, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


SNAPSHOT = jpeg_bytes()


def frigate_msg(kind="new", event_id="1712345678.123-abc", camera="driveway",
                label="person", score=0.85, has_snapshot=True, end_time=None):
    after = {
        "id": event_id,
        "camera": camera,
        "label": label,
        "top_score": score,
        "has_snapshot": has_snapshot,
    }
    if end_time is not None:
        after["end_time"] = end_time
    return json.dumps({"type": kind, "before": {}, "after": after})


@pytest.fixture
def consumer(tmp_path):
    config = SiteConfig(
        site_id="test",
        cameras=[],
        identity=IdentityConfig(vector_db_path=str(tmp_path / "vec")),
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/fr.db", media_dir=str(tmp_path / "media")
        ),
    )
    config.integrations.frigate.enabled = True
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    dispatcher = LocalBackend()
    dispatcher.register("identity", StubIdentity())
    resolver = IdentityResolver(
        config.identity, VectorStore(config.identity.vector_db_path)
    )
    fetches: list[str] = []

    def fetch(api_url, event_id):
        fetches.append(event_id)
        return SNAPSHOT

    c = FrigateConsumer(
        config, Session, dispatcher, resolver, snapshot_fetcher=fetch
    )
    c._test_fetches = fetches
    return c


def test_new_event_creates_event_and_identity(consumer):
    assert consumer.handle_message(frigate_msg()) is True
    with consumer.Session() as session:
        event = session.query(Event).one()
        assert event.external_id == "1712345678.123-abc"
        assert event.camera_id == "driveway"
        assert event.class_name == "person"
        assert event.best_confidence == 0.85
        assert event.best_crop_path is not None
        # Camera row auto-created for the Frigate camera.
        assert session.get(Camera, "driveway").adapter == "frigate"
        # Identity resolved and linked.
        assert session.query(Identity).count() == 1
        assert session.query(EventIdentity).count() == 1


def test_updates_dedupe_onto_one_event(consumer):
    consumer.handle_message(frigate_msg(kind="new"))
    consumer.cfg.update_interval_s = 0.0  # allow immediate reprocessing
    consumer.handle_message(frigate_msg(kind="update", score=0.92))
    with consumer.Session() as session:
        event = session.query(Event).one()  # still one event
        assert event.detection_count == 2
        assert event.best_confidence == 0.92
        # Same constant embedding -> still one identity, hit-counted.
        link = session.query(EventIdentity).one()
        assert link.hit_count == 2


def test_update_rate_limited(consumer):
    consumer.cfg.update_interval_s = 3600.0
    consumer.handle_message(frigate_msg(kind="new"))
    assert consumer.handle_message(frigate_msg(kind="update")) is False
    assert consumer.stats.by_reason.get("rate-limited") == 1
    assert len(consumer._test_fetches) == 1  # snapshot fetched once


def test_label_and_camera_filters(consumer):
    consumer.cfg.labels = ["person"]
    consumer.cfg.cameras = ["driveway"]
    assert consumer.handle_message(frigate_msg(label="bird")) is False
    assert consumer.handle_message(frigate_msg(camera="porch")) is False
    assert consumer.stats.by_reason == {"label": 1, "camera": 1}


def test_score_filter(consumer):
    consumer.cfg.min_score = 0.9
    assert consumer.handle_message(frigate_msg(score=0.7)) is False
    assert consumer.stats.by_reason.get("score") == 1


def test_end_event_stamps_duration(consumer):
    consumer.handle_message(frigate_msg(kind="new"))
    consumer.handle_message(frigate_msg(kind="end", end_time=1712345999.0))
    with consumer.Session() as session:
        event = session.query(Event).one()
        assert event.last_seen.year == 2024  # from end_time, not wall clock


def test_bad_json_and_missing_id_skipped(consumer):
    assert consumer.handle_message(b"{not json") is False
    assert consumer.handle_message(json.dumps({"type": "new", "after": {}})) is False
    assert consumer.stats.by_reason == {"bad-json": 1, "no-id": 1}


def test_snapshot_failure_counts_error(consumer):
    consumer.fetch = lambda api, eid: None
    assert consumer.handle_message(frigate_msg()) is False
    assert consumer.stats.errors == 1
    with consumer.Session() as session:
        assert session.query(Event).count() == 0
