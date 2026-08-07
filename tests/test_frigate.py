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


# -- input hostility (CLD-48 / CLD-75) --------------------------------------
#
# MQTT payloads are untrusted: the event id reaches a URL, the camera id
# reaches a filesystem path and a Camera row, the label reaches class
# routing. Every hostile value must be skipped as "bad-input" — never
# raised — leaving no rows, no files, and no snapshot fetch.


def assert_rejected(consumer, msg):
    assert consumer.handle_message(msg) is False
    assert consumer.stats.by_reason.get("bad-input", 0) >= 1
    assert consumer._test_fetches == []  # hostile id never reached the URL
    with consumer.Session() as session:
        assert session.query(Event).count() == 0
        assert session.query(Camera).count() == 0  # no Camera-row pollution


@pytest.mark.parametrize(
    "camera",
    [
        "../../etc",  # relative traversal
        "..",  # bare traversal, allowed by the char class alone
        ".hidden",  # leading dot
        "/etc/passwd",  # absolute path
        "a/b",  # separator
        "a" * 65,  # overlong
        "",  # empty
        "with space",
        5,  # non-string
        None,
    ],
)
def test_malicious_camera_rejected(consumer, camera):
    assert_rejected(consumer, frigate_msg(camera=camera))


@pytest.mark.parametrize(
    "event_id",
    [
        "../../../etc/passwd",  # would traverse both URL and filename
        "abc/../snap",
        "abc?crop=1&h=9999",  # URL query smuggling
        "abc#frag",
        "id with spaces",
        "x" * 65,
        7,  # non-string (truthy, so it passes the no-id gate)
    ],
)
def test_malicious_event_id_rejected(consumer, event_id):
    assert_rejected(consumer, frigate_msg(event_id=event_id))


@pytest.mark.parametrize(
    "score",
    [
        "not-a-number",
        float("nan"),  # json.dumps emits bare NaN, which json.loads accepts
        float("inf"),
        -3.0,
        1e9,  # far outside Frigate's [0, 1] confidence range
        {"x": 1},  # non-scalar
    ],
)
def test_malformed_score_rejected(consumer, score):
    assert_rejected(consumer, frigate_msg(score=score))


@pytest.mark.parametrize("label", ["../person", "a" * 65, "bad label", "", 3])
def test_malicious_label_rejected(consumer, label):
    assert_rejected(consumer, frigate_msg(label=label))


def test_non_dict_after_rejected(consumer):
    assert_rejected(consumer, json.dumps({"type": "new", "after": "not-a-dict"}))


def test_malformed_end_time_does_not_raise(consumer):
    consumer.handle_message(frigate_msg(kind="new"))
    before = None
    with consumer.Session() as session:
        before = session.query(Event).one().last_seen
    assert consumer.handle_message(frigate_msg(kind="end", end_time="garbage")) is False
    assert consumer.stats.by_reason.get("bad-input") == 1
    with consumer.Session() as session:
        assert session.query(Event).one().last_seen == before  # untouched


def test_save_snapshot_refuses_resolved_escape(consumer, tmp_path):
    # Defense in depth behind the id validation: even if a traversal
    # camera id reached _save_snapshot, the resolved-path check keeps the
    # write inside media_dir.
    from datetime import datetime
    from types import SimpleNamespace

    event = SimpleNamespace(camera_id="../escape", detection_count=1)
    path = consumer._save_snapshot(
        "1712345678.123-abc", event, SNAPSHOT, datetime(2026, 8, 1, 12, 0, 0)
    )
    assert path is None
    assert not (tmp_path / "escape").exists()


def test_legitimate_message_still_processes(consumer):
    # Positive control: a real Frigate payload (dotted id and all) passes
    # every gate and runs the pipeline end-to-end.
    assert consumer.handle_message(frigate_msg()) is True
    assert consumer.stats.processed == 1
    assert consumer.stats.by_reason.get("bad-input") is None
    with consumer.Session() as session:
        assert session.query(Event).one().external_id == "1712345678.123-abc"
        assert session.query(EventIdentity).count() == 1
