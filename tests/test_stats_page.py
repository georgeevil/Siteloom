"""The /stats page (CLD-17) — rendering, and the rules it must not break.

`tests/test_stats.py` covers the arithmetic. What is worth asserting at
the route level is that the page cannot show a precision figure stripped
of its denominator, and that the window actually filters.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import (
    Camera,
    Event,
    EventIdentity,
    Identity,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web.app import create_app

# Naive UTC, matching what ingest stores and what the route compares against.
NOW = datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.fixture
def client(tmp_path):
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="front", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/s.db", media_dir=str(tmp_path / "m")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(Camera(id="front", site_id="t", name="Front Yard"))
        face = Identity(
            identifier_key="face", class_name="person", label="Alice",
            first_seen=NOW, last_seen=NOW,
        )
        s.add(face)
        s.flush()
        # Three recent face claims: one confirmed, one wrong, one unjudged.
        for offset, verdict, similarity in [
            (1, "confirmed", 0.44),
            (2, "wrong", 0.37),
            (3, None, 0.41),
        ]:
            event = Event(
                camera_id="front", class_name="person",
                first_seen=NOW - timedelta(minutes=offset),
                last_seen=NOW - timedelta(minutes=offset),
                detection_count=3,
            )
            s.add(event)
            s.flush()
            s.add(EventIdentity(
                event_id=event.id, identity_id=face.id, identifier_key="face",
                matched_by="visual", similarity=similarity, verdict=verdict,
            ))
        # An old one, outside the default 24h window.
        stale = Event(
            camera_id="front", class_name="person",
            first_seen=NOW - timedelta(days=10), last_seen=NOW - timedelta(days=10),
            detection_count=3,
        )
        s.add(stale)
        s.flush()
        s.add(EventIdentity(
            event_id=stale.id, identity_id=face.id, identifier_key="face",
            matched_by="visual", similarity=0.9,
        ))
        # Cleared with nothing judged on it — the blind spot the page names.
        bare = Event(
            camera_id="front", class_name="car", first_seen=NOW, last_seen=NOW,
            detection_count=3, reviewed_at=NOW,
        )
        s.add(bare)
        s.commit()
    return TestClient(create_app(config))


def test_page_renders_per_identifier_rows(client):
    body = client.get("/stats").text
    assert body.count("face") >= 1
    assert "Front Yard" in body


def test_precision_never_appears_without_its_denominator(client):
    """One wrong of two reviewed is 50%, but 50% alone is the misleading
    number — the reviewed count and its share of all claims must ride
    along with it."""
    body = client.get("/stats").text
    assert "50%" in body
    assert "of 2 reviewed" in body
    # 2 of the 3 in-window claims were judged.
    assert "of 3" in body


def test_cleared_without_a_verdict_is_surfaced(client):
    body = client.get("/stats").text
    assert "cleared, no verdict" in body
    assert "blind spot" in body


def test_window_excludes_older_events(client):
    """The default 24h view must not silently average in last week."""
    day = client.get("/stats").text
    month = client.get("/stats", params={"days": 30}).text
    assert "3 visual matches" in day
    assert "4 visual matches" in month


def test_threshold_tradeoff_is_stated(client):
    """The point of the histogram is the decision it supports."""
    body = client.get("/stats", params={"days": 30}).text
    assert "Raising the cutoff" in body


def test_window_is_bounded(client):
    assert client.get("/stats", params={"days": 0}).status_code == 422
    assert client.get("/stats", params={"days": 9999}).status_code == 422
