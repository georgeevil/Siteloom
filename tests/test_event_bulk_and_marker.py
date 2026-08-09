"""Bulk claim detachment and the multi-subject marker (CLD-103).

Event 1401 carried fifteen identity claims. Clearing them one form at a
time is the work these endpoints remove, and the marker is what turns
that review moment into a corpus entry instead of a shrug.
"""

from __future__ import annotations

from datetime import datetime

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

TS = datetime(2026, 8, 6, 23, 33, 46)


@pytest.fixture
def env(tmp_path):
    config = SiteConfig(
        site_id="t",
        cameras=[CameraConfig(id="cam", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/e.db", media_dir=str(tmp_path / "m")
        ),
    )
    config.identity.enabled = False
    # Per-test vector store: embedded Qdrant is one client per path, so a
    # default path would collide with the real one on a dev machine.
    config.identity.vector_db_path = str(tmp_path / "vectors")
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(Camera(id="cam", site_id="t", name="Cam"))
        event = Event(
            camera_id="cam", track_id=42, class_name="person",
            first_seen=TS, last_seen=TS, detection_count=108,
        )
        s.add(event)
        s.flush()
        # Four claims, as a confused track accumulates them.
        for n, verdict in enumerate(("wrong", "wrong", None, "confirmed")):
            ident = Identity(
                identifier_key="person", class_name="person",
                first_seen=TS, last_seen=TS, appearance_count=1,
            )
            s.add(ident)
            s.flush()
            s.add(EventIdentity(
                event_id=event.id, identity_id=ident.id,
                identifier_key="person", matched_by="visual",
                similarity=0.85 + n / 100, verdict=verdict,
            ))
        s.commit()
    client = TestClient(create_app(config))
    client.sessionmaker = Session
    return client


def active(client) -> list[EventIdentity]:
    with client.sessionmaker() as s:
        return [
            link for link in s.query(EventIdentity).filter_by(event_id=1).all()
            if link.unlinked_at is None
        ]


def bulk(client, **data):
    return client.post("/events/1/identity/bulk", data=data,
                       follow_redirects=False)


# -- bulk detachment ------------------------------------------------------


def test_unlink_all_detaches_every_claim(env):
    assert len(active(env)) == 4
    assert bulk(env, action="unlink_all").status_code == 303
    assert active(env) == []


def test_unlink_wrong_leaves_the_others(env):
    """Two are marked wrong; the unjudged and the confirmed stay."""
    bulk(env, action="unlink_wrong")
    remaining = active(env)
    assert len(remaining) == 2
    assert {link.verdict for link in remaining} == {None, "confirmed"}


def test_keep_only_detaches_everything_else(env):
    """The common repair on a track that absorbed two people: one of
    these claims is right, the rest are the other person."""
    keep = active(env)[0].id
    bulk(env, action="keep_only", keep_id=keep)
    remaining = active(env)
    assert [link.id for link in remaining] == [keep]


def test_detached_claims_survive_as_a_record(env):
    """Unlinking is not deleting — "this event was once called someone
    else" is what a later reviewer needs, and re-attaching is how a
    correction gets corrected (CLD-36)."""
    bulk(env, action="unlink_all")
    with env.sessionmaker() as s:
        rows = s.query(EventIdentity).filter_by(event_id=1).all()
        assert len(rows) == 4
        assert all(r.unlinked_at is not None for r in rows)
        # Unlink asserts the pairing never held, which implies a verdict
        # /stats can count.
        assert all(r.verdict == "wrong" for r in rows)


def test_keep_only_needs_a_link_that_is_on_this_event(env):
    assert bulk(env, action="keep_only", keep_id=999).status_code == 404
    # Omitting keep_id is caught by the route with its own message rather
    # than falling through to a bare validation error.
    assert bulk(env, action="keep_only").status_code == 400
    assert len(active(env)) == 4


def test_an_unknown_bulk_action_is_refused(env):
    assert bulk(env, action="delete_everything").status_code == 400
    assert len(active(env)) == 4


def test_bulk_is_idempotent(env):
    """Double-submitting must not decrement counts twice."""
    bulk(env, action="unlink_all")
    assert bulk(env, action="unlink_all").status_code == 303
    assert active(env) == []


def test_bulk_redirect_cannot_leave_the_site(env):
    r = bulk(env, action="unlink_all", next_url="//evil.example/x")
    assert r.headers["location"] == "/events/1"


# -- the marker -----------------------------------------------------------


def test_marking_multi_subject_records_when(env):
    r = env.post("/events/1/multi-subject", data={"multi": "1"},
                 follow_redirects=False)
    assert r.status_code == 303
    with env.sessionmaker() as s:
        event = s.get(Event, 1)
        assert event.multi_subject is True
        assert event.multi_subject_at is not None


def test_the_marker_leaves_identity_claims_alone(env):
    """It records a tracking failure. Which claims to keep is a separate
    judgement the operator may not want to make in the same click."""
    env.post("/events/1/multi-subject", data={"multi": "1"})
    assert len(active(env)) == 4


def test_the_marker_can_be_retracted(env):
    env.post("/events/1/multi-subject", data={"multi": "1"})
    env.post("/events/1/multi-subject", data={"multi": "0"})
    with env.sessionmaker() as s:
        event = s.get(Event, 1)
        assert event.multi_subject is False
        assert event.multi_subject_at is None


def test_the_marker_404s_for_an_unknown_event(env):
    assert env.post("/events/999/multi-subject",
                    data={"multi": "1"}).status_code == 404


def test_the_event_page_offers_the_marker_and_bulk_actions(env):
    body = env.get("/events/1").text
    assert "more than one person here" in body
    assert "unlink all" in body
    assert "keep only" in body
    # Fifteen eagerly-rendered identity pickers were most of the clutter.
    assert body.count("<summary>reassign</summary>") == 4
