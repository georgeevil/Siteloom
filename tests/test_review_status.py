"""Event.review_status and its SQL twin must agree (CLD-20).

The triage screen reads the property per row but filters with the SQL
clause, so a disagreement between them shows the operator a row whose
chip contradicts the filter that selected it. This exhausts the state
space rather than spot-checking it.
"""

from __future__ import annotations

import itertools
from datetime import datetime

import pytest
from sqlalchemy import select

from siteloom.store import Camera, Event, EventIdentity, Identity
from siteloom.store.db import get_session, init_db, make_engine
from siteloom.store.models import REVIEW_STATUSES, status_clause, unmatched_clause

# Every combination an event's identity links can be in, up to two links.
VERDICT_SHAPES = [
    (),
    (None,),
    ("confirmed",),
    ("wrong",),
    (None, None),
    (None, "confirmed"),
    (None, "wrong"),
    ("confirmed", "confirmed"),
    ("confirmed", "wrong"),
    ("wrong", "wrong"),
]


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/status.db")
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(Camera(id="cam1", site_id="site", name="Cam One"))
        t = datetime(2026, 8, 6, 9, 0, 0)
        # One event per (verdict shape × missed flag × signed-off) combination.
        for i, (shape, missed, signed_off) in enumerate(
            itertools.product(VERDICT_SHAPES, [False, True], [False, True])
        ):
            event = Event(
                camera_id="cam1",
                track_id=i,
                class_name="person",
                first_seen=t,
                last_seen=t,
                detection_count=1,
                missed_identity=missed,
                reviewed_at=t if signed_off else None,
            )
            s.add(event)
            s.flush()
            for verdict in shape:
                identity = Identity(
                    identifier_key="face",
                    class_name="person",
                    first_seen=t,
                    last_seen=t,
                )
                s.add(identity)
                s.flush()
                s.add(
                    EventIdentity(
                        event_id=event.id,
                        identity_id=identity.id,
                        verdict=verdict,
                    )
                )
        s.commit()
        yield s


def test_every_status_matches_between_python_and_sql(session):
    """The property and the clause select exactly the same rows."""
    for status in REVIEW_STATUSES:
        by_sql = {
            e.id for e in session.scalars(select(Event).where(status_clause(status)))
        }
        by_property = {
            e.id
            for e in session.scalars(select(Event))
            if e.review_status == status
        }
        assert by_sql == by_property, (
            f"{status}: SQL selected {sorted(by_sql)}, "
            f"property selected {sorted(by_property)}"
        )


def test_statuses_partition_every_event(session):
    """No event is in two states, and none is in none of them."""
    events = list(session.scalars(select(Event)))
    assert events
    seen: dict[int, str] = {}
    for status in REVIEW_STATUSES:
        for event in session.scalars(select(Event).where(status_clause(status))):
            assert event.id not in seen, (
                f"event {event.id} matched both {seen[event.id]} and {status}"
            )
            seen[event.id] = status
    assert set(seen) == {e.id for e in events}


def test_a_recorded_miss_flags_until_signed_off(session):
    """missed_identity outranks every verdict combination, but not sign-off."""
    q = select(Event).where(Event.missed_identity.is_(True))
    for event in session.scalars(q):
        expected = "cleared" if event.reviewed_at else "flagged"
        assert event.review_status == expected


def test_sign_off_clears_regardless_of_verdicts(session):
    """The queue answers "what needs me", so a cleared event always leaves it."""
    q = select(Event).where(Event.reviewed_at.is_not(None))
    events = list(session.scalars(q))
    assert events
    for event in events:
        assert event.review_status == "cleared"


def test_an_unmatched_event_can_still_be_cleared(session):
    """The reason sign-off is a field: no identity links means no verdicts
    to infer review from, so such an event could never leave the queue."""
    unmatched = [
        e
        for e in session.scalars(select(Event))
        if not e.identities and not e.missed_identity
    ]
    assert unmatched
    assert {e.review_status for e in unmatched} == {"new", "cleared"}


def test_unmatched_means_no_identity_links(session):
    by_sql = {e.id for e in session.scalars(select(Event).where(unmatched_clause()))}
    by_python = {
        e.id for e in session.scalars(select(Event)) if not e.identities
    }
    assert by_sql == by_python
    assert by_sql, "fixture should include events with no identity links"
