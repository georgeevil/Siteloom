"""Accuracy readout arithmetic (CLD-17).

The queries are the easy part; the rules about *how* the numbers may be
presented are the part that goes wrong silently, so they are asserted
here: an unreviewed identifier has no precision, plate matches stay out of
the similarity view, and clearing an event without judging it is counted
rather than hidden.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from siteloom.stats import (
    camera_stats,
    identifier_stats,
    review_coverage,
    similarity_histograms,
)
from siteloom.store import (
    Camera,
    Event,
    EventIdentity,
    Identity,
    get_session,
    init_db,
    make_engine,
)

TS = datetime(2026, 8, 7, 12, 0, 0)


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/stats.db")
    init_db(engine)
    with get_session(engine)() as s:
        s.add(Camera(id="front", site_id="t", name="Front Yard"))
        s.add(Camera(id="back", site_id="t", name="Back Door"))
        s.commit()
        yield s


def add_event(session, camera="front", cls="person", at=TS, **kw) -> Event:
    event = Event(
        camera_id=camera,
        class_name=cls,
        first_seen=at,
        last_seen=at,
        detection_count=3,
        **kw,
    )
    session.add(event)
    session.flush()
    return event


def add_identity(session, key="face", cls="person", label=None) -> Identity:
    identity = Identity(
        identifier_key=key, class_name=cls, label=label, first_seen=TS, last_seen=TS
    )
    session.add(identity)
    session.flush()
    return identity


def link(session, event, identity=None, *, key="face", **kw) -> EventIdentity:
    row = EventIdentity(
        event_id=event.id,
        identity_id=identity.id if identity else None,
        identifier_key=key,
        **kw,
    )
    session.add(row)
    session.flush()
    return row


def by_key(stats) -> dict:
    return {stat.key: stat for stat in stats}


def test_unreviewed_identifier_has_no_precision(session):
    """None, not 0.0. A zero here reads as 'nothing went wrong', which is
    a claim nobody made."""
    identity = add_identity(session)
    link(session, add_event(session), identity, similarity=0.5, matched_by="visual")
    session.commit()

    face = by_key(identifier_stats(session))["face"]
    assert face.claims == 1
    assert face.reviewed == 0
    assert face.wrong_rate is None
    # Coverage is not undefined the way precision is — reviewing none of
    # one claim really is 0%, and saying so is the point.
    assert face.coverage == pytest.approx(0.0)


def test_precision_carries_its_denominators(session):
    """Two of three claims judged, one of those wrong: the rate is over
    the reviewed pair, and coverage says how partial that sample is."""
    identity = add_identity(session)
    for verdict in ("confirmed", "wrong", None):
        link(
            session,
            add_event(session),
            identity,
            similarity=0.5,
            matched_by="visual",
            verdict=verdict,
        )
    session.commit()

    face = by_key(identifier_stats(session))["face"]
    assert (face.claims, face.confirmed, face.wrong) == (3, 1, 1)
    assert face.reviewed == 2 and face.unreviewed == 1
    assert face.wrong_rate == pytest.approx(0.5)
    assert face.coverage == pytest.approx(2 / 3)


def test_misses_are_attributed_to_their_identifier(session):
    """A null-identity row is a recorded miss; recall is only computable
    because it names which identifier should have claimed something."""
    event = add_event(session)
    link(session, event, None, key="face", verdict="missed")
    link(session, add_event(session), add_identity(session), key="face", similarity=0.4)
    session.commit()

    face = by_key(identifier_stats(session))["face"]
    assert face.misses == 1
    assert face.claims == 1
    assert face.recall_denominator == 2


def test_plate_and_visual_matches_split(session):
    """PRD §6.4: both write the same vehicle identity, so only matched_by
    can tell them apart afterwards."""
    vehicle = add_identity(session, key="vehicle", cls="car")
    link(session, add_event(session, cls="car"), vehicle, key="vehicle",
         matched_by="plate", similarity=1.0)
    link(session, add_event(session, cls="car"), vehicle, key="vehicle",
         matched_by="visual", similarity=0.86, learned_plate=True)
    session.commit()

    stat = by_key(identifier_stats(session))["vehicle"]
    assert stat.plate_matches == 1
    assert stat.visual_matches == 1
    assert stat.learned_plates == 1
    assert stat.unattributed == 0


def test_plate_matches_stay_out_of_the_similarity_view(session):
    """A plate match carries a synthetic similarity. Letting it into the
    histogram puts a spike at a made-up value exactly where the operator
    is reading the cutoff."""
    vehicle = add_identity(session, key="vehicle", cls="car")
    link(session, add_event(session, cls="car"), vehicle, key="vehicle",
         matched_by="plate", similarity=1.0)
    link(session, add_event(session, cls="car"), vehicle, key="vehicle",
         matched_by="visual", similarity=0.83)
    session.commit()

    histogram = similarity_histograms(session, {"vehicle": 0.82})[0]
    assert histogram.total == 1  # the visual one only
    assert sum(b.count for b in histogram.bins) == 1
    assert histogram.threshold == 0.82


def test_histograms_are_per_identifier_never_pooled(session):
    """Face sits near 0.36 and generic near 0.8+; one shared histogram
    would describe the configuration rather than the data."""
    link(session, add_event(session), add_identity(session, key="face"),
         key="face", matched_by="visual", similarity=0.38)
    link(session, add_event(session, cls="car"),
         add_identity(session, key="vehicle", cls="car"),
         key="vehicle", matched_by="visual", similarity=0.84)
    session.commit()

    histograms = {h.key: h for h in similarity_histograms(
        session, {"face": 0.36, "vehicle": 0.82}
    )}
    assert set(histograms) == {"face", "vehicle"}
    assert histograms["face"].total == 1
    assert histograms["vehicle"].total == 1
    # Each lands in its own bin, against its own threshold.
    assert histograms["face"].bins[7].count == 1  # 0.35–0.40
    assert histograms["vehicle"].bins[16].count == 1  # 0.80–0.85


def test_histogram_flags_the_mass_either_side_of_the_cutoff(session):
    """Counted against the threshold itself, not the enclosing bin.

    0.36 falls inside the 0.35–0.40 bin, so anything derived from bin
    edges answers a question about 0.35 or 0.40 instead — which puts 0.37
    (wrong, and above the cutoff) on the wrong side of the line.
    """
    identity = add_identity(session)
    for similarity, verdict in [
        (0.42, "wrong"),
        (0.37, "wrong"),
        (0.38, "confirmed"),
        (0.30, "wrong"),
    ]:
        link(session, add_event(session), identity, key="face",
             matched_by="visual", similarity=similarity, verdict=verdict)
    session.commit()

    histogram = similarity_histograms(session, {"face": 0.36})[0]
    # Three wrong, but 0.30 never cleared the threshold — raising the
    # cutoff can only remove the two that did.
    assert histogram.wrong_above == 2
    # 0.38 sits within a bin-width above the cutoff: what raising it costs.
    assert histogram.confirmed_near == 1


def test_cleared_without_a_verdict_is_counted(session):
    """A soak can be worked to zero while producing no accuracy data.
    Clearing stays cheap; the gap has to be visible instead."""
    bare = add_event(session, reviewed_at=TS)
    judged = add_event(session, reviewed_at=TS)
    link(session, judged, add_identity(session), verdict="confirmed", similarity=0.5)
    add_event(session)  # still open
    session.commit()

    coverage = review_coverage(session)
    assert coverage.events == 3
    assert coverage.cleared == 2
    assert coverage.cleared_without_verdict == 1
    assert coverage.open == 1
    assert coverage.link_coverage == pytest.approx(1.0)
    assert bare.id != judged.id


def test_camera_breakdown_counts_visits_not_links(session):
    """Three claims on one event are still one visit.

    A car pulls up and someone gets out: the vehicle, the person and the
    face identifiers can each claim that event. One identity cannot hold
    two standing claims on it (CLD-133) — the several claims are several
    identities, and the camera breakdown counts events, not rows.
    """
    event = add_event(session, camera="front")
    for key, cls in (("face", "person"), ("person", "person"), ("vehicle", "car")):
        link(
            session,
            event,
            add_identity(session, key=key, cls=cls),
            key=key,
            similarity=0.5,
            matched_by="visual",
        )
    add_event(session, camera="front")  # unmatched
    add_event(session, camera="back")
    session.commit()

    stats = {c.camera_id: c for c in camera_stats(session)}
    assert stats["front"].events == 2
    assert stats["front"].matched_events == 1
    assert stats["front"].unmatched_events == 1
    assert stats["front"].name == "Front Yard"
    assert stats["back"].matched_events == 0


def test_window_filters_by_event_time(session):
    identity = add_identity(session)
    link(session, add_event(session, at=TS), identity, similarity=0.5)
    link(session, add_event(session, at=TS - timedelta(days=2)), identity, similarity=0.5)
    session.commit()

    windowed = by_key(identifier_stats(session, since=TS - timedelta(hours=1)))
    assert windowed["face"].claims == 1
    assert by_key(identifier_stats(session))["face"].claims == 2
    assert review_coverage(session, since=TS - timedelta(hours=1)).events == 1
