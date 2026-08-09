"""Keyset paging (CLD-104).

The bug this replaces is invisible in a static database and unmissable in
a live one: with OFFSET, a row arriving above the window while an
operator scrolls shifts everything down, so the next request re-serves a
row they already reviewed and skips one they never saw. So the load-
bearing test here inserts rows *between* two fetches, which is the only
way to tell the two schemes apart.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from siteloom.store import Event, get_session, init_db, make_engine
from siteloom.web.paging import (
    CursorError,
    Slice,
    after,
    decode_cursor,
    encode_cursor,
    take,
)

TS = datetime(2026, 8, 9, 12, 0)


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/p.db")
    init_db(engine)
    with get_session(engine)() as s:
        yield s


def add(session, n, ts=None, camera="c"):
    event = Event(
        camera_id=camera,
        track_id=n,
        class_name="person",
        first_seen=ts or (TS + timedelta(seconds=n)),
        last_seen=ts or (TS + timedelta(seconds=n)),
    )
    session.add(event)
    session.commit()
    return event


def page(session, size=3, cursor=None):
    """One slice, newest first — the shape every console list uses."""
    q = select(Event).order_by(Event.first_seen.desc(), Event.id.desc())
    if cursor is not None:
        q = q.where(after((Event.first_seen, Event.id), decode_cursor(cursor, (datetime, int))))
    rows = session.scalars(q.limit(size + 1)).all()
    return take(list(rows), size, lambda e: (e.first_seen, e.id))


# -- the reason this exists ----------------------------------------------


def test_a_row_arriving_mid_scroll_does_not_shift_the_window(session):
    """The OFFSET bug, stated as a test. With `offset(3)` the second
    fetch would re-serve event 3 and never show event 1."""
    for n in range(1, 7):
        add(session, n)

    first = page(session)
    seen = [e.track_id for e in first.rows]
    assert seen == [6, 5, 4]

    # Ingest keeps writing while the operator reads.
    add(session, 99)
    add(session, 98)

    second = page(session, cursor=first.next_cursor)
    assert [e.track_id for e in second.rows] == [3, 2, 1]
    assert not set(seen) & {e.track_id for e in second.rows}


def test_rows_sharing_a_timestamp_are_not_dropped(session):
    """Sampled cameras produce many events on one timestamp. A cursor on
    first_seen alone would skip every row tying with the last delivered."""
    same = TS + timedelta(minutes=5)
    for n in range(1, 7):
        add(session, n, ts=same)

    collected = []
    slice_ = page(session, size=2)
    while True:
        collected.extend(e.track_id for e in slice_.rows)
        if slice_.exhausted:
            break
        slice_ = page(session, size=2, cursor=slice_.next_cursor)

    assert sorted(collected) == [1, 2, 3, 4, 5, 6]
    assert len(collected) == len(set(collected))


def test_walking_to_the_end_terminates_and_says_so(session):
    for n in range(1, 5):
        add(session, n)
    slice_ = page(session, size=2)
    assert not slice_.exhausted
    slice_ = page(session, size=2, cursor=slice_.next_cursor)
    assert slice_.exhausted
    assert slice_.next_cursor is None


def test_an_exactly_full_page_is_not_reported_as_more(session):
    """Over-fetching by one is what answers this without a second query."""
    for n in range(1, 4):
        add(session, n)
    assert page(session, size=3).exhausted


def test_an_empty_list_is_exhausted_not_broken(session):
    result = page(session)
    assert result.rows == []
    assert result.exhausted


# -- cursor encoding -----------------------------------------------------


def test_a_cursor_round_trips(session):
    token = encode_cursor(TS, 41)
    assert decode_cursor(token, (datetime, int)) == (TS, 41)


def test_a_cursor_is_opaque(session):
    """Not a security boundary — a contract one. A legible ?after_id=41
    invites callers to mint their own, and then the sort key can never
    change without breaking them."""
    token = encode_cursor(TS, 41)
    assert "41" not in token
    assert TS.isoformat() not in token


@pytest.mark.parametrize(
    "token",
    ["not-base64!!", "", "YWJj", encode_cursor(TS), encode_cursor(TS, 1, 2)],
)
def test_a_cursor_the_server_did_not_mint_is_refused(token):
    """Every one of these must raise rather than quietly become "no
    cursor" — restarting the list from the top mid-scroll shows the
    operator duplicates, which reads as data corruption, not an error."""
    with pytest.raises(CursorError):
        decode_cursor(token, (datetime, int))


def test_a_non_datetime_component_is_refused():
    with pytest.raises(CursorError):
        decode_cursor(encode_cursor("banana", 1), (datetime, int))


# -- the ascending case, for lists that are not newest-first -------------


def test_ascending_order_is_supported(session):
    for n in range(1, 5):
        add(session, n)
    q = select(Event).order_by(Event.first_seen, Event.id)
    rows = list(session.scalars(q.limit(3)).all())
    first = take(rows, 2, lambda e: (e.first_seen, e.id))
    assert [e.track_id for e in first.rows] == [1, 2]

    values = decode_cursor(first.next_cursor, (datetime, int))
    rest = session.scalars(
        q.where(after((Event.first_seen, Event.id), values, descending=False))
    ).all()
    assert [e.track_id for e in rest] == [3, 4]


def test_slice_is_a_plain_container():
    """Nothing downstream should have to ask the database again to render
    a page it was already handed."""
    result = Slice(rows=[1, 2], next_cursor=None)
    assert result.exhausted and result.rows == [1, 2]
