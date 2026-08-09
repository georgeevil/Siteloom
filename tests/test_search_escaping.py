"""What the operator typed is a string, not a pattern (CLD-67).

`%` and `_` are LIKE wildcards and SQLAlchemy's `.like()` neither
escapes them nor emits an ESCAPE clause, so a plate or unit number
containing one used to match far more than it named. The second half of
the same route: a numeric query answered only "is there an event with
this id", which reads as "no such visit" for a unit number, a door code
or a plate fragment that happens to be all digits.
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
    LibraryItem,
    LibrarySource,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web.app import create_app, like_pattern

TS = datetime(2026, 8, 9, 9, 0, 0)


def _identity(session, label, plate=None):
    identity = Identity(
        identifier_key="vehicle",
        class_name="car",
        label=label,
        plate=plate,
        first_seen=TS,
        last_seen=TS,
    )
    session.add(identity)
    session.flush()
    return identity


@pytest.fixture
def client(tmp_path):
    """A gallery of names that are also LIKE patterns, plus the decoys
    an unescaped pattern would wrongly pull in."""
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="north", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/s.db", media_dir=str(tmp_path / "m")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(Camera(id="north", site_id="t", name="Gate 7"))
        s.add(Camera(id="dock", site_id="t", name="South Dock"))
        _identity(s, "Unit_4")
        _identity(s, "Unita4")  # "_" as a wildcard would match this
        _identity(s, "100% Courier")
        _identity(s, "Plain Courier")  # "%" as a wildcard would match this
        _identity(s, r"back\slash")
        van = _identity(s, "Van 7", plate="KJ-4471")

        s.add(LibrarySource(id=1, path="/a", name="A", added_at=TS))
        for name in ("gate_7.jpg", "gateX7.jpg"):
            s.add(
                LibraryItem(
                    source_id=1,
                    path=f"/a/{name}",
                    kind="image",
                    status="indexed",
                    mtime=TS,
                )
            )

        # Event 1 is reachable only through its matched identity's
        # plate; event 2 only by its own number.
        first = Event(
            camera_id="north",
            track_id=1,
            class_name="car",
            first_seen=TS,
            last_seen=TS,
            detection_count=3,
        )
        s.add(first)
        s.flush()
        s.add(EventIdentity(event_id=first.id, identity_id=van.id, similarity=0.9))
        s.add(
            Event(
                camera_id="dock",
                track_id=2,
                class_name="person",
                first_seen=TS,
                last_seen=TS,
                detection_count=3,
            )
        )
        s.commit()
    return TestClient(create_app(config))


def search(client, q: str) -> str:
    r = client.get("/search", params={"q": q})
    assert r.status_code == 200
    return r.text


def test_underscore_is_a_character_not_a_wildcard(client):
    body = search(client, "Unit_4")
    assert "Unit_4" in body
    assert "Unita4" not in body
    assert "Identities · 1" in body


def test_percent_is_a_character_not_a_wildcard(client):
    body = search(client, "100%")
    assert "100% Courier" in body
    assert "Plain Courier" not in body


def test_a_bare_wildcard_does_not_return_the_whole_table(client):
    """"%" alone used to match every row of every table — the loudest
    symptom, and the one that makes the box useless as a filter."""
    body = search(client, "%")
    assert "100% Courier" in body
    assert "Plain Courier" not in body
    assert "Identities · 1" in body
    assert "Library" not in body


def test_the_escape_character_itself_is_escaped(client):
    """A typed backslash is a backslash. Escaping `%`/`_` without also
    escaping the escape leaves the pattern malformed for exactly the
    query that contains one."""
    body = search(client, "back\\")
    assert "back\\slash" in body or "back\\\\slash" in body
    assert "Identities · 1" in body
    assert search(client, "\\").count("class=\"hit\"") == 1


def test_library_paths_are_escaped_too(client):
    body = search(client, "gate_7")
    assert "gate_7.jpg" in body
    assert "gateX7.jpg" not in body


def test_like_pattern_escapes_every_metacharacter():
    assert like_pattern("a%b") == "%a\\%b%"
    assert like_pattern("a_b") == "%a\\_b%"
    # The escape first, or the escapes it adds get escaped in turn.
    assert like_pattern("a\\b") == "%a\\\\b%"
    assert like_pattern("plain") == "%plain%"


def test_a_number_keeps_the_event_id_fast_path(client):
    """Typing an event number still lands on that event, even when
    nothing about it reads as the number."""
    body = search(client, "2")
    assert "Events · 1" in body
    assert "#2" in body


def test_a_number_also_runs_the_other_searches(client):
    """"7" is a camera name, a label and a plate fragment as much as it
    is an event id; answering only the id hid all three."""
    body = search(client, "7")
    assert "Van 7" in body  # identity by label
    assert "Events · 1" in body  # its visit, via camera name and plate
    assert "gate_7.jpg" in body  # library filename


def test_a_number_too_large_for_a_row_id_matches_nothing_and_does_not_crash(
    client,
):
    """A pasted phone number is digits: binding it as an id raises
    OverflowError out of the sqlite driver, 500ing the search page."""
    assert "Nothing matches" in search(client, "9" * 30)
