"""The events list's timeframe and evidence facets.

The timeframe picker's presets are living windows — "the last 24h",
judged when the page renders — so a pasted link means the same thing to
its recipient, unlike a frozen since/until pair. The absolute inputs
remain for a custom range and are an input boundary (CLD-100): an
operator types site wall-clock, the store holds naive UTC, and for a
while the two were compared raw, which shifted every custom window by
the site's UTC offset.

The facet chips (plate read / face match) narrow like Needs review does,
by evidence the event carries — they are not class kinds, because a
plated event is already a vehicle.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import (
    Event,
    EventIdentity,
    Identity,
    PlateRead,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web.app import create_app

NOW = datetime.now(timezone.utc).replace(tzinfo=None)


def build(tmp_path, timezone_name: str = ""):
    config = SiteConfig(
        site_id="t",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/tf.db", media_dir=str(tmp_path / "m")
        ),
        timezone=timezone_name,
    )
    config.identity.enabled = False
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    return config, get_session(engine)


def add_event(session, *, when, class_name="person", count=5):
    event = Event(
        camera_id="cam1",
        class_name=class_name,
        first_seen=when,
        last_seen=when,
        detection_count=count,
        confidence_sum=0.9 * count,
        best_confidence=0.9,
    )
    session.add(event)
    session.flush()
    return event


def test_a_preset_is_a_living_window(tmp_path):
    config, Session = build(tmp_path)
    with Session() as session:
        recent = add_event(session, when=NOW - timedelta(hours=1))
        stale = add_event(session, when=NOW - timedelta(days=3))
        session.commit()
        recent_id, stale_id = recent.id, stale.id
    client = TestClient(create_app(config))

    page = client.get("/?last=24h").text
    assert f'data-event="{recent_id}"' in page
    assert f'data-event="{stale_id}"' not in page
    # 7d covers both; the picker marks the active segment.
    week = client.get("/?last=7d").text
    assert f'data-event="{stale_id}"' in week
    assert 'class="tf-seg on"\n       href="/?last=7d"' in week or ">7d</a>" in week


def test_an_unknown_preset_degrades_to_all_time_not_a_500(tmp_path):
    config, Session = build(tmp_path)
    with Session() as session:
        add_event(session, when=NOW - timedelta(days=3))
        session.commit()
    client = TestClient(create_app(config))
    r = client.get("/?last=fortnight")
    assert r.status_code == 200
    assert "data-event=" in r.text  # nothing filtered out


def test_absolute_bounds_are_read_as_site_wall_clock(tmp_path):
    """CLD-100's input boundary: the operator types wall-clock time in
    the site's zone. In Los Angeles, an event at 20:00 UTC happened at
    13:00 local — a since of 12:00 must include it, and comparing the
    typed string raw against the UTC column (the old bug) would not."""
    config, Session = build(tmp_path, timezone_name="America/Los_Angeles")
    when = NOW.replace(hour=20, minute=0, second=0, microsecond=0)
    with Session() as session:
        event = add_event(session, when=when)
        session.commit()
        event_id = event.id
    client = TestClient(create_app(config))
    day = when.strftime("%Y-%m-%d")

    # 12:00 local == 19:00 UTC: the 20:00-UTC event is after it.
    assert f'data-event="{event_id}"' in client.get(f"/?since={day}T12:00").text
    # 14:00 local == 21:00 UTC: now it is before the window.
    assert f'data-event="{event_id}"' not in client.get(f"/?since={day}T14:00").text


def test_the_evidence_facets_narrow_to_plates_and_faces(tmp_path):
    config, Session = build(tmp_path)
    with Session() as session:
        plated = add_event(session, when=NOW, class_name="car")
        session.add(
            PlateRead(
                event_id=plated.id, at=NOW, text="AB1234", accepted=True,
            )
        )
        rejected = add_event(session, when=NOW, class_name="car")
        session.add(
            PlateRead(
                event_id=rejected.id, at=NOW, text="ZZ", accepted=False,
                reason="too-short",
            )
        )
        faced = add_event(session, when=NOW, class_name="person")
        identity = Identity(
            identifier_key="face", class_name="person",
            first_seen=NOW, last_seen=NOW,
        )
        session.add(identity)
        session.flush()
        session.add(
            EventIdentity(
                event_id=faced.id, identity_id=identity.id,
                identifier_key="face", similarity=0.5, matched_by="visual",
            )
        )
        plain = add_event(session, when=NOW, class_name="person")
        session.commit()
        ids = {
            "plated": plated.id, "rejected": rejected.id,
            "faced": faced.id, "plain": plain.id,
        }
    client = TestClient(create_app(config))

    plates = client.get("/?has_plate=1").text
    assert f'data-event="{ids["plated"]}"' in plates
    # A rejected read is not evidence of a plate; neither is no read.
    assert f'data-event="{ids["rejected"]}"' not in plates
    assert f'data-event="{ids["plain"]}"' not in plates

    faces = client.get("/?has_face=1").text
    assert f'data-event="{ids["faced"]}"' in faces
    assert f'data-event="{ids["plain"]}"' not in faces


def test_arrived_and_last_seen_carry_their_dates(tmp_path):
    """A bare clock is ambiguous the moment the list spans a midnight."""
    config, Session = build(tmp_path)
    when = datetime(2026, 8, 10, 12, 30, 0)
    with Session() as session:
        add_event(session, when=when)
        session.commit()
    client = TestClient(create_app(config))
    page = client.get("/").text
    assert "Aug 10" in page
    assert "12:30:00" in page


@pytest.mark.parametrize("preset", ["1h", "6h", "24h", "7d", "30d"])
def test_every_preset_the_picker_offers_is_understood(tmp_path, preset):
    config, Session = build(tmp_path)
    with Session() as session:
        add_event(session, when=NOW - timedelta(minutes=5))
        session.commit()
    client = TestClient(create_app(config))
    page = client.get(f"/?last={preset}")
    assert page.status_code == 200
    assert "data-event=" in page.text  # a 5-minute-old event clears every preset
