"""The /bookings screen (CLD-90).

`Event.guest_window` is what suppresses unknown-vehicle alarms during an
expected arrival — the PRD §12 success metric — and until this screen the
console could read that stamp and do nothing else with it. The properties
worth holding are therefore:

* a manual booking survives `sync-bookings` (the whole point: a
  correction the next sync reverts is worse than no correction),
* the empty state says which emptiness it is, per the CLD-26 rule,
* the count on a row and the list its link opens are the same set,
* and writing a booking is `edit` work, not `admin`, enforced by the one
  middleware rather than by a decorator on the route.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from siteloom.config import CameraConfig, GuestConfig, SiteConfig, StorageConfig
from siteloom.guests import GuestWindows, sync_bookings
from siteloom.store import (
    Booking,
    Camera,
    Event,
    User,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web.auth import hash_password, required_role
from siteloom.web.app import create_app
from siteloom.web.bookings_routes import (
    MANUAL_UID_PREFIX,
    BookingError,
    feed_label,
    parse_window,
)

#: A fixed "now" the sections are computed around is not available — the
#: page reads the wall clock — so fixtures are placed relative to it.
NOW = datetime.now(timezone.utc).replace(tzinfo=None)

ICS = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//test//test//EN
BEGIN:VEVENT
UID:booking-1@example.com
SUMMARY:Reserved - Unit A
DTSTART:20260810T150000Z
DTEND:20260814T110000Z
END:VEVENT
END:VCALENDAR
"""


def make_env(tmp_path, ical: str = "", cameras=None):
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=cameras
        if cameras is not None
        else [CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/b.db", media_dir=str(tmp_path / "m")
        ),
        guests=GuestConfig(ical=ical, arrival_pre_hours=2, arrival_post_hours=4),
    )
    config.identity.enabled = False
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(Camera(id="cam1", site_id="t", name="Cam One"))
        s.commit()
    return TestClient(create_app(config), follow_redirects=False), Session


def add_booking(Session, uid, summary, start, end, source="ical"):
    with Session() as s:
        s.add(
            Booking(uid=uid, summary=summary, start=start, end=end, source=source)
        )
        s.commit()


def add_event(Session, first_seen, last_seen, class_name="car", guest=False):
    with Session() as s:
        s.add(
            Event(
                camera_id="cam1",
                track_id=1,
                class_name=class_name,
                first_seen=first_seen,
                last_seen=last_seen,
                detection_count=3,
                guest_window=guest,
            )
        )
        s.commit()


def add_user(Session, username, role):
    with Session() as s:
        s.add(
            User(
                username=username,
                password_hash=hash_password("hunter2!"),
                role=role,
                created_at=NOW,
            )
        )
        s.commit()


def login(client, username):
    r = client.post(
        "/login",
        data={"username": username, "password": "hunter2!", "next": "/bookings"},
    )
    assert r.status_code == 303, r.text[:200]
    return client


def form(start: datetime, end: datetime, summary="Unit A, Okonjo"):
    return {
        "summary": summary,
        "start": start.strftime("%Y-%m-%dT%H:%M"),
        "end": end.strftime("%Y-%m-%dT%H:%M"),
    }


# -- the guarantee: a manual booking survives sync-bookings ----------------


@pytest.fixture
def sync_session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/sync.db")
    init_db(engine)
    with get_session(engine)() as s:
        yield s


def test_sync_leaves_a_manual_booking_alone(sync_session, tmp_path):
    """The reason `source` exists. An operator's correction must still be
    there — unchanged — after the calendar it corrects is re-read."""
    ics = tmp_path / "b.ics"
    ics.write_text(ICS)
    sync_session.add(
        Booking(
            uid=f"{MANUAL_UID_PREFIX}abc",
            summary="Late arrival, unit C",
            start=datetime(2026, 8, 11, 22, 0),
            end=datetime(2026, 8, 12, 10, 0),
            source="manual",
        )
    )
    sync_session.commit()

    sync_bookings(sync_session, GuestConfig(ical=str(ics)))

    rows = {b.uid: b for b in sync_session.query(Booking).all()}
    assert len(rows) == 2
    manual = rows[f"{MANUAL_UID_PREFIX}abc"]
    assert manual.summary == "Late arrival, unit C"
    assert manual.start == datetime(2026, 8, 11, 22, 0)
    assert manual.source == "manual"
    assert rows["booking-1@example.com"].source == "ical"


def test_a_feed_uid_colliding_with_a_manual_row_never_overwrites_it(
    sync_session, tmp_path
):
    """`uid` is unique, so only one of the two rows can exist. Of a feed
    entry and a human correction, the human's is the one that survives."""
    ics = tmp_path / "b.ics"
    ics.write_text(ICS)
    sync_session.add(
        Booking(
            uid="booking-1@example.com",  # the operator claimed the feed's uid
            summary="Corrected by hand",
            start=datetime(2026, 8, 10, 9, 0),
            end=datetime(2026, 8, 14, 9, 0),
            source="manual",
        )
    )
    sync_session.commit()

    sync_bookings(sync_session, GuestConfig(ical=str(ics)))

    row = sync_session.query(Booking).one()
    assert row.summary == "Corrected by hand"
    assert row.start == datetime(2026, 8, 10, 9, 0)
    assert row.source == "manual"


def test_a_manual_booking_suppresses_alarms_like_any_other(sync_session):
    """Provenance changes who may edit the row, not what it does: the
    arrival window is the point of entering one by hand."""
    sync_session.add(
        Booking(
            uid=f"{MANUAL_UID_PREFIX}x",
            summary="Unit A",
            start=datetime(2026, 8, 20, 15, 0),
            end=datetime(2026, 8, 23, 11, 0),
            source="manual",
        )
    )
    sync_session.commit()
    windows = GuestWindows(
        sync_session, GuestConfig(arrival_pre_hours=2, arrival_post_hours=4)
    )
    assert windows.contains(datetime(2026, 8, 20, 14, 0))
    assert not windows.contains(datetime(2026, 8, 20, 20, 0))


def test_the_source_column_is_added_to_an_existing_database(tmp_path):
    """Additive migration only (`_ensure_columns`): a database created
    before the column keeps its rows, and they read as feed rows —
    which is what they were."""
    db = tmp_path / "old.db"
    engine = make_engine(f"sqlite:///{db}")
    with engine.begin() as conn:
        conn.execute(
            text(
                'CREATE TABLE bookings (id INTEGER NOT NULL PRIMARY KEY, '
                'uid VARCHAR NOT NULL, summary VARCHAR, start DATETIME NOT NULL, '
                '"end" DATETIME NOT NULL)'
            )
        )
        conn.execute(
            text(
                'INSERT INTO bookings (uid, summary, start, "end") '
                "VALUES ('legacy@x', 'Old stay', '2026-08-01 10:00:00', "
                "'2026-08-03 10:00:00')"
            )
        )

    init_db(engine)

    with get_session(engine)() as s:
        row = s.scalars(select(Booking)).one()
        assert row.summary == "Old stay"
        assert row.source == "ical"
        assert not row.is_manual


# -- empty states (CLD-26 rule) -------------------------------------------


def test_no_feed_configured_is_not_the_same_as_no_bookings(tmp_path):
    client, _ = make_env(tmp_path, ical="")
    body = client.get("/bookings").text
    assert "No iCal feed is configured" in body
    assert "guests.ical" in body
    # The misleading reading this page exists to avoid.
    assert "No bookings have finished yet." not in body


def test_a_configured_feed_with_no_rows_says_it_has_not_been_read(tmp_path):
    client, _ = make_env(tmp_path, ical="/srv/cal/kai.ics")
    body = client.get("/bookings").text
    assert "no bookings have been read from it" in body
    assert "siteloom sync-bookings" in body
    assert "No iCal feed is configured" not in body


def test_an_empty_section_is_reported_per_section_not_as_a_dead_page(tmp_path):
    """With bookings present, an empty section is a fact about the
    calendar — it must not trigger the install-level empty state."""
    client, Session = make_env(tmp_path, ical="/srv/cal/kai.ics")
    add_booking(
        Session, "past@x", "Finished", NOW - timedelta(days=5), NOW - timedelta(days=4)
    )
    body = client.get("/bookings").text
    assert "No booking covers right now." in body
    assert "No bookings ahead." in body
    assert "No bookings have finished yet." not in body
    assert "No iCal feed is configured" not in body


def test_a_feed_url_is_named_without_handing_over_its_token():
    """A booking-site iCal URL is itself the credential."""
    label = feed_label("https://www.airbnb.com/calendar/ical/123.ics?s=SECRET")
    assert "SECRET" not in label
    assert "123.ics" not in label
    assert "airbnb.com" in label
    assert feed_label("/srv/cal/kai.ics") == "/srv/cal/kai.ics"
    assert feed_label("") == ""


def test_the_configured_feed_page_never_prints_the_raw_url(tmp_path):
    secret = "https://cal.example/ical/abc.ics?s=SUPERSECRET"
    client, _ = make_env(tmp_path, ical=secret)
    body = client.get("/bookings").text
    assert "SUPERSECRET" not in body
    assert "cal.example" in body


# -- sections and the link back to / --------------------------------------


def test_bookings_are_grouped_current_upcoming_past(tmp_path):
    client, Session = make_env(tmp_path)
    add_booking(
        Session, "now@x", "Staying now", NOW - timedelta(hours=2), NOW + timedelta(days=1)
    )
    add_booking(
        Session, "soon@x", "Arriving soon", NOW + timedelta(days=3), NOW + timedelta(days=5)
    )
    add_booking(
        Session, "gone@x", "Checked out", NOW - timedelta(days=9), NOW - timedelta(days=7)
    )
    body = client.get("/bookings").text
    # Order on the page is the order of the sections.
    assert (
        body.index("Staying now")
        < body.index("Arriving soon")
        < body.index("Checked out")
    )


def test_the_event_count_links_to_exactly_the_events_it_counted(tmp_path):
    """The link is the point of the screen, so the number on the row and
    the list it opens have to be the same set — which is why the count
    reuses the events list's own window comparison."""
    client, Session = make_env(tmp_path)
    start, end = NOW - timedelta(hours=6), NOW - timedelta(hours=2)
    add_booking(Session, "stay@x", "Unit A", start, end)
    add_event(Session, start + timedelta(minutes=10), start + timedelta(minutes=12))
    add_event(Session, start + timedelta(minutes=30), start + timedelta(minutes=33))
    add_event(Session, NOW - timedelta(days=3), NOW - timedelta(days=3))  # outside

    page = client.get("/bookings").text
    href = re.search(r'href="(/\?since=[^"]+)"', page).group(1)
    assert ">2</a>" in page

    listing = client.get(href.replace("&amp;", "&")).text
    assert "2 of 3 events" in listing


def test_the_guest_stamp_is_totalled_per_booking(tmp_path):
    """`Event.guest_window` was a per-event boolean nobody could total;
    this column is the PRD §12 metric made readable."""
    client, Session = make_env(tmp_path)
    check_in = NOW - timedelta(hours=3)
    add_booking(Session, "stay@x", "Unit A", check_in, NOW + timedelta(days=2))
    # Inside the arrival window (−2 h/+4 h) and stamped.
    add_event(Session, check_in + timedelta(minutes=5), check_in + timedelta(minutes=6),
              guest=True)
    # Inside the stay, but long after the arrival window closed.
    add_event(Session, NOW + timedelta(days=1), NOW + timedelta(days=1))
    body = client.get("/bookings").text
    counts = re.findall(r'<a href="/\?since=[^"]+">(\d+)</a>', body)
    assert counts == ["2", "1"]  # events in the stay, then guest-stamped


def test_a_booking_with_no_events_shows_a_plain_zero_not_a_link(tmp_path):
    client, Session = make_env(tmp_path)
    add_booking(
        Session, "quiet@x", "Unit B", NOW - timedelta(days=2), NOW - timedelta(days=1)
    )
    body = client.get("/bookings").text
    assert 'class="bk-zero">0<' in body


# -- manual bookings: add / edit / delete ---------------------------------


def test_adding_a_manual_booking_marks_its_provenance(tmp_path):
    client, Session = make_env(tmp_path)
    start, end = NOW + timedelta(days=1), NOW + timedelta(days=3)
    r = client.post("/bookings", data=form(start, end))
    assert r.status_code == 303
    with Session() as s:
        row = s.scalars(select(Booking)).one()
        assert row.source == "manual"
        assert row.uid.startswith(MANUAL_UID_PREFIX)
        assert row.summary == "Unit A, Okonjo"
        assert row.start.strftime("%Y-%m-%dT%H:%M") == start.strftime("%Y-%m-%dT%H:%M")


def test_a_manual_booking_can_be_edited_and_deleted(tmp_path):
    client, Session = make_env(tmp_path)
    client.post("/bookings", data=form(NOW + timedelta(days=1), NOW + timedelta(days=3)))
    with Session() as s:
        booking_id = s.scalars(select(Booking)).one().id

    r = client.post(
        f"/bookings/{booking_id}/edit",
        data=form(NOW + timedelta(days=2), NOW + timedelta(days=4), "Unit B, corrected"),
    )
    assert r.status_code == 303
    with Session() as s:
        assert s.scalars(select(Booking)).one().summary == "Unit B, corrected"

    assert client.post(f"/bookings/{booking_id}/delete").status_code == 303
    with Session() as s:
        assert s.scalars(select(Booking)).all() == []


def test_an_ical_booking_cannot_be_edited_or_deleted_in_place(tmp_path):
    """Explicitly out of scope: the feed owns those rows and the next
    sync would revert the change, so the console refuses rather than
    pretending the edit will hold."""
    client, Session = make_env(tmp_path, ical="/srv/cal/kai.ics")
    add_booking(Session, "feed@x", "From the feed", NOW, NOW + timedelta(days=1))
    with Session() as s:
        booking_id = s.scalars(select(Booking)).one().id

    edit = client.post(
        f"/bookings/{booking_id}/edit", data=form(NOW, NOW + timedelta(days=2), "Nope")
    )
    assert edit.status_code == 409
    assert client.post(f"/bookings/{booking_id}/delete").status_code == 409
    with Session() as s:
        assert s.scalars(select(Booking)).one().summary == "From the feed"


def test_the_page_offers_edit_only_for_manual_rows(tmp_path):
    client, Session = make_env(tmp_path, ical="/srv/cal/kai.ics")
    add_booking(Session, "feed@x", "From the feed", NOW, NOW + timedelta(days=1))
    assert "edit / delete" not in client.get("/bookings").text
    client.post("/bookings", data=form(NOW + timedelta(days=2), NOW + timedelta(days=3)))
    assert "edit / delete" in client.get("/bookings").text


def test_a_missing_booking_is_a_404(tmp_path):
    client, _ = make_env(tmp_path)
    assert client.post("/bookings/999/delete").status_code == 404


# -- input validation ------------------------------------------------------


def test_check_out_before_check_in_is_refused_with_a_reason(tmp_path):
    client, Session = make_env(tmp_path)
    r = client.post("/bookings", data=form(NOW + timedelta(days=3), NOW))
    assert r.status_code == 400
    assert "Check-out cannot be before check-in." in r.text
    with Session() as s:
        assert s.scalars(select(Booking)).all() == []


def test_an_unparseable_time_does_not_500(tmp_path):
    client, Session = make_env(tmp_path)
    r = client.post(
        "/bookings", data={"summary": "x", "start": "yesterday", "end": "soon"}
    )
    assert r.status_code == 400
    assert "check-in and a check-out time" in r.text
    with Session() as s:
        assert s.scalars(select(Booking)).all() == []


def test_a_summary_is_required_because_the_row_is_read_by_a_human(tmp_path):
    client, _ = make_env(tmp_path)
    r = client.post(
        "/bookings", data=form(NOW, NOW + timedelta(days=1), summary="   ")
    )
    assert r.status_code == 400
    assert "summary" in r.text.lower()


def test_parse_window_rejects_an_offset_rather_than_storing_a_mixed_frame():
    """Every stored timestamp is naive UTC; one row in another frame
    would move the arrival window that suppresses alarms."""
    with pytest.raises(BookingError):
        parse_window("2026-08-10T15:00+02:00", "2026-08-11T15:00")
    first, last = parse_window("2026-08-10T15:00", "2026-08-11T15:00")
    assert (first, last) == (datetime(2026, 8, 10, 15), datetime(2026, 8, 11, 15))


# -- roles -----------------------------------------------------------------


def test_writing_a_booking_is_edit_work_not_admin():
    """Correcting the calendar is judgment about what happened, not
    reconfiguration — so /bookings stays out of ADMIN_PREFIXES."""
    assert required_role("GET", "/bookings") == "view"
    assert required_role("POST", "/bookings") == "edit"
    assert required_role("POST", "/bookings/1/edit") == "edit"
    assert required_role("POST", "/bookings/1/delete") == "edit"


def test_a_viewer_can_read_the_calendar_but_not_change_it(tmp_path):
    client, Session = make_env(tmp_path)
    add_booking(Session, "feed@x", "From the feed", NOW, NOW + timedelta(days=1))
    add_user(Session, "vera", "view")
    login(client, "vera")
    assert client.get("/bookings").status_code == 200
    r = client.post("/bookings", data=form(NOW, NOW + timedelta(days=1)))
    assert r.status_code == 403
    with Session() as s:
        assert len(s.scalars(select(Booking)).all()) == 1


def test_an_editor_can_correct_the_calendar(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "eddy", "edit")
    login(client, "eddy")
    r = client.post("/bookings", data=form(NOW, NOW + timedelta(days=1)))
    assert r.status_code == 303
    with Session() as s:
        assert s.scalars(select(Booking)).one().source == "manual"


def test_bookings_are_reachable_from_the_sidebar(tmp_path):
    client, _ = make_env(tmp_path)
    body = client.get("/bookings").text
    assert 'href="/bookings"' in body
    assert body.count("sl-nav-item active") == 1
