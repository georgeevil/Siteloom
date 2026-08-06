from datetime import datetime

import pytest

from siteloom.config import GuestConfig
from siteloom.guests import GuestWindows, sync_bookings
from siteloom.store import Booking, get_session, init_db, make_engine

ICS = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//test//test//EN
BEGIN:VEVENT
UID:booking-1@example.com
SUMMARY:Reserved - Unit A
DTSTART;VALUE=DATE:20260810
DTEND;VALUE=DATE:20260814
END:VEVENT
BEGIN:VEVENT
UID:booking-2@example.com
SUMMARY:Reserved - Unit B
DTSTART:20260820T150000Z
DTEND:20260823T110000Z
END:VEVENT
END:VCALENDAR
"""


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/guests.db")
    init_db(engine)
    with get_session(engine)() as s:
        yield s


def test_sync_bookings_from_ics_file(session, tmp_path):
    ics_path = tmp_path / "bookings.ics"
    ics_path.write_text(ICS)
    cfg = GuestConfig(ical=str(ics_path))
    assert sync_bookings(session, cfg) == 2
    bookings = session.query(Booking).order_by(Booking.start).all()
    assert bookings[0].summary == "Reserved - Unit A"
    assert bookings[1].start == datetime(2026, 8, 20, 15, 0)
    # Re-sync upserts, never duplicates.
    assert sync_bookings(session, cfg) == 2
    assert session.query(Booking).count() == 2


def test_guest_windows(session, tmp_path):
    ics_path = tmp_path / "bookings.ics"
    ics_path.write_text(ICS)
    cfg = GuestConfig(ical=str(ics_path), arrival_pre_hours=2, arrival_post_hours=4)
    sync_bookings(session, cfg)
    windows = GuestWindows(session, cfg)
    # booking-2 check-in 2026-08-20 15:00 UTC: window 13:00-19:00
    assert windows.contains(datetime(2026, 8, 20, 14, 0))
    assert windows.contains(datetime(2026, 8, 20, 18, 59))
    assert not windows.contains(datetime(2026, 8, 20, 20, 0))
    assert not windows.contains(datetime(2026, 8, 19, 14, 0))


def test_no_ical_configured(session):
    assert sync_bookings(session, GuestConfig(ical="")) == 0
    assert not GuestWindows(session, GuestConfig()).contains(datetime(2026, 8, 20))
