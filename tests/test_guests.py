from datetime import datetime
from zoneinfo import ZoneInfo

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
    # No zone given: all-day dates take the check-in/out hours as UTC.
    assert bookings[0].start == datetime(2026, 8, 10, 15, 0)
    assert bookings[0].end == datetime(2026, 8, 14, 11, 0)
    assert bookings[1].start == datetime(2026, 8, 20, 15, 0)
    # Re-sync upserts, never duplicates.
    assert sync_bookings(session, cfg) == 2
    assert session.query(Booking).count() == 2


def test_all_day_dates_take_the_site_checkin_hour(session, tmp_path):
    """A feed date has no instant; the site's check-in hour gives it one,
    in the site's zone, and the exclusive DTEND date carries check-out."""
    ics_path = tmp_path / "bookings.ics"
    ics_path.write_text(ICS)
    cfg = GuestConfig(ical=str(ics_path), checkin_time="15:00", checkout_time="11:00")
    sync_bookings(session, cfg, ZoneInfo("America/Costa_Rica"))  # UTC-6
    unit_a = session.query(Booking).filter_by(summary="Reserved - Unit A").one()
    assert unit_a.start == datetime(2026, 8, 10, 21, 0)  # 15:00 CR
    assert unit_a.end == datetime(2026, 8, 14, 17, 0)  # 11:00 CR on the DTEND date
    # A timed event is already an instant: the zone must not move it.
    unit_b = session.query(Booking).filter_by(summary="Reserved - Unit B").one()
    assert unit_b.start == datetime(2026, 8, 20, 15, 0)


def test_several_feeds_land_in_one_table(session, tmp_path):
    one = tmp_path / "casa-1.ics"
    two = tmp_path / "casa-2.ics"
    one.write_text(ICS)
    two.write_text(ICS.replace("booking-1", "casa2-1").replace("booking-2", "casa2-2"))
    cfg = GuestConfig(ical=[str(one), "", str(two)])
    assert cfg.sources == [str(one), str(two)]
    assert sync_bookings(session, cfg) == 4
    assert session.query(Booking).count() == 4


def test_a_feed_anchor_is_not_a_booking(session, tmp_path):
    """Kai's direct calendar publishes a 1970 placeholder when it has no
    bookings; a row for it would be a stay that never happened."""
    ics_path = tmp_path / "empty.ics"
    ics_path.write_text(
        "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//t//t//EN\nBEGIN:VEVENT\n"
        "UID:anchor-casa-1@example\nDTSTART;VALUE=DATE:19700101\n"
        "DTEND;VALUE=DATE:19700102\nSUMMARY:no direct bookings yet\n"
        "END:VEVENT\nEND:VCALENDAR\n"
    )
    assert sync_bookings(session, GuestConfig(ical=str(ics_path))) == 0
    assert session.query(Booking).count() == 0


def test_checkin_time_must_be_a_time_of_day():
    with pytest.raises(ValueError):
        GuestConfig(checkin_time="3pm")


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
