"""Guest-correlation hook (PRD §6.7).

Syncs bookings from the site's iCal feed (the Kai Apartments iCal sync)
into the Booking table, and answers "does this timestamp fall inside a
known guest arrival window?" — used to stamp events guest_window=True so
unknown-vehicle alerts during expected arrivals are suppressed (the PoC
success metric in PRD §12).

Bookings can also be entered by hand from the console's /bookings screen
when the feed is wrong or does not carry them. Those rows are marked
`source="manual"` and this sync never touches them — see the collision
guard in `sync_bookings` (CLD-90).
"""

from __future__ import annotations

import logging
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from siteloom.config import GuestConfig
from siteloom.localtime import as_utc
from siteloom.store.models import Booking

log = logging.getLogger(__name__)


def _read_ical(source: str) -> bytes:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as resp:
            return resp.read()
    return Path(source).read_bytes()


#: Booking feeds publish a placeholder event dated at the epoch when they
#: have nothing to say (Kai's direct calendar: "no direct bookings yet",
#: 1970-01-01). A row for it is not a booking, so anything that old is a
#: feed anchor and is skipped.
ANCHOR_BEFORE = datetime(2000, 1, 1)


def _as_naive_utc(value, wall: time, zone: ZoneInfo | None) -> datetime:
    """iCal dates come as date or datetime, naive or aware — normalize.

    This is the iCal input boundary of the naive-UTC contract (CLD-100):
    an aware value is converted to UTC and stripped; a naive ("floating")
    value is stored as it stands, which the store reads as UTC. An
    all-day date carries no instant at all, so the site's check-in or
    check-out hour (`wall`) is attached to it in the site's zone and the
    result converted — the arrival window is measured from 15:00 local,
    not from midnight UTC.
    """
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, date):
        return as_utc(datetime.combine(value, wall), zone)
    raise TypeError(f"unsupported iCal date value: {value!r}")


def sync_bookings(
    session: Session, cfg: GuestConfig, zone: ZoneInfo | None = None
) -> int:
    """Upsert bookings from every configured iCal source. Returns count.

    `zone` is the site zone (`localtime.site_zone`), used only to place
    all-day dates; None is the unset-zone rung and reads as UTC.
    """
    import icalendar

    sources = cfg.sources
    if not sources:
        log.info("no iCal source configured; skipping booking sync")
        return 0
    checkin = time.fromisoformat(cfg.checkin_time)
    checkout = time.fromisoformat(cfg.checkout_time)
    count = 0
    for source in sources:
        calendar = icalendar.Calendar.from_ical(_read_ical(source))
        count += _sync_calendar(session, calendar, zone, checkin, checkout)
    session.commit()
    return count


def _sync_calendar(session, calendar, zone, checkin: time, checkout: time) -> int:
    count = 0
    for component in calendar.walk("VEVENT"):
        uid = str(component.get("UID", ""))
        if not uid:
            continue
        start = _as_naive_utc(component["DTSTART"].dt, checkin, zone)
        # An all-day DTEND is *exclusive* (RFC 5545): a stay over the
        # nights of the 10th–13th ends with DTEND=14th, and check-out at
        # 11:00 falls on that very date — no day to subtract.
        end = (
            _as_naive_utc(component["DTEND"].dt, checkout, zone)
            if "DTEND" in component
            else start
        )
        if end < ANCHOR_BEFORE:
            log.debug("skipping feed anchor %s (%s)", uid, start.date())
            continue
        summary = str(component.get("SUMMARY", ""))
        booking = session.query(Booking).filter_by(uid=uid).first()
        if booking is not None and booking.is_manual:
            # A manual booking is an operator's correction to this very
            # feed (CLD-90). `uid` is unique, so a feed carrying the same
            # UID cannot be stored alongside it — and of the two, the row
            # that must survive is the human's. Skipped loudly rather than
            # silently: a correction the next sync reverts is worse than
            # no correction at all.
            log.warning(
                "iCal uid %s collides with a manual booking (id=%s); "
                "keeping the manual row",
                uid,
                booking.id,
            )
            continue
        if booking is None:
            booking = Booking(uid=uid, source="ical")
            session.add(booking)
        booking.summary = summary
        booking.start = start
        booking.end = end
        count += 1
    return count


class GuestWindows:
    """Preloaded arrival windows for fast per-event checks during ingest."""

    def __init__(self, session: Session, cfg: GuestConfig):
        pre = timedelta(hours=cfg.arrival_pre_hours)
        post = timedelta(hours=cfg.arrival_post_hours)
        self._windows = [
            (b.start - pre, b.start + post) for b in session.query(Booking).all()
        ]

    def contains(self, ts: datetime) -> bool:
        ts = ts.replace(tzinfo=None)
        return any(start <= ts <= end for start, end in self._windows)
