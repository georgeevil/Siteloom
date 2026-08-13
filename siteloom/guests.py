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

from sqlalchemy.orm import Session

from siteloom.config import GuestConfig
from siteloom.store.models import Booking

log = logging.getLogger(__name__)


def _read_ical(source: str) -> bytes:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as resp:
            return resp.read()
    return Path(source).read_bytes()


def _as_naive_utc(value) -> datetime:
    """iCal dates come as date or datetime, naive or aware — normalize.

    This is the iCal input boundary of the naive-UTC contract (CLD-100):
    an aware value is converted to UTC and stripped; a naive ("floating")
    value and an all-day date are stored as they stand, which the store
    reads as UTC. Booking feeds ship aware stamps in practice, so the
    floating case is the degenerate one, kept as-is on purpose.
    """
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    raise TypeError(f"unsupported iCal date value: {value!r}")


def sync_bookings(session: Session, cfg: GuestConfig) -> int:
    """Upsert bookings from the configured iCal source. Returns count."""
    import icalendar

    if not cfg.ical:
        log.info("no iCal source configured; skipping booking sync")
        return 0
    calendar = icalendar.Calendar.from_ical(_read_ical(cfg.ical))
    count = 0
    for component in calendar.walk("VEVENT"):
        uid = str(component.get("UID", ""))
        if not uid:
            continue
        start = _as_naive_utc(component["DTSTART"].dt)
        end = _as_naive_utc(component["DTEND"].dt) if "DTEND" in component else start
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
    session.commit()
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
