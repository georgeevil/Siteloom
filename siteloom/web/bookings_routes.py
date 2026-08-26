"""Guest bookings (CLD-90) — the screen behind `Event.guest_window`.

`guests.py` syncs iCal into Booking rows and `GuestWindows.contains()`
stamps `Event.guest_window` at ingest, which is what suppresses
unknown-vehicle alarms during arrival windows — the PRD §12 success
metric. Until this screen the console could read that stamp and do
nothing else with it: no way to see the calendar it consulted, no way to
add a booking the feed does not carry, no way to correct one it got
wrong. An operator who disagreed with a suppression had nowhere to go.

Three decisions shape the page:

* **A dated list, not a calendar.** The question an operator actually
  asks is "was there a booking around 21:40 last Tuesday", which a month
  grid answers worse than three ordered sections do.
* **The link back to `/` is the point.** Every row's event count is a
  filter-preserving link built with app.py's own `_triage_url`, so the
  number and the list it opens can never drift apart.
* **A manual booking must survive `sync-bookings`.** `Booking.uid` is
  unique and the sync keys on it, so provenance lives in `Booking.source`
  and manual rows get a synthetic uid the feed cannot mint. The sync
  skips them (see `siteloom/guests.py`); this module never writes an
  iCal-sourced row.

Editing an iCal row in place is deliberately absent: the feed owns those
rows and the next sync would revert the edit. The fix for a wrong feed is
a manual booking.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from fastapi import Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func, select

from siteloom.store import Booking, Event
from siteloom.store.models import significance_clause
from siteloom.web import nav

log = logging.getLogger(__name__)

#: How many finished bookings the page renders. Past stays are the long
#: tail; the current and upcoming sections are never truncated because
#: they are what the screen is for.
PAST_LIMIT = 50

#: Marks a uid this console minted rather than one that arrived in a
#: feed. Only cosmetic — provenance is `Booking.source`; a uid is not a
#: permission check — but it makes a hand-entered row obvious in the
#: database and keeps collisions with real iCal UIDs vanishingly rare.
MANUAL_UID_PREFIX = "siteloom-manual:"


class BookingError(ValueError):
    """A submission the page rejects, with a reason to show the operator."""


def _now() -> datetime:
    """Naive UTC — the tz-free convention every stored timestamp uses."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_window(start: str, end: str, zone=None) -> tuple[datetime, datetime]:
    """The two timestamps a manual booking needs, in naive UTC, or raise.

    `datetime-local` inputs post `2026-08-10T15:00`, which is naive — and
    under the site-time contract (CLD-100) naive form input is the
    **site's** wall clock, converted to the naive UTC that
    `Booking.start`/`end` hold right here at the boundary. `zone` is the
    site zone; None is the unset rung, where site time *is* UTC and the
    conversion is the identity — the pre-CLD-100 behaviour, under which
    operators were implicitly typing UTC.

    An explicit offset is still refused: the form advertises one frame
    (the site's), and honouring a different one per row would move the
    arrival window that suppresses alarms.
    """
    from siteloom.localtime import as_utc

    try:
        first = datetime.fromisoformat(start.strip())
        last = datetime.fromisoformat(end.strip())
    except (ValueError, AttributeError) as exc:
        raise BookingError("Enter both a check-in and a check-out time.") from exc
    if first.tzinfo is not None or last.tzinfo is not None:
        raise BookingError("Times are site-local; drop the timezone offset.")
    if last < first:
        raise BookingError("Check-out cannot be before check-in.")
    return as_utc(first, zone), as_utc(last, zone)


def feed_label(ical: str) -> str:
    """A printable name for the configured feed, without its secrets.

    Booking-site iCal URLs carry an unguessable token in the path or
    query — that URL *is* the credential. The page needs to say which
    feed it read, not hand it to everyone with `view`, so a URL is
    reduced to its host and a local file to its path.
    """
    if not ical:
        return ""
    if ical.startswith(("http://", "https://")):
        parts = urlsplit(ical)
        return f"{parts.scheme}://{parts.netloc}/…"
    return ical


def arrival_window(booking: Booking, guests) -> tuple[datetime, datetime]:
    """The span `GuestWindows.contains()` actually tests against.

    Not the stay: a booking suppresses alarms around *check-in*, so this
    mirrors `GuestWindows.__init__` rather than re-deriving it loosely —
    a page that draws a different window from the one ingest applied
    would explain the wrong decision.
    """
    return (
        booking.start - timedelta(hours=guests.arrival_pre_hours),
        booking.start + timedelta(hours=guests.arrival_post_hours),
    )


def _overlap_clause(start: datetime, end: datetime):
    """Events overlapping a window, exactly as `/` filters them.

    `since` compares against `last_seen` and `until` against
    `first_seen` in app.py's index — an event that begins before the
    window and ends inside it belongs to that window. Repeating the same
    comparison here is what keeps the count on the row equal to the
    number of rows the link opens.
    """
    return (Event.last_seen >= start, Event.first_seen <= end)


def register(app, templates, Session, config) -> None:
    # app.py owns the events list's query-param vocabulary; borrowing its
    # builder rather than re-spelling `since`/`until` here means a rename
    # there cannot leave this page linking at nothing. Imported inside
    # register() because create_app imports this module, not the reverse.
    from siteloom.localtime import site_zone
    from siteloom.web.app import _triage_url

    # A tab of Events, beside Noise: expected arrivals are context for
    # the event stream, on the same read floor (what a restricted viewer
    # may see of a booking is redaction's business, not the sidebar's).
    nav.add("/bookings", "Bookings", "GB", tab_of="/")

    def _rows(session, bookings: list[Booking]) -> list[dict]:
        out = []
        for booking in bookings:
            arrive_from, arrive_to = arrival_window(booking, config.guests)
            stay = _overlap_clause(booking.start, booking.end)
            events = session.scalar(
                select(func.count())
                .select_from(Event)
                .where(*stay, significance_clause())
            )
            # Events the stamp actually suppressed. This is the number the
            # PRD §12 metric is made of, and the reason a booking screen
            # is worth having at all: without it `guest_window` is a
            # per-event boolean nobody can total.
            stamped = session.scalar(
                select(func.count())
                .select_from(Event)
                .where(
                    *_overlap_clause(arrive_from, arrive_to),
                    Event.guest_window.is_(True),
                    significance_clause(),
                )
            )
            out.append(
                {
                    "booking": booking,
                    "arrival_start": arrive_from,
                    "arrival_end": arrive_to,
                    "events": events or 0,
                    "stamped": stamped or 0,
                    "stay_url": _triage_url(
                        {},
                        since=booking.start.isoformat(),
                        until=booking.end.isoformat(),
                    ),
                    "arrival_url": _triage_url(
                        {},
                        since=arrive_from.isoformat(),
                        until=arrive_to.isoformat(),
                    ),
                }
            )
        return out

    def _page(request: Request, error: str | None = None, status: int = 200):
        now = _now()
        with Session() as session:
            everything = (
                session.scalars(select(Booking).order_by(Booking.start)).unique().all()
            )
            current = [b for b in everything if b.start <= now <= b.end]
            upcoming = [b for b in everything if b.start > now]
            # Most recently finished first: the past is read backwards.
            past = [b for b in everything if b.end < now][::-1]
            truncated = max(len(past) - PAST_LIMIT, 0)
            sections = [
                {
                    "key": "current",
                    "title": "Current",
                    "empty": "No booking covers right now.",
                    "rows": _rows(session, current),
                },
                {
                    "key": "upcoming",
                    "title": "Upcoming",
                    "empty": "No bookings ahead.",
                    "rows": _rows(session, upcoming),
                },
                {
                    "key": "past",
                    "title": "Past",
                    "empty": "No bookings have finished yet.",
                    "rows": _rows(session, past[:PAST_LIMIT]),
                },
            ]
            manual = sum(1 for b in everything if b.is_manual)
        # Two different emptinesses (the CLD-26 rule): a window with no
        # bookings is a fact about the calendar, while "no iCal feed is
        # configured" is a fact about this install — and a table that
        # cannot fill must never read as a quiet week.
        if everything:
            reason = None
        elif not config.guests.ical:
            reason = "unconfigured"
        else:
            reason = "unsynced"
        return templates.TemplateResponse(
            request,
            "bookings.html",
            {
                "site_name": config.site_name or config.site_id,
                "sections": sections,
                "total": len(everything),
                "manual_count": manual,
                "ical_count": len(everything) - manual,
                "truncated": truncated,
                "past_limit": PAST_LIMIT,
                "empty_reason": reason,
                "feed": feed_label(config.guests.ical),
                "pre_hours": config.guests.arrival_pre_hours,
                "post_hours": config.guests.arrival_post_hours,
                "now": now,
                "error": error,
            },
            status_code=status,
        )

    @app.get("/bookings", response_class=HTMLResponse)
    def bookings_page(request: Request):
        return _page(request)

    @app.post("/bookings")
    def add_booking(
        request: Request,
        summary: str = Form(""),
        start: str = Form(""),
        end: str = Form(""),
    ):
        """Record a booking the feed does not carry.

        The synthetic uid is what makes this row survive the next
        `sync-bookings`: the sync matches on uid and skips anything whose
        `source` is not "ical", so an operator's correction is never
        overwritten or deleted by the calendar it was correcting.
        """
        label = summary.strip()
        try:
            if not label:
                raise BookingError("Give the booking a summary.")
            first, last = parse_window(start, end, site_zone(config))
        except BookingError as exc:
            return _page(request, error=str(exc), status=400)
        with Session() as session:
            session.add(
                Booking(
                    uid=f"{MANUAL_UID_PREFIX}{uuid.uuid4()}",
                    summary=label,
                    start=first,
                    end=last,
                    source="manual",
                )
            )
            session.commit()
        return RedirectResponse("/bookings", status_code=303)

    def _manual(session, booking_id: int) -> Booking:
        booking = session.get(Booking, booking_id)
        if booking is None:
            raise HTTPException(404)
        if not booking.is_manual:
            # Not a permission failure — a scope one. The feed owns its
            # rows and would revert the change on the next sync, so the
            # console refuses instead of pretending the edit will hold.
            raise HTTPException(
                409,
                "This booking comes from the iCal feed and the next sync "
                "would revert an edit. Add a manual booking instead.",
            )
        return booking

    @app.post("/bookings/{booking_id}/edit")
    def edit_booking(
        request: Request,
        booking_id: int,
        summary: str = Form(""),
        start: str = Form(""),
        end: str = Form(""),
    ):
        label = summary.strip()
        try:
            if not label:
                raise BookingError("Give the booking a summary.")
            first, last = parse_window(start, end, site_zone(config))
        except BookingError as exc:
            return _page(request, error=str(exc), status=400)
        with Session() as session:
            booking = _manual(session, booking_id)
            booking.summary = label
            booking.start = first
            booking.end = last
            session.commit()
        return RedirectResponse("/bookings", status_code=303)

    @app.post("/bookings/{booking_id}/delete")
    def delete_booking(booking_id: int):
        with Session() as session:
            session.delete(_manual(session, booking_id))
            session.commit()
        return RedirectResponse("/bookings", status_code=303)
