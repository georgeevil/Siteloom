"""Guest bookings (CLD-89) — placeholder registration.

`guests.py` already syncs iCal into Booking rows and `GuestWindows`
already stamps `Event.guest_window` at ingest, which is what suppresses
unknown-vehicle alarms during arrival windows — the PRD §12 success
metric. The console can read that stamp and can do nothing else with it:
there is no way to see the calendar it came from, add a booking the feed
does not carry, or correct one it got wrong.

This module is the seam that screen registers through. It is deliberately
inert until then, so `main` gains no half-built page and no sidebar entry
pointing at a route that is not there.
"""

from __future__ import annotations


def register(app, templates, Session, config) -> None:
    """No routes yet — see CLD-89."""
