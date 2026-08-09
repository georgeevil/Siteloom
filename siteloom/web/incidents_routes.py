"""Incidents (CLD-96) — placeholder registration.

The triage rail's Escalate action has never been built, because nobody had
answered "escalate to whom". The answer, decided 2026-08-09: to a record,
not a person. For a property manager there is nobody on the other end
immediately, and somebody later — an insurer, the police, a tenant
dispute weeks on.

That makes an incident its own object rather than a fifth Event status.
One incident links many events across cameras, carries the operator's
prose, has an open/closed lifecycle, and exports as something a reader
outside the console can use. A per-event flag can express none of that,
which is the whole reason to build it.

This module is the seam that screen registers through — inert until then
so `main` gains no route the sidebar promises but cannot serve.
"""

from __future__ import annotations


def register(app, templates, Session, config) -> None:
    """No routes yet — see CLD-96."""
