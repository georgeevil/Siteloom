"""Backfill (CLD-91) — placeholder registration.

`siteloom backfill` and `backfill-unifi` are the only way to fill a gap
after a crash, and downloaded NVR clips are the one camera-derived path
that produces NoiseEvent rows at all. Both are terminal-only today: the
console shows the resulting events without offering any way to ask for
them.

This module is the seam that screen registers through — inert until then
so `main` gains no route the sidebar promises but cannot serve.
"""

from __future__ import annotations


def register(app, templates, Session, config) -> None:
    """No routes yet — see CLD-91."""
