"""Operator accounts and the audit trail (CLD-31) — placeholder.

Account management is CLI-only today, deliberately: there is no web path
to the first account, and a web-creatable account is a web-creatable
admin. That stays true. What is missing is everything *after* the first
account — seeing who exists, changing a role, disabling someone who left,
revoking a live session — none of which has a screen, and all of which an
admin needs without a shell.

The fourth role lands here too. `restricted` sits below `view`: it can
triage events and cannot reach identities, the face gallery, or training
data. That is biometric data minimisation (NFR5) with a real deployment
behind it — a night-shift operator who should judge whether an event
matters without browsing everyone's face.

The mechanism has to keep the single-middleware design: a second prefix
floor beside `ADMIN_PREFIXES`, never a decorator on a handler.

Inert until that screen exists, so `main` gains no route the sidebar
promises but cannot serve.
"""

from __future__ import annotations


def register(app, templates, Session, config) -> None:
    """No routes yet — see CLD-31."""
