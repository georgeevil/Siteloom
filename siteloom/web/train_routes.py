"""Model training and enrolment (CLD-92) — placeholder registration.

All five `siteloom train` commands — status, face, enroll,
export-detector, detector — have no console at all. `/training` is
labelling, not training, and the two get conflated precisely because only
one of them has a screen.

The load-bearing element of the screen that lands here is the held-out
evaluation: `EvalMetrics.valid` is False when a split cannot produce both
same- and different-person pairs, and an invalid score is not a zero
score. A UI that renders it as a number gets a model adopted on nothing.

Inert until that screen exists, so `main` gains no route the sidebar
promises but cannot serve.
"""

from __future__ import annotations


def register(app, templates, Session, config) -> None:
    """No routes yet — see CLD-92."""
