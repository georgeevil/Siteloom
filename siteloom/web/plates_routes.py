"""Plate reads (CLD-85) — the screen behind `Identity.plate`.

Until this table existed, plate OCR was judged by eye, once, and the
number could neither be reproduced nor compared against a later run:
`PlateReader.read()` returned `str | None` and everything else — the
detector's box confidence, any OCR confidence, the raw text before
normalization, the fact that a read had been attempted at all — was
discarded at the point of reading. The reads most likely to be wrong,
short/angled motorcycle plates falling under the four-character floor,
were exactly the ones that left no trace.

This page is the other half of that fix: the rows are useless if nobody
can look at them.

Three decisions shape it:

* **Failures are first-class rows, and the page leads with them.** A
  read that produced nothing carries its `reason` and its raw text. The
  class filter is here so "isolate the motorcycles" — CLD-9's actual
  question — is one click rather than a SQL prompt.
* **Every read shows the crop the OCR saw.** Not the vehicle thumbnail:
  the plate sub-region, saved as its own third image so that
  `Detection.crop_path` (display thumbnail *and* embedder input) is left
  exactly as it was.
* **A verdict persists.** Confirm/reject writes `PlateRead.verdict`, so
  the spike becomes "judge 20 rows" and the judgement survives it — the
  same reason `EventIdentity.verdict` exists, with the same vocabulary.

The floor itself is `IdentifierConfig.plate_min_chars`, not a literal, so
answering "is 4 too high?" is: move it, re-read this table. Reads that
failed it kept their raw text, so nothing has to be re-run.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from urllib.parse import urlencode

from fastapi import Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func, select

from siteloom.store import PLATE_VERDICTS, PlateRead
from siteloom.web import nav, paging

log = logging.getLogger(__name__)

#: Rows per slice. A read row is a thumbnail plus five short columns —
#: bigger than a triage row, smaller than an identity card.
PLATES_PAGE = 40

#: The status chips, as SQL. Kept here rather than in the template
#: because "rejected" is a filter over two columns and a page that
#: filtered in Python after the query would page wrongly (paging.py).
def _status_clause(status: str):
    if status == "accepted":
        return PlateRead.accepted.is_(True)
    if status == "rejected":
        return PlateRead.accepted.is_(False)
    if status == "unjudged":
        return PlateRead.verdict.is_(None)
    if status in PLATE_VERDICTS:
        return PlateRead.verdict == status
    raise ValueError(f"unknown plate-read status: {status!r}")


STATUSES = ("accepted", "rejected", "unjudged") + PLATE_VERDICTS

#: What each rejection reason means in words. The reason codes are
#: `identity/plates.py`'s; spelling them out here is what makes a row
#: readable by someone who has never opened that file.
REASON_LABELS = {
    "no-box": "no plate region found",
    "empty-crop": "plate box fell outside the crop",
    "no-text": "OCR read nothing",
    "too-short": "under the character floor",
}


def _now() -> datetime:
    """Naive UTC — the tz-free convention every stored timestamp uses."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def plates_url(base: dict, **overrides) -> str:
    """A link to this list with some filters changed, keeping the rest.

    No cursor, ever: this builds the links an operator clicks and copies,
    and every one has to open at the top of the set it names (CLD-104).
    """
    params: list[tuple[str, str]] = []
    merged = {**base, **overrides}
    for key in ("class", "status"):
        if merged.get(key):
            params.append((key, str(merged[key])))
    return "/plates?" + urlencode(params) if params else "/plates"


def register(app, templates, Session, config) -> None:
    from siteloom.web.app import _cursor_values, _more, _with_cursor

    nav.add("/plates", "Plate reads", "PR", after="/stats")

    def _filters(class_name: str | None, status: str | None) -> dict:
        return {
            "class": (class_name or "").strip() or None,
            "status": status if status in STATUSES else None,
        }

    @app.get("/plates", response_class=HTMLResponse)
    def plates(
        request: Request,
        after: str | None = None,
        status: str | None = None,
    ):
        # `class` is a Python keyword, so it cannot be a parameter name;
        # read straight off the query string instead of renaming the
        # parameter an operator sees in the URL.
        class_name = request.query_params.get("class")
        filters = _filters(class_name, status)
        clauses = []
        if filters["class"]:
            clauses.append(PlateRead.class_name == filters["class"])
        if filters["status"]:
            clauses.append(_status_clause(filters["status"]))

        sort = (PlateRead.at, PlateRead.id)
        with Session() as session:
            matching = (
                session.scalar(
                    select(func.count()).select_from(PlateRead).where(*clauses)
                )
                or 0
            )
            total = session.scalar(select(func.count()).select_from(PlateRead)) or 0
            accepted = (
                session.scalar(
                    select(func.count())
                    .select_from(PlateRead)
                    .where(PlateRead.accepted.is_(True))
                )
                or 0
            )
            judged = (
                session.scalar(
                    select(func.count())
                    .select_from(PlateRead)
                    .where(PlateRead.verdict.is_not(None))
                )
                or 0
            )
            wrong = (
                session.scalar(
                    select(func.count())
                    .select_from(PlateRead)
                    .where(PlateRead.verdict == "wrong")
                )
                or 0
            )
            # Class facets come from the rows, not from config: a class
            # can be added to detection.classes at any time (NFR3), and a
            # hard-coded list would hide the reads it produced.
            classes = [
                row[0]
                for row in session.execute(
                    select(PlateRead.class_name)
                    .where(PlateRead.class_name != "")
                    .group_by(PlateRead.class_name)
                    .order_by(PlateRead.class_name)
                ).all()
            ]
            query = (
                select(PlateRead)
                .where(*clauses)
                .order_by(PlateRead.at.desc(), PlateRead.id.desc())
            )
            values = _cursor_values(after, (datetime, int))
            if values is not None:
                query = query.where(paging.after(sort, values))
            fetched = session.scalars(query.limit(PLATES_PAGE + 1)).unique().all()
            slice_ = paging.take(list(fetched), PLATES_PAGE, lambda r: (r.at, r.id))
            rows = slice_.rows

        # Why the table is empty is never just "no vehicles came past"
        # (the CLD-26 rule). Three different installs land here and only
        # one of them has anything to wait for.
        plate_identifiers = [
            key
            for key, ident in config.identity.identifiers.items()
            if ident.plate_ocr
        ]
        if total:
            reason = None
        elif not config.identity.enabled:
            reason = "identity-off"
        elif not plate_identifiers:
            reason = "no-identifier"
        else:
            reason = "waiting"

        page_url = plates_url(filters)
        # Chips are built here, not in Jinja: `class` is the query
        # parameter's name and it cannot be spelled as a keyword argument
        # in a template expression.
        class_chips = [
            {
                "label": "All classes",
                "url": plates_url({"status": filters["status"]}),
                "active": not filters["class"],
            }
        ] + [
            {
                "label": name,
                "url": plates_url({"status": filters["status"]}, **{"class": name}),
                "active": filters["class"] == name,
            }
            for name in classes
        ]
        status_chips = [
            {
                "label": "Every read",
                "url": plates_url({"class": filters["class"]}),
                "active": not filters["status"],
            }
        ] + [
            {
                "label": name,
                "url": plates_url({"class": filters["class"]}, status=name),
                "active": filters["status"] == name,
            }
            for name in STATUSES
        ]
        return templates.TemplateResponse(
            request,
            "plates.html",
            {
                "site_name": config.site_name or config.site_id,
                "reads": rows,
                "filters": filters,
                "classes": classes,
                "class_chips": class_chips,
                "status_chips": status_chips,
                "back": page_url,
                "reason_labels": REASON_LABELS,
                "matching": matching,
                "total": total,
                "accepted": accepted,
                "judged": judged,
                "wrong": wrong,
                "empty_reason": reason,
                "plate_identifiers": plate_identifiers,
                "min_chars": {
                    key: config.identity.identifiers[key].plate_min_chars
                    for key in plate_identifiers
                },
                "more": _more(
                    "#plate-rows",
                    None
                    if slice_.exhausted
                    else _with_cursor(page_url, slice_.next_cursor),
                    f"Load {PLATES_PAGE} more reads",
                    f"End of the list — {matching} read"
                    f"{'' if matching == 1 else 's'} match this filter.",
                ),
            },
        )

    @app.post("/plates/{read_id}/verdict")
    def plate_verdict(
        read_id: int,
        verdict: str = Form(""),
        back: str = Form("/plates"),
    ):
        """Judge one read: confirmed, wrong, or back to unjudged.

        A wrong verdict is recorded, never deleted — the negatives-are-
        data philosophy again — and it deliberately changes nothing in the
        identity store. `Identity.plate` is write-once and a plate match
        beats visual similarity; unwinding that from here would be a
        second, larger decision made by accident.
        """
        choice = verdict.strip() or None
        if choice is not None and choice not in PLATE_VERDICTS:
            raise HTTPException(400, f"unknown verdict {verdict!r}")
        with Session() as session:
            read = session.get(PlateRead, read_id)
            if read is None:
                raise HTTPException(404)
            read.verdict = choice
            read.verdict_at = _now() if choice else None
            session.commit()
        # Back to the list the operator was judging from, filters intact.
        return RedirectResponse(
            back if back.startswith("/plates") else "/plates", status_code=303
        )
