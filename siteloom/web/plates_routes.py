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

The floors themselves are `IdentifierConfig` fields, not literals, so
answering "is 4 too high?" is: move it, re-read this table. Reads that
failed one kept their raw text, so nothing has to be re-run.

That extends to the image-quality floors, and it is what makes them
usable at all. OCR confidence turns out to be the weakest rejection
signal there is — a motion-smeared 60-pixel plate comes back at 0.86
because the model is genuinely confident about the characters it
interpolated — so what a read is judged on is measured off the image:
the plate region's width in pixels, its sharpness, and the *weakest*
character's probability rather than the mean that hides it. Every read
carries all three whether or not a floor is set, because a floor cannot
be chosen without first seeing the distribution it has to cut.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from urllib.parse import urlencode

from fastapi import Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import false, func, or_, select

from siteloom.identity.plates import normalize_plate
from siteloom.store import PLATE_VERDICTS, Identity, PlateRead, PlateWatch
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
    "too-small": "plate region too small to read",
    "too-blurry": "plate region too blurred to read",
    "too-short": "under the character floor",
    "low-confidence": "a character was a guess",
}

#: The configured floors, in the order they are applied, with the setting
#: that moves each. Rendered on the page because a floor an operator
#: cannot see is a floor they will debug as a bug — every rejected row
#: names the bar it failed, so the bar has to be named somewhere too.
FLOOR_FIELDS = (
    ("plate_min_width_px", "plate width", "px"),
    ("plate_min_sharpness", "sharpness", ""),
    ("plate_min_chars", "characters", ""),
    ("plate_min_char_confidence", "weakest character", ""),
)


def _floor_rows(config) -> list[dict]:
    """One line per plate identifier: which floors are in force.

    Only the ones actually set, plus the character floor, which is
    always in force. A list of four `0`s would read as configuration
    when it is the absence of it.
    """
    rows = []
    for key, ident in config.identity.identifiers.items():
        if not ident.plate_ocr:
            continue
        floors = [
            {"label": label, "value": f"{getattr(ident, field)}{unit}", "field": field}
            for field, label, unit in FLOOR_FIELDS
            if getattr(ident, field) or field == "plate_min_chars"
        ]
        rows.append({"key": key, "floors": floors})
    return rows


def _now() -> datetime:
    """Naive UTC — the tz-free convention every stored timestamp uses."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _plates_redirect(back: str) -> RedirectResponse:
    """Back to the /plates screen the operator acted from, filters intact
    — and nowhere else, since `back` arrives from the form."""
    return RedirectResponse(
        back if back.startswith("/plates") else "/plates", status_code=303
    )


def _empty_reason(config, total: int) -> str | None:
    """Why the table is empty — never just "no vehicles came past" (the
    CLD-26 rule). Three different installs land on an empty table and
    only one of them has anything to wait for."""
    if total:
        return None
    if not config.identity.enabled:
        return "identity-off"
    if not any(i.plate_ocr for i in config.identity.identifiers.values()):
        return "no-identifier"
    return "waiting"


def _chip_rows(filters: dict, classes: list[str]) -> tuple[list[dict], list[dict]]:
    """The class and status chip strips. Built here, not in Jinja:
    `class` is the query parameter's name and it cannot be spelled as a
    keyword argument in a template expression. Every chip keeps the other
    two filters, search included."""
    class_chips = [
        {
            "label": "All classes",
            "url": plates_url(filters, **{"class": None}),
            "active": not filters["class"],
        }
    ] + [
        {
            "label": name,
            "url": plates_url(filters, **{"class": name}),
            "active": filters["class"] == name,
        }
        for name in classes
    ]
    status_chips = [
        {
            "label": "Every read",
            "url": plates_url(filters, status=None),
            "active": not filters["status"],
        }
    ] + [
        {
            "label": name,
            "url": plates_url(filters, status=name),
            "active": filters["status"] == name,
        }
        for name in STATUSES
    ]
    return class_chips, status_chips


def _list_clauses(filters: dict) -> list:
    """The list page's WHERE, straight from its three filters."""
    clauses = []
    if filters["class"]:
        clauses.append(PlateRead.class_name == filters["class"])
    if filters["status"]:
        clauses.append(_status_clause(filters["status"]))
    if filters["q"]:
        clauses.append(search_clause(filters["q"]))
    return clauses


def _paged_reads(session, clauses: list, after: str | None):
    """One slice of reads, newest first — the shared shape of both
    screens' lists (cursor semantics per CLD-104, via app's helpers)."""
    from siteloom.web.app import _cursor_values

    sort = (PlateRead.at, PlateRead.id)
    query = (
        select(PlateRead)
        .where(*clauses)
        .order_by(PlateRead.at.desc(), PlateRead.id.desc())
    )
    values = _cursor_values(after, (datetime, int))
    if values is not None:
        query = query.where(paging.after(sort, values))
    fetched = session.scalars(query.limit(PLATES_PAGE + 1)).unique().all()
    return paging.take(list(fetched), PLATES_PAGE, lambda r: (r.at, r.id))


def _upsert_watch(session, plate: str, label: str, note: str) -> None:
    """Create the watch, or overwrite why it is watched — the form the
    operator just submitted is the current intent."""
    watch = session.scalar(select(PlateWatch).where(PlateWatch.plate == plate))
    if watch is None:
        session.add(
            PlateWatch(plate=plate, label=label, note=note, created_at=_now())
        )
    else:
        watch.label = label
        watch.note = note
    session.commit()


def _watch_rows(session) -> tuple[list[dict], set[str]]:
    """The watchlist with its sightings, and the plates in it.

    Sightings are computed from PlateRead at read time, never stored on
    the watch row — the watchlist must not be able to disagree with the
    reads table about when a plate was last seen.
    """
    watches = session.scalars(select(PlateWatch).order_by(PlateWatch.plate)).all()
    if not watches:
        return [], set()
    hits = {
        text: (count, last)
        for text, count, last in session.execute(
            select(PlateRead.text, func.count(), func.max(PlateRead.at))
            .where(
                PlateRead.accepted.is_(True),
                PlateRead.text.in_([w.plate for w in watches]),
            )
            .group_by(PlateRead.text)
        )
    }
    rows = [
        {
            "id": watch.id,
            "plate": watch.plate,
            "label": watch.label,
            "note": watch.note,
            "sightings": hits.get(watch.plate, (0, None))[0],
            "last_seen": hits.get(watch.plate, (0, None))[1],
        }
        for watch in watches
    ]
    return rows, {watch.plate for watch in watches}


def plates_url(base: dict, **overrides) -> str:
    """A link to this list with some filters changed, keeping the rest.

    No cursor, ever: this builds the links an operator clicks and copies,
    and every one has to open at the top of the set it names (CLD-104).
    """
    params: list[tuple[str, str]] = []
    merged = {**base, **overrides}
    for key in ("class", "status", "q"):
        if merged.get(key):
            params.append((key, str(merged[key])))
    return "/plates?" + urlencode(params) if params else "/plates"


def search_clause(q: str):
    """The WHERE for a plate search, matching how plates are matched.

    The operator's spacing and punctuation are stripped the same way the
    OCR's output was (`normalize_plate` is the one normalization in the
    system), so "ab-12" finds the row whose raw text was "AB 12". A query
    that normalizes to nothing matches nothing rather than everything —
    an explicit search must never silently become "no filter".

    Search matches the OCR's text OR an operator's correction: someone
    investigating a misread wants to find it by either spelling. (The
    per-plate page is stricter — it groups by the best-known truth,
    `coalesce(corrected_text, text)`.)
    """
    normalized = normalize_plate(q)
    if not normalized:
        return false()
    return or_(
        PlateRead.text.contains(normalized),
        PlateRead.corrected_text.contains(normalized),
    )


def register(app, templates, Session, config) -> None:
    from siteloom.web.app import _more, _with_cursor

    nav.add("/plates", "Plate reads", "PR", after="/stats")

    def _filters(class_name: str | None, status: str | None, q: str | None) -> dict:
        return {
            "class": (class_name or "").strip() or None,
            "status": status if status in STATUSES else None,
            "q": (q or "").strip() or None,
        }

    @app.get("/plates", response_class=HTMLResponse)
    def plates(
        request: Request,
        after: str | None = None,
        status: str | None = None,
        q: str | None = None,
    ):
        # `class` is a Python keyword, so it cannot be a parameter name;
        # read straight off the query string instead of renaming the
        # parameter an operator sees in the URL.
        class_name = request.query_params.get("class")
        filters = _filters(class_name, status, q)
        clauses = _list_clauses(filters)
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
            slice_ = _paged_reads(session, clauses, after)
            rows = slice_.rows
            watch_rows, watched_plates = _watch_rows(session)

        plate_identifiers = [
            key
            for key, ident in config.identity.identifiers.items()
            if ident.plate_ocr
        ]
        page_url = plates_url(filters)
        class_chips, status_chips = _chip_rows(filters, classes)
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
                "clear_search": plates_url(filters, q=None),
                "reason_labels": REASON_LABELS,
                "watches": watch_rows,
                "watched_plates": watched_plates,
                "matching": matching,
                "total": total,
                "accepted": accepted,
                "judged": judged,
                "wrong": wrong,
                "empty_reason": _empty_reason(config, total),
                "plate_identifiers": plate_identifiers,
                "floor_rows": _floor_rows(config),
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

    @app.get("/plates/p/{plate}", response_class=HTMLResponse)
    def plate_detail(request: Request, plate: str, after: str | None = None):
        """One plate's whole history: every read, and what it resolved to.

        The list page answers "how is the OCR doing"; this one answers
        "when was ABC123 here" — reads grouped by the normalized text
        plates are matched on, joined to the Identity row an accepted
        read wrote (write-once `Identity.plate`), with each read linking
        back to its Event.
        """
        canonical = normalize_plate(plate)
        if not canonical:
            raise HTTPException(404, "not a plate")
        if canonical != plate:
            # One URL per plate: a hand-typed "ab-12" lands on /AB12.
            return RedirectResponse(f"/plates/p/{canonical}", status_code=307)

        # A read belongs to the plate it is best known to say: the
        # operator's correction when there is one, else the OCR's text.
        # A misread corrected to this plate is evidence the vehicle was
        # here and appears; a read corrected *away* no longer does.
        belongs = func.coalesce(PlateRead.corrected_text, PlateRead.text) == canonical
        with Session() as session:
            total, accepted, first_at, last_at = session.execute(
                select(
                    func.count(),
                    func.count().filter(PlateRead.accepted.is_(True)),
                    func.min(PlateRead.at),
                    func.max(PlateRead.at),
                ).where(belongs)
            ).one()
            cameras = [
                row[0]
                for row in session.execute(
                    select(PlateRead.camera_id)
                    .where(belongs)
                    .where(PlateRead.camera_id.is_not(None))
                    .group_by(PlateRead.camera_id)
                    .order_by(PlateRead.camera_id)
                ).all()
            ]
            watch = session.scalar(
                select(PlateWatch).where(PlateWatch.plate == canonical)
            )
            # Usually one row ("vehicle"), but Identity.plate is per
            # identifier key, so a second plate identifier would show
            # both rather than hiding one.
            identities = session.scalars(
                select(Identity)
                .where(Identity.plate == canonical)
                .order_by(Identity.id)
            ).all()
            slice_ = _paged_reads(session, [belongs], after)
            # Detached rows are fine for the template (scalars only), but
            # the identity link needs display_name, computed while bound.
            identity_cards = [
                {
                    "id": identity.id,
                    "name": identity.display_name,
                    "identifier_key": identity.identifier_key,
                    "appearance_count": identity.appearance_count,
                    "best_crop_path": identity.best_crop_path,
                }
                for identity in identities
            ]

        page_url = f"/plates/p/{canonical}"
        return templates.TemplateResponse(
            request,
            "plate.html",
            {
                "site_name": config.site_name or config.site_id,
                "plate": canonical,
                "reads": slice_.rows,
                "total": total,
                "accepted": accepted,
                "first_at": first_at,
                "last_at": last_at,
                "cameras": cameras,
                "identities": identity_cards,
                "watch": watch,
                "back": page_url,
                "reason_labels": REASON_LABELS,
                "more": _more(
                    "#plate-rows",
                    None
                    if slice_.exhausted
                    else _with_cursor(page_url, slice_.next_cursor),
                    f"Load {PLATES_PAGE} more reads",
                    f"End of the list — {total} read"
                    f"{'' if total == 1 else 's'} of this plate.",
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
        return _plates_redirect(back)

    @app.post("/plates/{read_id}/correct")
    def plate_correct(
        read_id: int,
        text: str = Form(""),
        back: str = Form("/plates"),
    ):
        """Record what the plate actually says.

        Correcting is judging: the verdict follows from whether the
        correction agrees with what the OCR read, so a typed correction
        never coexists with a "confirmed" verdict on a misread. Like the
        verdict itself it changes nothing in the identity store —
        `Identity.plate` is write-once, and unwinding a plate match is a
        larger decision than judging a read. An empty submission clears
        the correction and leaves the verdict standing.
        """
        normalized = normalize_plate(text)
        if text.strip() and not normalized:
            # "!!!" is not a correction; refusing beats silently clearing.
            raise HTTPException(400, f"{text!r} normalizes to nothing")
        with Session() as session:
            read = session.get(PlateRead, read_id)
            if read is None:
                raise HTTPException(404)
            if not normalized:
                read.corrected_text = None
            else:
                read.corrected_text = normalized
                read.verdict = "confirmed" if normalized == read.text else "wrong"
                read.verdict_at = _now()
            session.commit()
        return _plates_redirect(back)

    @app.post("/plates/watchlist")
    def watch_add(
        plate: str = Form(""),
        label: str = Form(""),
        note: str = Form(""),
        back: str = Form("/plates"),
    ):
        """Watch a plate, or update why it is watched.

        The plate is stored normalized — the one form matching uses — and
        re-adding an already-watched plate overwrites its label and note
        rather than erroring: the form the operator just submitted is the
        current intent.
        """
        normalized = normalize_plate(plate)
        if not normalized:
            raise HTTPException(400, f"{plate!r} is not a plate")
        with Session() as session:
            _upsert_watch(session, normalized, label.strip(), note.strip())
        return _plates_redirect(back)

    @app.post("/plates/watchlist/{watch_id}/delete")
    def watch_delete(watch_id: int, back: str = Form("/plates")):
        """Stop watching. The sightings stay — they are PlateRead rows,
        and the watch row was only ever intent."""
        with Session() as session:
            watch = session.get(PlateWatch, watch_id)
            if watch is None:
                raise HTTPException(404)
            session.delete(watch)
            session.commit()
        return _plates_redirect(back)
