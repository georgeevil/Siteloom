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
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlencode

from fastapi import Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import case, false, func, or_, select, tuple_

from siteloom import localtime
from siteloom.identity.plates import normalize_plate
from siteloom.store import (
    PLATE_VERDICTS,
    EventIdentity,
    Identity,
    PlateRead,
    PlateWatch,
)
from siteloom.web import identity_ops, nav, paging

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

#: The default status view is accepted reads (CLD-131). The rejections
#: keep being recorded — the CLD-119 floors were derived from them and
#: CLD-114's consensus needs them — but an operator opening this screen
#: is asking "what plates came past?", and a default of newest-first
#: everything answered with `no-box` diagnostics and the reads the
#: system itself refused. "all" is the explicit ask for every read.
DEFAULT_STATUS = "accepted"

#: The two shapes of the list (CLD-130): "grouped" collapses a vehicle's
#: visit to one row per (event, best-known text) — the unit an operator
#: actually judges — and "reads" is the per-frame log underneath it.
VIEWS = ("grouped", "reads")
DEFAULT_VIEW = "grouped"

#: What a group's key text is: the operator's correction wins over the
#: OCR, and a read with no text at all groups under "" — never NULL,
#: because NULL breaks both GROUP BY identity and the keyset cursor.
BEST_TEXT = func.coalesce(PlateRead.corrected_text, PlateRead.text, "")

#: Bulk actions over a selection (CLD-131) — the same vocabulary as the
#: per-row verdict, plus the clear that undoes either.
BULK_ACTIONS = {"confirmed": "confirmed", "wrong": "wrong", "clear": None}

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


#: How a per-camera override field renders on the page, keyed as the
#: `PlateFloors` tuple spells them (the config-side names, CLD-128).
_OVERRIDE_LABELS = {
    "min_width_px": ("plate width", "px"),
    "min_sharpness": ("sharpness", ""),
    "min_chars": ("characters", ""),
    "min_char_confidence": ("weakest character", ""),
}


def _floor_rows(config) -> list[dict]:
    """One line per plate identifier: which floors are in force.

    Only the ones actually set, plus the character floor, which is
    always in force. A list of four `0`s would read as configuration
    when it is the absence of it. Cameras overriding a floor (CLD-128)
    are named too — a floor an operator cannot see is a floor they will
    debug as a bug, and that goes double for one that only bites on one
    camera.
    """
    camera_overrides = []
    for cam in config.cameras:
        override = cam.identity.plate_floors if cam.identity else None
        if override is None:
            continue
        parts = [
            f"{_OVERRIDE_LABELS[field][0]} ≥ {value}{_OVERRIDE_LABELS[field][1]}"
            for field, value in override.model_dump().items()
            if value is not None
        ]
        if parts:
            camera_overrides.append(f"{cam.id}: {', '.join(parts)}")
    rows = []
    for key, ident in config.identity.identifiers.items():
        if not ident.plate_ocr:
            continue
        floors = [
            {"label": label, "value": f"{getattr(ident, field)}{unit}", "field": field}
            for field, label, unit in FLOOR_FIELDS
            if getattr(ident, field) or field == "plate_min_chars"
        ]
        rows.append({"key": key, "floors": floors, "camera_overrides": camera_overrides})
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
    filters, search and timeframe included."""
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
    # Accepted leads because it is the default view; "Every read" is the
    # explicit opt into the rejections, never the resting state.
    status_chips = [
        {
            "label": name,
            "url": plates_url(filters, status=name),
            "active": filters["status"] == name,
        }
        for name in STATUSES
    ] + [
        {
            "label": "Every read",
            "url": plates_url(filters, status="all"),
            "active": filters["status"] == "all",
        }
    ]
    return class_chips, status_chips


def _time_clauses(filters: dict, config) -> list:
    """The timeframe's WHERE over `PlateRead.at` — the picker the events
    screen grew in CLD-121, adopted here (CLD-115/CLD-131).

    A preset (`last=24h`) is a living window judged at request time and
    wins over the absolute bounds — the two must never silently AND.
    Absolute bounds are operator-typed wall clock, an input boundary
    (CLD-100): converted site-local -> naive UTC here, never compared
    raw against the UTC column.
    """
    from siteloom.web.app import TIMEFRAME_PRESETS

    clauses = []
    if filters["last"]:
        clauses.append(
            PlateRead.at
            >= datetime.now(timezone.utc).replace(tzinfo=None)
            - TIMEFRAME_PRESETS[filters["last"]]
        )
        return clauses
    zone = localtime.site_zone(config)
    for key, column_op in (("since", "ge"), ("until", "le")):
        raw = filters[key]
        if not raw:
            continue
        try:
            bound = localtime.as_utc(datetime.fromisoformat(raw), zone)
        except ValueError:
            # A hand-mangled bound is a bad request, not a 500 and not a
            # silent "no filter" — an explicit ask must never widen.
            raise HTTPException(400, f"unreadable timeframe bound {raw!r}") from None
        clauses.append(PlateRead.at >= bound if column_op == "ge" else PlateRead.at <= bound)
    return clauses


def _list_clauses(filters: dict, config, *, status: bool = True) -> list:
    """The list page's WHERE, straight from its filters.

    `status=False` leaves the status filter out — that variant counts
    what the current status view hides, so an empty or filtered default
    view can say how many rejections sit one chip away (the /noise rule).
    """
    clauses = []
    if filters["class"]:
        clauses.append(PlateRead.class_name == filters["class"])
    if status and filters["status"] != "all":
        clauses.append(_status_clause(filters["status"]))
    if filters["q"]:
        clauses.append(search_clause(filters["q"]))
    if filters["event"]:
        clauses.append(PlateRead.event_id == filters["event"])
    clauses.extend(_time_clauses(filters, config))
    return clauses


#: The grouped view's sort key labels — spelled once because the cursor
#: must carry every column the ORDER BY does (CLD-104), and the group
#: key text rides along to break last_at/event ties between two texts
#: in one event.
_GROUP_SORT_TYPES = (datetime, int, str)


def _grouped_page(session, clauses: list, after: str | None):
    """One slice of (event, best text) groups, most recent visit first.

    Grouping is display only (CLD-130): the underlying rows stay — they
    are the measurement the floors were tuned from and the input the
    CLD-114 consensus gate needs — this query only presents a vehicle's
    visit as the single row an operator judges.
    """
    from siteloom.web.app import _cursor_values

    sub = (
        select(
            PlateRead.event_id.label("event_id"),
            BEST_TEXT.label("text"),
            func.count().label("reads"),
            func.min(PlateRead.at).label("first_at"),
            func.max(PlateRead.at).label("last_at"),
            func.sum(case((PlateRead.accepted.is_(True), 1), else_=0)).label(
                "accepted_n"
            ),
            func.sum(case((PlateRead.verdict == "confirmed", 1), else_=0)).label(
                "confirmed_n"
            ),
            func.sum(case((PlateRead.verdict == "wrong", 1), else_=0)).label(
                "wrong_n"
            ),
            func.max(PlateRead.camera_id).label("camera_id"),
            func.max(PlateRead.class_name).label("class_name"),
        )
        .where(*clauses)
        .group_by(PlateRead.event_id, BEST_TEXT)
        .subquery()
    )
    sort = (sub.c.last_at, sub.c.event_id, sub.c.text)
    query = select(sub).order_by(
        sub.c.last_at.desc(), sub.c.event_id.desc(), sub.c.text.desc()
    )
    values = _cursor_values(after, _GROUP_SORT_TYPES)
    if values is not None:
        query = query.where(paging.after(sort, values))
    fetched = session.execute(query.limit(PLATES_PAGE + 1)).all()
    return paging.take(
        list(fetched), PLATES_PAGE, lambda g: (g.last_at, g.event_id, g.text)
    )


def _group_crops(session, clauses: list, groups: list) -> dict:
    """The best crop per group on this page, keyed (event_id, text).

    "Best" prefers an accepted read, then the strongest weakest-character
    confidence, then sharpness — the same ordering a human picks the
    evidence image by. One query for the whole page, not one per group:
    a group can hold a thousand rows (the CLD-130 measurement), and this
    never fetches them, only the row a window function ranked first.
    """
    keys = [(g.event_id, g.text) for g in groups]
    if not keys:
        return {}
    rank = (
        func.row_number()
        .over(
            partition_by=(PlateRead.event_id, BEST_TEXT),
            order_by=(
                PlateRead.accepted.desc(),
                func.coalesce(PlateRead.ocr_min_confidence, -1.0).desc(),
                func.coalesce(PlateRead.sharpness, -1.0).desc(),
                PlateRead.id.desc(),
            ),
        )
        .label("rank")
    )
    inner = (
        select(
            PlateRead.event_id.label("event_id"),
            BEST_TEXT.label("text"),
            PlateRead.crop_path.label("crop_path"),
            rank,
        )
        .where(
            PlateRead.crop_path.is_not(None),
            tuple_(PlateRead.event_id, BEST_TEXT).in_(keys),
            *clauses,
        )
        .subquery()
    )
    rows = session.execute(select(inner).where(inner.c.rank == 1)).all()
    return {(r.event_id, r.text): r.crop_path for r in rows}


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


def _apply_targets(session, reads) -> dict[int, Identity]:
    """The identity each read could be applied to, where that is unambiguous.

    A read is appliable when an operator stands behind its value —
    `corrected_text`, or a `confirmed` verdict — and its event carries
    **exactly one** active claim of the read's own identifier. Keying on
    `read.identifier_key` rather than a config lookup of "which
    identifiers do plates" keeps the rule exact and config-free.

    Two claims is not a corner case: an event with a vehicle claimed on
    two identities is precisely where guessing would write the plate onto
    the wrong one, so the button is withheld and the identity page's edit
    field stays the unambiguous path. Zero is the event that was never
    linked at all.

    Two queries for the whole page — the claims, then the identities they
    name — never one per row: these lists run to hundreds of reads of one
    parked car, and a default page spans as many events as it has rows.
    """
    # Keyed by the pair, not by the event: one event can carry reads of
    # two identifiers, and keying on the event alone would let the later
    # one shadow the earlier and withhold its button.
    wanted = {
        (read.event_id, read.identifier_key)
        for read in reads
        if read.identifier_key and (read.corrected_text or read.verdict == "confirmed")
    }
    if not wanted:
        return {}
    claims: dict[tuple[int, str], list[EventIdentity]] = defaultdict(list)
    rows = session.scalars(
        select(EventIdentity).where(
            EventIdentity.event_id.in_({event_id for event_id, _ in wanted}),
            EventIdentity.unlinked_at.is_(None),
            EventIdentity.identity_id.is_not(None),
            EventIdentity.identifier_key.in_({key for _, key in wanted}),
        )
    ).all()
    for row in rows:
        claims[(row.event_id, row.identifier_key)].append(row)
    sole = {
        key: group[0].identity_id
        for key, group in claims.items()
        if key in wanted and len(group) == 1
    }
    if not sole:
        return {}
    identities = {
        identity.id: identity
        for identity in session.scalars(
            select(Identity).where(Identity.id.in_(set(sole.values())))
        ).all()
    }
    targets = {}
    for read in reads:
        identity_id = sole.get((read.event_id, read.identifier_key))
        if identity_id is None:
            continue
        target = identities.get(identity_id)
        if target is not None:
            targets[read.id] = target
    return targets


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


def _filters(
    class_name: str | None,
    status: str | None,
    q: str | None,
    view: str | None = None,
    last: str | None = None,
    since: str | None = None,
    until: str | None = None,
    event: int | None = None,
) -> dict:
    from siteloom.web.app import TIMEFRAME_PRESETS

    # Unknown tokens fall to the defaults rather than 400 — the chips
    # are the only things that mint them.
    return {
        "class": (class_name or "").strip() or None,
        "status": status if status in STATUSES + ("all",) else DEFAULT_STATUS,
        "q": (q or "").strip() or None,
        "view": view if view in VIEWS else DEFAULT_VIEW,
        "last": last if last in TIMEFRAME_PRESETS else None,
        "since": (since or "").strip() or None,
        "until": (until or "").strip() or None,
        "event": event,
    }


def _page_counts(session, filters: dict, clauses: list, config) -> dict:
    """The list page's numbers, one place.

    `hidden_by_status` is what the status chip is hiding within the same
    other filters — the /noise rule: a table defaulting to accepted reads
    must say how many rejections sit one chip away, or an empty default
    view reads as "nothing came past".
    """
    counts = {
        "matching": session.scalar(
            select(func.count()).select_from(PlateRead).where(*clauses)
        )
        or 0,
        "total": session.scalar(select(func.count()).select_from(PlateRead)) or 0,
        "accepted": session.scalar(
            select(func.count())
            .select_from(PlateRead)
            .where(PlateRead.accepted.is_(True))
        )
        or 0,
        "judged": session.scalar(
            select(func.count())
            .select_from(PlateRead)
            .where(PlateRead.verdict.is_not(None))
        )
        or 0,
        "wrong": session.scalar(
            select(func.count())
            .select_from(PlateRead)
            .where(PlateRead.verdict == "wrong")
        )
        or 0,
        "hidden_by_status": 0,
    }
    if filters["status"] != "all":
        in_timeframe = (
            session.scalar(
                select(func.count())
                .select_from(PlateRead)
                .where(*_list_clauses(filters, config, status=False))
            )
            or 0
        )
        counts["hidden_by_status"] = max(0, in_timeframe - counts["matching"])
    return counts


def _parse_groups(tokens: list[str]) -> list[tuple[int, str]]:
    """`<event_id>:<text>` pairs off the bulk form, or a 400.

    The text half is normalized-plate alphabet plus the empty no-text
    group, so the first colon is unambiguous.
    """
    groups: list[tuple[int, str]] = []
    for token in tokens:
        event_part, _, text_part = token.partition(":")
        try:
            groups.append((int(event_part), text_part))
        except ValueError:
            raise HTTPException(400, f"unreadable group {token!r}") from None
    return groups


def _bulk_targets(session, read_ids: list[int], groups: list[tuple[int, str]]):
    """Every read a bulk action names: listed ids, plus each group's rows
    by the same best-known-text key the grouped view groups on — so the
    buttons judge exactly the rows the row summarised."""
    targets = []
    if read_ids:
        targets.extend(
            session.scalars(select(PlateRead).where(PlateRead.id.in_(read_ids))).all()
        )
    for event_id, text in groups:
        targets.extend(
            session.scalars(
                select(PlateRead).where(
                    PlateRead.event_id == event_id, BEST_TEXT == text
                )
            ).all()
        )
    return targets


def _grouped_rows(session, clauses: list, filters: dict, after: str | None):
    """The grouped view's page: the slice plus template-ready dicts,
    each carrying its best crop and its expand-to-reads link."""
    slice_ = _grouped_page(session, clauses, after)
    groups = [dict(g._mapping) for g in slice_.rows]
    crops = _group_crops(session, clauses, slice_.rows)
    for g in groups:
        g["crop_path"] = crops.get((g["event_id"], g["text"]))
        g["unjudged_n"] = g["reads"] - g["confirmed_n"] - g["wrong_n"]
        # The per-frame log, narrowed to the visit and the group's text,
        # every status shown — the count already includes rejections.
        g["reads_url"] = plates_url(
            filters,
            view="reads",
            event=g["event_id"],
            q=g["text"] or None,
            status="all",
        )
    return slice_, groups


def plates_url(base: dict, **overrides) -> str:
    """A link to this list with some filters changed, keeping the rest.

    No cursor, ever: this builds the links an operator clicks and copies,
    and every one has to open at the top of the set it names (CLD-104).
    `last` *is* part of it — "the last 24h" is a filter, and a pasted
    link should mean the same living window for the recipient. The two
    defaults (grouped, accepted) are omitted so the bare `/plates` stays
    the canonical spelling of the default view.
    """
    params: list[tuple[str, str]] = []
    merged = {**base, **overrides}
    if merged.get("view") and merged["view"] != DEFAULT_VIEW:
        params.append(("view", str(merged["view"])))
    if merged.get("status") and merged["status"] != DEFAULT_STATUS:
        params.append(("status", str(merged["status"])))
    for key in ("class", "q", "last", "since", "until", "event"):
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


def register(app, templates, Session, config) -> None:  # noqa: C901 — route table
    from siteloom.web.app import _more, _with_cursor

    nav.add("/plates", "Plate reads", "PR", after="/stats")

    @app.get("/plates", response_class=HTMLResponse)
    def plates(
        request: Request,
        after: str | None = None,
        status: str | None = None,
        q: str | None = None,
        view: str | None = None,
        last: str | None = None,
        since: str | None = None,
        until: str | None = None,
        event: int | None = None,
    ):
        from siteloom.web.app import TIMEFRAME_PRESETS

        # `class` is a Python keyword, so it cannot be a parameter name;
        # read straight off the query string instead of renaming the
        # parameter an operator sees in the URL.
        class_name = request.query_params.get("class")
        filters = _filters(class_name, status, q, view, last, since, until, event)
        clauses = _list_clauses(filters, config)
        with Session() as session:
            counts = _page_counts(session, filters, clauses, config)
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
            grouped = filters["view"] == "grouped"
            if grouped:
                slice_, groups = _grouped_rows(session, clauses, filters, after)
                rows = []
            else:
                slice_ = _paged_reads(session, clauses, after)
                rows = slice_.rows
                groups = []
            # Inside the session, and one query for the page: the rows
            # detach when this block exits (CLD-134).
            apply_targets = _apply_targets(session, rows)
            watch_rows, watched_plates = _watch_rows(session)

        plate_identifiers = [
            key
            for key, ident in config.identity.identifiers.items()
            if ident.plate_ocr
        ]
        page_url = plates_url(filters)
        class_chips, status_chips = _chip_rows(filters, classes)
        # The timeframe picker's links: each preset keeps every other
        # filter, replaces the absolute window (the two would silently
        # AND together).
        timeframe_urls = {
            key: plates_url(filters, last=key, since=None, until=None)
            for key in TIMEFRAME_PRESETS
        }
        timeframe_urls["all"] = plates_url(filters, last=None, since=None, until=None)
        view_urls = {key: plates_url(filters, view=key) for key in VIEWS}
        unit = "visit" if grouped else "read"
        return templates.TemplateResponse(
            request,
            "plates.html",
            {
                "site_name": config.site_name or config.site_id,
                "apply_targets": apply_targets,
                "reads": rows,
                "groups": groups,
                "filters": filters,
                "classes": classes,
                "class_chips": class_chips,
                "status_chips": status_chips,
                "timeframe_urls": timeframe_urls,
                "timeframe_presets": list(TIMEFRAME_PRESETS),
                "view_urls": view_urls,
                "back": page_url,
                "clear_search": plates_url(filters, q=None),
                "clear_event": plates_url(filters, event=None),
                "show_all_url": plates_url(filters, status="all"),
                "hidden_by_status": counts["hidden_by_status"],
                "reason_labels": REASON_LABELS,
                "watches": watch_rows,
                "watched_plates": watched_plates,
                "matching": counts["matching"],
                "total": counts["total"],
                "accepted": counts["accepted"],
                "judged": counts["judged"],
                "wrong": counts["wrong"],
                "empty_reason": _empty_reason(config, counts["total"]),
                "plate_identifiers": plate_identifiers,
                "floor_rows": _floor_rows(config),
                "more": _more(
                    "#plate-rows",
                    None
                    if slice_.exhausted
                    else _with_cursor(page_url, slice_.next_cursor),
                    f"Load {PLATES_PAGE} more {unit}s",
                    f"End of the list — {counts['matching']} read"
                    f"{'' if counts['matching'] == 1 else 's'} match this filter.",
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
            apply_targets = _apply_targets(session, slice_.rows)
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
                "apply_targets": apply_targets,
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
        identity store. A plate match beats visual similarity, so moving
        one as a side effect of judging would be a second, larger
        decision made by accident. Writing a read onto its vehicle is the
        separate, explicit act next door (`plate_apply_identity`).
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

    @app.post("/plates/bulk")
    def plates_bulk(
        action: str = Form(...),
        read: list[int] = Form(default=[]),
        group: list[str] = Form(default=[]),
        back: str = Form("/plates"),
    ):
        """One verdict over a selection (CLD-131) or a visit (CLD-130).

        On a table where one parked car contributes hundreds of rows,
        per-row judging is not a workflow — and a verdict belongs to
        "this vehicle's visit", not to frame 743. `read` entries are
        individual row ids; `group` entries are `<event_id>:<text>`
        pairs naming every read whose best-known text (the operator's
        correction first, the OCR's text second) matches — the same
        definition the grouped view groups by, so the buttons judge
        exactly the rows the row summarised. Same vocabulary as the
        per-row verdict, and `clear` undoes either; corrections are
        untouched — clearing a verdict must not erase typed ground truth.
        """
        if action not in BULK_ACTIONS:
            raise HTTPException(400, f"unknown bulk action {action!r}")
        verdict = BULK_ACTIONS[action]
        stamp = _now() if verdict else None
        groups = _parse_groups(group)
        with Session() as session:
            for row in _bulk_targets(session, read, groups):
                row.verdict = verdict
                row.verdict_at = stamp
            session.commit()
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
        verdict itself it changes nothing in the identity store: knowing
        what a plate says and deciding which vehicle carries it are two
        acts, and the second one is `plate_apply_identity`, which this
        correction makes available. An empty submission clears the
        correction and leaves the verdict standing.
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

    @app.post("/plates/{read_id}/apply-identity")
    def plate_apply_identity(
        read_id: int,
        back: str = Form("/plates"),
    ):
        """Write this read's plate onto the vehicle its event claims (CLD-134).

        The deliberate second act after judging. Confirming a read says
        the OCR was right; this says the identity should carry it — and
        keeping them apart is what lets an operator work the queue
        without every keystroke reaching into the identity store.

        Refused unless the read's event carries exactly one active claim
        of the read's identifier: with two, writing to either would be a
        guess, and a plate match beats visual similarity outright
        (PRD §6.4), so the guess would move every future sighting of that
        number. The identity page's edit field is the unambiguous path.

        The same write as that field, through `set_identity_plate`, so
        the plate is normalized once and the result is operator-owned —
        including against the resolver re-learning over it.
        """
        with Session() as session:
            read = session.get(PlateRead, read_id)
            if read is None:
                raise HTTPException(404)
            value = read.corrected_text or read.text
            if not value or not (read.corrected_text or read.verdict == "confirmed"):
                raise HTTPException(
                    400,
                    "only a corrected or confirmed read can be applied — "
                    "judge it first",
                )
            claims = session.scalars(
                select(EventIdentity).where(
                    EventIdentity.event_id == read.event_id,
                    EventIdentity.unlinked_at.is_(None),
                    EventIdentity.identity_id.is_not(None),
                    EventIdentity.identifier_key == read.identifier_key,
                )
            ).all()
            if not claims:
                raise HTTPException(
                    400,
                    f"event {read.event_id} has no vehicle claim to apply this "
                    "read to — link one on the event page first",
                )
            if len(claims) > 1:
                names = ", ".join(
                    f"{i.display_name} (identity {i.id})"
                    for i in (session.get(Identity, c.identity_id) for c in claims)
                    if i is not None
                )
                raise HTTPException(
                    400,
                    f"event {read.event_id} claims {len(claims)} vehicles ({names}) "
                    "— set the plate on the right one from its identity page",
                )
            identity = session.get(Identity, claims[0].identity_id)
            # Not confirm=True: this button has no confirm checkbox, and
            # silently giving two identities one plate is the state the
            # duplicate check exists to keep the console from creating.
            # The 409 names the other identity, which is where the
            # operator has to decide anyway.
            identity_ops.set_identity_plate(session, identity, value)
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
