"""Library, labeling, class-management and training routes.

Split out of app.py to keep each file readable — registered by
create_app() onto the same FastAPI instance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from fastapi import Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.orm import selectinload

from siteloom.store import (
    VERIFIED_BY_HUMAN,
    Annotation,
    CustomClass,
    Event,
    EventIdentity,
    Identity,
    ItemTag,
    LibraryItem,
    LibrarySource,
    OperationRun,
    TrainingRun,
)
from siteloom.store.models import PLATE_SOURCE_OPERATOR
from siteloom.web import paging, params


log = logging.getLogger(__name__)

#: The site-wide event rules `/classes/events` may write, each paired
#: with the parser that says what kind of value it is (web/params.py).
#: The bounds are part of the rule, not decoration: `stitch_min_iou: 4`
#: asks for an overlap no two boxes can have, so stitching stops
#: entirely — and it used to be accepted, written to YAML, and noticed
#: days later as fragmentation nobody could explain. Zero is left
#: allowed wherever it means "gate off" (which is most of them);
#: negative is what has no meaning.
EVENT_RULES = {
    "min_detections": lambda v, f: params.as_int(v, f, low=0),
    "min_duration_s": lambda v, f: params.as_float(v, f, low=0.0),
    "min_confidence": params.as_confidence,
    "stitch_gap_s": lambda v, f: params.as_float(v, f, low=0.0),
    "stitch_min_iou": lambda v, f: params.as_float(v, f, low=0.0, high=1.0),
    "identify_min_confidence": params.as_confidence,
    "identify_min_crop_px": lambda v, f: params.as_int(v, f, low=0),
    "identify_only_significant": params.as_bool,
}

#: Top-level fields `/classes/detection` writes, and the per-identifier
#: settings it accepts inside them.
DETECTION_FIELDS = (
    "classes",
    "confidence",
    "class_confidence",
    "identifiers",
    "auto_add_classes",
    "auto_add_threshold",
)
IDENTIFIER_FIELDS = ("threshold", "applies_to", "plate_ocr")

#: What a review decision may ask for (`/api/training/review`).
REVIEW_ACTIONS = ("confirm", "classify", "reject", "unset")

#: The UI-triggered reindex runs as a background thread in the serve
#: process (which is what lets it reuse the shared vector store). One at
#: a time; observed via OperationRun like any other long job.
_reindex_state: dict = {"thread": None}

#: Same, for the import wizard's index step (CLD-27).
_import_state: dict = {"thread": None}


class ImportPathError(ValueError):
    """A path the wizard will not register, with a reason to show."""


def resolve_import_path(raw: str, roots: list[str]) -> Path:
    """The directory the wizard may register, or raise.

    The CLI takes any path because a shell already has the filesystem.
    The wizard's whole point is that an operator does not need one, so
    it must not become an arbitrary-path read of the host through a text
    input — `../../` or a plain `/etc` would otherwise be a source.

    Containment is the same gate the media route uses, and both halves
    matter for the same reasons: resolved paths compared with
    `is_relative_to`, never a string prefix (which would admit
    `/srv/media-x` under `/srv/media`, CLD-49), and the candidate
    resolved last so a symlink inside an allowed root pointing out of it
    is caught rather than trusted.
    """
    if not roots:
        raise ImportPathError(
            "Web import is off: no library.import_roots are configured."
        )
    text = (raw or "").strip()
    if not text:
        raise ImportPathError("Enter a directory to import.")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        raise ImportPathError("Enter an absolute path.")
    full = candidate.resolve()
    allowed = [Path(r).expanduser().resolve() for r in roots]
    if not any(full == root or full.is_relative_to(root) for root in allowed):
        raise ImportPathError(
            "That path is outside every configured import root: "
            + ", ".join(str(r) for r in allowed)
        )
    if not full.is_dir():
        raise ImportPathError(f"{full} is not a directory.")
    return full


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _optional_text(box: dict, key: str, where: str) -> str | None:
    """A text field the editor may send as a name, empty, or null."""
    value = box.get(key)
    if value is None or value == "":
        return None
    return params.as_name(value, f"{where}.{key}")


def _parse_box(box: object, index: int) -> dict:
    """One box from the annotation editor, read before it is stored.

    Every field is checked here rather than at the row it becomes,
    because this endpoint replaces an item's boxes wholesale: a value it
    cannot read halfway down the list would otherwise take the boxes
    before it with it.
    """
    where = f"annotations[{index}]"
    fields = params.as_object(box, where)
    if "bbox" not in fields:
        raise HTTPException(400, f"{where} must carry a bbox")
    corners = params.as_list(fields["bbox"], f"{where}.bbox")
    if len(corners) != 4:
        raise HTTPException(
            400,
            f"{where}.bbox must be four numbers [x1, y1, x2, y2], "
            f"got {len(corners)}",
        )
    frame_index = fields.get("frame_index")
    return {
        "id": (
            None
            if fields.get("id") is None
            else params.as_row_id(fields["id"], f"{where}.id")
        ),
        # Clamped rather than refused: dragging a box past the edge of
        # the frame is ordinary editing, and 0..1 is simply where it
        # lands. A bbox of the wrong *shape* is a different thing, and
        # is refused above.
        "bbox": [
            max(0.0, min(1.0, params.as_float(v, f"{where}.bbox[{i}]")))
            for i, v in enumerate(corners)
        ],
        "frame_index": (
            0
            if frame_index is None
            else params.as_int(frame_index, f"{where}.frame_index", low=0)
        ),
        "class_name": _optional_text(fields, "class_name", where) or "object",
        "custom_class": _optional_text(fields, "custom_class", where),
        "identity_id": (
            None
            if not fields.get("identity_id")
            else params.as_row_id(fields["identity_id"], f"{where}.identity_id")
        ),
        "verified": params.as_bool(
            fields.get("verified", False), f"{where}.verified"
        ),
        "rejected": params.as_bool(
            fields.get("rejected", False), f"{where}.rejected"
        ),
        # Absent means "leave whatever the row says"; present-and-empty
        # means "clear it". The two are not the same instruction.
        **(
            {"proposed_name": _optional_text(fields, "proposed_name", where)}
            if fields.get("proposed_name") is not None
            else {}
        ),
    }


#: Rows in one keyset slice of each grid (CLD-104). Page sizes, not
#: ceilings: every slice carries a cursor to the next, so these decide
#: how much arrives at once and nothing else.
LIBRARY_PAGE = 60
TRAINING_PAGE = 48

#: Crop-grid state filters, in the handoff's chip order.
CROP_FILTERS = {
    "needs_review": "Needs review",
    "verified": "Verified",
    "rejected": "Rejected",
    "unenrolled": "Unenrolled",
    "all": "All",
}
#: Which detection class the crops came from.
CROP_KINDS = {"faces": "Faces", "vehicles": "Vehicles", "any": "All classes"}

VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle", "bicycle")


def _crop_kind_filter(q, kind: str):
    if kind == "faces":
        return q.filter(Annotation.class_name == "face")
    if kind == "vehicles":
        return q.filter(Annotation.class_name.in_(VEHICLE_CLASSES))
    return q


def _crop_show_filter(q, show: str):
    if show == "needs_review":
        return q.filter(
            Annotation.verified.is_(False), Annotation.rejected.is_(False)
        )
    if show == "verified":
        return q.filter(Annotation.verified.is_(True), Annotation.rejected.is_(False))
    if show == "rejected":
        return q.filter(Annotation.rejected.is_(True))
    if show == "unenrolled":
        # Verified but with no vector: a label the system cannot actually
        # see (identity/enroll.py). These are the ones worth sweeping.
        return q.filter(
            Annotation.verified.is_(True),
            Annotation.rejected.is_(False),
            Annotation.enrolled.is_(False),
        )
    return q


def _face_max_vectors(config) -> int:
    face = config.identity.identifiers.get("face")
    return getattr(face, "max_vectors_per_identity", 20) if face else 20


#: Offered alongside whatever is already configured, so the common COCO
#: classes are one click away without pretending this is the whole list.
CLASS_CATALOG = (
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "boat",
    "bird", "cat", "dog", "horse", "sheep", "cow", "backpack", "umbrella",
    "handbag", "suitcase", "bottle", "cell phone", "laptop",
)

#: Hue per class family for the swatch, per the handoff's token list.
CLASS_HUES = {"person": 200, "package": 145}
VEHICLE_HUE = 260


def _class_hue(name: str) -> int:
    if name in CLASS_HUES:
        return CLASS_HUES[name]
    if name in VEHICLE_CLASSES:
        return VEHICLE_HUE
    # Stable per name rather than positional, so a class keeps its colour
    # when another is added or removed.
    return (sum(ord(c) for c in name) * 37) % 360


def _library_url(base: dict, **overrides) -> str:
    """A link to the library with some filters changed, keeping the rest.

    Filters only, and deliberately no scroll position: this builds the
    links an operator clicks, copies and sends, and every one of them has
    to open at the top of the set it describes. The cursor is composed in
    `_more_url` instead, which nothing but the load-more anchor calls.
    """
    params: list[tuple[str, str]] = []
    merged = {**base, **overrides}
    for key in ("source_id", "status", "person"):
        if merged.get(key):
            params.append((key, str(merged[key])))
    if merged.get("needs_review"):
        params.append(("needs_review", "true"))
    return "/library?" + urlencode(params) if params else "/library"


def _more_url(base: dict, cursor: str) -> str:
    """The load-more href: these filters, continued from one row.

    Separate from `_library_url` rather than one more keyword on it, so
    that a cursor can only reach a URL that a caller asked for by name.
    The anchor is real and the cursor is really in it — a client with no
    JavaScript walks the list by following it — but nothing composes it
    into a filter chip.
    """
    url = _library_url(base)
    joiner = "&" if "?" in url else "?"
    return url + joiner + urlencode({"after": cursor})


def _training_url(
    *,
    show: str,
    kind: str,
    group: str,
    size: str,
    source_id: int | None,
    person: str | None,
    cursor: str | None = None,
) -> str:
    """The crop grid's own URL, optionally continued from one row.

    The facet links are composed by the `nav_url` macro in the template,
    which never passes a cursor; this exists because the load-more anchor
    is built server-side from the same whole view state, and composing a
    query string by hand in Jinja is how a facet goes missing from it.
    """
    params: list[tuple[str, str]] = [
        ("show", show),
        ("kind", kind),
        ("group", group),
        ("size", size),
    ]
    if source_id:
        params.append(("source_id", str(source_id)))
    if person:
        params.append(("person", person))
    if cursor:
        params.append(("after", cursor))
    return "/training?" + urlencode(params)


def _class_rows(config, seen: dict, precision: dict | None = None) -> list[dict]:
    """One row per class the operator can track.

    "Active" is not a new flag: it is membership of `detection.classes`,
    which is what actually decides whether the detector reports the class.

    Precision arrives as `{identifier_key: IdentifierStat}` (CLD-87) and
    is deliberately carried as the whole stat, not a float: an identifier
    with nothing reviewed has a `wrong_rate` of None, and flattening that
    to a number here is exactly how a page ends up printing "100%" over
    zero verdicts. The row keeps the denominator so the template cannot.
    """
    active = list(config.detection.classes)
    names = active + [c for c in CLASS_CATALOG if c not in active]
    rows = []
    for name in names:
        identifier = next(
            (
                (key, ident)
                for key, ident in config.identity.identifiers.items()
                if name in (ident.applies_to or [])
            ),
            None,
        )
        rows.append(
            {
                "name": name,
                "active": name in active,
                "samples": seen.get(name, 0),
                "hue": _class_hue(name),
                "identifier": identifier[0] if identifier else None,
                "threshold": identifier[1].threshold if identifier else None,
                # Detection minimum for this class: per-class override or
                # the global floor. `overridden` tells the UI which.
                "det_conf": config.detection.class_confidence.get(
                    name, config.detection.confidence
                ),
                "det_conf_overridden": name in config.detection.class_confidence,
                # Registry auto-adds a generic identifier for an unseen
                # class when this is on, so "none configured" is not the
                # same as "will never be identified".
                "auto": identifier is None and config.identity.auto_add_classes,
                # None when the class has no identifier at all — there is
                # nothing whose precision this would be.
                "precision": (precision or {}).get(identifier[0])
                if identifier
                else None,
            }
        )
    return rows


#: Slider bounds per identification algorithm, plus the value that
#: algorithm's similarity distribution actually sits around. Face (SFace)
#: cosine scores cluster around 0.36 while generic appearance scores sit
#: above 0.80 — a single 0..1 track for both would put every usable face
#: value in its first third and invite the reading that one scale fits
#: all identifiers. Each row is edited on its own algorithm's scale.
THRESHOLD_BOUNDS: dict[str, dict[str, float]] = {
    "face": {"min": 0.15, "max": 0.70, "step": 0.01, "typical": 0.36},
    "generic": {"min": 0.50, "max": 0.99, "step": 0.01, "typical": 0.80},
}


def _identifier_rows(config) -> list[dict]:
    """One editable row per configured identifier (CLD-38).

    Thresholds belong to the identifier, not the class: "vehicle" covers
    car/truck/bus/motorcycle with one number, so the control lives here
    rather than on the class table where it would look per-class and
    silently move four classes at once.
    """
    rows = []
    for key, ident in config.identity.identifiers.items():
        bounds = THRESHOLD_BOUNDS.get(ident.algo, THRESHOLD_BOUNDS["generic"])
        rows.append(
            {
                "key": key,
                "algo": ident.algo,
                "applies_to": list(ident.applies_to or []),
                "threshold": ident.threshold,
                "plate_ocr": ident.plate_ocr,
                "min": bounds["min"],
                "max": bounds["max"],
                "step": bounds["step"],
                "typical": bounds["typical"],
                # Cameras that override this identifier's threshold —
                # shown so the site-wide slider never looks like the last
                # word for a camera that has its own (CLD-39).
                "camera_overrides": [
                    {"camera": cam.id, "threshold": cam.identity.thresholds[key]}
                    for cam in config.cameras
                    if cam.identity is not None and key in cam.identity.thresholds
                ],
            }
        )
    return rows


def _camera_override_rows(config) -> list[dict]:
    """Per-camera event/identity overrides, read-only (CLD-39).

    Editing per-camera values from the web UI is deliberately not offered
    yet: the page shows what the YAML says so an operator tuning the
    site-wide sliders can see which cameras will ignore them.
    """
    rows = []
    for cam in config.cameras:
        events = (
            {
                field: value
                for field, value in cam.events.model_dump().items()
                if value is not None
            }
            if cam.events is not None
            else {}
        )
        thresholds = dict(cam.identity.thresholds) if cam.identity is not None else {}
        rows.append(
            {
                "id": cam.id,
                "name": cam.name or cam.id,
                "events": events,
                "thresholds": thresholds,
                "count": len(events) + len(thresholds),
            }
        )
    return rows


def _threshold_preview(session, identifier_key: str, threshold: float, limit: int):
    """What moving one identifier's threshold would have done (CLD-38).

    Replayed from `EventIdentity.similarity`, which is recorded for both
    outcomes: a visual match stores the winning score, and a frame that
    minted a new identity stores its best *sub-threshold* near-miss. So
    the two counts that matter are both answerable without re-embedding
    anything:

    * `would_unmatch` — visual matches whose score falls below the
      candidate: those visits become unknowns.
    * `would_match` — near-misses that clear it: those new identities
      would instead have joined an existing one.

    Plate matches are reported but never counted: a plate outranks visual
    similarity outright (PRD §6.4), so no threshold moves it. This is an
    estimate over recorded scores, not a re-run of the resolver — a link
    keeps the strongest score across a visit's frames, and the margin and
    consistency gates (CLD-41) are not replayed. The UI says so.
    """
    from siteloom.store import EventIdentity

    rows = session.execute(
        select(EventIdentity.similarity, EventIdentity.matched_by)
        .where(
            EventIdentity.identifier_key == identifier_key,
            EventIdentity.identity_id.is_not(None),
        )
        .order_by(EventIdentity.id.desc())
        .limit(limit)
    ).all()
    matched = near_misses = plate = would_unmatch = would_match = 0
    for similarity, matched_by in rows:
        if matched_by == "plate":
            plate += 1
        elif matched_by == "visual":
            matched += 1
            if (similarity or 0.0) < threshold:
                would_unmatch += 1
        else:
            # No match mode recorded: the frame minted a new identity and
            # `similarity` is the best score that failed to clear the bar.
            near_misses += 1
            if (similarity or 0.0) >= threshold:
                would_match += 1
    return {
        "identifier": identifier_key,
        "threshold": threshold,
        "sampled": len(rows),
        "visual_matches": matched,
        "near_misses": near_misses,
        "plate_matches": plate,
        "would_unmatch": would_unmatch,
        "would_match": would_match,
    }


def _source_progress(session) -> list[dict]:
    """Per-source indexing progress for the sources rail.

    `failed` is reported separately from `pending` on purpose: nothing
    picks a failed item up again without `process(retry_failed=True)`, so
    folding the two together would show a source as nearly done when part
    of it will never be processed without an explicit opt-in.
    """
    counts: dict[int, dict[str, int]] = {}
    rows = session.execute(
        select(LibraryItem.source_id, LibraryItem.status, func.count()).group_by(
            LibraryItem.source_id, LibraryItem.status
        )
    ).all()
    for source_id, status, n in rows:
        counts.setdefault(source_id, {})[status] = n

    out = []
    for source in session.scalars(select(LibrarySource).order_by(LibrarySource.name)):
        by_status = counts.get(source.id, {})
        total = sum(by_status.values())
        done = by_status.get("indexed", 0) + by_status.get("skipped", 0)
        out.append(
            {
                "source": source,
                "total": total,
                "indexed": by_status.get("indexed", 0),
                "pending": by_status.get("pending", 0),
                "failed": by_status.get("failed", 0),
                "skipped": by_status.get("skipped", 0),
                "percent": round(done * 100 / total) if total else 0,
            }
        )
    return out


# -- Today's queue (CLD-8) -------------------------------------------------
#
# ~20 borderline judgments pinned atop /training, chosen because one label
# there moves the model most (Frigate's guidance, cited in
# docs/identity-management-analysis.md: label the clear borderline crops,
# not the 90%-confident ones). Everything below is SQL over columns that
# already exist — the queue must be buildable while ingest holds the
# vector store, so a signal that needs vectors is the wrong signal here.
#
# Three signal tiers, in the order they are trusted:
#
# 1. Unreviewed EventIdentity links whose recorded `similarity` sits
#    within QUEUE_SIMILARITY_BAND of their identifier's threshold — the
#    matches the resolver only just made. Plate matches are excluded the
#    way stats.py excludes them: a plate match carries a synthetic
#    similarity, which is not a borderline anything.
# 2. Unverified annotations carrying a `proposed_name` — the Takeout
#    pass-2 guesses. `Annotation` persists no similarity column, so
#    "closest to the face threshold first" cannot be honoured; the
#    proposal itself is the borderline signal (pass 1 already
#    auto-verified the certain ones) and the tier is not confidence-
#    banded, because a proposal's uncertainty lives in the name, not in
#    the detector's box score.
# 3. Unnamed annotations whose detection `confidence` sits in
#    QUEUE_CONFIDENCE_BAND — the clear-but-uncertain crops, never the top
#    of the range.

#: How many judgments a day's queue aims at — small enough to clear in
#: about ten minutes, which is the entire product. The day's actual
#: membership hovers around this (see `daily_queue`); it is a target, not
#: an exact page size.
DAILY_QUEUE_TARGET = 20

#: How far from its identifier's threshold a link's similarity may sit
#: and still count as borderline. Absolute, applied on each identifier's
#: own scale (face ≈0.36, generic ≈0.80) — the band is about nearness to
#: *that* cutoff, never a pooled score range.
QUEUE_SIMILARITY_BAND = 0.08

#: The middle of the detection-confidence range for tier 3. The top of
#: the range is deliberately outside it: a 0.95 crop teaches the model
#: nothing it does not already know.
QUEUE_CONFIDENCE_BAND = (0.35, 0.70)

#: Newest rows considered per tier. A bound, not a page: the queue reads
#: light columns over recent history and the borderline crops worth a
#: label today are not ten thousand rows back.
QUEUE_RECENT_ROWS = 2000


def _queue_today() -> date:
    """The queue's day — UTC, the timezone every stored timestamp is in.

    A module function so tests can pin the day; the route reads it
    through the module attribute.
    """
    return datetime.now(timezone.utc).date()


def _queue_hash(day: date, kind: str, row_id: int) -> float:
    """A per-day, per-item priority in [0, 1) — stable across processes.

    sha256 rather than `hash()` because Python's is salted per process,
    and two workers (or a reload) must agree on today's queue.
    """
    digest = hashlib.sha256(f"{day.isoformat()}:{kind}:{row_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def _queue_threshold(config, identifier_key: str | None) -> float:
    """The cosine cutoff a link was judged against — site-wide value, or
    the auto-add default for an identifier the registry minted at runtime
    that the YAML never named."""
    threshold = config.identity.threshold_for(identifier_key or "")
    return threshold if threshold is not None else config.identity.auto_add_threshold


def daily_queue(session, config, day: date) -> dict:
    """Today's queue: deterministic for the day, judged items leave it,
    and it does not refill until tomorrow.

    Membership is per-item and independent, which is what delivers the
    no-refill property without persisting queue state anywhere: an item
    is in today's queue iff it is a pending candidate created before the
    day started AND its per-day hash falls under a cutoff. Judging an
    item removes *it*; nobody else's hash moves, so nothing slides in to
    replace it — a bottomless queue is how ten minutes becomes an hour.

    The cutoff is DAILY_QUEUE_TARGET over the day's candidate population,
    where the population counts pending items plus items judged since the
    day started (EventIdentity judgments carry `verdict_at`/`unlinked_at`
    and confirmations carry `verified_at`, so the morning population is
    reconstructable; a *rejected* annotation is the one judgment with no
    timestamp, so each rejection shrinks the population by one and can in
    principle flip an item whose hash sits within ~cutoff/population of
    the boundary — a sliver accepted rather than adding a column). When
    the population is at or under the target the cutoff is 1.0 and every
    pending candidate is shown — fewer qualifying than ~20 means showing
    what exists.

    Rows created today are tomorrow's queue, never a mid-session refill.
    SQL only, by contract: this renders while ingest holds the vector
    store.
    """
    day_start = datetime(day.year, day.month, day.day)

    # Tier 1 — links near their identifier's threshold. Light columns
    # first: the band depends on per-identifier config, so it is applied
    # in Python over a bounded recent window.
    link_rows = session.execute(
        select(
            EventIdentity.id,
            EventIdentity.identifier_key,
            EventIdentity.similarity,
            EventIdentity.verdict,
            EventIdentity.unlinked_at,
        )
        .join(Event, EventIdentity.event_id == Event.id)
        .where(
            EventIdentity.identity_id.is_not(None),
            EventIdentity.matched_by == "visual",
            Event.first_seen < day_start,
            Event.best_crop_path.is_not(None),
            or_(
                and_(
                    EventIdentity.verdict.is_(None),
                    EventIdentity.unlinked_at.is_(None),
                ),
                EventIdentity.verdict_at >= day_start,
                EventIdentity.unlinked_at >= day_start,
            ),
        )
        .order_by(EventIdentity.id.desc())
        .limit(QUEUE_RECENT_ROWS)
    ).all()
    population = 0
    pending_links: list[int] = []
    for row in link_rows:
        threshold = _queue_threshold(config, row.identifier_key)
        if abs((row.similarity or 0.0) - threshold) > QUEUE_SIMILARITY_BAND:
            continue
        population += 1
        if row.verdict is None and row.unlinked_at is None:
            pending_links.append(row.id)

    def annotation_tier(named: bool) -> list[int]:
        nonlocal population
        q = select(Annotation.id, Annotation.verified, Annotation.rejected).where(
            Annotation.crop_path.is_not(None),
            Annotation.created_at < day_start,
            or_(
                and_(
                    Annotation.verified.is_(False),
                    Annotation.rejected.is_(False),
                ),
                Annotation.verified_at >= day_start,
            ),
        )
        if named:
            q = q.where(Annotation.proposed_name.is_not(None))
        else:
            q = q.where(
                Annotation.proposed_name.is_(None),
                Annotation.confidence >= QUEUE_CONFIDENCE_BAND[0],
                Annotation.confidence <= QUEUE_CONFIDENCE_BAND[1],
            )
        rows = session.execute(
            q.order_by(Annotation.id.desc()).limit(QUEUE_RECENT_ROWS)
        ).all()
        population += len(rows)
        return [r.id for r in rows if not r.verified and not r.rejected]

    pending_named = annotation_tier(named=True)
    pending_unnamed = annotation_tier(named=False)

    cutoff = DAILY_QUEUE_TARGET / max(population, DAILY_QUEUE_TARGET)

    def members(kind: str, ids: list[int]) -> list[int]:
        scored = sorted(
            (h, i) for i in ids if (h := _queue_hash(day, kind, i)) < cutoff
        )
        return [i for _, i in scored]

    # Tier order is the trust order; within a tier the day's hash decides,
    # which is the "seeded by the date" the behaviour asks for — the same
    # order all day, a different rotation through the borderline region
    # tomorrow.
    link_ids = members("link", pending_links)
    named_ids = members("proposal", pending_named)
    crop_ids = members("crop", pending_unnamed)

    links_by_id: dict[int, EventIdentity] = {}
    if link_ids:
        links_by_id = {
            link.id: link
            for link in session.scalars(
                select(EventIdentity)
                .options(
                    selectinload(EventIdentity.identity),
                    selectinload(EventIdentity.event),
                )
                .where(EventIdentity.id.in_(link_ids))
            )
            .unique()
            .all()
        }
    annotations_by_id: dict[int, Annotation] = {}
    if named_ids or crop_ids:
        annotations_by_id = {
            a.id: a
            for a in session.scalars(
                select(Annotation)
                .options(selectinload(Annotation.item))
                .where(Annotation.id.in_(named_ids + crop_ids))
            )
            .unique()
            .all()
        }

    entries: list[dict] = []
    for link_id in link_ids:
        link = links_by_id.get(link_id)
        if link is not None:
            entries.append(
                {
                    "kind": "link",
                    "link": link,
                    "threshold": _queue_threshold(config, link.identifier_key),
                }
            )
    for annotation_id in named_ids + crop_ids:
        annotation = annotations_by_id.get(annotation_id)
        if annotation is not None:
            entries.append({"kind": "annotation", "annotation": annotation})

    # The acknowledgement, from timestamps that exist: human confirmations
    # (verified_at, CLD-95) and link verdicts (verdict_at). Rejections are
    # deliberately un-timestamped in the schema, so a reject-heavy session
    # under-counts here — the honest direction for a number that must not
    # become a score to chase.
    judged_today = (
        session.scalar(
            select(func.count())
            .select_from(Annotation)
            .where(
                Annotation.verified_by == VERIFIED_BY_HUMAN,
                Annotation.verified_at >= day_start,
            )
        )
        or 0
    ) + (
        session.scalar(
            select(func.count())
            .select_from(EventIdentity)
            .where(EventIdentity.verdict_at >= day_start)
        )
        or 0
    )

    return {
        "day": day.isoformat(),
        "entries": entries,
        "judged_today": judged_today,
        "pending_total": len(pending_links) + len(pending_named) + len(pending_unnamed),
    }


def index_backlog(session, source_id: int | None = None) -> dict:
    """What has been scanned but not indexed, and which source to act on.

    Two-phase indexing is deliberate — `scan()` is cheap and registers
    rows as `pending`, `process()` is expensive and bounded — but it
    leaves a legitimate state where the library is full of rows and
    empty of pictures, because `thumb_path` is only written by the
    second pass. The grid rendered that as several hundred blank
    placeholders and said nothing, which reads as "my import failed"
    (CLD-126). These counts already existed per source for the sources
    panel; what was missing was saying so where it looks broken.

    `failed` is reported separately from `pending` for the reason it
    always is: nothing picks a failed item up again, so folding the two
    together would promise a run that will never happen.

    `target` is the source a single button may act on — the one being
    filtered to, or the only one with a backlog. When several have one,
    there is deliberately no target: a Takeout archive and a plain
    directory need different passes (CLD-92), so the caller offers a
    choice rather than guessing.
    """
    q = select(LibraryItem.source_id, LibraryItem.status, func.count()).group_by(
        LibraryItem.source_id, LibraryItem.status
    )
    if source_id:
        q = q.filter(LibraryItem.source_id == source_id)
    per_source: dict[int, dict[str, int]] = {}
    for sid, status, count in session.execute(q):
        per_source.setdefault(sid, {})[status] = count

    names = dict(
        session.execute(
            select(LibrarySource.id, func.coalesce(LibrarySource.name, ""))
        ).all()
    )
    sources = [
        {
            "id": sid,
            "name": names.get(sid) or f"source {sid}",
            "pending": counts.get("pending", 0),
            "failed": counts.get("failed", 0),
        }
        for sid, counts in sorted(per_source.items())
        if counts.get("pending") or counts.get("failed")
    ]
    target = source_id or (sources[0]["id"] if len(sources) == 1 else None)
    return {
        "pending": sum(s["pending"] for s in sources),
        "failed": sum(s["failed"] for s in sources),
        "sources": sources,
        "target": target,
    }


def register(app, templates, Session, config):  # noqa: C901 — route table
    def ctx(**kw) -> dict:
        return {"site_name": config.site_name or config.site_id, **kw}

    # -- import wizard (CLD-27) -------------------------------------------

    def _build_indexer():
        """A LibraryIndexer wired for the *serving* process.

        The vector store comes from the process-wide one, never a fresh
        VectorStore: embedded Qdrant takes an exclusive lock per path,
        and the recognition API and enrollment already hold this one. A
        store held by *another* process is an actionable 503, not a
        traceback — `identity_ops.shared_store` is where that is said.
        """
        from siteloom.identity import IdentityResolver
        from siteloom.ingest import build_dispatcher
        from siteloom.library import LibraryIndexer
        from siteloom.web.identity_ops import shared_store

        resolver = None
        if config.identity.enabled:
            resolver = IdentityResolver(
                config.identity, shared_store(config, "import")
            )
        return LibraryIndexer(config, Session, build_dispatcher(config), resolver)

    def _start_index_run(
        source_id: int,
        *,
        identify: bool,
        auto_verify: bool = False,
        retry_failed: bool = False,
    ) -> str | None:
        """Start one background index pass over a source. Returns why it
        could not, or None once the thread is running.

        The wizard's step 2 and the library banner's "Start indexing" are
        the same job behind two buttons, so they share this — and above
        all they share the guard, which exists because two passes would
        fight over the same pending rows and the one embedded vector
        store.

        Always **one source**, never "everything pending", and that is
        the CLD-92 rule rather than a limitation: which pass runs is
        decided by the source's kind, so a Takeout archive gets the
        importer (sidecar people tags, name proposals) and anything else
        gets the ordinary indexer. A run spanning both kinds would have
        to pick one, and picking the plain pass writes face annotations —
        which is exactly what makes `takeout import` skip an item later.
        Indexing a Takeout source with the wrong pass is not slow, it is
        lossy.
        """
        thread = _import_state["thread"]
        if thread is not None and thread.is_alive():
            return "An index run is already going."

        with Session() as session:
            source = session.get(LibrarySource, source_id)
            if source is None:
                raise HTTPException(404)
            source_name = source.name
            source_path = source.path
            source_kind = source.kind

        if config.identity.enabled:
            # Resolved here rather than inside the worker, for the reason
            # /train/enroll gives: embedded Qdrant allows one client per
            # path per machine, so a backfill or live ingest holding it
            # is ordinary — and a 503 naming it beats a job that starts,
            # dies in a thread, and leaves the operator watching /jobs
            # for a run that never appears. The store is process-wide, so
            # the indexer built below reuses this one.
            from siteloom.web.identity_ops import shared_store

            shared_store(config, "index run")

        def work():
            from siteloom.progress import ProgressReporter

            try:
                indexer = _build_indexer()
                if source_kind == "takeout":
                    from siteloom.library.takeout import TakeoutImporter

                    # The resume command has to carry the auto-verify
                    # choice. Dropping a flag on resume is exactly the
                    # bug _resume_command was written for, and this is
                    # the flag whose loss writes unreviewed rows that
                    # training/dataset.py reads as ground truth.
                    flag = "" if auto_verify else " --no-auto-verify"
                    with ProgressReporter(
                        Session,
                        "takeout-import",
                        target=source_path,
                        bar=False,
                        resume_command=(
                            f"siteloom takeout import {source_path}{flag}"
                        ),
                    ) as progress:
                        TakeoutImporter(
                            indexer,
                            auto_verify_unambiguous=auto_verify,
                            progress=progress,
                        ).import_tree(
                            source_path,
                            name=source_name,
                            batch_size=config.library.batch_size,
                        )
                    return

                retry = " --retry-failed" if retry_failed else ""
                with ProgressReporter(
                    Session,
                    "library-index",
                    target=source_name,
                    bar=False,
                    resume_command=(
                        f"siteloom library index --source {source_id} --all{retry}"
                    ),
                ) as progress:
                    indexer.process(
                        source_id=source_id,
                        # Same sentinel `library index --all` uses: process
                        # is batch-committed and interruptible internally,
                        # so "everything pending" is a limit, not a
                        # single transaction.
                        limit=10**9,
                        identify=identify,
                        progress=progress,
                        retry_failed=retry_failed,
                    )
            except Exception:  # pragma: no cover — surfaced via OperationRun
                log.exception("library index run failed")
            finally:
                _import_state["thread"] = None

        thread = threading.Thread(target=work, name="siteloom-import", daemon=True)
        _import_state["thread"] = thread
        thread.start()
        return None

    def _import_ctx(step: str, **kw) -> dict:
        roots = list(config.library.import_roots)
        running = _import_state["thread"]
        return ctx(
            step=step,
            roots=roots,
            enabled=bool(roots),
            video_frames=config.library.video_frames,
            batch_size=config.library.batch_size,
            identify_default=config.library.identify_on_index,
            indexing=running is not None and running.is_alive(),
            **kw,
        )

    @app.get("/library/import")
    def import_wizard(request: Request):
        return templates.TemplateResponse(
            request, "import.html", _import_ctx("source")
        )

    @app.post("/library/import/source")
    def import_add_source(
        request: Request,
        path: str = Form(""),
        kind: str = Form("directory"),
        name: str = Form(""),
    ):
        """Step 1 → 2: register the directory, then scan it.

        Scan is cheap and decodes nothing, so it runs inline here rather
        than as a job — the operator sees real counts before committing
        to the expensive step.
        """
        try:
            full = resolve_import_path(path, config.library.import_roots)
        except ImportPathError as exc:
            return templates.TemplateResponse(
                request,
                "import.html",
                _import_ctx("source", error=str(exc), path=path, kind=kind),
                status_code=400,
            )
        kind = kind if kind in ("directory", "takeout") else "directory"
        indexer = _build_indexer()
        source = indexer.add_source(full, name=name, kind=kind)

        if kind == "takeout":
            # Deliberately NOT indexer.scan(). A Takeout tree's `-edited`
            # derivatives are skipped by import_tree but would be
            # registered by scan, left pending forever, and eventually
            # picked up by a later `library index` — seeding the gallery
            # with near-duplicates. preview_tree registers nothing and
            # counts what the import will actually take.
            from siteloom.library.takeout import preview_tree

            preview = preview_tree(full)
            return templates.TemplateResponse(
                request,
                "import.html",
                _import_ctx(
                    "scan", source=source, preview=preview, kind=kind
                ),
            )

        result = indexer.scan(source.id)
        with Session() as session:
            sample = (
                session.scalars(
                    select(LibraryItem)
                    .filter_by(source_id=source.id)
                    .order_by(LibraryItem.id)
                    .limit(8)
                )
                .unique()
                .all()
            )
        return templates.TemplateResponse(
            request,
            "import.html",
            _import_ctx(
                "scan", source=source, result=result, sample=sample, kind=kind
            ),
        )

    @app.post("/library/import/index")
    def import_start_index(
        request: Request,
        source_id: int = Form(...),
        identify: str = Form("0"),
        auto_verify: str = Form("0"),
    ):
        """Step 2 → 3: start the expensive pass in the background.

        Bounded and resumable like every other index run — this starts
        one batch-driven job whose progress lands in OperationRun, which
        is what /jobs already renders. Nothing here re-implements a
        progress bar the platform owns.
        """
        refusal = _start_index_run(
            source_id,
            identify=identify == "1",
            auto_verify=auto_verify == "1",
        )
        if refusal:
            # To /jobs, where the run that refused this one is visible,
            # rather than back to step 2: that step renders the scan
            # result it was reached with, which this request no longer
            # has — it raised UndefinedError instead of saying anything.
            # The notice is the same channel cancel and reap answer on.
            return RedirectResponse(
                "/jobs?" + urlencode({"notice": refusal}), status_code=303
            )
        return RedirectResponse(
            f"/library/import/done?source_id={source_id}", status_code=303
        )

    @app.post("/library/index")
    def library_start_index(
        source_id: int = Form(...),
        retry_failed: str = Form("0"),
    ):
        """Start indexing from the library's own banner (CLD-126).

        The wizard is where an archive is registered; this is for the
        library an operator is already looking at, whose items are
        registered and blank. Same job, same guard, same OperationRun —
        so it lands on /jobs, which is where a long run is watched.

        `identify` is not offered as a choice here. `library.identify_on_index`
        is the site's answer to that question and the CLI already reads
        it; a second, differently-defaulted switch on a banner would make
        two runs of "the same" pass behave differently.

        A refusal lands on /jobs too, carrying its reason as a notice.
        This is a form a person clicked, so the answer has to be a page —
        and the operator's next question ("then what *is* running?") is
        answered by where they land.
        """
        refusal = _start_index_run(
            source_id,
            identify=config.library.identify_on_index,
            retry_failed=retry_failed == "1",
        )
        if refusal:
            return RedirectResponse(
                "/jobs?" + urlencode({"notice": refusal}), status_code=303
            )
        return RedirectResponse("/jobs", status_code=303)

    @app.get("/library/import/done")
    def import_done(request: Request, source_id: int):
        with Session() as session:
            source = session.get(LibrarySource, source_id)
            if source is None:
                raise HTTPException(404)
            counts = dict(
                session.execute(
                    select(LibraryItem.status, func.count())
                    .filter_by(source_id=source_id)
                    .group_by(LibraryItem.status)
                ).all()
            )
            faces = None
            if source.kind == "takeout":
                # `indexed`/`pending` are the wrong facts for this run:
                # the importer registers items and attaches face
                # annotations without ever marking an item indexed, so a
                # successful import would read as "0 indexed, 26k
                # pending" — a failure, in the same numbers.
                base = (
                    select(func.count())
                    .select_from(Annotation)
                    .join(LibraryItem, Annotation.item_id == LibraryItem.id)
                    .where(
                        LibraryItem.source_id == source_id,
                        Annotation.class_name == "face",
                    )
                )
                faces = {
                    "detected": session.scalar(base),
                    "proposed": session.scalar(
                        base.where(Annotation.proposed_name.isnot(None))
                    ),
                    "verified": session.scalar(base.where(Annotation.verified.is_(True))),
                }
        return templates.TemplateResponse(
            request,
            "import.html",
            _import_ctx("done", source=source, counts=counts, faces=faces),
        )

    # -- library browser ---------------------------------------------------

    @app.get("/library")
    def library(
        request: Request,
        source_id: int | None = None,
        status: str | None = None,
        needs_review: bool = False,
        person: str | None = None,
        after: str | None = None,
    ):
        """The archive grid, one keyset slice at a time (CLD-104).

        Ordered by `LibraryItem.id` ascending and nothing else, so the
        cursor is that one column — the primary key, so no two rows tie
        and there is no tie-break to leave out.

        Ascending order means a freshly scanned file lands at the end and
        shifts nothing. What does shift this window is a row *leaving* it
        — a re-scan pruning files that are gone, a source cleaned up —
        and under OFFSET that costs the operator exactly as many rows as
        disappeared above them, chosen from the ones they had not reached
        yet and dropped without a trace. A cursor names the last row
        delivered, so nothing below it can move.
        """
        page_size = LIBRARY_PAGE
        filters = {
            "source_id": source_id or "",
            "status": status or "",
            "needs_review": needs_review,
            "person": person or "",
        }
        with Session() as session:
            q = select(LibraryItem).order_by(LibraryItem.id)
            if source_id:
                q = q.filter(LibraryItem.source_id == source_id)
            if status:
                q = q.filter(LibraryItem.status == status)
            if needs_review:
                # Items holding at least one unverified, unrejected box.
                q = q.filter(
                    LibraryItem.id.in_(
                        select(Annotation.item_id).filter(
                            Annotation.verified.is_(False),
                            Annotation.rejected.is_(False),
                        )
                    )
                )
            if person:
                q = q.filter(
                    LibraryItem.id.in_(
                        select(ItemTag.item_id).filter(
                            ItemTag.kind == "person", ItemTag.value == person
                        )
                    )
                )
            # The count is taken from the filtered query, never from the
            # windowed one: "matched" describes the filter, and a cursor
            # is not a filter — counting after it would report the tail
            # the operator has not scrolled past yet as the whole set.
            matched = session.scalar(select(func.count()).select_from(q.subquery()))
            windowed = q
            if after:
                try:
                    values = paging.decode_cursor(after, (int,))
                except paging.CursorError as exc:
                    # Refused, not ignored. Falling back to no cursor would
                    # silently restart the list at the top mid-scroll, and
                    # the operator would read the repeat as duplicate rows.
                    raise HTTPException(400, f"bad library cursor: {exc}") from None
                windowed = q.where(
                    paging.after((LibraryItem.id,), values, descending=False)
                )
            slice_ = paging.take(
                list(session.scalars(windowed.limit(page_size + 1)).unique().all()),
                page_size,
                lambda i: (i.id,),
            )
            items = slice_.rows
            sources = session.scalars(select(LibrarySource)).all()
            counts = dict(
                session.execute(
                    select(LibraryItem.status, func.count()).group_by(
                        LibraryItem.status
                    )
                ).all()
            )
            people = [
                row[0]
                for row in session.execute(
                    select(ItemTag.value)
                    .filter(ItemTag.kind == "person")
                    .group_by(ItemTag.value)
                    .order_by(func.count().desc())
                    .limit(60)
                )
            ]
            box_counts = dict(
                session.execute(
                    select(Annotation.item_id, func.count())
                    .filter(Annotation.item_id.in_([i.id for i in items] or [0]))
                    .group_by(Annotation.item_id)
                ).all()
            )
            source_names = {s.id: (s.name or s.path) for s in sources}
            backlog = index_backlog(session, source_id)
        return templates.TemplateResponse(
            request,
            "library.html",
            ctx(
                items=items,
                sources=sources,
                counts=counts,
                backlog=backlog,
                matched=matched,
                total=sum(counts.values()),
                source_names=source_names,
                status_tabs=("", "indexed", "pending", "failed", "skipped"),
                library_url=_library_url,
                people=people,
                box_counts=box_counts,
                filters=filters,
                # None exactly when the list is exhausted, which is what
                # lets the footer say "that is everything" rather than
                # leaving a list that has simply stopped.
                more_url=(
                    None
                    if slice_.exhausted
                    else _more_url(filters, slice_.next_cursor or "")
                ),
                # Only ever true on the no-JS walk: a client running
                # infinite.js appends in place and never navigates to a
                # cursor, so nobody with scripting can land mid-list
                # without a way back to the top of it.
                top_url=_library_url(filters) if after else None,
            ),
        )

    # -- labeling ----------------------------------------------------------

    @app.get("/library/{item_id}")
    def library_item(request: Request, item_id: int, frame: int = 0):
        with Session() as session:
            item = session.scalar(
                select(LibraryItem)
                .options(selectinload(LibraryItem.tags))
                .filter_by(id=item_id)
            )
            if item is None:
                raise HTTPException(404)
            annotations = (
                session.scalars(
                    select(Annotation)
                    .options(selectinload(Annotation.identity))
                    .filter_by(item_id=item_id)
                    .order_by(Annotation.frame_index, Annotation.id)
                )
                .unique()
                .all()
            )
            identities = session.scalars(
                select(Identity)
                .filter(Identity.label.is_not(None))
                .order_by(Identity.label)
            ).all()
            custom_classes = session.scalars(
                select(CustomClass).order_by(CustomClass.name)
            ).all()
            detection_classes = sorted(
                set(config.detection.classes)
                | {a.class_name for a in annotations}
                | {"face"}
            )
            neighbours = session.execute(
                select(
                    func.max(LibraryItem.id).filter(LibraryItem.id < item_id),
                    func.min(LibraryItem.id).filter(LibraryItem.id > item_id),
                )
            ).one()
            payload = [
                {
                    "id": a.id,
                    "bbox": json.loads(a.bbox),
                    "class_name": a.class_name,
                    "custom_class": a.custom_class,
                    "identity_id": a.identity_id,
                    "identity_name": a.identity.display_name if a.identity else None,
                    "proposed_name": a.proposed_name,
                    "proposal_basis": a.proposal_basis,
                    "confidence": a.confidence,
                    "source": a.source,
                    "verified": a.verified,
                    "rejected": a.rejected,
                    "frame_index": a.frame_index,
                    "crop_path": a.crop_path,
                }
                for a in annotations
            ]
        return templates.TemplateResponse(
            request,
            "library_item.html",
            ctx(
                item=item,
                annotations=payload,
                annotations_json=json.dumps(payload),
                identities=identities,
                custom_classes=custom_classes,
                detection_classes=detection_classes,
                prev_id=neighbours[0],
                next_id=neighbours[1],
                frame=frame,
                tags=[t for t in item.tags],
            ),
        )

    @app.get("/api/items/{item_id}/annotations")
    def get_annotations(item_id: int):
        with Session() as session:
            rows = session.scalars(
                select(Annotation).filter_by(item_id=item_id)
            ).all()
            return JSONResponse(
                [
                    {
                        "id": a.id,
                        "bbox": json.loads(a.bbox),
                        "class_name": a.class_name,
                        "custom_class": a.custom_class,
                        "identity_id": a.identity_id,
                        "proposed_name": a.proposed_name,
                        "verified": a.verified,
                        "rejected": a.rejected,
                        "frame_index": a.frame_index,
                    }
                    for a in rows
                ]
            )

    @app.post("/api/items/{item_id}/annotations")
    async def save_annotations(item_id: int, request: Request):
        """Persist the box editor's state for one item.

        Boxes are sent whole rather than diffed: the editor is the source
        of truth for an item while it is open, and a full replace avoids
        an entire class of merge bugs from partial updates.

        Which makes the shape check load-bearing: a full replace deletes
        every box it was not sent, so a body this endpoint cannot read is
        refused outright rather than half-applied. Coordinates are
        normalized 0..1 (they have to survive thumbnailing), and a bbox
        that is not four numbers is malformed — clamping whatever arrived
        used to store a two-element box that the editor could not draw.
        """
        body = await params.json_object(request)
        boxes = [
            _parse_box(box, index)
            for index, box in enumerate(
                params.as_list(body.get("annotations", []), "annotations")
            )
        ]
        with Session() as session:
            item = session.get(LibraryItem, item_id)
            if item is None:
                raise HTTPException(404)
            existing = {
                a.id: a
                for a in session.scalars(
                    select(Annotation).filter_by(item_id=item_id)
                ).all()
            }
            kept: set[int] = set()
            for box in boxes:
                bbox = box["bbox"]
                annotation_id = box["id"]
                annotation = existing.get(annotation_id) if annotation_id else None
                if annotation is None:
                    annotation = Annotation(
                        item_id=item_id,
                        created_at=_now(),
                        source="human",
                        frame_index=box["frame_index"],
                    )
                    session.add(annotation)
                elif json.loads(annotation.bbox) != bbox and annotation.source == "auto":
                    # A moved machine box becomes a human correction.
                    annotation.source = "human"
                annotation.bbox = json.dumps(bbox)
                annotation.class_name = box["class_name"]
                annotation.custom_class = box["custom_class"]
                annotation.identity_id = box["identity_id"]
                # Transitions only. This endpoint replaces an item's boxes
                # wholesale, so it re-sends `verified` for rows nobody
                # touched; stamping on every save would relabel the
                # importer's own auto-verifications as human sign-off
                # because somebody opened the editor and pressed save.
                # Flipping the flag on *is* an explicit act, so that one
                # is recorded.
                wants_verified = box["verified"]
                if wants_verified and not annotation.verified:
                    annotation.mark_verified(VERIFIED_BY_HUMAN, _now())
                elif not wants_verified:
                    annotation.clear_verified()
                annotation.rejected = box["rejected"]
                if "proposed_name" in box:
                    annotation.proposed_name = box["proposed_name"]
                session.flush()
                kept.add(annotation.id)
            for annotation_id, annotation in existing.items():
                if annotation_id not in kept:
                    session.delete(annotation)
            item.reviewed = True
            session.commit()
            count = len(kept)
        return JSONResponse({"ok": True, "saved": count})

    @app.post("/api/items/{item_id}/tags")
    async def save_tags(item_id: int, request: Request):
        body = await params.json_object(request)
        values = params.as_names(body.get("tags", []), "tags")
        with Session() as session:
            if session.get(LibraryItem, item_id) is None:
                raise HTTPException(404)
            session.execute(
                delete(ItemTag).where(
                    ItemTag.item_id == item_id, ItemTag.kind == "user"
                )
            )
            for value in values:
                session.add(ItemTag(item_id=item_id, kind="user", value=value))
            session.commit()
        return JSONResponse({"ok": True, "tags": values})

    # -- class management --------------------------------------------------

    @app.get("/classes")
    def classes_page(request: Request):
        with Session() as session:
            custom = session.scalars(
                select(CustomClass).order_by(CustomClass.name)
            ).all()
            seen = dict(
                session.execute(
                    select(Annotation.class_name, func.count()).group_by(
                        Annotation.class_name
                    )
                ).all()
            )
            # Precision per identifier (CLD-87). Unwindowed on purpose:
            # /stats answers "how did last night go", this column answers
            # "is this threshold any good", and the second question wants
            # every verdict ever filed rather than a 24 h slice.
            from siteloom import stats as stats_mod

            precision = {s.key: s for s in stats_mod.identifier_stats(session)}
        return templates.TemplateResponse(
            request,
            "classes.html",
            ctx(
                detection_classes=config.detection.classes,
                identifiers=config.identity.identifiers,
                custom_classes=custom,
                seen=seen,
                auto_add=config.identity.auto_add_classes,
                auto_add_threshold=config.identity.auto_add_threshold,
                confidence=config.detection.confidence,
                event_rules=config.events,
                class_rows=_class_rows(config, seen, precision),
                identifier_rows=_identifier_rows(config),
                camera_rows=_camera_override_rows(config),
                model_line=(
                    f"{config.detection.model} · {config.detection.device} · "
                    f"conf {config.detection.confidence:.2f}"
                ),
            ),
        )

    @app.post("/classes/detection")
    async def update_detection_classes(request: Request):
        """Rewrite the tracked class list and per-identifier settings.

        Writes back to the live config object AND to the YAML file so the
        change survives a restart — class definition is meant to be an
        operator action, not an edit-the-file-and-redeploy action (NFR3).

        Which is exactly why the whole body is parsed before any of it is
        applied (CLD-61). This mutates the config the serving process is
        running on and then writes it to disk; a body that fails on its
        fourth field after three have landed leaves a half-edited site
        config in memory *and* in YAML, with nothing to say so.
        """
        body = await params.json_object(request)
        params.only_keys(body, DETECTION_FIELDS, "detection setting")

        # -- parse ---------------------------------------------------------
        classes = None
        if "classes" in body:
            classes = params.as_names(body["classes"], "classes")
            if not classes:
                raise HTTPException(
                    400,
                    "classes must name at least one class to track — a "
                    "detector with an empty class list finds nothing",
                )
        confidence = (
            params.as_confidence(body["confidence"], "confidence")
            if "confidence" in body
            else None
        )
        class_confidence = None
        if "class_confidence" in body:
            # Full-replace semantics: the page always posts the complete
            # per-class map; a class matching the global floor is omitted
            # by the UI so it keeps following the global value.
            raw = params.as_object(
                body["class_confidence"] or {}, "class_confidence"
            )
            class_confidence = {
                params.as_name(key, "every key of class_confidence"): (
                    params.as_confidence(value, f"class_confidence[{key}]")
                )
                for key, value in raw.items()
            }
        identifier_updates: list[tuple[str, dict]] = []
        for key, values in params.as_object(
            body.get("identifiers") or {}, "identifiers"
        ).items():
            if key not in config.identity.identifiers:
                # Skipping it silently reported success for a threshold
                # that was never stored — the operator's next look at the
                # page showed the old value with no explanation.
                raise HTTPException(
                    400,
                    f"unknown identifier {key!r} — configured identifiers: "
                    + ", ".join(sorted(config.identity.identifiers)),
                )
            settings = params.as_object(values, f"identifiers[{key}]")
            params.only_keys(
                settings, IDENTIFIER_FIELDS, f"setting for identifier {key}"
            )
            parsed: dict = {}
            if "threshold" in settings:
                parsed["threshold"] = params.as_similarity(
                    settings["threshold"], f"{key} threshold"
                )
            if "applies_to" in settings:
                parsed["applies_to"] = params.as_names(
                    settings["applies_to"], f"{key} applies_to"
                )
            if "plate_ocr" in settings:
                parsed["plate_ocr"] = params.as_bool(
                    settings["plate_ocr"], f"{key} plate_ocr"
                )
            identifier_updates.append((key, parsed))
        auto_add = (
            params.as_bool(body["auto_add_classes"], "auto_add_classes")
            if "auto_add_classes" in body
            else None
        )
        # The threshold a class with no identifier of its own gets on
        # first sighting — always the generic scale, since auto-added
        # identifiers are always generic.
        auto_threshold = (
            params.as_similarity(body["auto_add_threshold"], "auto_add_threshold")
            if "auto_add_threshold" in body
            else None
        )

        # -- apply ---------------------------------------------------------
        if classes is not None:
            config.detection.classes = classes
        if confidence is not None:
            config.detection.confidence = confidence
        if class_confidence is not None:
            config.detection.class_confidence = class_confidence
        for key, parsed in identifier_updates:
            ident = config.identity.identifiers[key]
            for field, value in parsed.items():
                setattr(ident, field, value)
        if auto_add is not None:
            config.identity.auto_add_classes = auto_add
        if auto_threshold is not None:
            config.identity.auto_add_threshold = auto_threshold
        written = _persist_config(config)
        return JSONResponse({"ok": True, "written_to": written})

    @app.get("/classes/thresholds/preview")
    def preview_threshold(identifier: str, threshold: float, limit: int = 200):
        """Dry-run one identifier's threshold against recorded matches.

        Read-only: it answers "what would have happened" from
        EventIdentity scores so a threshold can be explored before it is
        saved — and, because serve and ingest are separate processes,
        long before it reaches live ingest.
        """
        if identifier not in config.identity.identifiers:
            raise HTTPException(404, "no such identifier")
        if not (0.0 <= threshold <= 1.0):
            raise HTTPException(400, "threshold must be in 0..1")
        limit = max(1, min(int(limit), 2000))
        with Session() as session:
            body = _threshold_preview(session, identifier, threshold, limit)
        body["current"] = config.identity.identifiers[identifier].threshold
        return JSONResponse(body)

    @app.post("/classes/events")
    async def update_event_rules(request: Request):
        """Rewrite the site-wide event rules (significance + stitching).

        Same contract as /classes/detection: live config object + YAML
        write-back, everything parsed before anything is applied.
        Applies to ingest on restart — `siteloom serve` and `siteloom run`
        are separate processes. Per-camera overrides stay YAML-only
        (CameraConfig.events).
        """
        body = await params.json_object(request)
        params.only_keys(body, EVENT_RULES, "event rule")
        updates = {
            field: EVENT_RULES[field](value, field)
            for field, value in body.items()
        }
        rules = config.events
        for field, value in updates.items():
            setattr(rules, field, value)
        written = _persist_config(config)
        return JSONResponse({"ok": True, "written_to": written})

    @app.post("/classes/custom")
    def create_custom_class(
        name: str = Form(...),
        parent_class: str = Form(""),
        description: str = Form(""),
        threshold: float = Form(0.85),
    ):
        slug = name.strip().lower().replace(" ", "-")
        if not slug:
            raise HTTPException(400, "name required")
        # A custom class votes by cosine similarity like everything else
        # (identity/classes.py), so its cutoff is on the same 0..1 scale
        # — a class saved at 8.5 can never match, and says nothing about
        # why.
        threshold = params.as_similarity(threshold, "threshold")
        with Session() as session:
            existing = session.scalar(select(CustomClass).filter_by(name=slug))
            if existing is None:
                session.add(
                    CustomClass(
                        name=slug,
                        parent_class=parent_class.strip(),
                        description=description.strip(),
                        threshold=threshold,
                        created_at=_now(),
                    )
                )
            else:
                existing.parent_class = parent_class.strip()
                existing.description = description.strip()
                existing.threshold = threshold
            session.commit()
        return RedirectResponse("/classes", status_code=303)

    @app.post("/classes/custom/{class_id}/delete")
    def delete_custom_class(class_id: int):
        with Session() as session:
            custom = session.get(CustomClass, class_id)
            if custom is not None:
                # Clear the label off annotations rather than deleting
                # them — the boxes and their identities remain valid work.
                session.execute(
                    Annotation.__table__.update()
                    .where(Annotation.custom_class == custom.name)
                    .values(custom_class=None)
                )
                session.delete(custom)
                session.commit()
        return RedirectResponse("/classes", status_code=303)

    # -- identity merge / split -------------------------------------------

    @app.post("/identities/{identity_id}/merge")
    def merge_identity(identity_id: int, target_id: int = Form(...)):
        """Fold this identity into another: vectors are re-pointed, links
        moved — folding where the event already carries a claim on the
        target — stats summed, and the now-empty source row deleted."""
        if identity_id == target_id:
            raise HTTPException(400, "cannot merge an identity into itself")
        from siteloom.store import EventIdentity
        from siteloom.store.claims import active_claim, fold_claim
        from siteloom.web.identity_ops import shared_store

        with Session() as session:
            source = session.get(Identity, identity_id)
            target = session.get(Identity, target_id)
            if source is None or target is None:
                raise HTTPException(404)
            if source.identifier_key != target.identifier_key:
                raise HTTPException(
                    400, "identities from different identifiers cannot be merged"
                )
            # Merging without re-pointing the vectors would strand them
            # on a deleted identity, so a locked store refuses rather
            # than merging halfway (web/identity_ops.py).
            vectors = shared_store(config, "merge")
            moved = vectors.reassign_identity(
                source.identifier_key, source.id, target.id
            )
            # Merging is exactly what an operator does after the resolver
            # split one subject across buckets that appeared on the *same*
            # events, so a collision on the target is the normal case, not
            # the corner one — a blind re-point stacked duplicates there
            # (CLD-133).
            for row in session.scalars(
                select(EventIdentity).where(EventIdentity.identity_id == source.id)
            ).all():
                if row.unlinked_at is not None:
                    # Repudiated claims follow the merge untouched. They
                    # must move, not be left behind: deleting the source
                    # identity would null their identity_id, turning "this
                    # was claimed and it was wrong" into a recorded miss —
                    # a different fact.
                    row.identity_id = target.id
                    continue
                # Autoflushes, so a row moved earlier in this loop is
                # visible here: two source claims on one event fold
                # together rather than colliding.
                keeper = active_claim(session, row.event_id, target.id)
                if keeper is None:
                    row.identity_id = target.id
                    continue
                fold_claim(keeper, row)
                session.delete(row)
            # Flush the deletes before `session.delete(source)` below, or
            # the ORM nulls the FK on rows it is about to remove.
            session.flush()
            session.execute(
                Annotation.__table__.update()
                .where(Annotation.identity_id == source.id)
                .values(identity_id=target.id)
            )
            # Summed, not recomputed: the counter is per identified frame
            # (identity/resolver.py), so it tracks Σ hit_count over the
            # links — which the fold above preserves.
            target.appearance_count += source.appearance_count
            target.vector_count += moved
            target.first_seen = min(target.first_seen, source.first_seen)
            target.last_seen = max(target.last_seen, source.last_seen)
            target.label = target.label or source.label
            # The plate travels with its provenance, and an operator lock
            # travels even when there is no plate to carry it (CLD-134).
            # A cleared-and-locked source folded into an unlocked target
            # would hand the junk plate straight back to the resolver on
            # the survivor — the same failure this issue fixed, through
            # another door.
            if not target.plate and source.plate:
                target.plate = source.plate
                target.plate_source = source.plate_source
            elif not target.plate and PLATE_SOURCE_OPERATOR in (
                target.plate_source,
                source.plate_source,
            ):
                target.plate_source = PLATE_SOURCE_OPERATOR
            # The target keeps its own cover; only an empty one adopts the
            # source's — and then it adopts the lock with it (CLD-137).
            # Otherwise a merge silently downgrades an operator's choice
            # to automatic, and the next recompute clobbers it.
            if not target.best_crop_path and source.best_crop_path:
                target.best_crop_path = source.best_crop_path
                target.cover_locked = source.cover_locked
            session.delete(source)
            session.commit()
        return RedirectResponse(f"/identities/{target_id}", status_code=303)

    @app.post("/identities/{identity_id}/split")
    def split_identity(identity_id: int, annotation_ids: str = Form("")):
        """Pull selected annotations out into a fresh identity.

        Used when a cluster has absorbed two people. Vectors move with
        the annotations: the fresh identity is enrolled from the verified
        crops it takes (a label without vectors is a name the system
        cannot see), and the source loses exactly those crops' vectors,
        because a polluted gallery keeps re-attracting the wrong faces.

        The source's gallery is edited, never rebuilt. Most of an
        identity's vectors come from live camera matching
        (identity/resolver.py adds them directly) and have no Annotation
        row to re-embed them from, so dropping the gallery and
        reconstructing it from annotations would delete everything the
        cameras learned. What can be removed safely is the moved crops'
        own vectors: by the annotation recorded on the payload where
        there is one (CLD-84), and by numerical identity for points
        written before provenance existed — see
        VectorStore.delete_by_annotations and delete_duplicates_of.
        """
        from siteloom.identity.enroll import embed_annotations, enroll_embedded
        from siteloom.web.identity_ops import identifier_embedder, shared_store

        tokens = [t.strip() for t in annotation_ids.split(",") if t.strip()]
        malformed = [t for t in tokens if not t.isdigit()]
        if malformed:
            # Distinct from the empty-selection case: silently dropping a
            # bad token would split off a subset and report success.
            raise HTTPException(
                400,
                "annotation ids must be integers, got: " + ", ".join(malformed),
            )
        # De-duplicate while keeping submission order.
        ids = list(dict.fromkeys(int(t) for t in tokens))
        if not ids:
            raise HTTPException(400, "select at least one annotation to split off")
        with Session() as session:
            source = session.get(Identity, identity_id)
            if source is None:
                raise HTTPException(404)
            rows = {
                a.id: a
                for a in session.scalars(
                    select(Annotation).filter(Annotation.id.in_(ids))
                )
            }
            # Ownership guard: this endpoint only claims to split the
            # identity in the URL. Naming the offending ids — rather than
            # silently skipping them — is how the caller learns that less
            # than asked (or nothing) would have moved.
            wrong = [
                str(i)
                for i in ids
                if i not in rows or rows[i].identity_id != identity_id
            ]
            if wrong:
                raise HTTPException(
                    400,
                    "annotations do not belong to this identity: "
                    + ", ".join(wrong),
                )
            # Moving rows without moving vectors is exactly the bug this
            # endpoint used to be, so a locked store refuses rather than
            # splitting halfway (web/identity_ops.py).
            vectors = shared_store(config, "split")
            embedder = identifier_embedder(
                config, source.identifier_key, app.state.embedders
            )
            ident_cfg = config.identity.identifiers.get(source.identifier_key)
            max_vectors = ident_cfg.max_vectors_per_identity if ident_cfg else 20
            moved = [rows[i] for i in ids]
            fresh = Identity(
                identifier_key=source.identifier_key,
                class_name=source.class_name,
                first_seen=source.first_seen,
                last_seen=source.last_seen,
            )
            session.add(fresh)
            session.flush()
            for annotation in moved:
                annotation.identity_id = fresh.id
            # Enroll the fresh identity from the verified crops it now
            # owns, then take those same vectors off the source. The
            # embeddings are computed once and used for both halves, so
            # the vector deleted from the source is provably the one the
            # crop contributed. Historical EventIdentity claims stay with
            # the source: library annotations carry no link to camera
            # events, so there is nothing to derive "this event was the
            # split-off subject" from — per-event verdicts (CLD-16)
            # remain the tool for correcting individual past claims.
            embedded = embed_annotations(embedder, moved)
            enroll_embedded(vectors, fresh, embedded, max_vectors)
            # By origin first (CLD-84): a vector enrolled from one of
            # these annotations records which one, so it comes out
            # exactly. The numeric pass then catches points written
            # before payloads carried provenance — for those, a
            # re-embedded crop at cosine ≈ 1.0 is still the only
            # available proof of which vector a crop contributed.
            removed = vectors.delete_by_annotations(
                source.identifier_key, source.id, [a.id for a in moved]
            )
            removed += vectors.delete_duplicates_of(
                source.identifier_key,
                source.id,
                [embedding for _, embedding in embedded],
            )
            # Read both counts back from the store rather than doing
            # arithmetic on them: vector_count is incremented by several
            # writers and drifts, and a split is exactly when an operator
            # is looking at the number.
            fresh.vector_count = vectors.count_identity(
                fresh.identifier_key, fresh.id
            )
            source.vector_count = vectors.count_identity(
                source.identifier_key, source.id
            )
            # Appearance counts tally sightings; a library crop that was
            # enrolled counted as one on the source (identity/enroll.py),
            # so exactly those move. Live camera sightings stay with the
            # source along with their event links.
            fresh.appearance_count = removed
            source.appearance_count = max(source.appearance_count - removed, 0)
            moved_paths = {a.crop_path for a in moved if a.crop_path}
            if source.best_crop_path and source.best_crop_path in moved_paths:
                # The source handed its face to the new identity; pick it
                # a new thumbnail from what it kept.
                #
                # A lock does not survive that: the operator has just said
                # this crop is someone else, which contradicts having
                # chosen it as this identity's cover, and the later
                # statement wins (CLD-137).
                source.cover_locked = False
                fresh.best_crop_path = source.best_crop_path
                remaining = session.scalars(
                    select(Annotation)
                    .filter(
                        Annotation.identity_id == identity_id,
                        Annotation.rejected.is_(False),
                        Annotation.crop_path.is_not(None),
                    )
                    .order_by(Annotation.verified.desc(), Annotation.id)
                ).all()
                source.best_crop_path = next(
                    (a.crop_path for a in remaining if a.crop_path), None
                )
            else:
                fresh.best_crop_path = next(
                    (a.crop_path for a in moved if a.crop_path), None
                )
            session.commit()
            new_id = fresh.id
        return RedirectResponse(
            f"/identities/{new_id}?split_from={identity_id}", status_code=303
        )

    # -- jobs dashboard ----------------------------------------------------

    def _run_payload(run: OperationRun) -> dict:
        from siteloom.localtime import display, site_zone
        from siteloom.progress import humanize

        return {
            "id": run.id,
            "kind": run.kind,
            "target": run.target,
            "phase": run.phase,
            "status": "stale" if run.is_stale else run.status,
            "current": run.current,
            "total": run.total,
            "percent": round(run.percent, 1),
            "rate": round(run.rate, 2),
            "elapsed": humanize(run.elapsed_s),
            "eta": humanize(run.eta_s),
            "counters": json.loads(run.counters or "{}"),
            "phase_timings": json.loads(run.phase_timings or "{}"),
            "started_at": display(run.started_at, site_zone(config), "%Y-%m-%d %H:%M"),
            "resume_command": run.resume_command,
            "message": run.message,
        }

    @app.get("/jobs")
    def jobs_page(request: Request, notice: str | None = None):
        with Session() as session:
            runs = session.scalars(
                select(OperationRun).order_by(OperationRun.id.desc()).limit(25)
            ).all()
            payload = [_run_payload(r) for r in runs]
        return templates.TemplateResponse(
            request,
            "jobs.html",
            ctx(
                runs=payload,
                running=[r for r in payload if r["status"] == "running"],
                stale=[r for r in payload if r["status"] == "stale"],
                # Echoed back after a cancel/reap; bounded because it
                # arrives in a URL anyone can craft.
                notice=(notice or "")[:300],
                unifi_cameras=[c for c in config.cameras if c.adapter == "unifi"],
                reindex_running=_reindex_state["thread"] is not None
                and _reindex_state["thread"].is_alive(),
            ),
        )

    def _jobs_redirect(detail: str) -> RedirectResponse:
        return RedirectResponse(
            "/jobs?" + urlencode({"notice": detail}), status_code=303
        )

    @app.post("/jobs/{run_id}/cancel")
    def cancel_job(run_id: int):
        """Ask a running job to stop — the console half of `jobs cancel`.

        The same `progress.request_cancel` the CLI calls, for the same
        reason the issue asks for it: a job the wizard started in this
        process has no terminal to Ctrl-C, and a second stop mechanism
        would be a second definition of what a cancelled run means.

        It is a request, not a kill. When it cannot be delivered — the
        run belongs to another host, or nothing is behind it any more —
        this says so with a status code rather than redirecting to a page
        that looks like it worked.
        """
        from siteloom.progress import request_cancel

        with Session() as session:
            result = request_cancel(session, run_id)
        if not result.ok:
            return JSONResponse(
                {"error": result.detail, "reason": result.reason},
                status_code=404 if result.reason == "not_found" else 409,
            )
        return _jobs_redirect(result.detail)

    @app.post("/jobs/{run_id}/reap")
    def reap_job(run_id: int):
        """Close one dead row out as `abandoned`.

        Refuses anything that is not stale: a live run must be cancelled,
        which asks it to save its work, not reaped, which would leave the
        process running against a row that says it stopped.
        """
        from siteloom.progress import reap_runs, stale_runs

        with Session() as session:
            runs = stale_runs(session, [run_id])
            if not runs:
                return JSONResponse(
                    {
                        "error": (
                            f"run #{run_id} is not stale — only a row whose "
                            f"process is gone can be reaped"
                        ),
                        "reason": "not_stale",
                    },
                    status_code=409,
                )
            reap_runs(session, runs)
        return _jobs_redirect(
            f"reaped #{run_id}; its position and resume command are preserved"
        )

    @app.post("/jobs/reap")
    def reap_jobs():
        """Bulk reap — every stale row at once, as `jobs reap` does."""
        from siteloom.progress import reap_runs, stale_runs

        with Session() as session:
            runs = stale_runs(session)
            if not runs:
                return JSONResponse(
                    {"error": "nothing to reap", "reason": "not_stale"},
                    status_code=409,
                )
            count = reap_runs(session, runs)
        return _jobs_redirect(
            f"reaped {count} run(s); positions and resume commands are preserved"
        )

    @app.post("/jobs/reindex")
    def start_reindex(
        hours: float = Form(6.0),
        cameras: list[str] = Form([]),
    ):
        """Drop-and-reindex a recent window from the NVR, in the background.

        Purges the window's events/detections/crops and re-runs the UniFi
        backfill through the live pipeline so current settings (stitching,
        gating, per-class confidence) apply. Progress lands in
        OperationRun via ProgressReporter, so this page shows it live.
        One at a time — the pipeline holds per-camera tracker state and
        the embedded vector store.
        """
        from datetime import datetime, timedelta, timezone

        thread = _reindex_state["thread"]
        if thread is not None and thread.is_alive():
            return JSONResponse(
                {"error": "a reindex is already running"}, status_code=409
            )
        wanted = [
            c
            for c in config.cameras
            if c.adapter == "unifi" and (not cameras or c.id in cameras)
        ]
        if not wanted:
            return JSONResponse(
                {"error": "no matching unifi cameras"}, status_code=400
            )
        hours = max(0.1, min(hours, 24 * 14))
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)

        def work():
            from siteloom.ingest import IngestService
            from siteloom.progress import ProgressReporter
            from siteloom.reindex import reindex_window

            try:
                service = IngestService(config)
                with ProgressReporter(
                    Session,
                    "reindex",
                    target=f"last {hours:g}h · " + ", ".join(c.id for c in wanted),
                    bar=False,
                ) as progress:
                    reindex_window(service, wanted, start, end, progress=progress)
            except Exception:  # pragma: no cover — surfaced via OperationRun
                log.exception("reindex failed")
            finally:
                _reindex_state["thread"] = None

        thread = threading.Thread(target=work, name="siteloom-reindex", daemon=True)
        _reindex_state["thread"] = thread
        thread.start()
        return RedirectResponse("/jobs", status_code=303)

    @app.get("/api/jobs")
    def jobs_api():
        """Polled by the dashboard so a run started in a terminal is
        visible in the browser without a page reload."""
        with Session() as session:
            runs = session.scalars(
                select(OperationRun).order_by(OperationRun.id.desc()).limit(25)
            ).all()
            return JSONResponse([_run_payload(r) for r in runs])

    @app.get("/api/jobs/stream")
    def jobs_stream(updates: int = 0, interval: float = 2.0):
        """Server-sent events over the same run payloads (CLD-26/27).

        This is the decided observation channel for long-running jobs:
        the import wizard's index step and /jobs both subscribe here
        instead of each polling. The stream re-reads OperationRun every
        tick, so it observes work owned by *other processes* — a CLI
        import keeps its own ProgressReporter heartbeat and the browser
        merely watches, which is what lets a closed tab never kill a run.

        `updates` bounds the number of ticks (0 = until the client
        disconnects); tests use updates=1 for a single snapshot.
        `interval` is clamped so a client cannot ask the server to spin.
        """
        import time

        interval = max(0.5, min(interval, 30.0))

        def gen():
            sent = 0
            while True:
                with Session() as session:
                    runs = session.scalars(
                        select(OperationRun)
                        .order_by(OperationRun.id.desc())
                        .limit(25)
                    ).all()
                    payload = [_run_payload(r) for r in runs]
                yield f"data: {json.dumps(payload)}\n\n"
                sent += 1
                if updates and sent >= updates:
                    return
                time.sleep(interval)

        return StreamingResponse(gen(), media_type="text/event-stream")

    # -- training review ---------------------------------------------------

    @app.get("/training")
    def training_page(  # noqa: C901 — one screen, several independent facets
        request: Request,
        person: str | None = None,
        source_id: int | None = None,
        show: str = "needs_review",
        kind: str = "faces",
        group: str = "name",
        size: str = "m",
        after: str | None = None,
    ):
        """The crop grid, one keyset slice at a time (CLD-104).

        The order here is `verified, id` — unreviewed crops first — and
        `verified` is a boolean, so it ties across thousands of rows. The
        cursor therefore carries the pair: a cursor on `verified` alone
        would skip every crop sharing the last delivered row's flag,
        which is to say almost the whole queue.

        It travels as 0/1 rather than "False"/"True" because a cursor
        component is decoded by calling the type on the text, and
        `bool("False")` is True.
        """
        page_size = TRAINING_PAGE
        show = show if show in CROP_FILTERS else "needs_review"
        kind = kind if kind in CROP_KINDS else "faces"
        group = group if group == "name" else "flat"
        size = size if size in ("s", "m", "l") else "m"
        with Session() as session:
            q = (
                select(Annotation)
                .options(selectinload(Annotation.item))
                .order_by(Annotation.verified, Annotation.id)
            )
            q = _crop_kind_filter(q, kind)
            q = _crop_show_filter(q, show)
            if source_id:
                q = q.filter(
                    Annotation.item_id.in_(
                        select(LibraryItem.id).filter(
                            LibraryItem.source_id == source_id
                        )
                    )
                )
            if person:
                q = q.filter(Annotation.proposed_name == person)
            crop_total = session.scalar(
                select(func.count()).select_from(q.subquery())
            )
            windowed = q
            if after:
                try:
                    verified, last_id = paging.decode_cursor(after, (int, int))
                except paging.CursorError as exc:
                    raise HTTPException(400, f"bad training cursor: {exc}") from None
                windowed = q.where(
                    paging.after(
                        (Annotation.verified, Annotation.id),
                        (bool(verified), last_id),
                        descending=False,
                    )
                )
            slice_ = paging.take(
                list(session.scalars(windowed.limit(page_size + 1)).unique().all()),
                page_size,
                lambda a: (int(a.verified), a.id),
            )
            proposals = slice_.rows

            by_basis = dict(
                session.execute(
                    select(Annotation.proposal_basis, func.count())
                    .filter(Annotation.class_name == "face")
                    .group_by(Annotation.proposal_basis)
                ).all()
            )
            coverage = session.execute(
                select(Annotation.proposed_name, func.count())
                .filter(
                    Annotation.class_name == "face",
                    Annotation.proposed_name.is_not(None),
                    Annotation.rejected.is_(False),
                )
                .group_by(Annotation.proposed_name)
                .order_by(func.count().desc())
            ).all()
            verified_coverage = dict(
                session.execute(
                    select(Annotation.proposed_name, func.count())
                    .filter(
                        Annotation.class_name == "face",
                        Annotation.verified.is_(True),
                        Annotation.rejected.is_(False),
                    )
                    .group_by(Annotation.proposed_name)
                ).all()
            )
            totals = {
                "faces": session.scalar(
                    select(func.count())
                    .select_from(Annotation)
                    .filter(Annotation.class_name == "face")
                )
                or 0,
                "verified": session.scalar(
                    select(func.count())
                    .select_from(Annotation)
                    .filter(
                        Annotation.class_name == "face",
                        Annotation.verified.is_(True),
                        Annotation.rejected.is_(False),
                    )
                )
                or 0,
                "rejected": session.scalar(
                    select(func.count())
                    .select_from(Annotation)
                    .filter(
                        Annotation.class_name == "face",
                        Annotation.rejected.is_(True),
                    )
                )
                or 0,
            }
            runs = session.scalars(
                select(TrainingRun).order_by(TrainingRun.id.desc()).limit(10)
            ).all()
            sources = _source_progress(session)
            # Grouped here, not in the template: Jinja's groupby sorts by
            # the key, and proposed_name is nullable — comparing None with
            # str raises. Unnamed crops sort last under their own heading.
            groups = None
            if group == "name":
                buckets: dict[str, list] = {}
                for a in proposals:
                    buckets.setdefault(a.proposed_name or "", []).append(a)
                groups = sorted(
                    buckets.items(), key=lambda kv: (kv[0] == "", kv[0].lower())
                )
            # The inspector labels a selection onto an identity, so it
            # offers the ones that already exist; a new name is still just
            # typed in, which is what keeps label-and-learn pre-enrolment
            # free (PRD 6.3).
            identities = session.scalars(
                select(Identity)
                .filter(Identity.identifier_key == "face", Identity.label.is_not(None))
                .order_by(Identity.last_seen.desc())
                .limit(60)
            ).all()
            custom_classes = session.scalars(
                select(CustomClass).order_by(CustomClass.name)
            ).all()
            # Today's queue (CLD-8), pinned above the grid whatever the
            # filters say. Read through the module attribute so tests can
            # pin the day. SQL only — never the vector store.
            queue = daily_queue(session, config, _queue_today())
        return templates.TemplateResponse(
            request,
            "training.html",
            ctx(
                queue=queue,
                proposals=proposals,
                by_basis=by_basis,
                coverage=coverage,
                verified_coverage=verified_coverage,
                totals=totals,
                runs=[
                    {"run": r, "metrics": json.loads(r.metrics or "{}")} for r in runs
                ],
                filters={"person": person or ""},
                more_url=(
                    None
                    if slice_.exhausted
                    else _training_url(
                        show=show,
                        kind=kind,
                        group=group,
                        size=size,
                        source_id=source_id,
                        person=person,
                        cursor=slice_.next_cursor,
                    )
                ),
                top_url=(
                    _training_url(
                        show=show,
                        kind=kind,
                        group=group,
                        size=size,
                        source_id=source_id,
                        person=person,
                    )
                    if after
                    else None
                ),
                min_samples=config.training.min_samples_per_person,
                sources=sources,
                groups=groups,
                selected_source=source_id,
                crop_filters=CROP_FILTERS,
                crop_kinds=CROP_KINDS,
                crop_total=crop_total,
                view={"show": show, "kind": kind, "group": group, "size": size},
                identities=identities,
                custom_classes=custom_classes,
                max_vectors=_face_max_vectors(config),
            ),
        )

    # Enrollment resources, built lazily: the embedder and vector store
    # are only loaded once someone actually confirms a proposal.
    _enroll_state: dict = {}

    def _enroll_resources(action: str = "confirmation"):
        if not _enroll_state:
            from siteloom.identity.embedders import FaceEmbedder
            from siteloom.web.identity_ops import shared_store

            # Shared process-wide client — a second one on the same path
            # would deadlock against it (identity/vectors.py) — and an
            # actionable 503 when another *process* holds it, which is
            # the ordinary state while ingest or an index run is going
            # (CLD-62).
            _enroll_state["vectors"] = shared_store(config, action)
            _enroll_state["embedder"] = FaceEmbedder(
                projection_path=config.identity.face_projection_path or None
            )
        return _enroll_state["vectors"], _enroll_state["embedder"]

    def _class_resources(action: str = "class assignment"):
        """Vector store + the *generic* embedder, for custom classes.

        Not the face embedder `_enroll_resources` hands out: custom
        classes are k-NN over the shared appearance embedding
        (identity/classes.py), and `classes rebuild` re-derives every
        example with GenericEmbedder. Mixing the two would put live
        assignment in a different vector space from the rebuild that is
        supposed to reproduce it — voting would degrade silently, with
        nothing to show for it but worse answers.
        """
        if "class_embedder" not in _enroll_state:
            from siteloom.identity.embedders import GenericEmbedder
            from siteloom.web.identity_ops import shared_store

            _enroll_state["vectors"] = shared_store(config, action)
            _enroll_state["class_embedder"] = GenericEmbedder(
                device=config.detection.device
            )
        return _enroll_state["vectors"], _enroll_state["class_embedder"]

    @app.post("/api/training/review")
    async def review_proposals(request: Request):
        """Bulk confirm / reject / rename face proposals.

        Confirming also enrolls the face's embedding into the identity
        store, so a person verified here is recognized on live cameras,
        by the Frigate consumer, and via the recognition API immediately
        — a label without vectors is a name the system cannot see.

        That coupling is why the two refusals here matter more than they
        would on a plain form. The batch is read whole before a row is
        touched (CLD-61): a decision the endpoint cannot parse takes the
        400 with it rather than leaving the first half of a review
        applied. And the vector store the enrolment needs is the embedded
        one, which allows a single client per path per machine — a
        backfill, an index run or live ingest holding it is the *ordinary*
        state, not an exotic failure, so it answers with the same 503
        merge and split give (CLD-62, web/identity_ops.py). Nothing is
        committed until the end of the batch, so either refusal leaves
        the database exactly as it was.
        """
        from siteloom.identity.enroll import enroll_annotation, identity_for_label

        body = await params.json_object(request)
        params.only_keys(body, ("decisions",), "review field")
        decisions = [
            _parse_decision(decision, index)
            for index, decision in enumerate(
                params.as_list(body.get("decisions", []), "decisions")
            )
        ]
        confirmed = rejected = enrolled = 0
        classified = examples = skipped = missing = 0
        touched_classes: set[str] = set()
        max_vectors = 20
        face_cfg = config.identity.identifiers.get("face")
        if face_cfg:
            max_vectors = face_cfg.max_vectors_per_identity

        # Built on first use, and only if a classify decision arrives —
        # loading the generic embedder costs a model load, and the common
        # case here is a face review that never needs one.
        _classifier: list = []

        def classifier():
            if not _classifier:
                from siteloom.identity.classes import CustomClassifier

                vectors, _ = _class_resources()
                _classifier.append(CustomClassifier(vectors))
            return _classifier[0]

        def _embed_crop(path: str):
            import cv2

            # Read before building the embedder, not after: a crop whose
            # file is gone must not cost a ResNet load. That also keeps
            # this path reachable in tests, which may not have weights.
            image = cv2.imread(path)
            if image is None:
                return None
            _, embedder = _class_resources()
            return embedder.embed(image)

        with Session() as session:
            # An unknown class name must not become one by being typed —
            # CustomClass rows carry the threshold and parent class that
            # make a name mean anything, so assignment picks from what
            # exists rather than creating on the fly.
            known_classes = {
                c.name for c in session.scalars(select(CustomClass)).all()
            }
            for decision in decisions:
                annotation = session.get(Annotation, decision["id"])
                if annotation is None:
                    # A crop deleted or re-indexed since the grid was
                    # rendered. Counted rather than dropped: a caller
                    # told "confirmed: 4" of six must be able to see the
                    # other two went nowhere.
                    missing += 1
                    continue
                action = decision["action"]
                if action == "confirm":
                    # An empty name means "use the crop's own proposal",
                    # which is what the grid's confirm button sends when
                    # nothing is typed. With neither there is nothing to
                    # confirm the crop *as*.
                    name = decision["name"] or annotation.proposed_name or ""
                    if not name:
                        skipped += 1
                        continue
                    annotation.proposed_name = name
                    # Re-stamped unconditionally: a row the importer had
                    # already auto-verified becomes "human" the moment a
                    # person confirms it. That transition is the one the
                    # old schema lost — `source` stays "import" forever.
                    annotation.mark_verified(VERIFIED_BY_HUMAN, _now())
                    annotation.rejected = False
                    annotation.enrolled = False  # (re)enroll under this name
                    annotation.identity_id = identity_for_label(session, name).id
                    confirmed += 1
                    vectors, embedder = _enroll_resources("confirmation")
                    if enroll_annotation(
                        session, annotation, vectors, embedder, max_vectors
                    ):
                        enrolled += 1
                elif action == "classify":
                    # Assigning a custom class is not enrolment (CLD-29).
                    # Examples live in the `class-examples` collection with
                    # the class name in the payload; nothing here may touch
                    # an identity's gallery, and identity_id is left exactly
                    # as it was — a crop can be both "Alice" and
                    # "delivery-van" without either claim moving the other.
                    name = decision["custom_class"] or ""
                    if not name or name not in known_classes:
                        skipped += 1
                        continue
                    annotation.custom_class = name
                    annotation.mark_verified(VERIFIED_BY_HUMAN, _now())
                    annotation.rejected = False
                    classified += 1
                    touched_classes.add(name)
                    if annotation.crop_path:
                        vector = _embed_crop(annotation.crop_path)
                        if vector is not None:
                            classifier().add_example(vector, name)
                            examples += 1
                elif action == "reject":
                    annotation.rejected = True
                    # Rejection is a human act, but `rejected` already
                    # records that: no importer ever rejects. What it is
                    # not is a verification, so the sign-off comes off
                    # with it — see Annotation's `rejected` comment.
                    annotation.clear_verified()
                    rejected += 1
                elif action == "unset":
                    annotation.clear_verified()
                    annotation.rejected = False
            # Keep the chips' counts honest. Derived from the annotations
            # rather than incremented, so re-assigning an already
            # classified crop cannot inflate them.
            #
            # `crop_path is not null` is part of the definition, not an
            # optimisation: this must agree with CustomClassifier.rebuild,
            # which skips crops it cannot embed. The chip therefore counts
            # examples that can actually vote, which is the number worth
            # showing — a class whose examples cannot be embedded does not
            # discriminate better for having been labelled.
            for name in touched_classes:
                custom = session.scalar(select(CustomClass).filter_by(name=name))
                if custom is not None:
                    custom.example_count = (
                        session.scalar(
                            select(func.count())
                            .select_from(Annotation)
                            .filter(
                                Annotation.custom_class == name,
                                Annotation.verified.is_(True),
                                Annotation.rejected.is_(False),
                                Annotation.crop_path.is_not(None),
                            )
                        )
                        or 0
                    )
            session.commit()
        return JSONResponse(
            {
                "ok": True,
                "confirmed": confirmed,
                "rejected": rejected,
                "enrolled": enrolled,
                "classified": classified,
                # Assigned but not embedded — a crop with no file on disk
                # still carries the label, but cannot vote. Reported so a
                # caller is never told N examples landed when fewer did.
                "examples": examples,
                "skipped": skipped,
                # Decisions whose annotation no longer exists — a deleted
                # or re-indexed crop, not a malformed request.
                "missing": missing,
            }
        )


def _parse_decision(decision: object, index: int) -> dict:
    """One review decision, read before any of the batch is applied.

    An action nobody implements used to fall through every branch and be
    reported as a success with all counters at zero — the same silence
    `split_identity` refuses for an unknown annotation id.
    """
    where = f"decisions[{index}]"
    fields = params.as_object(decision, where)
    if "id" not in fields:
        raise HTTPException(
            400, f"{where} must name the annotation id it decides"
        )
    return {
        "id": params.as_row_id(fields["id"], f"{where}.id"),
        "action": params.one_of(
            fields.get("action"), f"{where}.action", REVIEW_ACTIONS
        ),
        "name": _optional_text(fields, "name", where),
        "custom_class": _optional_text(fields, "custom_class", where),
    }


def _persist_config(config) -> str | None:
    """Write the live config back to its YAML file, if we know the path."""
    from siteloom.config import save_config

    if not getattr(config, "_source_path", None):
        return None  # config built in-memory (tests); nothing to write
    try:
        return save_config(config)
    except OSError:
        return None
