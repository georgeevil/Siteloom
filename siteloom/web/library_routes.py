"""Library, labeling, class-management and training routes.

Split out of app.py to keep each file readable — registered by
create_app() onto the same FastAPI instance.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from fastapi import Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from sqlalchemy import delete, func, select
from sqlalchemy.orm import selectinload

from siteloom.store import (
    VERIFIED_BY_HUMAN,
    Annotation,
    CustomClass,
    Identity,
    ItemTag,
    LibraryItem,
    LibrarySource,
    OperationRun,
    TrainingRun,
)
from siteloom.web import paging


log = logging.getLogger(__name__)

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


def register(app, templates, Session, config):  # noqa: C901 — route table
    def ctx(**kw) -> dict:
        return {"site_name": config.site_name or config.site_id, **kw}

    # -- import wizard (CLD-27) -------------------------------------------

    def _build_indexer():
        """A LibraryIndexer wired for the *serving* process.

        The vector store comes from get_shared_store, never a fresh
        VectorStore: embedded Qdrant takes an exclusive lock per path,
        and the recognition API and enrollment already hold this one.
        """
        from siteloom.identity import IdentityResolver, get_shared_store
        from siteloom.ingest import build_dispatcher
        from siteloom.library import LibraryIndexer

        resolver = None
        if config.identity.enabled:
            resolver = IdentityResolver(
                config.identity, get_shared_store(config.identity.vector_db_path)
            )
        return LibraryIndexer(config, Session, build_dispatcher(config), resolver)

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

        Which pass runs is decided by the source kind, not by this
        endpoint's caller. A Takeout source gets TakeoutImporter: sidecar
        people tags, face detection, two-pass name proposals. Anything
        else gets the ordinary indexer. Choosing "Google Takeout" in step
        1 and then running a plain directory index — which is what this
        did before CLD-92 — succeeds silently and proposes nothing.
        """
        thread = _import_state["thread"]
        if thread is not None and thread.is_alive():
            return templates.TemplateResponse(
                request,
                "import.html",
                _import_ctx("scan", error="An index run is already going."),
                status_code=409,
            )
        with Session() as session:
            source = session.get(LibrarySource, source_id)
            if source is None:
                raise HTTPException(404)
            source_name = source.name
            source_path = source.path
            source_kind = source.kind

        wants_identify = identify == "1"
        wants_auto_verify = auto_verify == "1"

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
                    flag = "" if wants_auto_verify else " --no-auto-verify"
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
                            auto_verify_unambiguous=wants_auto_verify,
                            progress=progress,
                        ).import_tree(
                            source_path,
                            name=source_name,
                            batch_size=config.library.batch_size,
                        )
                    return

                with ProgressReporter(
                    Session,
                    "library-index",
                    target=source_name,
                    bar=False,
                    resume_command=(
                        f"siteloom library index --source {source_id} --all"
                    ),
                ) as progress:
                    indexer.process(
                        source_id=source_id,
                        # Same sentinel `library index --all` uses: process
                        # is batch-committed and interruptible internally,
                        # so "everything pending" is a limit, not a
                        # single transaction.
                        limit=10**9,
                        identify=wants_identify,
                        progress=progress,
                    )
            except Exception:  # pragma: no cover — surfaced via OperationRun
                log.exception("library import indexing failed")
            finally:
                _import_state["thread"] = None

        thread = threading.Thread(target=work, name="siteloom-import", daemon=True)
        _import_state["thread"] = thread
        thread.start()
        return RedirectResponse(
            f"/library/import/done?source_id={source_id}", status_code=303
        )

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
        return templates.TemplateResponse(
            request,
            "library.html",
            ctx(
                items=items,
                sources=sources,
                counts=counts,
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
        """
        body = await request.json()
        boxes = body.get("annotations", [])
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
                bbox = [max(0.0, min(1.0, float(v))) for v in box["bbox"]]
                annotation_id = box.get("id")
                annotation = existing.get(annotation_id) if annotation_id else None
                if annotation is None:
                    annotation = Annotation(
                        item_id=item_id,
                        created_at=_now(),
                        source="human",
                        frame_index=int(box.get("frame_index", 0)),
                    )
                    session.add(annotation)
                elif json.loads(annotation.bbox) != bbox and annotation.source == "auto":
                    # A moved machine box becomes a human correction.
                    annotation.source = "human"
                annotation.bbox = json.dumps(bbox)
                annotation.class_name = box.get("class_name") or "object"
                annotation.custom_class = box.get("custom_class") or None
                annotation.identity_id = box.get("identity_id") or None
                # Transitions only. This endpoint replaces an item's boxes
                # wholesale, so it re-sends `verified` for rows nobody
                # touched; stamping on every save would relabel the
                # importer's own auto-verifications as human sign-off
                # because somebody opened the editor and pressed save.
                # Flipping the flag on *is* an explicit act, so that one
                # is recorded.
                wants_verified = bool(box.get("verified", False))
                if wants_verified and not annotation.verified:
                    annotation.mark_verified(VERIFIED_BY_HUMAN, _now())
                elif not wants_verified:
                    annotation.clear_verified()
                annotation.rejected = bool(box.get("rejected", False))
                if box.get("proposed_name") is not None:
                    annotation.proposed_name = box["proposed_name"] or None
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
        body = await request.json()
        values = [v.strip() for v in body.get("tags", []) if v.strip()]
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
        """
        body = await request.json()
        classes = [
            c.strip()
            for c in body.get("classes", [])
            if isinstance(c, str) and c.strip()
        ]
        if classes:
            config.detection.classes = classes
        if "confidence" in body:
            config.detection.confidence = float(body["confidence"])
        if "class_confidence" in body:
            # Full-replace semantics: the page always posts the complete
            # per-class map; a class matching the global floor is omitted
            # by the UI so it keeps following the global value.
            config.detection.class_confidence = {
                str(k): float(v)
                for k, v in (body["class_confidence"] or {}).items()
            }
        for key, values in (body.get("identifiers") or {}).items():
            ident = config.identity.identifiers.get(key)
            if ident is None:
                continue
            if "threshold" in values:
                threshold = float(values["threshold"])
                if not (0.0 <= threshold <= 1.0):
                    raise HTTPException(
                        400, f"{key} threshold must be a cosine similarity in 0..1"
                    )
                ident.threshold = threshold
            if "applies_to" in values:
                ident.applies_to = [v for v in values["applies_to"] if v]
            if "plate_ocr" in values:
                ident.plate_ocr = bool(values["plate_ocr"])
        if "auto_add_classes" in body:
            config.identity.auto_add_classes = bool(body["auto_add_classes"])
        if "auto_add_threshold" in body:
            # The threshold a class with no identifier of its own gets on
            # first sighting — always the generic scale, since auto-added
            # identifiers are always generic.
            auto_threshold = float(body["auto_add_threshold"])
            if not (0.0 <= auto_threshold <= 1.0):
                raise HTTPException(
                    400, "auto_add_threshold must be a cosine similarity in 0..1"
                )
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
        write-back. Applies to ingest on restart — `siteloom serve` and
        `siteloom run` are separate processes. Per-camera overrides stay
        YAML-only (CameraConfig.events).
        """
        body = await request.json()
        rules = config.events
        for field, cast in (
            ("min_detections", int),
            ("min_duration_s", float),
            ("min_confidence", float),
            ("stitch_gap_s", float),
            ("stitch_min_iou", float),
            ("identify_min_confidence", float),
            ("identify_min_crop_px", int),
        ):
            if field in body:
                setattr(rules, field, cast(body[field]))
        if "identify_only_significant" in body:
            rules.identify_only_significant = bool(body["identify_only_significant"])
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
        moved, stats summed, and the now-empty source row deleted."""
        if identity_id == target_id:
            raise HTTPException(400, "cannot merge an identity into itself")
        from siteloom.store import EventIdentity
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
            session.execute(
                EventIdentity.__table__.update()
                .where(EventIdentity.identity_id == source.id)
                .values(identity_id=target.id)
            )
            session.execute(
                Annotation.__table__.update()
                .where(Annotation.identity_id == source.id)
                .values(identity_id=target.id)
            )
            target.appearance_count += source.appearance_count
            target.vector_count += moved
            target.first_seen = min(target.first_seen, source.first_seen)
            target.last_seen = max(target.last_seen, source.last_seen)
            target.label = target.label or source.label
            target.plate = target.plate or source.plate
            target.best_crop_path = target.best_crop_path or source.best_crop_path
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
            "started_at": run.started_at.strftime("%Y-%m-%d %H:%M"),
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
        return templates.TemplateResponse(
            request,
            "training.html",
            ctx(
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

    def _enroll_resources():
        if not _enroll_state:
            from siteloom.identity import get_shared_store
            from siteloom.identity.embedders import FaceEmbedder

            # Shared process-wide client — a second one on the same path
            # would deadlock against it (identity/vectors.py).
            _enroll_state["vectors"] = get_shared_store(
                config.identity.vector_db_path
            )
            _enroll_state["embedder"] = FaceEmbedder(
                projection_path=config.identity.face_projection_path or None
            )
        return _enroll_state["vectors"], _enroll_state["embedder"]

    def _class_resources():
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
            from siteloom.identity import get_shared_store
            from siteloom.identity.embedders import GenericEmbedder

            _enroll_state["vectors"] = get_shared_store(
                config.identity.vector_db_path
            )
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
        """
        from siteloom.identity.enroll import enroll_annotation, identity_for_label

        body = await request.json()
        decisions = body.get("decisions", [])
        confirmed = rejected = enrolled = 0
        classified = examples = skipped = 0
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
                annotation = session.get(Annotation, int(decision["id"]))
                if annotation is None:
                    continue
                action = decision.get("action")
                if action == "confirm":
                    name = (decision.get("name") or annotation.proposed_name or "").strip()
                    if not name:
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
                    vectors, embedder = _enroll_resources()
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
                    name = (decision.get("custom_class") or "").strip()
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
            }
        )


def _persist_config(config) -> str | None:
    """Write the live config back to its YAML file, if we know the path."""
    from siteloom.config import save_config

    if not getattr(config, "_source_path", None):
        return None  # config built in-memory (tests); nothing to write
    try:
        return save_config(config)
    except OSError:
        return None
