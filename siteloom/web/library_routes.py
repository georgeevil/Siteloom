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
    Annotation,
    CustomClass,
    Identity,
    ItemTag,
    LibraryItem,
    LibrarySource,
    OperationRun,
    TrainingRun,
)


log = logging.getLogger(__name__)

#: The UI-triggered reindex runs as a background thread in the serve
#: process (which is what lets it reuse the shared vector store). One at
#: a time; observed via OperationRun like any other long job.
_reindex_state: dict = {"thread": None}


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
    """A link to the library with some filters changed, keeping the rest."""
    params: list[tuple[str, str]] = []
    merged = {**base, **overrides}
    for key in ("source_id", "status", "person"):
        if merged.get(key):
            params.append((key, str(merged[key])))
    if merged.get("needs_review"):
        params.append(("needs_review", "true"))
    if merged.get("page") and int(merged["page"]) > 1:
        params.append(("page", str(merged["page"])))
    return "/library?" + urlencode(params) if params else "/library"


def _class_rows(config, seen: dict) -> list[dict]:
    """One row per class the operator can track.

    "Active" is not a new flag: it is membership of `detection.classes`,
    which is what actually decides whether the detector reports the class.
    Precision has no source in the schema yet, so no column claims one.
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

    # -- library browser ---------------------------------------------------

    @app.get("/library")
    def library(
        request: Request,
        source_id: int | None = None,
        status: str | None = None,
        needs_review: bool = False,
        person: str | None = None,
        page: int = 1,
    ):
        page_size = 60
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
            items = (
                session.scalars(q.offset((page - 1) * page_size).limit(page_size + 1))
                .unique()
                .all()
            )
            has_next = len(items) > page_size
            items = items[:page_size]
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
            matched = session.scalar(select(func.count()).select_from(q.subquery()))
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
                filters={
                    "source_id": source_id or "",
                    "status": status or "",
                    "needs_review": needs_review,
                    "person": person or "",
                },
                page=page,
                has_next=has_next,
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
                annotation.verified = bool(box.get("verified", False))
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
                class_rows=_class_rows(config, seen),
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
        from siteloom.identity import get_shared_store
        from siteloom.store import EventIdentity

        with Session() as session:
            source = session.get(Identity, identity_id)
            target = session.get(Identity, target_id)
            if source is None or target is None:
                raise HTTPException(404)
            if source.identifier_key != target.identifier_key:
                raise HTTPException(
                    400, "identities from different identifiers cannot be merged"
                )
            # Reuse the process-wide client — embedded Qdrant allows ONE
            # client per path per machine (see identity/vectors.py). A
            # RuntimeError here means a different process (a backfill,
            # library index, or frigate job) holds the store; merging
            # without re-pointing the vectors would strand them on a
            # deleted identity, so refuse instead of merging halfway.
            try:
                vectors = get_shared_store(config.identity.vector_db_path)
            except RuntimeError as exc:
                raise HTTPException(
                    503,
                    "the vector store is locked by another process — a "
                    "backfill, library index, or frigate job is likely "
                    "running against the same database. Wait for it to "
                    "finish (see /jobs), then retry the merge. "
                    f"({exc})",
                )
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

    # Embedders for split's gallery rebuild, cached per algo: re-embedding
    # a stored crop must use the same embedder live matching used ("one
    # crop, two jobs"), and building one is expensive (model load).
    _split_embedders: dict = {}

    def _identifier_embedder(identifier_key: str):
        from siteloom.identity import embedders

        ident = config.identity.identifiers.get(identifier_key)
        # Auto-added classes have no configured identifier; they are
        # generic by construction (identity/registry.py).
        algo = ident.algo if ident else "generic"
        if algo not in _split_embedders:
            _split_embedders[algo] = embedders.build_embedder(
                algo,
                device=config.detection.device,
                projection_path=config.identity.face_projection_path or None,
            )
        return _split_embedders[algo]

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
        own vectors, identified by re-embedding them and matching on
        numerical identity — see VectorStore.delete_duplicates_of.
        """
        from siteloom.identity import get_shared_store
        from siteloom.identity.enroll import embed_annotations, enroll_embedded

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
            # Reuse the process-wide client — embedded Qdrant allows ONE
            # client per path per machine (see identity/vectors.py). A
            # RuntimeError here means a different process holds the store;
            # moving rows without moving vectors is exactly the bug this
            # endpoint used to be, so refuse instead of splitting halfway.
            try:
                vectors = get_shared_store(config.identity.vector_db_path)
            except RuntimeError as exc:
                raise HTTPException(
                    503,
                    "the vector store is locked by another process — a "
                    "backfill, library index, or frigate job is likely "
                    "running against the same database. Wait for it to "
                    "finish (see /jobs), then retry the split. "
                    f"({exc})",
                )
            embedder = _identifier_embedder(source.identifier_key)
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
            removed = vectors.delete_duplicates_of(
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
    def jobs_page(request: Request):
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
                unifi_cameras=[c for c in config.cameras if c.adapter == "unifi"],
                reindex_running=_reindex_state["thread"] is not None
                and _reindex_state["thread"].is_alive(),
            ),
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
        page: int = 1,
    ):
        page_size = 48
        show = show if show in CROP_FILTERS else "needs_review"
        kind = kind if kind in CROP_KINDS else "faces"
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
            proposals = (
                session.scalars(q.offset((page - 1) * page_size).limit(page_size + 1))
                .unique()
                .all()
            )
            has_next = len(proposals) > page_size
            proposals = proposals[:page_size]

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
                page=page,
                has_next=has_next,
                min_samples=config.training.min_samples_per_person,
                sources=sources,
                groups=groups,
                selected_source=source_id,
                crop_filters=CROP_FILTERS,
                crop_kinds=CROP_KINDS,
                crop_total=crop_total,
                view={
                    "show": show,
                    "kind": kind,
                    "group": group,
                    "size": size if size in ("s", "m", "l") else "m",
                },
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
        max_vectors = 20
        face_cfg = config.identity.identifiers.get("face")
        if face_cfg:
            max_vectors = face_cfg.max_vectors_per_identity
        with Session() as session:
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
                    annotation.verified = True
                    annotation.rejected = False
                    annotation.enrolled = False  # (re)enroll under this name
                    annotation.identity_id = identity_for_label(session, name).id
                    confirmed += 1
                    vectors, embedder = _enroll_resources()
                    if enroll_annotation(
                        session, annotation, vectors, embedder, max_vectors
                    ):
                        enrolled += 1
                elif action == "reject":
                    annotation.rejected = True
                    annotation.verified = False
                    rejected += 1
                elif action == "unset":
                    annotation.verified = False
                    annotation.rejected = False
            session.commit()
        return JSONResponse(
            {
                "ok": True,
                "confirmed": confirmed,
                "rejected": rejected,
                "enrolled": enrolled,
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
