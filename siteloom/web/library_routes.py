"""Library, labeling, class-management and training routes.

Split out of app.py to keep each file readable — registered by
create_app() onto the same FastAPI instance.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
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


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
        return templates.TemplateResponse(
            request,
            "library.html",
            ctx(
                items=items,
                sources=sources,
                counts=counts,
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
                confidence=config.detection.confidence,
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
        classes = [c.strip() for c in body.get("classes", []) if c.strip()]
        if classes:
            config.detection.classes = classes
        if "confidence" in body:
            config.detection.confidence = float(body["confidence"])
        for key, values in (body.get("identifiers") or {}).items():
            ident = config.identity.identifiers.get(key)
            if ident is None:
                continue
            if "threshold" in values:
                ident.threshold = float(values["threshold"])
            if "applies_to" in values:
                ident.applies_to = [v for v in values["applies_to"] if v]
            if "plate_ocr" in values:
                ident.plate_ocr = bool(values["plate_ocr"])
        if "auto_add_classes" in body:
            config.identity.auto_add_classes = bool(body["auto_add_classes"])
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
        from siteloom.identity import VectorStore
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
            vectors = VectorStore(config.identity.vector_db_path)
            try:
                moved = vectors.reassign_identity(
                    source.identifier_key, source.id, target.id
                )
            finally:
                vectors.close()
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

        Used when a cluster has absorbed two people. The old identity's
        vectors are dropped and rebuilt from what remains, because a
        polluted gallery keeps re-attracting the wrong faces.
        """
        ids = [int(v) for v in annotation_ids.split(",") if v.strip().isdigit()]
        if not ids:
            raise HTTPException(400, "select at least one annotation to split off")
        with Session() as session:
            source = session.get(Identity, identity_id)
            if source is None:
                raise HTTPException(404)
            fresh = Identity(
                identifier_key=source.identifier_key,
                class_name=source.class_name,
                first_seen=source.first_seen,
                last_seen=source.last_seen,
            )
            session.add(fresh)
            session.flush()
            session.execute(
                Annotation.__table__.update()
                .where(Annotation.id.in_(ids))
                .values(identity_id=fresh.id)
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
            ctx(runs=payload, running=[r for r in payload if r["status"] == "running"]),
        )

    @app.get("/api/jobs")
    def jobs_api():
        """Polled by the dashboard so a run started in a terminal is
        visible in the browser without a page reload."""
        with Session() as session:
            runs = session.scalars(
                select(OperationRun).order_by(OperationRun.id.desc()).limit(25)
            ).all()
            return JSONResponse([_run_payload(r) for r in runs])

    # -- training review ---------------------------------------------------

    @app.get("/training")
    def training_page(request: Request, person: str | None = None, page: int = 1):
        page_size = 48
        with Session() as session:
            q = (
                select(Annotation)
                .options(selectinload(Annotation.item))
                .filter(Annotation.class_name == "face")
                .order_by(Annotation.verified, Annotation.id)
            )
            if person:
                q = q.filter(Annotation.proposed_name == person)
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
            ),
        )

    # Enrollment resources, built lazily: the embedder and vector store
    # are only loaded once someone actually confirms a proposal.
    _enroll_state: dict = {}

    def _enroll_resources():
        if not _enroll_state:
            from siteloom.identity import VectorStore
            from siteloom.identity.embedders import FaceEmbedder

            _enroll_state["vectors"] = VectorStore(config.identity.vector_db_path)
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
