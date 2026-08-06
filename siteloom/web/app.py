"""Operator web UI: browse events, filter by camera/class/time.

The label-and-learn workflow (naming unknown faces/vehicles) lands here
when the face/plate modules arrive; this slice ships the event browser
those workflows build on.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from siteloom.config import SiteConfig, load_config
from siteloom.store import (
    Camera,
    Detection,
    Event,
    EventIdentity,
    Identity,
    NoiseEvent,
    get_session,
    init_db,
    make_engine,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app(config: SiteConfig) -> FastAPI:
    app = FastAPI(title="Siteloom")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    media_root = Path(config.storage.media_dir).resolve()

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        camera: str | None = None,
        class_name: str | None = Query(None, alias="class"),
        since: str | None = None,
        until: str | None = None,
        page: int = 1,
    ):
        page_size = 50
        with Session() as session:
            q = (
                select(Event)
                .options(selectinload(Event.camera))
                .order_by(Event.last_seen.desc())
            )
            if camera:
                q = q.filter(Event.camera_id == camera)
            if class_name:
                q = q.filter(Event.class_name == class_name)
            if since:
                q = q.filter(Event.last_seen >= datetime.fromisoformat(since))
            if until:
                q = q.filter(Event.first_seen <= datetime.fromisoformat(until))
            events = (
                session.scalars(q.offset((page - 1) * page_size).limit(page_size + 1))
                .unique()
                .all()
            )
            has_next = len(events) > page_size
            events = events[:page_size]
            cameras = session.scalars(select(Camera)).all()
            classes = sorted(
                {c for (c,) in session.execute(select(Event.class_name).distinct())}
            )
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "site_name": config.site_name or config.site_id,
                "events": events,
                "cameras": cameras,
                "classes": classes,
                "filters": {
                    "camera": camera or "",
                    "class": class_name or "",
                    "since": since or "",
                    "until": until or "",
                },
                "page": page,
                "has_next": has_next,
            },
        )

    @app.get("/events/{event_id}", response_class=HTMLResponse)
    def event_detail(request: Request, event_id: int):
        with Session() as session:
            event = session.get(Event, event_id)
            if event is None:
                raise HTTPException(404)
            detections = (
                session.scalars(
                    select(Detection)
                    .filter_by(event_id=event_id)
                    .order_by(Detection.timestamp)
                )
                .unique()
                .all()
            )
            camera = session.get(Camera, event.camera_id)
            identity_links = (
                session.scalars(
                    select(EventIdentity)
                    .options(selectinload(EventIdentity.identity))
                    .filter_by(event_id=event_id)
                )
                .unique()
                .all()
            )
        return templates.TemplateResponse(
            request,
            "event.html",
            {
                "site_name": config.site_name or config.site_id,
                "event": event,
                "camera": camera,
                "detections": [
                    {
                        "d": d,
                        "bbox": json.loads(d.bbox),
                        "zones": json.loads(d.zones),
                    }
                    for d in detections
                ],
                "identity_links": identity_links,
            },
        )

    @app.get("/identities", response_class=HTMLResponse)
    def identities(
        request: Request,
        identifier: str | None = None,
        unlabeled: bool = False,
    ):
        with Session() as session:
            q = select(Identity).order_by(Identity.last_seen.desc())
            if identifier:
                q = q.filter(Identity.identifier_key == identifier)
            if unlabeled:
                q = q.filter(Identity.label.is_(None))
            rows = session.scalars(q.limit(200)).unique().all()
            identifier_keys = sorted(
                {k for (k,) in session.execute(select(Identity.identifier_key).distinct())}
            )
        return templates.TemplateResponse(
            request,
            "identities.html",
            {
                "site_name": config.site_name or config.site_id,
                "identities": rows,
                "identifier_keys": identifier_keys,
                "filters": {"identifier": identifier or "", "unlabeled": unlabeled},
            },
        )

    @app.get("/identities/{identity_id}", response_class=HTMLResponse)
    def identity_detail(request: Request, identity_id: int):
        from siteloom.store import Annotation

        with Session() as session:
            identity = session.get(Identity, identity_id)
            if identity is None:
                raise HTTPException(404)
            links = (
                session.scalars(
                    select(EventIdentity)
                    .options(
                        selectinload(EventIdentity.event).selectinload(Event.camera)
                    )
                    .filter_by(identity_id=identity_id)
                    .order_by(EventIdentity.id.desc())
                    .limit(100)
                )
                .unique()
                .all()
            )
            # Library crops attributed to this identity — the material a
            # split operates on.
            annotations = (
                session.scalars(
                    select(Annotation)
                    .filter_by(identity_id=identity_id)
                    .order_by(Annotation.id)
                    .limit(60)
                )
                .unique()
                .all()
            )
            merge_candidates = session.scalars(
                select(Identity)
                .filter(
                    Identity.identifier_key == identity.identifier_key,
                    Identity.id != identity_id,
                )
                .order_by(Identity.label.is_(None), Identity.label, Identity.id)
                .limit(200)
            ).all()
        return templates.TemplateResponse(
            request,
            "identity.html",
            {
                "site_name": config.site_name or config.site_id,
                "identity": identity,
                "links": links,
                "annotations": annotations,
                "merge_candidates": merge_candidates,
            },
        )

    @app.post("/identities/{identity_id}/label")
    def label_identity(identity_id: int, label: str = Form("")):
        """Label-and-learn (PRD §6.3): name an unknown identity. All past
        and future matches inherit the label instantly — embeddings are
        already grouped under this identity in the vector store."""
        with Session() as session:
            identity = session.get(Identity, identity_id)
            if identity is None:
                raise HTTPException(404)
            identity.label = label.strip() or None
            session.commit()
        return RedirectResponse(f"/identities/{identity_id}", status_code=303)

    @app.get("/noise", response_class=HTMLResponse)
    def noise(request: Request):
        with Session() as session:
            rows = (
                session.scalars(
                    select(NoiseEvent).order_by(NoiseEvent.start.desc()).limit(200)
                )
                .unique()
                .all()
            )
        return templates.TemplateResponse(
            request,
            "noise.html",
            {"site_name": config.site_name or config.site_id, "noise_events": rows},
        )

    from siteloom.web import library_routes

    library_routes.register(app, templates, Session, config)

    @app.get("/media/{path:path}")
    def media(path: str):
        # Crop paths in the DB are relative to the working directory;
        # resolve and confine them to the media dir.
        full = (Path(path).resolve() if Path(path).is_absolute() else Path(path).resolve())
        if not str(full).startswith(str(media_root)) or not full.is_file():
            raise HTTPException(404)
        return FileResponse(full)

    return app


def create_app_from_env() -> FastAPI:
    """Uvicorn factory entrypoint: SITELOOM_CONFIG=site.yaml."""
    import os

    return create_app(load_config(os.environ.get("SITELOOM_CONFIG", "site.yaml")))
