"""Operator web UI: browse events, filter by camera/class/time.

The label-and-learn workflow (naming unknown faces/vehicles) lands here
when the face/plate modules arrive; this slice ships the event browser
those workflows build on.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from siteloom.config import SiteConfig, load_config
from siteloom.store import Camera, Detection, Event, get_session, init_db, make_engine

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
            },
        )

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
