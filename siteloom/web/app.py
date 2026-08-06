"""Operator web UI: browse events, filter by camera/class/time.

The label-and-learn workflow (naming unknown faces/vehicles) lands here
when the face/plate modules arrive; this slice ships the event browser
those workflows build on.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, not_, or_, select
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
from siteloom.store.models import status_clause, unmatched_clause

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Triage's class-kind chips. "other" is the residual rather than a fixed
# list, so a class added to detection.classes (or auto-added by the
# registry) stays reachable in the UI without a code change (NFR3).
CLASS_KINDS = {
    "people": {"person"},
    "vehicles": {"car", "truck", "bus", "motorcycle", "bicycle"},
    "other": set(),
}


def _kind_clause(kind: str):
    named = CLASS_KINDS["people"] | CLASS_KINDS["vehicles"]
    if kind == "other":
        return Event.class_name.not_in(named)
    return Event.class_name.in_(CLASS_KINDS[kind])


def _triage_url(base: dict, **overrides) -> str:
    """A link to the events list with some filters changed.

    Toggling a chip must preserve every other filter — losing the camera
    or time window when you tick "Unmatched" makes the chips unusable —
    so links are built from the whole live filter state, not composed
    from the one field being changed.
    """
    params: list[tuple[str, str]] = []
    merged = {**base, **overrides}
    for key in ("camera", "class", "since", "until"):
        if merged.get(key):
            params.append((key, str(merged[key])))
    for flag in ("needs_review", "unmatched"):
        if merged.get(flag):
            params.append((flag, "1"))
    for k in merged.get("kinds") or []:
        params.append(("kind", k))
    if merged.get("selected"):
        params.append(("selected", str(merged["selected"])))
    if merged.get("page", 1) and int(merged.get("page") or 1) > 1:
        params.append(("page", str(merged["page"])))
    return "/?" + urlencode(params) if params else "/"


def _safe_next(next_url: str, event_id: int) -> str:
    """Confine a form-supplied redirect target to this site.

    The triage rail round-trips the operator back to the filtered list
    they were working, so the target comes from the page. That makes it
    attacker-supplied: anything not a plain absolute path on this origin
    (scheme, host, protocol-relative `//evil`) falls back to the event.
    """
    if (
        next_url.startswith("/")
        and not next_url.startswith("//")
        and "\\" not in next_url
    ):
        return next_url
    return f"/events/{event_id}"


def _rail_context(session, event_id: int) -> dict | None:
    """Everything the triage detail rail shows for one event."""
    event = session.get(Event, event_id)
    if event is None:
        return None
    detections = (
        session.scalars(
            select(Detection).filter_by(event_id=event_id).order_by(Detection.timestamp)
        )
        .unique()
        .all()
    )
    links = (
        session.scalars(
            select(EventIdentity)
            .options(selectinload(EventIdentity.identity))
            .filter_by(event_id=event_id)
        )
        .unique()
        .all()
    )
    zones: list[str] = []
    for d in detections:
        for zone in json.loads(d.zones):
            if zone not in zones:
                zones.append(zone)
    return {
        "event": event,
        "camera": session.get(Camera, event.camera_id),
        "detections": detections,
        "identity_links": links,
        "zones": zones,
        "status": event.review_status,
    }


def create_app(config: SiteConfig, recognition_service=None) -> FastAPI:
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
        needs_review: bool = False,
        unmatched: bool = False,
        kind: list[str] = Query(default=[]),
        selected: int | None = None,
        page: int = 1,
    ):
        page_size = 50
        kinds = [k for k in kind if k in CLASS_KINDS]
        with Session() as session:
            q = (
                select(Event)
                .options(
                    selectinload(Event.camera),
                    # Through to the Identity: the row prints its display
                    # name, so stopping at the link both detaches after the
                    # session closes and costs a query per row.
                    selectinload(Event.identities).selectinload(
                        EventIdentity.identity
                    ),
                )
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
            # Triage chips. The state chips narrow (AND); the class-kind
            # chips are alternatives (OR), so ticking People and Vehicles
            # widens the list rather than emptying it.
            if needs_review:
                q = q.filter(not_(status_clause("cleared")))
            if unmatched:
                q = q.filter(unmatched_clause())
            if kinds:
                q = q.filter(or_(*(_kind_clause(k) for k in kinds)))

            total = session.scalar(
                select(func.count()).select_from(Event)
            )
            matched = session.scalar(
                select(func.count()).select_from(q.subquery())
            )
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
            # The rail is server-rendered so a deep link works without JS;
            # the fragment endpoint below swaps it in place when JS is on.
            rail = _rail_context(session, selected) if selected else None

        state = {
            "camera": camera,
            "class": class_name,
            "since": since,
            "until": until,
            "needs_review": needs_review,
            "unmatched": unmatched,
            "kinds": kinds,
        }
        # Selecting a row is a filter-preserving link, and every chip
        # toggle drops the selection (the rail would outlive its row).
        chip_urls = {
            "needs_review": _triage_url(state, needs_review=not needs_review),
            "unmatched": _triage_url(state, unmatched=not unmatched),
        }
        for k in CLASS_KINDS:
            chip_urls[k] = _triage_url(
                state, kinds=[x for x in kinds if x != k] if k in kinds else kinds + [k]
            )
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "chip_urls": chip_urls,
                "row_urls": {e.id: _triage_url(state, selected=e.id) for e in events},
                "back_url": _triage_url(state, selected=selected, page=page),
                "clear_url": _triage_url({}),
                "kind_labels": {
                    "people": "People",
                    "vehicles": "Vehicles",
                    "other": "Other",
                },
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
                "chips": {
                    "needs_review": needs_review,
                    "unmatched": unmatched,
                    "kinds": kinds,
                },
                "chip_count": int(needs_review) + int(unmatched) + len(kinds),
                "matched": matched,
                "total": total,
                "selected": selected,
                "rail": rail,
                "page": page,
                "has_next": has_next,
                "prev_url": _triage_url(state, selected=selected, page=page - 1),
                "next_url": _triage_url(state, selected=selected, page=page + 1),
            },
        )

    @app.get("/events/{event_id}/rail", response_class=HTMLResponse)
    def event_rail(request: Request, event_id: int, back: str = "/"):
        """The triage detail rail on its own, for in-place swapping."""
        with Session() as session:
            rail = _rail_context(session, event_id)
            if rail is None:
                raise HTTPException(404)
            return templates.TemplateResponse(
                request,
                "_event_rail.html",
                {
                    "site_name": config.site_name or config.site_id,
                    "rail": rail,
                    "back_url": _safe_next(back, event_id),
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

    @app.post("/events/{event_id}/identity/{link_id}/verdict")
    def set_identity_verdict(event_id: int, link_id: int, verdict: str = Form(...)):
        """Record a human verdict on one identity claim (CLD-16).

        Persists the judgment only — a wrong verdict must not touch the
        vector store; resolver-side learning from verdicts is separate
        work."""
        if verdict not in ("confirmed", "wrong", "clear"):
            raise HTTPException(400, "verdict must be confirmed, wrong, or clear")
        with Session() as session:
            link = session.get(EventIdentity, link_id)
            if link is None or link.event_id != event_id:
                raise HTTPException(404)
            if verdict == "clear":
                link.verdict = None
                link.verdict_at = None
            else:
                link.verdict = verdict
                link.verdict_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()
        return RedirectResponse(f"/events/{event_id}", status_code=303)

    @app.post("/events/{event_id}/missed")
    def set_missed_identity(event_id: int, missed: str = Form(...)):
        """Mark/unmark an event as a missed identification: an
        identifiable subject was there, the system claimed nothing."""
        with Session() as session:
            event = session.get(Event, event_id)
            if event is None:
                raise HTTPException(404)
            event.missed_identity = missed == "1"
            event.missed_at = (
                datetime.now(timezone.utc).replace(tzinfo=None)
                if event.missed_identity
                else None
            )
            session.commit()
        return RedirectResponse(f"/events/{event_id}", status_code=303)

    @app.post("/events/{event_id}/review")
    def set_event_reviewed(
        event_id: int,
        reviewed: str = Form(...),
        next_url: str = Form("/"),
    ):
        """Operator sign-off: clear an event out of the triage queue, or
        reopen it. Identity verdicts are left untouched — clearing says
        the event needs nothing further, not that every claim was right."""
        with Session() as session:
            event = session.get(Event, event_id)
            if event is None:
                raise HTTPException(404)
            event.reviewed_at = (
                datetime.now(timezone.utc).replace(tzinfo=None)
                if reviewed == "1"
                else None
            )
            session.commit()
        return RedirectResponse(_safe_next(next_url, event_id), status_code=303)

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

    if config.integrations.recognition_api.enabled:
        from siteloom.web import recognition_api

        service = recognition_service or recognition_api.RecognitionService(
            config, Session
        )
        recognition_api.register(app, config, service)

    @app.get("/media/{path:path}")
    def media(path: str):
        # Crop paths in the DB are relative to the working directory;
        # resolve and confine them to the media dir.
        full = (Path(path).resolve() if Path(path).is_absolute() else Path(path).resolve())
        if not str(full).startswith(str(media_root)) or not full.is_file():
            raise HTTPException(404)
        return FileResponse(full)

    # -- supervision -------------------------------------------------------

    @app.get("/healthz")
    def healthz():
        """Liveness: the process is up and serving. Deliberately touches
        nothing else, so a slow database cannot get the server killed and
        restarted into the same slow database."""
        return {"status": "ok", "site": config.site_id, "pid": os.getpid()}

    @app.get("/readyz")
    def readyz():
        """Readiness: can this process actually do its job? Runs the
        cheap half of `siteloom doctor` — never the vector store, which
        this process is already holding."""
        from siteloom.health import LIVE_CHECKS, run_checks

        report = run_checks(config, LIVE_CHECKS)
        return JSONResponse(report.as_dict(), status_code=200 if report.ok else 503)

    return app


def create_app_from_env() -> FastAPI:
    """Uvicorn factory entrypoint: SITELOOM_CONFIG=site.yaml."""
    import os

    return create_app(load_config(os.environ.get("SITELOOM_CONFIG", "site.yaml")))
