"""Operator web UI: browse events, filter by camera/class/time.

The label-and-learn workflow (naming unknown faces/vehicles) lands here
when the face/plate modules arrive; this slice ships the event browser
those workflows build on.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, not_, or_, select
from sqlalchemy.orm import selectinload

from siteloom.config import SiteConfig, load_config
from siteloom.web import auth, identity_ops
from siteloom.store import (
    Camera,
    Detection,
    Event,
    EventIdentity,
    Identity,
    NoiseEvent,
    WebSession,
    get_session,
    init_db,
    make_engine,
)
from siteloom.store.models import significance_clause, status_clause, unmatched_clause

log = logging.getLogger(__name__)

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
    for key in ("camera", "class", "since", "until", "min_conf", "min_count"):
        if merged.get(key):
            params.append((key, str(merged[key])))
    for flag in ("needs_review", "unmatched", "show_ephemeral"):
        if merged.get(flag):
            params.append((flag, "1"))
    for k in merged.get("kinds") or []:
        params.append(("kind", k))
    if merged.get("selected"):
        params.append(("selected", str(merged["selected"])))
    if merged.get("page", 1) and int(merged.get("page") or 1) > 1:
        params.append(("page", str(merged["page"])))
    return "/?" + urlencode(params) if params else "/"


def _identities_url(base: dict, **overrides) -> str:
    """A link to the identities list with some filters changed."""
    params: list[tuple[str, str]] = []
    merged = {**base, **overrides}
    if merged.get("identifier"):
        params.append(("identifier", str(merged["identifier"])))
    for flag in ("unlabeled", "unenrolled"):
        if merged.get(flag):
            params.append((flag, "1"))
    if merged.get("selected"):
        params.append(("selected", str(merged["selected"])))
    return "/identities?" + urlencode(params) if params else "/identities"


def _identity_rail(session, identity_id: int) -> dict | None:
    """Everything the identity detail rail shows."""
    from siteloom.store import Annotation

    identity = session.get(Identity, identity_id)
    if identity is None:
        return None
    links = (
        session.scalars(
            select(EventIdentity)
            .options(selectinload(EventIdentity.event).selectinload(Event.camera))
            .filter_by(identity_id=identity_id)
            # Claims an operator detached (CLD-36) are not this identity's
            # visits — showing them here would put the wrong person's
            # events back in front of whoever is reviewing the gallery.
            .filter(EventIdentity.unlinked_at.is_(None))
            .order_by(EventIdentity.id.desc())
            .limit(12)
        )
        .unique()
        .all()
    )
    samples = (
        session.scalars(
            select(Annotation)
            .filter_by(identity_id=identity_id)
            .order_by(Annotation.id)
            .limit(12)
        )
        .unique()
        .all()
    )
    return {
        "identity": identity,
        "links": links,
        "samples": samples,
        # A named identity with no vectors cannot be recognised at all, so
        # the rail says so rather than showing a reassuring zero.
        "unenrolled": bool(identity.label) and identity.vector_count == 0,
    }


def _safe_next(next_url: str, event_id: int) -> str:
    """The triage rail's redirect target, confined to this site.

    One validator (`auth.safe_next`) decides what "on this site" means
    for every redirect in the console; this only supplies the rail's
    fallback, which is the event the operator was judging.
    """
    return auth.safe_next(next_url, f"/events/{event_id}")


def _rail_context(session, event_id: int) -> dict | None:
    """Everything the triage detail rail shows for one event."""
    # Eager-load what the rail template touches: on the index page the
    # template renders AFTER this session closes, so a lazy load there is
    # a DetachedInstanceError. That bites on search deep-links
    # (/?selected=<id>) whenever the event is not on the loaded page and
    # thus not already in the identity map with its relations loaded.
    event = session.scalar(
        select(Event)
        .options(
            selectinload(Event.camera),
            selectinload(Event.identities),
        )
        .where(Event.id == event_id)
    )
    if event is None:
        return None
    detections = (
        session.scalars(
            select(Detection).filter_by(event_id=event_id).order_by(Detection.timestamp)
        )
        .unique()
        .all()
    )
    rows = (
        session.scalars(
            select(EventIdentity)
            .options(selectinload(EventIdentity.identity))
            .filter_by(event_id=event_id)
        )
        .unique()
        .all()
    )
    links = [r for r in rows if r.is_active]
    misses = [r for r in rows if r.identity_id is None]
    # Repudiated claims (CLD-36) are shown, not hidden: the operator
    # needs to see that this event was once called someone else, and
    # re-attaching is how a correction gets corrected.
    unlinked = [
        r for r in rows if r.identity_id is not None and r.unlinked_at is not None
    ]
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
        "misses": misses,
        "unlinked": unlinked,
        "candidates": _identity_candidates(session),
        "zones": zones,
        "status": event.review_status,
    }


def _identity_candidates(session) -> list[Identity]:
    """Who this event could be — the picker's options (CLD-36).

    Not filtered to one identifier: the operator, not the resolver, is
    deciding, and the face identifier missing someone is exactly when
    they need to attach the person identity by hand. Labeled identities
    sort first because naming a visit is the common case; the unknown
    buckets stay reachable for merging two sightings of one stranger.
    """
    return list(
        session.scalars(
            select(Identity)
            .order_by(
                Identity.label.is_(None), Identity.label, Identity.last_seen.desc()
            )
            .limit(200)
        )
        .unique()
        .all()
    )


def create_app(config: SiteConfig, recognition_service=None) -> FastAPI:
    app = FastAPI(title="Siteloom")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    # Embedders are built lazily and cached for the app's lifetime (model
    # load is expensive); on the app rather than the module so the cache
    # cannot outlive the config it was built from.
    app.state.embedders = {}
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    media_root = Path(config.storage.media_dir).resolve()
    #: Per-app so a second app in the same process (tests) starts clean.
    auth_gate = auth.AuthGate()
    login_throttle = auth.LoginThrottle()

    @app.middleware("http")
    async def auth_and_audit(request: Request, call_next):
        """One gate for the whole console (see web/auth.py).

        Also the one audit writer: every mutating request that reaches a
        handler leaves a row, so a new POST route cannot forget to audit.
        """
        path = request.url.path
        with Session() as session:
            enabled = auth_gate.enabled(session)
            user = auth.resolve_user(
                session, request.cookies.get(auth.SESSION_COOKIE)
            )
        request.state.user = user
        request.state.auth_enabled = enabled

        exempt = path in auth.PUBLIC_PATHS or path.startswith(auth.EXEMPT_PREFIXES)
        if enabled and not exempt:
            if user is None:
                if request.method in ("GET", "HEAD"):
                    # Quoted so a crafted path cannot smuggle extra query
                    # parameters into the login URL it is pasted into.
                    target = quote(path, safe="/")
                    return RedirectResponse(f"/login?next={target}", status_code=303)
                return JSONResponse({"detail": "login required"}, status_code=401)
            if not auth.has_role(user, auth.required_role(request.method, path)):
                return JSONResponse({"detail": "insufficient role"}, status_code=403)

        response = await call_next(request)

        if request.method not in ("GET", "HEAD", "OPTIONS") and not exempt:
            if response.status_code < 400:
                with Session() as session:
                    auth.record_audit(
                        session, user, request.method, path, response.status_code
                    )
                    session.commit()
        return response

    def _login_page(request: Request, next: str, error: str | None, status: int):
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "site_name": config.site_name or config.site_id,
                # Sanitized on the way in as well as on the way out: the
                # value is echoed into the form's hidden field, so an
                # unchecked one survives the round trip to the redirect.
                "next": auth.safe_next(next),
                "error": error,
            },
            status_code=status,
        )

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request, next: str = "/"):
        return _login_page(request, next, None, 200)

    @app.post("/login", response_class=HTMLResponse)
    def login_submit(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
        next: str = Form("/"),
    ):
        caller = request.client.host if request.client else "unknown"
        wait = login_throttle.retry_after(caller)
        if wait > 0:
            # Not audited: nothing was attempted, and a row per hammered
            # request would turn the defence into unbounded table growth.
            response = _login_page(
                request,
                next,
                f"Too many sign-in attempts. Try again in {int(wait) + 1}s.",
                429,
            )
            response.headers["Retry-After"] = str(int(wait) + 1)
            return response

        with Session() as session:
            user = auth.authenticate(session, username, password)
            if user is None:
                login_throttle.record_failure(caller)
                # A failed sign-in is an event even though nothing
                # happened — it is the only trace a guessing run leaves.
                auth.record_audit(
                    session,
                    None,
                    "POST",
                    "/login",
                    401,
                    username=auth.failed_actor(username),
                )
                session.commit()
                # One message for both failures — do not confirm usernames.
                return _login_page(
                    request, next, "Wrong username or password.", 401
                )
            login_throttle.record_success(caller)
            auth.purge_expired_sessions(session)
            token = auth.create_session(session, user)
            auth.record_audit(session, user, "POST", "/login", 303)
            session.commit()
        response = RedirectResponse(auth.safe_next(next), status_code=303)
        response.set_cookie(
            auth.SESSION_COOKIE,
            token,
            httponly=True,
            samesite="lax",
            max_age=int(auth.SESSION_TTL.total_seconds()),
        )
        return response

    @app.post("/logout")
    def logout(request: Request):
        token = request.cookies.get(auth.SESSION_COOKIE)
        if token:
            with Session() as session:
                row = session.get(WebSession, token)
                if row is not None:
                    session.delete(row)
                auth.purge_expired_sessions(session)
                session.commit()
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(auth.SESSION_COOKIE)
        return response

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        camera: str | None = None,
        class_name: str | None = Query(None, alias="class"),
        since: str | None = None,
        until: str | None = None,
        needs_review: bool = False,
        unmatched: bool = False,
        show_ephemeral: bool = False,
        min_conf: float | None = None,
        min_count: int | None = None,
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
            if min_conf is not None:
                q = q.filter(Event.best_confidence >= min_conf)
            if min_count is not None:
                q = q.filter(Event.detection_count >= min_count)
            # The significance gate is the default view: ephemeral events
            # (fragments below EventConfig's thresholds) stay hidden until
            # asked for. SQL-side, like every other filter, so paging
            # counts only rows the operator sees.
            hidden = 0
            if not show_ephemeral:
                hidden = session.scalar(
                    select(func.count()).select_from(
                        q.filter(not_(significance_clause())).subquery()
                    )
                )
                q = q.filter(significance_clause())

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
            "show_ephemeral": show_ephemeral,
            "min_conf": min_conf,
            "min_count": min_count,
            "kinds": kinds,
        }
        # Selecting a row is a filter-preserving link, and every chip
        # toggle drops the selection (the rail would outlive its row).
        chip_urls = {
            "needs_review": _triage_url(state, needs_review=not needs_review),
            "unmatched": _triage_url(state, unmatched=not unmatched),
            "show_ephemeral": _triage_url(state, show_ephemeral=not show_ephemeral),
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
                "identifier_keys": list(config.identity.identifiers.keys()),
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
                    "min_conf": min_conf if min_conf is not None else "",
                    "min_count": min_count if min_count is not None else "",
                },
                "chips": {
                    "needs_review": needs_review,
                    "unmatched": unmatched,
                    "show_ephemeral": show_ephemeral,
                    "kinds": kinds,
                },
                "chip_count": int(needs_review)
                + int(unmatched)
                + int(show_ephemeral)
                + len(kinds),
                "matched": matched,
                "total": total,
                "hidden": hidden,
                "selected": selected,
                "rail": rail,
                "page": page,
                "has_next": has_next,
                "prev_url": _triage_url(state, selected=selected, page=page - 1),
                "next_url": _triage_url(state, selected=selected, page=page + 1),
            },
        )

    @app.get("/search", response_class=HTMLResponse)
    def search(request: Request, q: str = ""):
        """One box over events, people and plates — the top bar's promise.

        Substring match per entity, ranked by recency. SQLite LIKE is
        case-insensitive for ASCII and these tables are PoC-sized;
        FTS5 is the V1 upgrade path once an archive gets big enough to
        feel it, and it slots in behind this same route.
        """
        term = q.strip()
        results: dict = {"identities": [], "events": [], "library": []}
        if term:
            like = f"%{term}%"
            with Session() as session:
                results["identities"] = (
                    session.scalars(
                        select(Identity)
                        .filter(
                            or_(
                                Identity.label.like(like),
                                Identity.plate.like(like),
                            )
                        )
                        .order_by(Identity.last_seen.desc())
                        .limit(20)
                    )
                    .unique()
                    .all()
                )
                event_q = (
                    select(Event)
                    .options(
                        selectinload(Event.camera),
                        selectinload(Event.identities).selectinload(
                            EventIdentity.identity
                        ),
                    )
                    .join(Camera, Event.camera_id == Camera.id)
                    .filter(
                        or_(
                            Event.class_name.like(like),
                            Camera.name.like(like),
                            # Events surface by whom they matched, too —
                            # searching a name should find the visits.
                            Event.id.in_(
                                select(EventIdentity.event_id)
                                .join(
                                    Identity,
                                    EventIdentity.identity_id == Identity.id,
                                )
                                .where(
                                    or_(
                                        Identity.label.like(like),
                                        Identity.plate.like(like),
                                    )
                                )
                            ),
                        )
                        if not term.isdigit()
                        else Event.id == int(term)
                    )
                    .order_by(Event.last_seen.desc())
                    .limit(20)
                )
                results["events"] = session.scalars(event_q).unique().all()
                from siteloom.store import LibraryItem

                results["library"] = (
                    session.scalars(
                        select(LibraryItem)
                        .filter(LibraryItem.path.like(like))
                        .order_by(LibraryItem.id.desc())
                        .limit(20)
                    )
                    .unique()
                    .all()
                )
        total = sum(len(v) for v in results.values())
        return templates.TemplateResponse(
            request,
            "search.html",
            {
                "site_name": config.site_name or config.site_id,
                "q": term,
                "results": results,
                "total": total,
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
                    "identifier_keys": list(config.identity.identifiers.keys()),
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
            rows = (
                session.scalars(
                    select(EventIdentity)
                    .options(selectinload(EventIdentity.identity))
                    .filter(
                        EventIdentity.event_id == event_id,
                        EventIdentity.identity_id.is_not(None),
                    )
                )
                .unique()
                .all()
            )
            identity_links = [r for r in rows if r.unlinked_at is None]
            unlinked = [r for r in rows if r.unlinked_at is not None]
            candidates = _identity_candidates(session)
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
                "unlinked": unlinked,
                "candidates": candidates,
                "identifier_keys": list(config.identity.identifiers.keys()),
            },
        )

    @app.get("/identities", response_class=HTMLResponse)
    def identities(
        request: Request,
        identifier: str | None = None,
        unlabeled: bool = False,
        unenrolled: bool = False,
        selected: int | None = None,
    ):
        with Session() as session:
            q = select(Identity).order_by(Identity.last_seen.desc())
            if identifier:
                q = q.filter(Identity.identifier_key == identifier)
            if unlabeled:
                q = q.filter(Identity.label.is_(None))
            if unenrolled:
                # Named, but with nothing in the vector store — recognition
                # cannot match on this person at all (identity/enroll.py).
                q = q.filter(Identity.label.is_not(None), Identity.vector_count == 0)
            rows = session.scalars(q.limit(200)).unique().all()
            identifier_keys = sorted(
                {k for (k,) in session.execute(select(Identity.identifier_key).distinct())}
            )
            total = session.scalar(select(func.count()).select_from(Identity)) or 0
            rail = _identity_rail(session, selected) if selected else None

        state = {
            "identifier": identifier,
            "unlabeled": unlabeled,
            "unenrolled": unenrolled,
        }
        return templates.TemplateResponse(
            request,
            "identities.html",
            {
                "site_name": config.site_name or config.site_id,
                "identities": rows,
                "identifier_keys": identifier_keys,
                "filters": {"identifier": identifier or "", "unlabeled": unlabeled},
                "chips": state,
                "chip_urls": {
                    "all": _identities_url({}),
                    "unlabeled": _identities_url(state, unlabeled=not unlabeled),
                    "unenrolled": _identities_url(state, unenrolled=not unenrolled),
                    "identifier": {
                        k: _identities_url(
                            state, identifier=None if identifier == k else k
                        )
                        for k in identifier_keys
                    },
                },
                "card_urls": {i.id: _identities_url(state, selected=i.id) for i in rows},
                "matched": len(rows),
                "total": total,
                "selected": selected,
                "rail": rail,
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
                    .filter(EventIdentity.unlinked_at.is_(None))
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

        A verdict is a judgment, not an edit: it must not touch the
        vector store — resolver-side learning from embeddings is separate
        work, and correcting the gallery is what link/unlink/reassign
        below are for. It does revert a plate this very match taught the
        identity: plate matches win outright (PRD §6.4), so a mis-learned
        plate would poison every future sighting of that number. That is
        correction, not learning, and it is scoped to exactly the
        evidence being repudiated."""
        if verdict not in ("confirmed", "wrong", "clear"):
            raise HTTPException(400, "verdict must be confirmed, wrong, or clear")
        with Session() as session:
            link = session.get(EventIdentity, link_id)
            if link is None or link.event_id != event_id or link.identity_id is None:
                raise HTTPException(404)
            if verdict == "clear":
                link.verdict = None
                link.verdict_at = None
            else:
                link.verdict = verdict
                link.verdict_at = datetime.now(timezone.utc).replace(tzinfo=None)
            if verdict == "wrong":
                identity_ops.revert_learned_plate(session, link, event_id)
            session.commit()
        return RedirectResponse(f"/events/{event_id}", status_code=303)

    def _resolve_target(
        session, event: Event, identity_id: str, identifier: str, label: str
    ) -> Identity:
        """The identity an attach/reassign names: an existing one, or a
        new one minted here.

        "New identity" is not a convenience — it is the answer when the
        resolver merged two people into one row, and the correct target
        does not exist yet. It starts unlabeled unless the operator typed
        a name, which is the same label-and-learn shape as everywhere
        else (PRD §6.3).
        """
        if identity_id == "new":
            if identifier not in config.identity.identifiers:
                raise HTTPException(
                    400,
                    f"unknown identifier {identifier!r}; expected one of "
                    + ", ".join(sorted(config.identity.identifiers)),
                )
            identity = Identity(
                identifier_key=identifier,
                class_name=event.class_name,
                label=label.strip() or None,
                first_seen=event.first_seen,
                last_seen=event.last_seen,
                best_crop_path=event.best_crop_path,
            )
            session.add(identity)
            session.flush()
            return identity
        if not identity_id.isdigit():
            raise HTTPException(400, "identity_id must be an integer or 'new'")
        identity = session.get(Identity, int(identity_id))
        if identity is None:
            raise HTTPException(404, f"no identity {identity_id}")
        return identity

    def _attach(session, vectors, event: Event, identity: Identity, enroll: bool):
        """Create (or revive) the operator's claim that this event is
        this identity, and make the claim visible to matching.

        A manual link is `matched_by="human"` and `verdict="confirmed"`
        by construction: the provenance of a correction matters as much
        as the provenance of a match (CLD-32), and an operator saying who
        someone is has already reviewed it. `similarity` stays 0.0 — a
        human link has no cosine score, and inventing one would corrupt
        the plate-vs-visual accuracy numbers that read this column.

        `vectors` is None when the caller never opened the store (an
        attach with enrollment off); nothing here reads or writes it
        then, and the identity's vector count is left as it was.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        link = (
            session.query(EventIdentity)
            .filter_by(event_id=event.id, identity_id=identity.id)
            .order_by(EventIdentity.id.desc())
            .first()
        )
        if link is None:
            link = EventIdentity(
                event_id=event.id,
                identity_id=identity.id,
                identifier_key=identity.identifier_key,
                similarity=0.0,
                matched_by="human",
            )
            session.add(link)
            new_claim = True
        else:
            # Re-attaching a previously unlinked claim revives that row
            # rather than stacking a second one: the operator changed
            # their mind about this pairing, which is one decision, not
            # two. Re-affirming a claim that already stands is not a new
            # sighting at all — counting it again, or enrolling the same
            # crop a second time, would let a stuck refresh inflate both.
            new_claim = link.unlinked_at is not None
        link.unlinked_at = None
        link.verdict = "confirmed"
        link.verdict_at = now
        enrolled = False
        if not new_claim:
            return link, enrolled
        if enroll and vectors is not None:
            enrolled = identity_ops.enroll_event_crop(
                config, vectors, event, identity, app.state.embedders
            )
        identity.last_seen = max(identity.last_seen, event.last_seen)
        identity.first_seen = min(identity.first_seen, event.first_seen)
        identity.appearance_count += 1
        if not identity.best_crop_path:
            identity.best_crop_path = event.best_crop_path
        if vectors is not None:
            identity_ops.refresh_vector_count(session, vectors, identity)
        return link, enrolled

    @app.post("/events/{event_id}/identity")
    def attach_identity(
        event_id: int,
        identity_id: str = Form(...),
        identifier: str = Form("face"),
        label: str = Form(""),
        enroll: str = Form("1"),
        next_url: str = Form(""),
    ):
        """Say who this event actually was (CLD-36).

        The other half of the verdict buttons: an operator could say a
        claim was wrong but never say what was right, so a wrong name
        kept rendering forever and the system learned nothing from the
        correction.

        Enrollment is on by default because a link the matcher cannot see
        fixes one event and leaves the next one to go wrong the same way
        — "a label without vectors is a name the system cannot see". It
        is still optional: a crop can be too poor to want in a gallery.
        """
        want_enroll = enroll == "1"
        with Session() as session:
            event = session.get(Event, event_id)
            if event is None:
                raise HTTPException(404)
            # Only reach for the store when enrollment needs it, so a
            # console with a backfill running can still record who
            # someone was — with enrollment off, which the operator sees.
            vectors = (
                identity_ops.shared_store(config, "link") if want_enroll else None
            )
            identity = _resolve_target(session, event, identity_id, identifier, label)
            _attach(session, vectors, event, identity, enroll=want_enroll)
            session.commit()
            target = identity.id
        return RedirectResponse(
            _safe_next(next_url, event_id) if next_url else f"/identities/{target}",
            status_code=303,
        )

    @app.post("/events/{event_id}/identity/{link_id}/unlink")
    def unlink_identity(event_id: int, link_id: int, next_url: str = Form("")):
        """Detach a claim this event should never have carried (CLD-36).

        The row survives with its identity, similarity and matched_by
        intact — that is the record of what the system got wrong, and
        negatives are data (the Annotation philosophy). What does not
        survive is the claim's effect on matching: the vectors this
        event taught that identity are removed from its gallery, because
        a polluted gallery keeps re-attracting the same wrong match, and
        a plate learned from this claim is reverted.

        This is the deliberate difference from a "wrong" verdict, which
        judges without editing: unlink is the operator asserting the
        pairing never held.
        """
        with Session() as session:
            link = session.get(EventIdentity, link_id)
            if link is None or link.event_id != event_id or link.identity_id is None:
                raise HTTPException(404)
            event = session.get(Event, event_id)
            identity = session.get(Identity, link.identity_id)
            vectors = identity_ops.shared_store(config, "unlink")
            removed = vectors.delete_by_crops(
                identity.identifier_key,
                identity.id,
                identity_ops.event_crop_paths(session, event),
            )
            log.info(
                "unlinked identity %s from event %s (%s vectors removed)",
                identity.id,
                event_id,
                removed,
            )
            link.unlinked_at = datetime.now(timezone.utc).replace(tzinfo=None)
            link.verdict = "wrong"
            link.verdict_at = link.unlinked_at
            identity_ops.revert_learned_plate(session, link, event_id)
            identity.appearance_count = max(identity.appearance_count - 1, 0)
            identity_ops.refresh_vector_count(session, vectors, identity)
            session.commit()
        return RedirectResponse(
            _safe_next(next_url, event_id) if next_url else f"/events/{event_id}",
            status_code=303,
        )

    @app.post("/events/{event_id}/identity/{link_id}/reassign")
    def reassign_identity(
        event_id: int,
        link_id: int,
        identity_id: str = Form(...),
        identifier: str = Form("face"),
        label: str = Form(""),
        next_url: str = Form(""),
    ):
        """Point a wrong claim at the right person, vectors and all.

        Unlink + attach as one decision, which is what makes it more
        than the sum: the vectors this event contributed *move* from the
        wrong gallery to the right one (they are traceable by the crop
        they came from — CLD-84), so the same correction that fixes the
        record also stops the wrong identity re-matching and starts the
        right one matching. Where nothing could be moved — a legacy
        vector with no provenance, or an event whose crops never entered
        a gallery — the event's best crop is enrolled instead, so the
        new claim is never invisible to matching.
        """
        with Session() as session:
            link = session.get(EventIdentity, link_id)
            if link is None or link.event_id != event_id or link.identity_id is None:
                raise HTTPException(404)
            event = session.get(Event, event_id)
            vectors = identity_ops.shared_store(config, "reassign")
            target = _resolve_target(session, event, identity_id, identifier, label)
            if target.id == link.identity_id:
                raise HTTPException(400, "that identity is already linked")
            old = session.get(Identity, link.identity_id)
            crops = identity_ops.event_crop_paths(session, event)
            moved = 0
            if old.identifier_key == target.identifier_key:
                # Across identifiers a vector cannot move: a face
                # embedding is not a vehicle embedding, and dropping one
                # into the other's collection would be a dimension
                # mismatch at best and silent nonsense at worst. Strip it
                # from the wrong gallery and let the attach enroll fresh.
                moved = vectors.move_by_crops(
                    old.identifier_key, old.id, target.id, crops
                )
            else:
                vectors.delete_by_crops(old.identifier_key, old.id, crops)
            link.unlinked_at = datetime.now(timezone.utc).replace(tzinfo=None)
            link.verdict = "wrong"
            link.verdict_at = link.unlinked_at
            identity_ops.revert_learned_plate(session, link, event_id)
            old.appearance_count = max(old.appearance_count - 1, 0)
            identity_ops.refresh_vector_count(session, vectors, old)
            _attach(session, vectors, event, target, enroll=moved == 0)
            session.commit()
            new_id = target.id
        return RedirectResponse(
            _safe_next(next_url, event_id) if next_url else f"/identities/{new_id}",
            status_code=303,
        )

    @app.post("/events/{event_id}/missed")
    def set_missed_identity(
        event_id: int,
        missed: str = Form(...),
        identifier: str = Form("face"),
    ):
        """Mark/unmark a missed identification, attributed to an identifier.

        A miss is a null-identity EventIdentity row (verdict="missed",
        identifier_key set) so per-identifier recall is computable
        (CLD-17) — "the face identifier missed the person" and "plate OCR
        missed the car" are different failures on the same event.
        `Event.missed_identity` mirrors "any miss rows exist" and this
        endpoint is its single writer.
        """
        with Session() as session:
            event = session.get(Event, event_id)
            if event is None:
                raise HTTPException(404)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            misses = (
                session.query(EventIdentity)
                .filter_by(event_id=event_id, identity_id=None)
                .all()
            )
            if missed == "1":
                if not any(m.identifier_key == identifier for m in misses):
                    session.add(
                        EventIdentity(
                            event_id=event_id,
                            identity_id=None,
                            identifier_key=identifier,
                            verdict="missed",
                            verdict_at=now,
                        )
                    )
                event.missed_identity = True
                event.missed_at = event.missed_at or now
            else:
                # Retracting the mark removes the miss rows — the same
                # semantics "clear" has for verdicts: an operator taking
                # back their own judgment, not deleting system evidence.
                for m in misses:
                    session.delete(m)
                event.missed_identity = False
                event.missed_at = None
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

    @app.get("/stats", response_class=HTMLResponse)
    def stats(request: Request, days: int = Query(1, ge=1, le=90)):
        """Accuracy readout over a window (CLD-17).

        Defaults to one day because the question this answers is "how did
        last night's soak go" — a lifetime average buries the run you are
        actually trying to judge.
        """
        from siteloom import stats as stats_mod

        since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        thresholds = {
            key: ident.threshold
            for key, ident in config.identity.identifiers.items()
        }
        with Session() as session:
            context = {
                "site_name": config.site_name or config.site_id,
                "days": days,
                "identifiers": stats_mod.identifier_stats(session, since=since),
                "cameras": stats_mod.camera_stats(session, since=since),
                "coverage": stats_mod.review_coverage(session, since=since),
                "histograms": stats_mod.similarity_histograms(
                    session, thresholds, since=since
                ),
                "thresholds": thresholds,
            }
        return templates.TemplateResponse(request, "stats.html", context)

    from siteloom.web import library_routes

    library_routes.register(app, templates, Session, config)

    if config.integrations.recognition_api.enabled:
        from siteloom.web import recognition_api

        service = recognition_service or recognition_api.RecognitionService(
            config, Session
        )
        recognition_api.register(app, config, service)

    # -- live view ---------------------------------------------------------

    from siteloom.web.live import LiveHub

    hub = LiveHub(config)
    app.router.on_shutdown.append(hub.stop)

    @app.get("/live", response_class=HTMLResponse)
    def live(request: Request):
        return templates.TemplateResponse(
            request,
            "live.html",
            {
                "site_name": config.site_name or config.site_id,
                "live_cameras": hub.cameras(),
            },
        )

    @app.get("/live/{camera_id}/stream.mjpeg")
    def live_stream(camera_id: str):
        """Shared-reader MJPEG stream (see web/live.py).

        Each open stream occupies one serving-threadpool thread for its
        whole lifetime — fine for an operator console's worth of tiles,
        not a public endpoint to embed forty times.
        """
        if not any(c.id == camera_id for c in hub.cameras()):
            raise HTTPException(404)

        def gen():
            for jpeg in hub.frames(camera_id):
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )

        return StreamingResponse(
            gen(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/live/{camera_id}/snapshot.jpg")
    def live_snapshot(camera_id: str):
        if not any(c.id == camera_id for c in hub.cameras()):
            raise HTTPException(404)
        jpeg = hub.snapshot(camera_id)
        if jpeg is None:
            raise HTTPException(503, detail="camera stream unavailable")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/media/{path:path}")
    def media(path: str):
        """Serve a stored crop, confined to media_dir.

        The request component is resolved as given, never joined onto
        media_root. That is deliberate: ingest stores crop_path with the
        media_dir prefix already on it, so the component is absolute
        whenever media_dir is — the normal case, not an attack. Joining
        would also be no defence at all, since Path("/a") / "/etc/passwd"
        is "/etc/passwd": an absolute component silently discards the
        base. Containment is therefore the only gate.

        That gate compares *resolved* paths with is_relative_to. Both
        halves matter. A string-prefix test would admit a sibling
        directory — "/var/media-x/secret" starts with "/var/media"
        (CLD-49) — and resolving the full path last, rather than
        trusting a resolved root, is what catches a symlink inside
        media_dir pointing outside it.
        """
        full = Path(path).resolve()
        if not full.is_relative_to(media_root) or not full.is_file():
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
