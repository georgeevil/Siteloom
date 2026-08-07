"""Operator web UI: browse events, filter by camera/class/time.

The label-and-learn workflow (naming unknown faces/vehicles) lands here
when the face/plate modules arrive; this slice ships the event browser
those workflows build on.
"""

from __future__ import annotations

import json
import logging
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
    Response,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, not_, or_, select
from sqlalchemy.orm import selectinload

from siteloom.config import SiteConfig, load_config
from siteloom.web import auth
from siteloom.store import (
    Camera,
    Detection,
    Event,
    EventIdentity,
    Identity,
    NoiseEvent,
    User,
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
    links = [r for r in rows if r.identity_id is not None]
    misses = [r for r in rows if r.identity_id is None]
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

    @app.middleware("http")
    async def auth_and_audit(request: Request, call_next):
        """One gate for the whole console (see web/auth.py).

        Also the one audit writer: every mutating request that reaches a
        handler leaves a row, so a new POST route cannot forget to audit.
        """
        path = request.url.path
        with Session() as session:
            enabled = auth.auth_enabled(session)
            user = auth.resolve_user(
                session, request.cookies.get(auth.SESSION_COOKIE)
            )
        request.state.user = user
        request.state.auth_enabled = enabled

        exempt = path in auth.PUBLIC_PATHS or path.startswith(auth.EXEMPT_PREFIXES)
        if enabled and not exempt:
            if user is None:
                if request.method in ("GET", "HEAD"):
                    return RedirectResponse(f"/login?next={path}", status_code=303)
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

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request, next: str = "/"):
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "site_name": config.site_name or config.site_id,
                "next": _safe_next(next, 0) if next != "/" else "/",
                "error": None,
            },
        )

    @app.post("/login", response_class=HTMLResponse)
    def login_submit(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
        next: str = Form("/"),
    ):
        with Session() as session:
            user = session.scalar(
                select(User).filter_by(username=username.strip())
            )
            if (
                user is None
                or user.disabled
                or not auth.verify_password(password, user.password_hash)
            ):
                # One message for both failures — do not confirm usernames.
                return templates.TemplateResponse(
                    request,
                    "login.html",
                    {
                        "site_name": config.site_name or config.site_id,
                        "next": next,
                        "error": "Wrong username or password.",
                    },
                    status_code=401,
                )
            token = auth.create_session(session, user)
            auth.record_audit(session, user, "POST", "/login", 303)
            session.commit()
        target = next if next.startswith("/") and not next.startswith("//") else "/"
        response = RedirectResponse(target, status_code=303)
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
            identity_links = (
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

        A wrong verdict must not touch the vector store — resolver-side
        learning from embeddings is separate work — but it does revert a
        plate this very match taught the identity: plate matches win
        outright (PRD §6.4), so a mis-learned plate would poison every
        future sighting of that number. That is correction, not learning,
        and it is scoped to exactly the evidence being repudiated."""
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
            if verdict == "wrong" and link.learned_plate:
                identity = session.get(Identity, link.identity_id)
                if identity is not None and identity.plate:
                    log.info(
                        "reverting plate %s learned on event %s from identity %s",
                        identity.plate,
                        event_id,
                        identity.id,
                    )
                    identity.plate = None
            session.commit()
        return RedirectResponse(f"/events/{event_id}", status_code=303)

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
