"""Operator web UI: browse events, filter by camera/class/time.

The label-and-learn workflow (naming unknown faces/vehicles) lands here
when the face/plate modules arrive; this slice ships the event browser
those workflows build on.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
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
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, not_, or_, select
from sqlalchemy.orm import selectinload

from siteloom.config import SiteConfig, load_config
from siteloom.web import auth, identity_ops, nav, paging, redaction
from siteloom.store import (
    Camera,
    Detection,
    Event,
    EventIdentity,
    Identity,
    NoiseEvent,
    PlateRead,
    User,
    WebSession,
    get_session,
    init_db,
    make_engine,
)
from siteloom.store import claims
from siteloom.store.models import (
    PLATE_SOURCE_OPERATOR,
    significance_clause,
    status_clause,
    unmatched_clause,
)

log = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

# Triage's class-kind chips. "other" is the residual rather than a fixed
# list, so a class added to detection.classes (or auto-added by the
# registry) stays reachable in the UI without a code change (NFR3).
CLASS_KINDS = {
    "people": {"person"},
    "vehicles": {"car", "truck", "bus", "motorcycle", "bicycle"},
    "other": set(),
}


@dataclass(frozen=True)
class EventColumn:
    """One column the events list can offer (CLD-108).

    The offer is data in the spirit of nav.py's NAV, not a template fork
    per column: index.html iterates this table for the header, the row
    cells, the picker's checkboxes and the per-column hide rules, so
    adding a column later is one entry here plus one cell branch in the
    row macro. `key` is the column's identity everywhere it travels — the
    `c-<key>` cell class, the `hide-<key>` class the picker toggles on
    the list container, and the token stored in localStorage — so it must
    stay CSS-class- and JSON-safe, and renaming one orphans saved
    preferences (they degrade to the default set, never to an error).
    """

    key: str
    label: str
    #: In the server-rendered default set. The default is today's
    #: composition plus Last seen — the sort column visible out of the
    #: box, nothing an operator relies on gone (CLD-108).
    default: bool
    #: The column the list is ordered by. Marked in the header — the sort
    #: stays on `last_seen` and the keyset cursor is untouched; this only
    #: makes that fact visible instead of leaving the operator to infer
    #: it from timestamps that stop being monotonic down the page.
    sort: bool = False


#: The events list's columns on offer, in render order. Status is not
#: here on purpose: the review-status chip is the triage signal itself,
#: the one cell the queue cannot mean anything without, so it is fixed
#: rather than offered.
EVENT_COLUMNS: tuple[EventColumn, ...] = (
    EventColumn("arrived", "Arrived", default=True),
    EventColumn("last-seen", "Last seen", default=True, sort=True),
    EventColumn("duration", "Duration", default=False),
    EventColumn("detection", "Detection", default=True),
    EventColumn("camera", "Camera", default=True),
    EventColumn("identity", "Identity", default=True),
    EventColumn("confidence", "Conf", default=True),
    EventColumn("detections", "Detections", default=False),
)

#: The columns the server hides by class on the list container. Every
#: row always renders every cell — visibility is CSS keyed off these
#: classes, which is what lets load-more fragments and the client-side
#: picker share one mechanism (the fragment inherits the container's
#: classes; the picker only edits them).
DEFAULT_HIDDEN_COLUMNS: tuple[str, ...] = tuple(
    c.key for c in EVENT_COLUMNS if not c.default
)


def duration_text(
    first_seen: datetime,
    last_seen: datetime,
    *,
    now: datetime | None = None,
    active_gap_s: float = 0.0,
) -> str:
    """An event's dwell as a duration the operator can read (CLD-108).

    "14 min", not two timestamps to subtract. Naive-UTC in, plain text
    out — a difference has no zone, so this deliberately bypasses
    `local_time` (there is nothing to convert; the *timestamps* still
    render through it).

    Renderings, decided here so they are testable:

    * under a second — "< 1 s". A single-frame event has no measured
      span; "0 s" would claim a measurement that was never made.
    * under a minute — whole seconds ("38 s").
    * under an hour — whole minutes ("14 min"), floored: triage wants
      magnitude, not precision.
    * an hour and up — "2 h 05", minutes zero-padded so columns align.
    * possibly still growing — a trailing " +" ("14 min +") when
      `last_seen` is within `active_gap_s` of `now`. The store has no
      open/closed bit — de-fragmentation can extend `last_seen` for as
      long as the subject keeps being seen — so "still active" is
      necessarily a horizon, and the honest horizon is the track-link
      gap: the window inside which ingest would still extend this very
      event. The marker claims "may still be growing", never "live".
    """
    span = (last_seen - first_seen).total_seconds()
    if span < 1:
        text = "< 1 s"
    elif span < 60:
        text = f"{int(span)} s"
    elif span < 3600:
        text = f"{int(span // 60)} min"
    else:
        text = f"{int(span // 3600)} h {int(span % 3600 // 60):02d}"
    ongoing = (
        now is not None
        and active_gap_s > 0
        and (now - last_seen).total_seconds() <= active_gap_s
    )
    return text + " +" if ongoing else text


# LIKE's own wildcards, and the character that turns them back into
# literals. SQLAlchemy's `.like()` emits the pattern verbatim and adds no
# ESCAPE clause, so both halves are the caller's job (CLD-67).
LIKE_ESCAPE = "\\"

# SQLite binds Python ints as signed 64-bit; a longer digit string (a
# phone number pasted into the search box) raises OverflowError out of
# the driver rather than simply matching nothing.
_MAX_ROW_ID = 2**63 - 1

# How long a /readyz answer is reused before the checks run again
# (CLD-54). The endpoint is public, so its cost must not scale with how
# hard someone hammers it; well under any sane probe interval, so a
# supervisor still sees a state change within one poll.
READYZ_CACHE_S = 5.0


def like_pattern(term: str) -> str:
    """A LIKE pattern matching `term` as a literal substring.

    Operators type `%` and `_` — in plates, unit numbers, filenames —
    and unescaped they are "any run" and "any character": searching
    "unit_4" would return "unita4", and a bare "%" would return the
    whole table. Pair with escape=LIKE_ESCAPE; the ESCAPE clause is not
    implied by the pattern.
    """
    for char in (LIKE_ESCAPE, "%", "_"):
        term = term.replace(char, LIKE_ESCAPE + char)
    return f"%{term}%"


def _contains(column, term: str):
    """`column` contains `term` literally."""
    return column.like(like_pattern(term), escape=LIKE_ESCAPE)


def _as_row_id(term: str) -> int | None:
    """`term` read as a row id, or None when it is not one."""
    if not term.isdigit():
        return None
    value = int(term)
    return value if 0 < value <= _MAX_ROW_ID else None


def _media_candidates(raw: str, media_root: Path) -> list[Path]:
    """Every location a stored media path can mean, likeliest first.

    Two shapes are current. An absolute path is taken as it stands —
    joining it onto media_root would be no-op anyway, since Path("/a") /
    "/etc/passwd" is "/etc/passwd" — and a relative one is anchored on
    media_root, which is the only reading that does not depend on where
    the process was started (CLD-64).

    Legacy rows carry a third: a relative path written when media_dir
    itself was relative, so the media_dir prefix is still on the front
    ("media/cam1/…" for media_dir "media"). Those are re-anchored by
    stripping the prefix rather than left permanently unreachable.
    Containment is re-checked on whatever this returns, so an extra
    candidate cannot widen what is servable.
    """
    candidate = Path(raw)
    if candidate.is_absolute():
        return [candidate]
    parts = candidate.parts
    root_parts = media_root.parts
    out = [media_root / candidate]
    for n in range(1, len(parts)):
        if root_parts[-n:] == parts[:n]:
            out.append(media_root.joinpath(*parts[n:]))
    return out


def _kind_clause(kind: str):
    named = CLASS_KINDS["people"] | CLASS_KINDS["vehicles"]
    if kind == "other":
        return Event.class_name.not_in(named)
    return Event.class_name.in_(CLASS_KINDS[kind])


def _has_plate_clause():
    """Events carrying at least one accepted plate read.

    An evidence facet, not a class kind: a plated event is already a
    vehicle, so this narrows (AND) like Needs review does, rather than
    widening like the kind chips."""
    return (
        select(PlateRead.id)
        .where(PlateRead.event_id == Event.id, PlateRead.accepted.is_(True))
        .exists()
    )


def _has_face_clause():
    """Events with a face-identifier match. Same facet shape as plates:
    the person kind says what was seen, this says the face pipeline
    recognised someone in it."""
    return (
        select(EventIdentity.id)
        .where(
            EventIdentity.event_id == Event.id,
            EventIdentity.identifier_key == "face",
            EventIdentity.identity_id.is_not(None),
        )
        .exists()
    )


#: How many rows each keyset-paged list hands over at a time. Sized to
#: the shape of the screen, not to a shared constant: a triage row is
#: 52px and an identity card is a grid tile.
EVENTS_PAGE = 50
IDENTITIES_PAGE = 60
NOISE_PAGE = 100

#: The query parameter carrying a keyset cursor. One name across the
#: console so `_with_cursor` is the only place that spells it.
CURSOR_PARAM = "after"


#: The relative-timeframe presets (observability convention: "the last
#: N", judged against request time). Spelled once: the route computes
#: the window from this table and the template renders its labels, so a
#: preset added here appears in the picker with no second list.
TIMEFRAME_PRESETS: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "6h": timedelta(hours=6),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}


def _triage_url(base: dict, **overrides) -> str:
    """A link to the events list with some filters changed.

    Toggling a chip must preserve every other filter — losing the camera
    or time window when you tick "Unmatched" makes the chips unusable —
    so links are built from the whole live filter state, not composed
    from the one field being changed.

    A cursor is deliberately not part of that state (CLD-104). It
    describes one client's position in one scroll, so a link built here
    — a chip, a row, the rail's "back" — opens at the top of the
    filtered set for whoever it is pasted to. Only `_with_cursor` adds
    one, and only to the load-more request. `last` *is* part of it —
    "the last 24h" is a filter, and a pasted link should mean the same
    living window for the recipient.
    """
    params: list[tuple[str, str]] = []
    merged = {**base, **overrides}
    for key in ("camera", "class", "last", "since", "until", "min_conf", "min_count"):
        if merged.get(key):
            params.append((key, str(merged[key])))
    for flag in ("needs_review", "unmatched", "has_plate", "has_face", "show_ephemeral"):
        if merged.get(flag):
            params.append((flag, "1"))
    for k in merged.get("kinds") or []:
        params.append(("kind", k))
    if merged.get("selected"):
        params.append(("selected", str(merged["selected"])))
    return "/?" + urlencode(params) if params else "/"


def _with_cursor(url: str, cursor: str) -> str:
    """The same filtered list, continued after `cursor`.

    Appended to a URL the filter helpers built rather than threaded
    through them, so every other link in the console stays cursor-free
    by construction.
    """
    sep = "&" if "?" in url else "?"
    return url + sep + urlencode({CURSOR_PARAM: cursor})


def _cursor_values(token: str | None, types: tuple[type, ...]) -> tuple | None:
    """A cursor decoded, or None when there is none — never a shrug.

    A token this server did not mint is a 400. Falling back to "no
    cursor" would restart the list from the top halfway down a scroll,
    and the operator sees rows they already worked appear again: that
    reads as duplicated data, not as a bad request.
    """
    if not token:
        return None
    try:
        return paging.decode_cursor(token, types)
    except paging.CursorError as exc:
        raise HTTPException(400, f"unreadable cursor: {exc}") from None


def _more(target: str, next_url: str | None, label: str, end: str) -> dict:
    """The load-more footer's context (templates/_load_more.html).

    `next_url` is None exactly when the list is exhausted, which is what
    lets the footer say "that is all of them" instead of going quiet —
    silence is indistinguishable from a failed fetch.
    """
    return {"target": target, "next_url": next_url, "label": label, "end": end}


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


def _viewer(request: Request) -> User | None:
    """The signed-in operator, or None for the open single-operator mode.

    The middleware sets this on every request, so None here means there
    are no User rows at all (auth is off) rather than "not signed in" —
    when auth is on, an anonymous request never reaches a handler. The
    getattr is for a handler exercised without the middleware; it answers
    open mode, which is what a console with no accounts is.
    """
    return getattr(request.state, "user", None)


def _rail_context(session, event_id: int, user: User | None, config) -> dict | None:
    """Everything the triage detail rail shows for one event.

    `user` is the viewer, and decides only whether the identity picker
    gets its options (`_identity_candidates`). The event's *own* matched
    identities are shown to everyone who may read the event — withholding
    those would leave nothing to judge; the leak this closes was the list
    of everybody else. `config` scopes the picker and the identifier
    selects to identifiers that consume this event's class — the offer
    side of the rule the attach POST enforces.
    """
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
    allowed = compatible_identifier_keys(config, event.class_name)
    return {
        "event": event,
        "camera": session.get(Camera, event.camera_id),
        "detections": detections,
        "identity_links": links,
        "misses": misses,
        "unlinked": unlinked,
        "candidates": _identity_candidates(session, user, allowed),
        # For the "new identity" / "missed by" selects: the configured
        # identifiers that consume this class, so the form cannot offer
        # what the POST refuses. (An auto-added key is not configured,
        # so a dynamically-added class falls back to its own key.)
        "identifier_options": sorted(
            k for k in config.identity.identifiers if k in allowed
        )
        or sorted(allowed),
        "zones": zones,
        "status": event.review_status,
    }


def compatible_identifier_keys(config, class_name: str) -> set[str]:
    """The identifier keys whose pipeline consumes this event class.

    The one compatibility rule for a manual association, shared by the
    picker (what to offer) and the attach POST (what to accept) so the
    two cannot drift. Keyed on the identifier, not on the Identity's own
    class_name: a "car" identity may legitimately claim a "truck" event
    because the vehicle identifier spans both — detector flapping across
    vehicle classes is normal. What it refuses is the cross-kind link: a
    person event attached to a vehicle identity poisons that gallery for
    every future match, and nothing in the UI said so.

    An identifier the registry auto-added is keyed by the class it was
    minted for (identity/registry.py), hence the class name itself.
    """
    keys = {
        key
        for key, ident in config.identity.identifiers.items()
        if class_name in ident.applies_to
    }
    keys.add(class_name)
    return keys


def _identity_candidates(
    session, user: User | None, allowed_keys: set[str] | None = None
) -> list[Identity]:
    """Who this event could be — the picker's options (CLD-36).

    Not filtered to one identifier — the face identifier missing someone
    is exactly when the operator attaches the person identity by hand —
    but it IS filtered to the identifiers that consume the event's class
    (`allowed_keys`): offering a vehicle identity on a person event
    invites the cross-kind link the attach POST now refuses. Labeled
    identities sort first because naming a visit is the common case; the
    unknown buckets stay reachable for merging two sightings of one
    stranger.

    Below the `/identities` floor the list is not built at all (CLD-103).
    The gate is here rather than at each call site because this is the
    one query that reads the directory: a screen added tomorrow that
    wants a picker gets the floor with it, and a template `{% if %}`
    would have shipped every name to the browser regardless.

    `redaction.may_name` is that floor, and the same question the name of
    a *matched* identity is put through (CLD-111) — "may this viewer be
    told who someone is" is one decision, not one per surface.
    """
    if not redaction.may_name(user):
        return []
    q = select(Identity).order_by(
        Identity.label.is_(None), Identity.label, Identity.last_seen.desc()
    )
    if allowed_keys is not None:
        q = q.where(Identity.identifier_key.in_(allowed_keys))
    return list(session.scalars(q.limit(200)).unique().all())


def create_app(config: SiteConfig, recognition_service=None) -> FastAPI:
    app = FastAPI(title="Siteloom")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    # The sidebar is data (web/nav.py) and reached as a Jinja global, so a
    # screen cannot render chrome-less by forgetting to pass it.
    templates.env.globals["nav_items"] = nav.items
    # The sidebar names the rung the way the rest of the console does —
    # "Writer", not the stored `edit` (CLD-103). A global rather than
    # per-screen context for the same reason nav_items is one: base.html
    # renders on every page and must not depend on each of them
    # remembering to pass it.
    templates.env.globals["role_label"] = auth.role_label
    # What this viewer may be told about who someone is (CLD-111). Bound
    # as globals for the same reason nav_items is: these are called from
    # templates owned by four modules, and they resolve the viewer from
    # the request themselves so no screen can render a name by forgetting
    # to ask.
    redaction.install(templates)
    # Every timestamp the console shows goes through `local_time` — the
    # display side of the naive-UTC contract (CLD-100, siteloom/localtime).
    # Same written-once shape as redaction: a screen added later inherits
    # site-local rendering instead of re-deciding it with a bare strftime.
    # The zone is resolved from the live config per render, so an admin
    # edit on /classes applies without a restart; per-screen formats stay
    # the screen's own (the helper owns the zone, not a format string).
    from siteloom import localtime

    def local_time(value, fmt: str = localtime.DEFAULT_FORMAT) -> str:
        return localtime.display(value, localtime.site_zone(config), fmt)

    templates.env.globals["local_time"] = local_time
    templates.env.globals["site_tz"] = lambda: localtime.zone_name(config)
    templates.env.globals["site_tz_source"] = lambda: localtime.source_label(config)
    templates.env.globals["site_tz_set"] = lambda: bool(config.timezone)
    templates.env.globals["site_tz_options"] = localtime.zone_options
    # Vendored fonts and anything else the console needs to render without
    # reaching the internet (CLD-86). Mounted rather than routed: this
    # serves a fixed directory shipped with the package, not user paths,
    # so it needs none of the containment /media does.
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    # Embedders are built lazily and cached for the app's lifetime (model
    # load is expensive); on the app rather than the module so the cache
    # cannot outlive the config it was built from.
    app.state.embedders = {}
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    # load_config has already anchored a relative media_dir to the config
    # file's directory (CLD-64); resolve() here only settles symlinks and
    # covers a config built in-process, which has no file to anchor to.
    media_root = Path(config.storage.media_dir).expanduser().resolve()

    def cover_src(path: str | None) -> str | None:
        """`/media/...` for a crop that is actually there, else None.

        Reuses `_media_candidates` — the same resolution and containment
        the /media route applies — so a cover this says is renderable is
        one that route will serve, and the two cannot drift into a
        broken-image icon that only shows up in production.

        Needed because `purge_window` deletes crop FILES for the events
        it removes while leaving Identity rows standing: the identity
        keeps a path to a file that is gone. Event crops cannot reach
        that state — their rows go in the same pass — which is why this
        guards the identity covers and nothing else.
        """
        if not path:
            return None
        for candidate in _media_candidates(path, media_root):
            full = candidate.resolve()
            if full.is_relative_to(media_root) and full.is_file():
                return f"/media/{path}"
        return None

    templates.env.globals["cover_src"] = cover_src
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
        # CSRF (CLD-58): a browser's cross-site mutation is refused first,
        # before the session is even resolved, so a new POST route is
        # covered by default — same reasoning as the auth gate itself.
        # Deliberately independent of auth: it holds in the open
        # single-operator mode too, where there is no cookie to ride but
        # every mutation is otherwise anyone's. /api/v1/ is exempt — its
        # clients authenticate per-request with x-api-key, not an ambient
        # cookie, so they are not CSRF-reachable and a browser-provenance
        # demand would break Double Take. Refusals are not audited:
        # nothing happened, the same rule as a role denial.
        if request.method not in auth.SAFE_METHODS and not path.startswith(
            auth.EXEMPT_PREFIXES
        ):
            reason = auth.cross_site_reason(
                request.method,
                request.headers.get("origin"),
                request.headers.get("sec-fetch-site"),
                request.headers.get("host"),
                request.url.scheme,
                request.headers.get("x-forwarded-proto"),
                request.headers.get("x-forwarded-host"),
            )
            if reason is not None:
                log.warning("refused %s %s: %s", request.method, path, reason)
                return JSONResponse(
                    {"detail": "cross-site request refused"}, status_code=403
                )
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

        if request.method not in auth.SAFE_METHODS and not exempt:
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
        # HttpOnly: script must never read the token. SameSite=Lax: the
        # cookie rides top-level navigations only — the belt to the
        # middleware's cross-site check. Secure exactly when the browser
        # arrived over HTTPS (directly or attested by a proxy's
        # X-Forwarded-Proto): hardcoding it would break the plain-HTTP
        # LAN login that is the current deployment (CLD-58).
        response.set_cookie(
            auth.SESSION_COOKIE,
            token,
            httponly=True,
            samesite="lax",
            secure=auth.cookie_secure(
                request.url.scheme, request.headers.get("x-forwarded-proto")
            ),
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
        # Same attributes as set_cookie above: a deletion only reaches
        # the cookie it names, attributes included.
        response.delete_cookie(
            auth.SESSION_COOKIE,
            httponly=True,
            samesite="lax",
            secure=auth.cookie_secure(
                request.url.scheme, request.headers.get("x-forwarded-proto")
            ),
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        camera: str | None = None,
        class_name: str | None = Query(None, alias="class"),
        last: str | None = None,
        since: str | None = None,
        until: str | None = None,
        needs_review: bool = False,
        unmatched: bool = False,
        has_plate: bool = False,
        has_face: bool = False,
        show_ephemeral: bool = False,
        min_conf: float | None = None,
        min_count: int | None = None,
        kind: list[str] = Query(default=[]),
        selected: int | None = None,
        after: str | None = None,
    ):
        page_size = EVENTS_PAGE
        kinds = [k for k in kind if k in CLASS_KINDS]
        # The sort key, spelled once: the cursor must carry every column
        # the ORDER BY does or it drops rows. Sampled cameras close many
        # tracks on the same frame, so `last_seen` ties constantly and a
        # cursor on it alone would skip every row tying with the last one
        # delivered — hence the id tie-break (CLD-104).
        sort = (Event.last_seen, Event.id)
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
                .order_by(Event.last_seen.desc(), Event.id.desc())
            )
            if camera:
                q = q.filter(Event.camera_id == camera)
            if class_name:
                q = q.filter(Event.class_name == class_name)
            # The timeframe. A preset (`last=24h`) wins over the absolute
            # inputs: it is a living window judged at request time, which
            # is what a pasted link should keep meaning. Unknown tokens
            # fall through to "all time" rather than 400 — the picker is
            # the only thing that mints them.
            if last and last in TIMEFRAME_PRESETS:
                q = q.filter(
                    Event.last_seen
                    >= datetime.now(timezone.utc).replace(tzinfo=None)
                    - TIMEFRAME_PRESETS[last]
                )
            else:
                last = None
                # Operator-typed wall-clock bounds are an input boundary
                # (CLD-100): converted site-local -> naive UTC here, not
                # compared raw against UTC columns as they briefly were.
                zone = localtime.site_zone(config)
                if since:
                    q = q.filter(
                        Event.last_seen
                        >= localtime.as_utc(datetime.fromisoformat(since), zone)
                    )
                if until:
                    q = q.filter(
                        Event.first_seen
                        <= localtime.as_utc(datetime.fromisoformat(until), zone)
                    )
            # Triage chips. The state chips narrow (AND); the class-kind
            # chips are alternatives (OR), so ticking People and Vehicles
            # widens the list rather than emptying it.
            if needs_review:
                q = q.filter(not_(status_clause("cleared")))
            if unmatched:
                q = q.filter(unmatched_clause())
            if has_plate:
                q = q.filter(_has_plate_clause())
            if has_face:
                q = q.filter(_has_face_clause())
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
            # Counted over the filtered set, not the sliced one: "1 of 12"
            # while showing 50 rows would be a worse lie than the OFFSET
            # this replaces. The cursor narrows what is rendered, never
            # what is counted.
            matched = session.scalar(
                select(func.count()).select_from(q.subquery())
            )
            values = _cursor_values(after, (datetime, int))
            rows_q = q if values is None else q.where(paging.after(sort, values))
            # One row more than the page, so "is there more" is answered
            # by the same query that produced the page — a second COUNT
            # can disagree with it while ingest writes between the two.
            events = (
                session.scalars(rows_q.limit(page_size + 1)).unique().all()
            )
            slice_ = paging.take(
                list(events), page_size, lambda e: (e.last_seen, e.id)
            )
            events = slice_.rows
            cameras = session.scalars(select(Camera)).all()
            classes = sorted(
                {c for (c,) in session.execute(select(Event.class_name).distinct())}
            )
            # The rail is server-rendered so a deep link works without JS;
            # the fragment endpoint below swaps it in place when JS is on.
            rail = (
                _rail_context(session, selected, _viewer(request), config)
                if selected
                else None
            )

        state = {
            "camera": camera,
            "class": class_name,
            "last": last,
            "since": since,
            "until": until,
            "needs_review": needs_review,
            "unmatched": unmatched,
            "has_plate": has_plate,
            "has_face": has_face,
            "show_ephemeral": show_ephemeral,
            "min_conf": min_conf,
            "min_count": min_count,
            "kinds": kinds,
        }
        # The timeframe picker's links: each preset keeps every other
        # filter, replaces the absolute window (the two would silently
        # AND together), and drops the selection like any chip does.
        timeframe_urls = {
            key: _triage_url(state, last=key, since=None, until=None)
            for key in TIMEFRAME_PRESETS
        }
        timeframe_urls["all"] = _triage_url(state, last=None, since=None, until=None)
        # Selecting a row is a filter-preserving link, and every chip
        # toggle drops the selection (the rail would outlive its row).
        chip_urls = {
            "needs_review": _triage_url(state, needs_review=not needs_review),
            "unmatched": _triage_url(state, unmatched=not unmatched),
            "has_plate": _triage_url(state, has_plate=not has_plate),
            "has_face": _triage_url(state, has_face=not has_face),
            "show_ephemeral": _triage_url(state, show_ephemeral=not show_ephemeral),
        }
        for k in CLASS_KINDS:
            chip_urls[k] = _triage_url(
                state, kinds=[x for x in kinds if x != k] if k in kinds else kinds + [k]
            )
        # The load-more link is this same view, filters and selection
        # intact, continued after the last row actually delivered. It is
        # a real href: JavaScript only intercepts the click, so a client
        # without it walks the list one navigation at a time.
        page_url = _triage_url(state, selected=selected)
        more = _more(
            "#event-rows",
            None if slice_.exhausted else _with_cursor(page_url, slice_.next_cursor),
            f"Load {page_size} more events",
            f"End of the list — {matched} event"
            f"{'' if matched == 1 else 's'} match these filters.",
        )
        # Bound per request so "still growing" is judged against the
        # moment this page rendered, with the same horizon ingest itself
        # uses to extend an event (EventConfig.track_link_gap_s).
        page_now = datetime.now(timezone.utc).replace(tzinfo=None)

        def _duration(first_seen: datetime, last_seen: datetime) -> str:
            return duration_text(
                first_seen,
                last_seen,
                now=page_now,
                active_gap_s=config.events.track_link_gap_s,
            )

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "columns": EVENT_COLUMNS,
                "hidden_columns": DEFAULT_HIDDEN_COLUMNS,
                "duration_text": _duration,
                "chip_urls": chip_urls,
                "identifier_keys": list(config.identity.identifiers.keys()),
                "row_urls": {e.id: _triage_url(state, selected=e.id) for e in events},
                "back_url": _triage_url(state, selected=selected),
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
                    "last": last or "",
                    "since": since or "",
                    "until": until or "",
                    "min_conf": min_conf if min_conf is not None else "",
                    "min_count": min_count if min_count is not None else "",
                },
                "chips": {
                    "needs_review": needs_review,
                    "unmatched": unmatched,
                    "has_plate": has_plate,
                    "has_face": has_face,
                    "show_ephemeral": show_ephemeral,
                    "kinds": kinds,
                },
                "chip_count": int(needs_review)
                + int(unmatched)
                + int(has_plate)
                + int(has_face)
                + int(show_ephemeral)
                + len(kinds),
                "timeframe_urls": timeframe_urls,
                "timeframe_presets": list(TIMEFRAME_PRESETS),
                "matched": matched,
                "total": total,
                "hidden": hidden,
                "selected": selected,
                "rail": rail,
                "more": more,
            },
        )

    @app.get("/search", response_class=HTMLResponse)
    def search(request: Request, q: str = ""):
        """One box over events, people and plates — the top bar's promise.

        Substring match per entity, ranked by recency — literal
        substrings, via _contains: what the operator typed is a string,
        never a pattern. SQLite LIKE is case-insensitive for ASCII and
        these tables are PoC-sized; FTS5 is the V1 upgrade path once an
        archive gets big enough to feel it, and it slots in behind this
        same route.
        """
        term = q.strip()
        results: dict = {"identities": [], "events": [], "library": []}
        if term:
            named = or_(
                _contains(Identity.label, term), _contains(Identity.plate, term)
            )
            with Session() as session:
                results["identities"] = (
                    session.scalars(
                        select(Identity)
                        .filter(named)
                        .order_by(Identity.last_seen.desc())
                        .limit(20)
                    )
                    .unique()
                    .all()
                )
                matches = [
                    _contains(Event.class_name, term),
                    _contains(Camera.name, term),
                    # Events surface by whom they matched, too —
                    # searching a name should find the visits.
                    Event.id.in_(
                        select(EventIdentity.event_id)
                        .join(Identity, EventIdentity.identity_id == Identity.id)
                        .where(named)
                    ),
                ]
                event_id = _as_row_id(term)
                if event_id is not None:
                    # The event-number fast path stays — typing "412"
                    # must land on event 412 — but it is one more way to
                    # match, not a replacement for the others: "7" is
                    # equally a plate fragment, a camera name and a unit
                    # number, and answering only the id reads as "no such
                    # visit" for all of them (CLD-67).
                    matches.insert(0, Event.id == event_id)
                event_q = (
                    select(Event)
                    .options(
                        selectinload(Event.camera),
                        selectinload(Event.identities).selectinload(
                            EventIdentity.identity
                        ),
                    )
                    .join(Camera, Event.camera_id == Camera.id)
                    .filter(or_(*matches))
                    .order_by(Event.last_seen.desc())
                    .limit(20)
                )
                results["events"] = session.scalars(event_q).unique().all()
                from siteloom.store import LibraryItem

                results["library"] = (
                    session.scalars(
                        select(LibraryItem)
                        .filter(_contains(LibraryItem.path, term))
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
            rail = _rail_context(session, event_id, _viewer(request), config)
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
            # The full event page renders the same picker as the rail, so
            # it takes the same floor (CLD-103) — one leak with two URLs.
            candidates = _identity_candidates(session, _viewer(request))
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
        after: str | None = None,
    ):
        # This screen used to take `LIMIT 200` and say nothing about it,
        # so on a site with more identities than that the 201st did not
        # exist as far as the console was concerned — the same failure
        # CLD-26 fixed on /noise. It is now walkable to the end, and the
        # count in the toolbar is the real one (CLD-104).
        sort = (Identity.last_seen, Identity.id)
        with Session() as session:
            q = select(Identity).order_by(
                Identity.last_seen.desc(), Identity.id.desc()
            )
            if identifier:
                q = q.filter(Identity.identifier_key == identifier)
            if unlabeled:
                q = q.filter(Identity.label.is_(None))
            if unenrolled:
                # Named, but with nothing in the vector store — recognition
                # cannot match on this person at all (identity/enroll.py).
                q = q.filter(Identity.label.is_not(None), Identity.vector_count == 0)
            matched = session.scalar(select(func.count()).select_from(q.subquery())) or 0
            values = _cursor_values(after, (datetime, int))
            rows_q = q if values is None else q.where(paging.after(sort, values))
            fetched = session.scalars(rows_q.limit(IDENTITIES_PAGE + 1)).unique().all()
            slice_ = paging.take(
                list(fetched), IDENTITIES_PAGE, lambda i: (i.last_seen, i.id)
            )
            rows = slice_.rows
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
                "matched": matched,
                "total": total,
                "selected": selected,
                "rail": rail,
                "more": _more(
                    "#identity-cards",
                    None
                    if slice_.exhausted
                    else _with_cursor(
                        _identities_url(state, selected=selected),
                        slice_.next_cursor,
                    ),
                    f"Load {IDENTITIES_PAGE} more identities",
                    f"End of the list — {matched} identit"
                    f"{'y' if matched == 1 else 'ies'} match this filter.",
                ),
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
            cover_options = identity_ops.cover_candidates(session, identity)
        return templates.TemplateResponse(
            request,
            "identity.html",
            {
                "site_name": config.site_name or config.site_id,
                "identity": identity,
                "links": links,
                "annotations": annotations,
                "merge_candidates": merge_candidates,
                "cover_options": cover_options,
                # Whether this identifier does plates at all. The section
                # also renders without it when a plate is already set — a
                # plate learned under an older config has to stay
                # removable (CLD-134).
                "plate_ocr": bool(
                    getattr(
                        config.identity.identifiers.get(identity.identifier_key),
                        "plate_ocr",
                        False,
                    )
                ),
            },
        )

    def _plate_identity(session, identity_id: int) -> Identity:
        identity = session.get(Identity, identity_id)
        if identity is None:
            raise HTTPException(404)
        return identity

    @app.post("/identities/{identity_id}/plate")
    def set_identity_plate(
        identity_id: int,
        plate: str = Form(""),
        confirm: str = Form(""),
        relearn: str = Form(""),
    ):
        """Set, overwrite or clear this identity's plate (CLD-134).

        The resolver is write-once: it fills an empty plate and never
        touches a full one, which left a junk plate — `111111` on a
        vehicle whose real plate is TYB506 — unfixable from the console
        even after the claim that taught it was marked wrong. An operator
        outranks OCR, so this is the one path that may overwrite.

        Clearing is deliberately available even when the identifier no
        longer does plates: a plate learned under an older config must
        still be removable, which is the whole complaint. Setting one is
        not — a face identity has no plate, and letting the console write
        one would put a value where nothing reads it.

        A clear is remembered as an operator act (`plate_source`), which
        is what makes it stick: an empty plate is exactly the condition
        the resolver's learn path fires on, so without that mark the next
        sighting re-learns the junk and the correction undoes itself.
        `relearn` is the opposite request — hand this vehicle back to
        automatic learning — and so clears the mark instead of setting
        it.

        Clearing an identity that never had a plate is therefore not the
        no-op it looks like: it locks that vehicle out of automatic plate
        learning, deliberately, because "never put a plate on this thing"
        is a thing an operator may well mean. It says so on the page
        afterwards, and the relearn button appears in exactly that state,
        so the lock is visible and one click from being undone.
        """
        with Session() as session:
            identity = _plate_identity(session, identity_id)
            wants_plate = bool(plate and plate.strip())
            if wants_plate:
                ident_cfg = config.identity.identifiers.get(identity.identifier_key)
                if ident_cfg is None or not ident_cfg.plate_ocr:
                    raise HTTPException(
                        400,
                        f"{identity.identifier_key!r} identities do not carry "
                        "plates — only clearing one is allowed",
                    )
            identity_ops.set_identity_plate(
                session, identity, plate, confirm=bool(confirm)
            )
            if not wants_plate and relearn:
                # Not "an operator set this to nothing" but "stop treating
                # my last answer as final" — the only way back to
                # automatic learning once a clear has locked it.
                identity.plate_source = None
            session.commit()
        return RedirectResponse(f"/identities/{identity_id}", status_code=303)

    @app.post("/identities/{identity_id}/plate/move")
    def move_identity_plate(
        identity_id: int,
        target_id: int = Form(...),
        confirm: str = Form(""),
    ):
        """Move a plate from one identity to another (CLD-134).

        The correction for a plate learned onto the wrong vehicle, which
        is two edits everywhere else: clearing it here and typing it
        there, with a window in between where the plate belongs to
        nobody and the two halves can be left half-done.

        Both ends end up operator-owned, the source included. The
        operator has just said this plate is not that vehicle's, so
        letting the resolver re-learn the same string onto it would be
        the same mistake by another route.
        """
        with Session() as session:
            identity = _plate_identity(session, identity_id)
            if target_id == identity_id:
                raise HTTPException(400, "cannot move a plate to the same identity")
            target = session.get(Identity, target_id)
            if target is None:
                raise HTTPException(404)
            if target.identifier_key != identity.identifier_key:
                # Same rule, and the same reason, as merge_identity's: a
                # vehicle plate on a face identity is nonsense the store
                # should not hold.
                raise HTTPException(
                    400, "identities from different identifiers cannot swap plates"
                )
            if not identity.plate:
                raise HTTPException(400, "this identity has no plate to move")
            if target.plate and target.plate != identity.plate and not confirm:
                raise HTTPException(
                    409,
                    f"{target.display_name} already carries {target.plate} — "
                    f"confirm to replace it with {identity.plate}",
                )
            target.plate = identity.plate
            target.plate_source = PLATE_SOURCE_OPERATOR
            identity.plate = None
            identity.plate_source = PLATE_SOURCE_OPERATOR
            session.commit()
        # The operator's attention has moved with the plate.
        return RedirectResponse(f"/identities/{target_id}", status_code=303)

    @app.post("/events/{event_id}/identity/{link_id}/verdict")
    def set_identity_verdict(
        event_id: int,
        link_id: int,
        verdict: str = Form(...),
        next_url: str = Form("/"),
    ):
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
        # Back to the filtered list, like the clear/reopen form beside it.
        # Verdicts are filed in runs — the runbook's scenario B is "work
        # the needs-review chip" — and a fixed /events/{id} redirect drops
        # the operator's filters and their place in the queue on every
        # single click.
        return RedirectResponse(_safe_next(next_url, event_id), status_code=303)

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
        # The compatibility rule, enforced where the write happens: the
        # picker only OFFERS compatible identities, but a crafted or
        # stale form must not get past it — a person event attached to a
        # vehicle identity poisons that gallery for every future match,
        # and unwinding it is identity surgery. Same set the picker was
        # built from (`compatible_identifier_keys`), so offer and
        # enforcement cannot drift.
        allowed = compatible_identifier_keys(config, event.class_name)
        if identity_id == "new":
            if identifier not in config.identity.identifiers:
                raise HTTPException(
                    400,
                    f"unknown identifier {identifier!r}; expected one of "
                    + ", ".join(sorted(config.identity.identifiers)),
                )
            if identifier not in allowed:
                raise HTTPException(
                    400,
                    f"identifier {identifier!r} does not apply to "
                    f"{event.class_name!r} events; expected one of "
                    + ", ".join(sorted(allowed)),
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
        if identity.identifier_key not in allowed:
            raise HTTPException(
                400,
                f"identity {identity.id} is a {identity.identifier_key!r} "
                f"identity and cannot claim a {event.class_name!r} event — "
                "cross-kind links poison the gallery for future matching.",
            )
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
        # The standing claim first, and only then the newest unlinked one.
        # Taking the newest row in *any* state — as this did — revives a
        # repudiation that a live claim already supersedes, which is a
        # second active row for the pair (CLD-133).
        link = claims.active_claim(session, event.id, identity.id)
        if link is not None:
            # Re-affirming a claim that already stands is not a new
            # sighting at all — counting it again, or enrolling the same
            # crop a second time, would let a stuck refresh inflate both.
            new_claim = False
        else:
            unlinked = (
                session.query(EventIdentity)
                .filter_by(event_id=event.id, identity_id=identity.id)
                .filter(EventIdentity.unlinked_at.is_not(None))
                .order_by(EventIdentity.id.desc())
                .first()
            )
            if unlinked is not None:
                # Re-attaching a previously unlinked claim revives that
                # row rather than stacking a second one: the operator
                # changed their mind about this pairing, which is one
                # decision, not two.
                link, new_claim = claims.revive_claim(session, unlinked)
            else:
                link, new_claim = claims.insert_claim(
                    session,
                    EventIdentity(
                        event_id=event.id,
                        identity_id=identity.id,
                        identifier_key=identity.identifier_key,
                        similarity=0.0,
                        matched_by="human",
                    ),
                    # A double-submit that loses the race leaves
                    # appearance_count alone (new_claim is False), so the
                    # winner must not gain a frame either — the two move
                    # together or the counter stops meaning anything.
                    count_sighting=False,
                )
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
            vectors = identity_ops.shared_store(config, "unlink")
            removed = identity_ops.unlink_claim(
                session,
                vectors,
                event,
                link,
                when=datetime.now(timezone.utc).replace(tzinfo=None),
            )
            log.info(
                "unlinked identity %s from event %s (%s vectors removed)",
                link.identity_id,
                event_id,
                removed,
            )
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
            # The old identity loses this event's crops, so it may lose
            # its cover with them (CLD-137). Its own line rather than a
            # call to unlink_claim: reassign duplicates those steps
            # deliberately, because it moves vectors where unlink deletes
            # them. Flushed first so the candidate query cannot see the
            # link just detached.
            session.flush()
            identity_ops.recompute_cover(session, old, dropped=crops)
            _attach(session, vectors, event, target, enroll=moved == 0)
            session.commit()
            new_id = target.id
        return RedirectResponse(
            _safe_next(next_url, event_id) if next_url else f"/identities/{new_id}",
            status_code=303,
        )

    @app.post("/events/{event_id}/multi-subject")
    def set_multi_subject(
        event_id: int,
        multi: str = Form(...),
        next_url: str = Form("/"),
    ):
        """Mark this event as containing more than one subject (CLD-103).

        A different statement from every other control on this page. A
        verdict grades one identity claim; this says the *event* is wrong
        — its crops are not all the same subject — so no claim on it can
        be right and grading them individually is wasted work.

        It is also the cheapest corpus contribution available. The mark
        carries a camera and a time window, which is exactly what
        `scripts/track_ab.py` needs to define a clip, and the operator is
        already looking at the event when they make it.

        Identity links are deliberately left alone: this records a
        tracking failure, and deciding which of the claims to keep is a
        separate judgement the operator may not want to make now.
        """
        with Session() as session:
            event = session.get(Event, event_id)
            if event is None:
                raise HTTPException(404)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            if multi == "1":
                event.multi_subject = True
                event.multi_subject_at = event.multi_subject_at or now
            else:
                event.multi_subject = False
                event.multi_subject_at = None
            session.commit()
        return RedirectResponse(_safe_next(next_url, event_id), status_code=303)

    @app.post("/events/{event_id}/identity/bulk")
    def bulk_identity_action(
        event_id: int,
        action: str = Form(...),
        keep_id: int | None = Form(None),
        next_url: str = Form("/"),
    ):
        """Detach many identity claims at once (CLD-103).

        A track that absorbed two people accumulates a dozen claims, and
        clearing them one form at a time is the work this endpoint
        removes — event 1401 had fifteen.

        Unlinking, not deleting: `unlinked_at` is set and the row stays,
        because "this event was once called someone else" is exactly what
        a reviewer needs to see later, and re-attaching is how a
        correction gets corrected (CLD-36).
        """
        if action not in ("unlink_all", "unlink_wrong", "keep_only"):
            raise HTTPException(400, "unknown bulk action")
        with Session() as session:
            event = session.get(Event, event_id)
            if event is None:
                raise HTTPException(404)
            links = [
                link
                for link in session.query(EventIdentity)
                .filter_by(event_id=event_id)
                .all()
                if link.is_active
            ]
            if action == "keep_only":
                if keep_id is None:
                    raise HTTPException(400, "keep_only needs keep_id")
                if not any(link.id == keep_id for link in links):
                    raise HTTPException(404, "no such active link on this event")
                targets = [link for link in links if link.id != keep_id]
            elif action == "unlink_wrong":
                targets = [link for link in links if link.verdict == "wrong"]
            else:
                targets = links

            now = datetime.now(timezone.utc).replace(tzinfo=None)
            vectors = identity_ops.shared_store(config, "bulk-unlink")
            for link in targets:
                identity_ops.unlink_claim(
                    session, vectors, event, link, when=now
                )
            session.commit()
            log.info(
                "bulk %s on event %s: detached %d of %d claims",
                action, event_id, len(targets), len(links),
            )
        return RedirectResponse(_safe_next(next_url, event_id), status_code=303)

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

    @app.post("/identities/{identity_id}/cover")
    def set_identity_cover(identity_id: int, crop_path: str = Form("")):
        """Choose the crop that represents this identity (CLD-137).

        The cover used to be first-write-wins at every writer, so an
        identity founded by a wrong match wore that match's face in every
        list on the console, permanently — unlinking the bad event did
        not take the picture with it.

        Choosing one locks it, which is what keeps automatic recompute
        off an operator's decision. An empty submission is the undo: it
        unlocks and re-derives, deliberately the same shape as CLD-134's
        `relearn`, so an operator who made a bad choice hands the field
        back to the system rather than hunting for the crop the system
        would have picked.
        """
        with Session() as session:
            identity = session.get(Identity, identity_id)
            if identity is None:
                raise HTTPException(404)
            chosen = crop_path.strip()
            if chosen:
                if not identity_ops.owns_crop(session, identity, chosen):
                    # This screen renders the cover as /media/{path}, so
                    # an unchecked field would point one identity's cover
                    # at anything that route will serve.
                    raise HTTPException(
                        400,
                        f"{chosen!r} is not a crop of {identity.display_name} — "
                        "choose one of this identity's own sightings",
                    )
                identity.best_crop_path = chosen
                identity.cover_locked = True
            else:
                identity.cover_locked = False
                identity_ops.recompute_cover(session, identity, dropped=None)
            session.commit()
        return RedirectResponse(f"/identities/{identity_id}", status_code=303)

    @app.get("/noise", response_class=HTMLResponse)
    def noise(request: Request, after: str | None = None):
        # `LIMIT 200` with nothing said about it was the same class of
        # failure this page's empty states were written for (CLD-26): a
        # screen that cannot show the truth reading as though it has. A
        # long soak makes 200 episodes easily, and episode 201 simply did
        # not exist here. Keyset-paged now, with the real count stated.
        sort = (NoiseEvent.start, NoiseEvent.id)
        with Session() as session:
            q = select(NoiseEvent).order_by(
                NoiseEvent.start.desc(), NoiseEvent.id.desc()
            )
            noise_total = (
                session.scalar(select(func.count()).select_from(NoiseEvent)) or 0
            )
            values = _cursor_values(after, (datetime, int))
            rows_q = q if values is None else q.where(paging.after(sort, values))
            fetched = session.scalars(rows_q.limit(NOISE_PAGE + 1)).unique().all()
            slice_ = paging.take(
                list(fetched), NOISE_PAGE, lambda n: (n.start, n.id)
            )
            rows = slice_.rows
        # Why this page may be empty, which is not the same as "it was
        # quiet" (CLD-26). Audio never reaches AudioModule on a live
        # stream: Frame carries only an image, OpenCV's FFmpeg backend
        # discards audio, and ingest calls _process_audio only for the
        # file adapter. So on a site of UniFi cameras this table cannot
        # fill, no matter how loud it gets, and `audio.enabled: true`
        # changes nothing. Saying so beats an empty table that reads as
        # a clean bill of health.
        # Why the table is empty, which is never just "it was quiet".
        # Mirrors ingest's gates exactly: _process_audio runs only for the
        # file adapter, and only when the site has audio on and the camera
        # lists the module. Each reason gets its own message — collapsing
        # them would mean explaining the live-stream limitation to someone
        # whose real problem is that audio is switched off.
        wants_audio = [
            c.id
            for c in config.cameras
            if config.audio.enabled and "audio" in (c.modules or [])
        ]
        producing = [
            c.id for c in config.cameras if c.adapter == "file" and c.id in wants_audio
        ]
        if producing:
            reason = "quiet"
        elif wants_audio:
            reason = "inert"  # asked for, but no camera can deliver it
        elif config.audio.enabled:
            reason = "unconfigured"  # on site-wide, no camera lists it
        else:
            reason = "off"
        return templates.TemplateResponse(
            request,
            "noise.html",
            {
                "site_name": config.site_name or config.site_id,
                "noise_events": rows,
                "noise_total": noise_total,
                "more": _more(
                    "#noise-rows",
                    None
                    if slice_.exhausted
                    else _with_cursor("/noise", slice_.next_cursor),
                    f"Load {NOISE_PAGE} more episodes",
                    f"End of the list — {noise_total} episode"
                    f"{'' if noise_total == 1 else 's'} recorded.",
                ),
                "producing_cameras": producing,
                "empty_reason": reason,
                # Cameras asked for audio that cannot deliver it — the
                # module is listed and the live path silently ignores it.
                "inert_cameras": [c for c in wants_audio if c not in producing],
            },
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

    # Route modules split out of this file to keep each readable. Each
    # takes the same four arguments, registers its own routes onto this
    # app, and declares its own sidebar entry via web/nav.add().
    from siteloom.web import (
        backfill_routes,
        bookings_routes,
        incidents_routes,
        library_routes,
        plates_routes,
        timezone_routes,
        train_routes,
        users_routes,
    )

    for module in (
        library_routes,
        bookings_routes,
        backfill_routes,
        plates_routes,
        train_routes,
        incidents_routes,
        timezone_routes,
        users_routes,
    ):
        module.register(app, templates, Session, config)

    if config.integrations.recognition_api.enabled:
        from siteloom.web import recognition_api

        service = recognition_service or recognition_api.RecognitionService(
            config, Session
        )
        recognition_api.register(app, config, service)

    # -- live view ---------------------------------------------------------

    from siteloom.web.live import LiveHub

    hub = LiveHub(config)
    # Two registrations, because the lifespan hook alone is unreachable
    # for the case that matters (CLD-132). uvicorn's shutdown waits for
    # open connections *first* and runs the lifespan shutdown after — so
    # `on_shutdown` fires only once every stream has already ended, and
    # an MJPEG stream is a response that never ends on its own. The
    # lifespan hook still covers TestClient and any other host; `serve`
    # reaches `app.state.live_hub` directly, ahead of the wait.
    app.state.live_hub = hub
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

        The request component is read through _media_candidates: absolute
        as given, relative anchored on media_root (never on the CWD, which
        is what made the same config serve different files depending on
        how the process was launched — CLD-64).

        Containment is the gate, and it is applied to every candidate. It
        compares *resolved* paths with is_relative_to, and both halves
        matter. A string-prefix test would admit a sibling directory —
        "/var/media-x/secret" starts with "/var/media" (CLD-49) — and
        resolving the full path last, rather than trusting a resolved
        root, is what catches a symlink inside media_dir pointing out of
        it.
        """
        for candidate in _media_candidates(path, media_root):
            full = candidate.resolve()
            if full.is_relative_to(media_root) and full.is_file():
                return FileResponse(full)
        raise HTTPException(404)

    # -- supervision -------------------------------------------------------
    #
    # Both probes are public — a probe that needs a cookie gets the
    # service killed — and the console is treated as internet-exposed
    # (CLD-54), which sets two rules for them: bodies carry nothing that
    # maps the install (no paths, no hostnames, no pids, no versions, no
    # dependency errors — that detail is `siteloom doctor`, at a
    # terminal), and a hit performs bounded, read-only work. /readyz
    # answers from a short-lived cache so hammering it costs one check
    # run per TTL, not one per request.
    ready_lock = threading.Lock()
    ready_state: dict = {"until": 0.0, "cached": None}

    @app.get("/healthz")
    def healthz():
        """Liveness: the process is up and serving. Deliberately touches
        nothing else, so a slow database cannot get the server killed and
        restarted into the same slow database. The answer is a yes, not a
        fact sheet: liveness needs one bit."""
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        """Readiness: can this process actually do its job? Runs the
        cheap, read-only half of `siteloom doctor` — never the vector
        store, which this process is already holding, and never a
        migration: `init_db` belongs to startup (CLD-54)."""
        from siteloom.health import LIVE_CHECKS, run_checks

        now = time.monotonic()
        with ready_lock:
            cached = ready_state["cached"] if now < ready_state["until"] else None
        if cached is None:
            report = run_checks(config, LIVE_CHECKS)
            # Name and status only — Check.detail and .remedy name the
            # db url, the media path and worse, which an operator wants
            # from `doctor` and an unauthenticated caller must not get
            # from a public endpoint.
            cached = (
                200 if report.ok else 503,
                {
                    "ok": report.ok,
                    "checks": [
                        {"name": c.name, "status": c.status} for c in report.checks
                    ],
                },
            )
            with ready_lock:
                ready_state["until"] = time.monotonic() + READYZ_CACHE_S
                ready_state["cached"] = cached
        status_code, body = cached
        return JSONResponse(body, status_code=status_code)

    return app


def create_app_from_env() -> FastAPI:
    """Uvicorn factory entrypoint: SITELOOM_CONFIG=site.yaml."""
    return create_app(load_config(os.environ.get("SITELOOM_CONFIG", "site.yaml")))
