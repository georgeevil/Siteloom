"""Authentication and authorization for the operator console.

Design (CLD auth milestone):

* Roles are a strict ladder: restricted < view < edit < admin. Reads need
  restricted, reads of personal data need view, mutations need edit,
  configuration needs admin. There is no permission matrix — the rungs
  are "look at the queue", "look at everything", "judge", "reconfigure",
  and which rung a request needs is decided by the prefix lists below,
  never by a decorator on a handler.
* Auth turns on when the first User row exists. Before that the console
  runs in the open single-operator mode the PoC started with, so adding
  this layer breaks nothing until someone opts in with
  `siteloom users add`.
* Enforcement lives in one middleware, not per-route decorators, so a
  newly added route is secured (and audited) by default rather than by
  remembering.
* A floor refuses a whole screen, which is the wrong instrument for a
  screen `restricted` must keep: the triage queue has to stay workable
  while it stops printing who everyone is. That second mechanism —
  substitution rather than refusal — is `web/redaction.py`, and it takes
  its floor from `required_role("GET", "/identities")` here rather than
  spelling a rung of its own (CLD-111).
* The recognition API (/api/v1/…) keeps its own x-api-key scheme —
  machine clients, per CompreFace convention — and is exempt here.
* Sign-in is slowed by address, never locked by username (LoginThrottle),
  costs the same whether or not the account exists (`authenticate`), and
  leaves an audit row either way.
* Cross-site request forgery is refused in the same middleware, by
  browser-attested request provenance (`cross_site_reason`) rather than
  per-form tokens — a token each new template must remember is a check
  that will be forgotten, which is the same reasoning as the one
  middleware itself (CLD-58). The check binds to nothing about auth, so
  it protects the open single-operator mode too.

Passwords are scrypt (stdlib) with a per-user salt; the cookie stores
only an opaque token whose server side row can be revoked.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from sqlalchemy import delete, select

from siteloom.store import AuditLog, User, WebSession

#: The ladder, low to high. Stored on `User.role` as these strings and
#: only ever compared through this mapping — adding `restricted` at the
#: bottom renumbered every rung above it, and nothing migrated because
#: nothing anywhere stores or compares the integer (CLD-31).
ROLES = {"restricted": 0, "view": 1, "edit": 2, "admin": 3}
#: Below every rung, for a role string this build does not recognise (a
#: row written by a newer version, or edited by hand). It must stay below
#: the *lowest* rung rather than at 0, which is now a real one.
NO_ROLE = -1

#: What each rung is called on screen, and what it means. The stored
#: values are unchanged — a rename would touch every role comparison,
#: every test asserting on a role string and every existing User row, and
#: buy nothing the labels do not (CLD-31 product decision).
ROLE_LABELS = {
    "restricted": "Restricted",
    "view": "Viewer",
    "edit": "Writer",
    "admin": "Admin",
}
ROLE_SUMMARIES = {
    "restricted": "Triage queue only — no identities, gallery or training data",
    "view": "Sees everything, changes nothing",
    "edit": "Judges: verdicts, labels, corrections",
    "admin": "Reconfigures: accounts, thresholds, models, jobs",
}


def role_label(role: str) -> str:
    """The operator-facing name for a stored role value."""
    return ROLE_LABELS.get(role, role)


SESSION_COOKIE = "sl_session"
#: Absolute session lifetime, enforced server-side (`resolve_user`
#: rejects an expired row no matter what the cookie claims) and pruned at
#: sign-in/sign-out (`purge_expired_sessions`, CLD-65). There is
#: deliberately no *idle* timeout (CLD-58): the console is a wall screen
#: an operator glances at across a shift, and a session that logs itself
#: out between glances trains people toward weaker passwords and shared
#: terminals left signed in. Ending a session sooner is sign-out,
#: disabling the account, or revoking its sessions on /users.
SESSION_TTL = timedelta(days=14)

#: Methods that read. Everything else is treated as a mutation — by the
#: role floors (`required_role`), by the audit writer, and by the
#: cross-site check (`cross_site_reason`) — so the three can never
#: disagree about what "a write" is.
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")

#: Paths that must work logged-out: the login flow itself and the probes
#: a service manager hits. Everything else needs at least `view` once
#: auth is enabled.
PUBLIC_PATHS = {"/login", "/healthz", "/readyz"}
#: Prefixes with their own auth scheme (x-api-key).
EXEMPT_PREFIXES = ("/api/v1/",)
#: Path prefixes whose mutations reconfigure the system rather than
#: review its output — admin only. A trailing slash is load-bearing on
#: two of these: `/jobs/` covers every mutation on the jobs console
#: (reindex, cancel, reap) while leaving GET /jobs a readable screen, and
#: `/train/` keeps this off `/training`, which is labelling and needs
#: only `edit`. `/audit` and `/users` appear here and in
#: ADMIN_READ_PREFIXES below — reads *and* writes are admin there, and
#: listing them twice is what stops a mutation added under either prefix
#: tomorrow landing on `edit` by default.
ADMIN_PREFIXES = (
    "/audit",
    "/backfill",
    "/classes/detection",
    "/classes/events",
    "/jobs/",
    # Enrolling, fine-tuning and adopting a model reconfigure what the
    # system recognizes (CLD-94).
    "/train/",
    "/users",
)
#: Prefixes whose *reads* are administrative too. Deliberately a separate
#: list from ADMIN_PREFIXES rather than applying that one to GETs: the
#: backfill and jobs screens are things a viewer may watch and only act
#: on with admin, while the account list and the audit trail are not
#: things a viewer may read at all.
ADMIN_READ_PREFIXES = (
    "/audit",
    "/users",
)
#: The second floor (CLD-31), and the mirror image of ADMIN_PREFIXES: a
#: read under one of these needs `view`, so the `restricted` rung below it
#: cannot reach identities, the face gallery or the training corpus.
#: Biometric data minimisation (NFR5) for a night-shift operator who
#: should judge whether an event matters without browsing everyone's face.
#:
#: `/train` with no trailing slash on purpose — unlike ADMIN_PREFIXES this
#: list wants `/training` too, and both are the labelling and modelling
#: surface. `/search` is here because it is a directory of people and
#: plates by another name: leaving it open would make the `/identities`
#: floor cosmetic (see the module docstring of web/users_routes.py for
#: what that costs).
RESTRICTED_DENIED_PREFIXES = (
    # The library's own read API, and therefore the training corpus by
    # another URL: `/api/items/{id}/annotations` hands back every box on
    # an item with its `proposed_name` — the Takeout importer's guess at
    # who a face is (CLD-111). `/library` was on this list from the
    # start; the JSON behind it was not, because it does not begin with
    # `/library`. Nothing here can be substituted usefully: an annotation
    # editor without names is an editor of nothing, and `restricted` has
    # no business in one.
    "/api/items",
    "/classes",
    "/identities",
    # An incident is built from events a restricted operator may read, but
    # its detail view and its export both render `identity.display_name`
    # for every event on the timeline — so the incident is a list of who
    # was there, which is the one thing this rung withholds. Escalating is
    # `edit` work and out of reach anyway; reading is what needed closing.
    "/incidents",
    "/library",
    # Every OCR'd plate on the site, with its crop (CLD-85). A plate is a
    # vehicle's identifier and the screen is a list of them — the same
    # directory `/identities` and `/search` are closed for, reached by a
    # third URL. Note the plate crops live under
    # `media_dir/<camera>/<date>/plates/`, not under `library/`, so the
    # gallery-media gate does not cover them and this prefix is the only
    # thing that does.
    "/plates",
    "/search",
    "/train",
)
#: Crops are served by one route for two purposes, so the prefix cannot
#: tell them apart — the path below it can. Event crops are written to
#: `media_dir/<camera-id>/<date>/`, while everything that makes up the
#: face gallery and the training corpus (library thumbnails, library
#: crops, Takeout faces) is written under `media_dir/library/`. So the
#: gate is a path *segment*, which is what survives both spellings
#: `/media/{path}` accepts — a relative path anchored on media_dir and an
#: absolute one — and the case-insensitivity of the macOS filesystem this
#: primarily targets.
MEDIA_PREFIX = "/media/"
GALLERY_MEDIA_SEGMENT = "library"

_SCRYPT = {"n": 2**14, "r": 8, "p": 1}

#: Login backoff (CLD-51). The first few misses are free — an operator
#: fat-fingering their own password must not be punished — after which
#: each further miss doubles the wait, up to the cap.
LOGIN_FREE_ATTEMPTS = 3
LOGIN_BASE_DELAY_S = 2.0
LOGIN_MAX_DELAY_S = 300.0
#: An address that stops guessing for this long starts fresh.
LOGIN_FORGET_S = 900.0
#: Cap on the attacker-supplied name copied into an audit row.
FAILED_ACTOR_MAX = 64


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt_hex), **_SCRYPT
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def safe_next(next_url: str, fallback: str = "/") -> str:
    """Confine a caller-supplied redirect target to this origin.

    Both the login form and the triage rail round-trip an operator back
    to where they were, so the target arrives from the page and is
    therefore attacker-supplied. Only a plain absolute path on this
    origin survives; anything else becomes `fallback`.

    The rules are all one bug in different clothes — a target the server
    reads as local and the browser resolves as external:

    * `https://evil` / `javascript:…` — a scheme.
    * `//evil` — protocol-relative, host in the "path".
    * `/\\evil` — browsers normalize the backslash to a slash, so this
      reaches the network as `//evil`. This is the rule POST /login was
      missing (CLD-52); it lives here now so there is one validator
      rather than a copy per call site that can drift again.
    * Control characters — browsers strip tab/CR/LF from URLs before
      resolving them, so `/\\t/evil` is another spelling of the above,
      and a bare CR/LF in a Location header is response splitting.
    """
    if (
        next_url.startswith("/")
        and not next_url.startswith("//")
        and "\\" not in next_url
        and all(ch.isprintable() for ch in next_url)
    ):
        return next_url
    return fallback


#: Ports a browser omits from Origin, keyed by scheme — the reason
#: `http://cam.lan` and `Host: cam.lan:80` must compare equal.
_DEFAULT_PORTS = {"http": ":80", "https": ":443"}


def effective_scheme(scheme: str, forwarded_proto: str | None) -> str:
    """The scheme the *browser* used, seen from behind a proxy or not.

    A reverse proxy or tunnel terminates TLS and forwards plain HTTP, so
    the socket says http while the operator's browser said https;
    `X-Forwarded-Proto` is how the proxy relays that, and the first value
    wins when proxies chain. Trusting the header unconditionally is safe
    for the two things it decides here: it can only *add* the cookie's
    Secure flag (an attacker forging it on a plain-HTTP login breaks
    their own cookie, nobody else's), and in the origin comparison it
    only settles default-port elision. A forged header never widens
    anything.
    """
    proto = (forwarded_proto or "").split(",")[0].strip().lower()
    return "https" if proto == "https" else scheme


def cookie_secure(scheme: str, forwarded_proto: str | None) -> bool:
    """Whether the session cookie should carry `Secure` (CLD-58).

    Decided per-request rather than hardcoded on: the current deployment
    is plain HTTP on a LAN, where a Secure cookie is one the browser
    refuses to send back — a login that silently never sticks. Any HTTPS
    arrival (direct, or via a proxy that sets X-Forwarded-Proto) gets the
    flag with zero configuration.
    """
    return effective_scheme(scheme, forwarded_proto) == "https"


def _bare_netloc(netloc: str, scheme: str) -> str:
    """host[:port], case-folded, with the scheme's default port dropped."""
    netloc = netloc.strip().lower()
    default = _DEFAULT_PORTS.get(scheme)
    if default and netloc.endswith(default):
        netloc = netloc[: -len(default)]
    return netloc


def cross_site_reason(
    method: str,
    origin: str | None,
    sec_fetch_site: str | None,
    host: str | None,
    scheme: str,
    forwarded_proto: str | None = None,
    forwarded_host: str | None = None,
) -> str | None:
    """Why this request must be refused as cross-site, or None to allow.

    The CSRF control (CLD-58): every console mutation is a form POST
    authenticated by an ambient cookie, the exact shape CSRF exploits.
    The defence is the request provenance browsers already attest —
    every browser that can carry the session cookie cross-site also
    stamps `Origin` (and `Sec-Fetch-Site`) on cross-site POSTs, and an
    attacker's page cannot forge either on a victim's behalf — so
    rejecting a foreign origin here protects every form without a token
    in any template. The rules, in order:

    * Safe methods are never checked — they mutate nothing.
    * An `Origin` that parses is compared to this request's own host
      (`Host`, or `X-Forwarded-Host` for a proxy that rewrites Host),
      case-folded, default ports elided per scheme. Match allows,
      mismatch refuses.
    * `Origin: null` (sandboxed iframe, data: URL) or unparsable
      refuses: a browser that will not name the sender is not one to
      honour a mutation from.
    * No Origin but `Sec-Fetch-Site` present: `same-origin` and `none`
      (user-initiated: address bar, bookmark) allow; `cross-site` and
      `same-site` refuse — a sibling subdomain is not this console.
    * **Neither header allows.** That is a non-browser client — curl, a
      script, the test suite — which carries no ambient browser state to
      be confused into using; fetch-metadata guidance, and also what
      keeps every existing CLI and TestClient caller working unchanged.

    Returned reasons are fixed strings for the log line, never echoes of
    attacker-controlled header values.
    """
    if method in SAFE_METHODS:
        return None
    if origin is not None:
        parts = urlsplit(origin)
        if not parts.scheme or not parts.netloc:
            return "origin is null or unparsable"
        req_scheme = effective_scheme(scheme, forwarded_proto)
        ours = {_bare_netloc(host or "", req_scheme)}
        if forwarded_host:
            first = forwarded_host.split(",")[0]
            ours.add(_bare_netloc(first, req_scheme))
        if _bare_netloc(parts.netloc, parts.scheme.lower()) not in ours:
            return "origin does not match this host"
        return None
    if sec_fetch_site is not None and sec_fetch_site.strip().lower() not in (
        "same-origin",
        "none",
    ):
        return "cross-site by fetch metadata"
    return None


def auth_enabled(session) -> bool:
    return session.scalar(select(User.id).limit(1)) is not None


class AuthGate:
    """Per-app cache of `auth_enabled` (CLD-65).

    Only the "on" answer is cached, and deliberately so. Auth turns on
    when the first User row appears — usually from a *different* process
    (`siteloom users add`) — and there is no supported way back to open
    mode, so a True answer can never go stale while a False one would
    leave the console open for the cache lifetime after the operator
    enabled it. That is the one moment staleness would matter, which
    makes a TTL exactly the wrong shape here.
    """

    def __init__(self) -> None:
        self._enabled = False

    def enabled(self, session) -> bool:
        if not self._enabled:
            self._enabled = auth_enabled(session)
        return self._enabled


def create_session(session, user: User) -> str:
    token = secrets.token_urlsafe(32)
    session.add(
        WebSession(
            token=token,
            user_id=user.id,
            created_at=_now(),
            expires_at=_now() + SESSION_TTL,
        )
    )
    return token


def purge_expired_sessions(session, now: datetime | None = None) -> int:
    """Delete web sessions whose TTL has passed (CLD-65).

    Called where a row is *created* — sign-in and sign-out — so every
    login pays for the sessions it obsoletes and the table cannot grow
    without someone logging in. Deliberately not in the request path:
    `resolve_user` runs on every hit and must stay a read.
    """
    result = session.execute(
        delete(WebSession).where(WebSession.expires_at < (now or _now()))
    )
    return result.rowcount or 0


def resolve_user(session, token: str | None) -> User | None:
    if not token:
        return None
    row = session.get(WebSession, token)
    if row is None or row.expires_at < _now():
        return None
    user = session.get(User, row.user_id)
    if user is None or user.disabled:
        return None
    return user


_dummy_hash_cache: str | None = None
_dummy_lock = threading.Lock()


def _dummy_hash() -> str:
    """A hash of a password nobody knows, built once per process.

    Not computed at import: scrypt at these parameters costs ~100 ms and
    every CLI command imports this module.
    """
    global _dummy_hash_cache
    with _dummy_lock:
        if _dummy_hash_cache is None:
            _dummy_hash_cache = hash_password(secrets.token_urlsafe(24))
        return _dummy_hash_cache


def authenticate(session, username: str, password: str) -> User | None:
    """Check a username/password pair in constant work (CLD-51).

    A missing user is verified against a throwaway hash instead of
    returning early, so "no such account" and "wrong password" cost the
    same scrypt round and cannot be told apart by a stopwatch — the
    timing half of the same leak the shared error message closes. The
    disabled check comes after the verification for the same reason.
    """
    user = session.scalar(select(User).filter_by(username=username.strip()))
    stored = user.password_hash if user is not None else _dummy_hash()
    ok = verify_password(password, stored)
    if user is None or user.disabled or not ok:
        return None
    return user


class LoginThrottle:
    """Exponential backoff for failed sign-ins, keyed by client address.

    Keyed by address and *not* by username on purpose: a username-keyed
    lockout hands any stranger who can reach the port a way to lock the
    operator out of their own console by failing logins as them. Keying
    on the caller bounds guessing from that caller no matter how many
    usernames it tries, and the operator's own address is only ever
    slowed by their own mistakes.

    Nothing here is permanent. The delay is a wait, not a lock: it
    expires on its own (capped at LOGIN_MAX_DELAY_S), a successful
    sign-in clears it, an idle address is forgotten after
    LOGIN_FORGET_S, and the whole thing is in-process memory that a
    restart of `siteloom serve` drops.

    Lock-serialized because FastAPI serves sync endpoints from a
    threadpool (same reasoning as the recognition API's RateLimiter),
    and swept on access so an attacker rotating source addresses cannot
    turn the defence into a slow memory leak.
    """

    def __init__(
        self,
        free_attempts: int = LOGIN_FREE_ATTEMPTS,
        base_delay_s: float = LOGIN_BASE_DELAY_S,
        max_delay_s: float = LOGIN_MAX_DELAY_S,
        forget_s: float = LOGIN_FORGET_S,
    ):
        self.free_attempts = free_attempts
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.forget_s = forget_s
        # key -> [failures, blocked_until, last_failure]
        self._state: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def retry_after(self, key: str, now: float | None = None) -> float:
        """Seconds the caller must wait; 0.0 when it may try now."""
        now = time.monotonic() if now is None else now
        with self._lock:
            entry = self._state.get(key)
            if entry is None:
                return 0.0
            return max(0.0, entry[1] - now)

    def record_failure(self, key: str, now: float | None = None) -> float:
        """Count a failed attempt; return the wait it just earned."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._sweep(now)
            entry = self._state.get(key)
            if entry is None or now - entry[2] > self.forget_s:
                entry = [0.0, 0.0, now]
                self._state[key] = entry
            entry[0] += 1
            entry[2] = now
            over = int(entry[0]) - self.free_attempts
            if over <= 0:
                return 0.0
            delay = min(self.base_delay_s * 2 ** (over - 1), self.max_delay_s)
            entry[1] = now + delay
            return delay

    def record_success(self, key: str) -> None:
        with self._lock:
            self._state.pop(key, None)

    def _sweep(self, now: float) -> None:
        stale = [
            k
            for k, e in self._state.items()
            if e[1] <= now and now - e[2] > self.forget_s
        ]
        for k in stale:
            del self._state[k]


def failed_actor(attempted: str) -> str:
    """The audit `username` for a sign-in that did not succeed.

    The attempted name is what makes a brute-force run readable in the
    log, but it is attacker-controlled text, so it is printable-filtered,
    truncated, and wrapped in a marker that no successful actor string
    can collide with. The row's `user_id` stays NULL: nobody acted.
    """
    name = "".join(c for c in attempted.strip() if c.isprintable())
    return f"(failed:{name[:FAILED_ACTOR_MAX]})"


def is_gallery_media(path: str) -> bool:
    """Whether a /media request is for face-gallery or training material.

    The one media route serves two kinds of image from two directory
    trees: event crops under `media_dir/<camera-id>/<date>/`, which are
    what an operator triaging a visit is looking at, and the library's
    thumbnails, crops and imported faces under `media_dir/library/`,
    which are the gallery and the training corpus. Only the second is
    withheld from `restricted`.

    Matching on the segment rather than on a URL prefix is what makes
    that hold for every spelling the route accepts: `_media_candidates`
    reads a relative path against media_dir *and* an absolute one, so
    `/media/library/crops/x.jpg` and `/media//srv/media/library/crops/x.jpg`
    are the same file, and only one of them starts with `/media/library`.
    Lower-cased because the primary deployment target is a case-
    insensitive filesystem, where `/media/Library/...` serves the same
    file that `/media/library/...` does.

    What this cannot see is a symlink inside media_dir under some other
    name pointing at `library/` — the media route's containment check
    stops paths escaping media_dir, not paths taking a scenic route
    inside it. An operator who creates one has re-plumbed the crop
    layout this reads.
    """
    if not path.startswith(MEDIA_PREFIX):
        return False
    rest = path[len(MEDIA_PREFIX) :]
    return GALLERY_MEDIA_SEGMENT in (s.lower() for s in rest.split("/"))


def withheld_from_restricted(path: str) -> bool:
    """Whether reading `path` needs `view` rather than `restricted`."""
    return any(
        path.startswith(p) for p in RESTRICTED_DENIED_PREFIXES
    ) or is_gallery_media(path)


def required_role(method: str, path: str) -> str:
    """The lowest rung that may make this request.

    Four answers, decided by which prefix list the path falls in, so a
    route added tomorrow lands on a floor without anyone remembering to
    gate it. Reads are the interesting half now: the default read floor
    is `restricted` (the triage queue, an event, the live view), reads of
    people and training material need `view`, and the account list and
    audit trail need `admin` — a read a viewer may not make at all.
    """
    if method in SAFE_METHODS:
        if any(path.startswith(p) for p in ADMIN_READ_PREFIXES):
            return "admin"
        return "view" if withheld_from_restricted(path) else "restricted"
    if any(path.startswith(p) for p in ADMIN_PREFIXES):
        return "admin"
    return "edit"


def has_role(user: User, role: str) -> bool:
    """Whether `user` stands at or above the `role` rung.

    Every comparison in the console goes through here and through ROLES;
    nothing compares a rung's integer, which is why adding a rung at the
    bottom moved all three existing numbers and migrated no data.
    """
    return ROLES.get(user.role, NO_ROLE) >= ROLES[role]


def record_audit(
    session,
    user: User | None,
    method: str,
    path: str,
    status_code: int,
    username: str | None = None,
) -> None:
    """Record one action. `username` overrides the denormalized actor
    string for rows with no User behind them — a failed sign-in, say."""
    session.add(
        AuditLog(
            at=_now(),
            user_id=user.id if user else None,
            username=username or (user.username if user else "(open)"),
            method=method,
            path=path,
            status_code=status_code,
        )
    )
