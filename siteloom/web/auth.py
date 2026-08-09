"""Authentication and authorization for the operator console.

Design (CLD auth milestone):

* Roles are a strict ladder: view < edit < admin. GETs need view,
  mutations need edit, configuration needs admin. There is no permission
  matrix — three rungs cover "look", "judge", "reconfigure".
* Auth turns on when the first User row exists. Before that the console
  runs in the open single-operator mode the PoC started with, so adding
  this layer breaks nothing until someone opts in with
  `siteloom users add`.
* Enforcement lives in one middleware, not per-route decorators, so a
  newly added route is secured (and audited) by default rather than by
  remembering.
* The recognition API (/api/v1/…) keeps its own x-api-key scheme —
  machine clients, per CompreFace convention — and is exempt here.
* Sign-in is slowed by address, never locked by username (LoginThrottle),
  costs the same whether or not the account exists (`authenticate`), and
  leaves an audit row either way.

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

from sqlalchemy import delete, select

from siteloom.store import AuditLog, User, WebSession

ROLES = {"view": 0, "edit": 1, "admin": 2}
SESSION_COOKIE = "sl_session"
SESSION_TTL = timedelta(days=14)

#: Paths that must work logged-out: the login flow itself and the probes
#: a service manager hits. Everything else needs at least `view` once
#: auth is enabled.
PUBLIC_PATHS = {"/login", "/healthz", "/readyz"}
#: Prefixes with their own auth scheme (x-api-key).
EXEMPT_PREFIXES = ("/api/v1/",)
#: Path prefixes whose mutations reconfigure the system rather than
#: review its output — admin only.
ADMIN_PREFIXES = (
    "/classes/detection",
    "/classes/events",
    "/jobs/reindex",
    "/users",
    # Enrolling, fine-tuning and adopting a model reconfigure what the
    # system recognizes; the trailing slash keeps this off /training,
    # which is labelling and only needs `edit` (CLD-94).
    "/train/",
)

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


def required_role(method: str, path: str) -> str:
    if method in ("GET", "HEAD", "OPTIONS"):
        return "view"
    if any(path.startswith(p) for p in ADMIN_PREFIXES):
        return "admin"
    return "edit"


def has_role(user: User, role: str) -> bool:
    return ROLES.get(user.role, -1) >= ROLES[role]


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
