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

Passwords are scrypt (stdlib) with a per-user salt; the cookie stores
only an opaque token whose server side row can be revoked.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

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
ADMIN_PREFIXES = ("/classes/detection", "/classes/events", "/jobs/reindex", "/users")

_SCRYPT = {"n": 2**14, "r": 8, "p": 1}


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


def required_role(method: str, path: str) -> str:
    if method in ("GET", "HEAD", "OPTIONS"):
        return "view"
    if any(path.startswith(p) for p in ADMIN_PREFIXES):
        return "admin"
    return "edit"


def has_role(user: User, role: str) -> bool:
    return ROLES.get(user.role, -1) >= ROLES[role]


def record_audit(
    session, user: User | None, method: str, path: str, status_code: int
) -> None:
    session.add(
        AuditLog(
            at=_now(),
            user_id=user.id if user else None,
            username=user.username if user else "(open)",
            method=method,
            path=path,
            status_code=status_code,
        )
    )
