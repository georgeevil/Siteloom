"""Cross-site request forgery and cookie hardening (CLD-58).

Every mutation in the console is a form POST authenticated by an ambient
cookie — the textbook CSRF shape once the console is internet-exposed.
The defence is browser-attested provenance (Origin, with Sec-Fetch-Site
as the backstop) checked in the one middleware, not per-form tokens: a
route added tomorrow is covered without any template carrying anything,
and a client that sends neither header (curl, this suite, Double Take)
is a non-browser client with no ambient state to be confused into using.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient
from sqlalchemy import select

from siteloom.store import AuditLog, get_session, init_db, make_engine
from siteloom.web.app import create_app
from siteloom.web.auth import SAFE_METHODS, cookie_secure, cross_site_reason
from test_auth import add_user, login, make_env

CROSS_SITE = {"Origin": "https://evil.example"}


# -- the pure decision -------------------------------------------------------


def test_cross_origin_refused_same_origin_allowed():
    assert cross_site_reason("POST", "https://evil.example", None, "cam.lan", "http")
    assert cross_site_reason("POST", "http://cam.lan", None, "cam.lan", "http") is None


def test_origin_comparison_survives_ports_case_and_proxies():
    # Browsers elide default ports from Origin; Host may spell them out.
    assert (
        cross_site_reason("POST", "http://cam.lan", None, "cam.lan:80", "http") is None
    )
    assert (
        cross_site_reason(
            "POST", "https://cam.lan", None, "cam.lan:443", "http", "https"
        )
        is None
    )
    assert (
        cross_site_reason("POST", "http://CAM.LAN:8000", None, "cam.lan:8000", "http")
        is None
    )
    # A proxy that rewrites Host forwards the original in X-Forwarded-Host.
    assert (
        cross_site_reason(
            "POST",
            "https://cam.example.com",
            None,
            "localhost:8000",
            "http",
            "https",
            "cam.example.com",
        )
        is None
    )
    # A different port is a different origin — another service on the
    # same host must not be able to drive this one.
    assert cross_site_reason(
        "POST", "http://cam.lan:8001", None, "cam.lan:8000", "http"
    )


def test_null_or_garbled_origin_is_refused():
    # A browser that will not name the sender (sandboxed iframe, data:
    # URL) is not one to honour a mutation from.
    assert cross_site_reason("POST", "null", None, "cam.lan", "http")
    assert cross_site_reason("POST", "not a url", None, "cam.lan", "http")


def test_fetch_metadata_backstop():
    assert cross_site_reason("POST", None, "cross-site", "cam.lan", "http")
    # A sibling subdomain is not this console.
    assert cross_site_reason("POST", None, "same-site", "cam.lan", "http")
    assert cross_site_reason("POST", None, "same-origin", "cam.lan", "http") is None
    # User-initiated: address bar, bookmark.
    assert cross_site_reason("POST", None, "none", "cam.lan", "http") is None


def test_headerless_and_safe_requests_are_never_checked():
    # Neither header: a non-browser client — curl, scripts, this suite.
    assert cross_site_reason("POST", None, None, "cam.lan", "http") is None
    # Reads mutate nothing.
    assert (
        cross_site_reason("GET", "https://evil.example", None, "cam.lan", "http")
        is None
    )


def test_cookie_secure_honours_the_first_forwarded_proto():
    assert not cookie_secure("http", None)
    assert cookie_secure("https", None)
    assert cookie_secure("http", "https")
    assert cookie_secure("http", "https, http")  # first proxy in a chain wins
    assert not cookie_secure("http", "http")


# -- the middleware ----------------------------------------------------------


def test_cross_site_post_is_refused_in_open_mode(tmp_path):
    """The check binds to nothing about auth, deliberately: in open mode
    there is no cookie to ride, but a hostile page could otherwise drive
    every mutation of an unauthenticated console. And the refusal is not
    audited — nothing happened, the same rule as a role denial."""
    client, Session = make_env(tmp_path)
    r = client.post("/events/1/review", data={"reviewed": "1"}, headers=CROSS_SITE)
    assert r.status_code == 403
    assert r.json() == {"detail": "cross-site request refused"}
    with Session() as s:
        assert s.scalars(select(AuditLog)).all() == []


def test_same_origin_post_passes(tmp_path):
    client, _ = make_env(tmp_path)
    r = client.post(
        "/events/1/review",
        data={"reviewed": "1"},
        headers={"Origin": "http://testserver"},
    )
    assert r.status_code == 303


def test_cross_site_post_is_refused_with_a_valid_session(tmp_path):
    """The attack this exists for: a signed-in operator's browser lends
    the session cookie to a hostile page's form post."""
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    login(client, "ana")
    r = client.post("/events/1/review", data={"reviewed": "1"}, headers=CROSS_SITE)
    assert r.status_code == 403
    # The session itself is untouched — only the forged request died.
    assert (
        client.post("/events/1/review", data={"reviewed": "1"}).status_code == 303
    )


def test_every_mutating_route_refuses_a_cross_site_origin(tmp_path):
    """The same walk test_auth runs for the auth gate, for the same
    reason: the design is one middleware covering routes nobody
    remembered to think about, which is only true while every mutating
    route actually passes through it. Runs in open mode — coverage must
    not depend on auth being on. /login is deliberately *in* the walk:
    login CSRF (signing the victim into the attacker's account) is CSRF.
    """
    client, _ = make_env(tmp_path)
    checked = 0
    for route in client.app.routes:
        methods = getattr(route, "methods", set()) - set(SAFE_METHODS)
        path = getattr(route, "path", "")
        if not methods or path.startswith("/api/v1/"):
            continue
        concrete = re.sub(r"\{[^}]+\}", "1", path)
        for method in sorted(methods):
            r = client.request(method, concrete, headers=CROSS_SITE)
            assert r.status_code == 403, f"{method} {concrete} -> {r.status_code}"
            assert r.json() == {"detail": "cross-site request refused"}
            checked += 1
    assert checked > 5  # the walk found routes, not an empty table


def test_api_v1_is_exempt(tmp_path):
    """A cross-site Origin on /api/v1 must reach the API's own routing
    and auth, not the console's CSRF gate: its clients authenticate per
    request with x-api-key, not an ambient cookie, so they are not
    CSRF-reachable. Here nothing is mounted under the prefix, so passing
    the middleware means 404 from the router — a 403 would mean the gate
    ate it."""
    client, _ = make_env(tmp_path)
    r = client.post("/api/v1/recognition/recognize", headers=CROSS_SITE)
    assert r.status_code == 404


def test_double_take_clients_keep_working_cross_origin(tmp_path):
    """The same exemption against the real API: correct key, foreign
    Origin, and the request must succeed — a browser-provenance demand
    here would break Double Take for no gain."""
    from siteloom.identity import VectorStore
    from siteloom.web.recognition_api import RecognitionService
    from test_recognition_api import StubFace, empty_photo, make_config

    config = make_config(tmp_path)
    config.integrations.recognition_api.enabled = True
    config.integrations.recognition_api.api_key = "sekrit"
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    vectors = VectorStore(config.identity.vector_db_path)
    try:
        service = RecognitionService(
            config, get_session(engine), vectors=vectors, embedder=StubFace()
        )
        client = TestClient(create_app(config, recognition_service=service))
        r = client.post(
            "/api/v1/recognition/recognize",
            files={"file": ("f.jpg", empty_photo(), "image/jpeg")},
            headers={**CROSS_SITE, "x-api-key": "sekrit"},
        )
        assert r.status_code == 200
        assert r.json()["result"] == []
    finally:
        vectors.close()


# -- cookie flags ------------------------------------------------------------


def _login_cookie(client, **headers) -> str:
    r = client.post(
        "/login",
        data={"username": "ana", "password": "hunter2!", "next": "/"},
        headers=headers or None,
    )
    assert r.status_code == 303
    return r.headers["set-cookie"].lower()


def test_session_cookie_flags_on_plain_http(tmp_path):
    """HttpOnly and SameSite always; Secure NOT hardcoded — the current
    deployment is plain HTTP on a LAN, where a Secure cookie is one the
    browser refuses to send back: a login that silently never sticks."""
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    cookie = _login_cookie(client)
    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert "secure" not in cookie


def test_session_cookie_is_secure_behind_https(tmp_path):
    """A proxy or tunnel that terminated TLS says so in
    X-Forwarded-Proto, and the cookie picks up Secure with zero
    configuration."""
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    assert "secure" in _login_cookie(client, **{"X-Forwarded-Proto": "https"})


def test_logout_deletion_carries_the_same_attributes(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    login(client, "ana")
    r = client.post("/logout", headers={"X-Forwarded-Proto": "https"})
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "secure" in cookie
    assert 'sl_session=""' in cookie or "sl_session=;" in cookie
