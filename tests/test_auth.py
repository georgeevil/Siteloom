"""Authentication, roles and the audit trail (CLD auth milestone).

The property that matters most is the first one: an empty users table
means the open single-operator mode the PoC started with, so adding the
auth layer breaks nothing until someone opts in.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import (
    AuditLog,
    Camera,
    Event,
    User,
    WebSession,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web import auth as auth_module
from siteloom.web.app import create_app
from siteloom.web.auth import (
    FAILED_ACTOR_MAX,
    LOGIN_FREE_ATTEMPTS,
    LoginThrottle,
    failed_actor,
    hash_password,
    purge_expired_sessions,
    required_role,
    safe_next,
    verify_password,
)

TS = datetime(2026, 8, 6, 9, 0, 0)


def make_env(tmp_path):
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/w.db", media_dir=str(tmp_path / "m")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(Camera(id="cam1", site_id="t", name="Cam One"))
        s.add(
            Event(
                camera_id="cam1",
                track_id=1,
                class_name="car",
                first_seen=TS,
                last_seen=TS,
                detection_count=1,
            )
        )
        s.commit()
    return TestClient(create_app(config), follow_redirects=False), Session


def add_user(Session, username, role):
    with Session() as s:
        s.add(
            User(
                username=username,
                password_hash=hash_password("hunter2!"),
                role=role,
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        s.commit()


def login(client, username):
    r = client.post(
        "/login", data={"username": username, "password": "hunter2!", "next": "/"}
    )
    assert r.status_code == 303, r.text[:200]
    return client  # cookie persisted on the client


# -- open mode -------------------------------------------------------------


def test_no_users_means_open_access(tmp_path):
    client, Session = make_env(tmp_path)
    assert client.get("/").status_code == 200
    r = client.post("/events/1/review", data={"reviewed": "1"})
    assert r.status_code == 303
    # Mutations are still audited, attributed to the open mode.
    with Session() as s:
        row = s.scalars(select(AuditLog)).one()
        assert row.username == "(open)"
        assert row.path == "/events/1/review"


# -- enforcement -----------------------------------------------------------


def test_first_user_turns_the_gate_on(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    r = client.get("/")
    assert r.status_code == 303
    assert r.headers["location"].startswith("/login")
    # Probes must keep working logged-out or the service manager kills us.
    assert client.get("/healthz").status_code == 200
    # Mutations get 401, not a redirect a script would follow blindly.
    assert client.post("/events/1/review", data={"reviewed": "1"}).status_code == 401


def test_login_logout_round_trip(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    login(client, "ana")
    page_after = client.get("/")
    assert page_after.status_code == 200
    assert "ana" in page_after.text  # footer shows who is signed in
    client.post("/logout")
    assert client.get("/").status_code == 303


def test_wrong_password_is_rejected_without_confirming_usernames(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    wrong_pw = client.post(
        "/login", data={"username": "ana", "password": "nope", "next": "/"}
    )
    no_user = client.post(
        "/login", data={"username": "ghost", "password": "nope", "next": "/"}
    )
    assert wrong_pw.status_code == no_user.status_code == 401
    assert "Wrong username or password." in wrong_pw.text
    assert "Wrong username or password." in no_user.text


def test_view_can_look_but_not_judge(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "vera", "view")
    login(client, "vera")
    assert client.get("/").status_code == 200
    assert client.post("/events/1/review", data={"reviewed": "1"}).status_code == 403


def test_edit_can_judge_but_not_reconfigure(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "eddy", "edit")
    login(client, "eddy")
    assert client.post("/events/1/review", data={"reviewed": "1"}).status_code == 303
    r = client.post(
        "/classes/detection", json={"classes": ["person"]}
    )
    assert r.status_code == 403
    assert client.post("/classes/events", json={"min_detections": 2}).status_code == 403


def test_admin_can_reconfigure(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "root", "admin")
    login(client, "root")
    assert (
        client.post("/classes/detection", json={"classes": ["person"]}).status_code
        == 200
    )
    assert client.post("/classes/events", json={"min_detections": 2}).status_code == 200


def test_disabled_user_cannot_sign_in(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    with Session() as s:
        s.scalar(select(User).filter_by(username="ana")).disabled = True
        s.commit()
    r = client.post(
        "/login", data={"username": "ana", "password": "hunter2!", "next": "/"}
    )
    assert r.status_code == 401


def test_recognition_api_keeps_its_own_scheme(tmp_path):
    """/api/v1 is machine-auth (x-api-key) — the cookie gate must not
    stack a redirect in front of CompreFace-compatible clients."""
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    r = client.get("/api/v1/recognition/subjects")
    # 401/403/404 from the API's own key check is fine; a 303 to /login is not.
    assert r.status_code != 303


# -- audit -----------------------------------------------------------------


def test_mutations_are_audited_with_the_actor(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "eddy", "edit")
    login(client, "eddy")
    client.post("/events/1/review", data={"reviewed": "1"})
    with Session() as s:
        rows = s.scalars(select(AuditLog).order_by(AuditLog.id)).all()
        actions = {(r.username, r.path) for r in rows}
        assert ("eddy", "/login") in actions
        assert ("eddy", "/events/1/review") in actions


def test_denied_requests_are_not_audited_as_actions(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "vera", "view")
    login(client, "vera")
    client.post("/events/1/review", data={"reviewed": "1"})  # 403
    with Session() as s:
        paths = [r.path for r in s.scalars(select(AuditLog))]
        assert "/events/1/review" not in paths  # nothing happened to record


# -- primitives ------------------------------------------------------------


def test_password_hashing_round_trip():
    stored = hash_password("s3cret pass")
    assert stored.startswith("scrypt$")
    assert verify_password("s3cret pass", stored)
    assert not verify_password("s3cret pas", stored)
    assert not verify_password("s3cret pass", "garbage")


def test_role_ladder_for_paths():
    assert required_role("GET", "/") == "view"
    assert required_role("POST", "/events/1/review") == "edit"
    assert required_role("POST", "/classes/detection") == "admin"
    assert required_role("POST", "/classes/events") == "admin"
    assert required_role("POST", "/users/anything") == "admin"


def test_expired_session_is_rejected(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    login(client, "ana")
    with Session() as s:
        for row in s.scalars(select(WebSession)):
            row.expires_at = datetime(2020, 1, 1)
        s.commit()
    assert client.get("/").status_code == 303


def test_reindex_requires_admin():
    assert required_role("POST", "/jobs/reindex") == "admin"


def test_threshold_write_is_admin_but_its_dry_run_is_not():
    """Saving a similarity threshold reconfigures the system, so it goes
    through the admin-gated /classes/detection. Previewing one only reads
    recorded scores — a viewer may explore without being able to save."""
    assert required_role("POST", "/classes/detection") == "admin"
    assert required_role("GET", "/classes/thresholds/preview") == "view"


def test_edit_role_can_preview_but_not_save_a_threshold(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "eddy", "edit")
    login(client, "eddy")
    r = client.get(
        "/classes/thresholds/preview", params={"identifier": "face", "threshold": 0.4}
    )
    assert r.status_code == 200
    assert (
        client.post(
            "/classes/detection", json={"identifiers": {"face": {"threshold": 0.4}}}
        ).status_code
        == 403
    )

# -- redirect containment (CLD-52) -----------------------------------------

OFFSITE = [
    "//evil.example",  # protocol-relative: host in the "path"
    "/\\evil.example",  # browsers normalize \ to / — the missing rule
    "\\\\evil.example",
    "/\\/evil.example",
    "https://evil.example/x",
    "javascript:alert(1)",
    "/ok\r\nSet-Cookie: sl_session=stolen",  # response splitting
    "/ok\tevil",  # browsers strip tabs before resolving
]


def test_safe_next_confines_every_shape_of_offsite_target():
    for bad in OFFSITE:
        assert safe_next(bad, "/fallback") == "/fallback", bad
    for good in ["/", "/events/3", "/?camera=cam1&class=car", "/identities"]:
        assert safe_next(good, "/fallback") == good


@pytest.mark.parametrize("target", OFFSITE)
def test_login_never_redirects_off_site(tmp_path, target):
    """POST /login used to check only the leading-slash rules, so a
    backslash target it read as local resolved externally in a browser —
    a credential-phishing hop straight off a successful sign-in."""
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    r = client.post(
        "/login", data={"username": "ana", "password": "hunter2!", "next": target}
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/"


def test_login_keeps_a_genuine_local_target(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    r = client.post(
        "/login",
        data={"username": "ana", "password": "hunter2!", "next": "/?camera=cam1"},
    )
    assert r.headers["location"] == "/?camera=cam1"


def test_login_form_sanitizes_the_target_it_echoes(tmp_path):
    """The form round-trips `next` through a hidden field, so a target
    left unchecked on the way in comes back on the way out."""
    client, _ = make_env(tmp_path)
    r = client.get("/login", params={"next": "/\\evil.example"})
    assert r.status_code == 200
    assert "evil.example" not in r.text
    assert 'name="next" value="/"' in r.text


# -- brute force (CLD-51) ---------------------------------------------------


def test_backoff_grows_and_is_a_wait_not_a_lock():
    t = LoginThrottle(free_attempts=2, base_delay_s=1.0, max_delay_s=8.0)
    now = 100.0
    assert t.retry_after("a", now) == 0.0
    assert t.record_failure("a", now) == 0.0  # the free misses cost nothing
    assert t.record_failure("a", now) == 0.0
    assert t.record_failure("a", now) == 1.0
    assert t.retry_after("a", now) == 1.0
    # It expires by itself — no operator action, no permanent lockout.
    assert t.retry_after("a", now + 1.5) == 0.0
    assert t.record_failure("a", now + 1.5) == 2.0
    assert t.record_failure("a", now + 4) == 4.0
    assert t.record_failure("a", now + 9) == 8.0
    assert t.record_failure("a", now + 20) == 8.0  # capped


def test_backoff_is_keyed_by_caller_not_by_username():
    """A username-keyed lockout would let any stranger who can reach the
    port lock the operator out of their own console by failing logins as
    them; only the address doing the guessing may be slowed."""
    t = LoginThrottle(free_attempts=0, base_delay_s=5.0)
    t.record_failure("10.0.0.9", 0.0)
    assert t.retry_after("10.0.0.9", 0.0) > 0
    assert t.retry_after("192.168.1.5", 0.0) == 0.0


def test_a_successful_sign_in_clears_the_backoff():
    t = LoginThrottle(free_attempts=1, base_delay_s=5.0)
    t.record_failure("a", 0.0)
    t.record_failure("a", 0.0)
    assert t.retry_after("a", 0.0) > 0
    t.record_success("a")
    assert t.retry_after("a", 0.0) == 0.0
    assert t.record_failure("a", 0.0) == 0.0  # and the budget is fresh


def test_an_idle_caller_is_forgotten():
    t = LoginThrottle(free_attempts=1, base_delay_s=5.0, forget_s=60.0)
    t.record_failure("a", 0.0)
    assert t.record_failure("a", 500.0) == 0.0  # a free miss again


def test_rotating_addresses_cannot_grow_the_throttle_without_bound():
    """The traffic a throttle attracts rotates source addresses and never
    comes back; without the sweep the defence becomes a slow OOM."""
    t = LoginThrottle(free_attempts=0, base_delay_s=1.0, forget_s=10.0)
    for i in range(500):
        t.record_failure(f"10.0.0.{i}", float(i))
    assert len(t._state) < 100


def test_repeated_failures_start_costing_the_caller_time(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    for _ in range(LOGIN_FREE_ATTEMPTS + 1):
        r = client.post(
            "/login", data={"username": "ana", "password": "nope", "next": "/"}
        )
        assert r.status_code == 401
    # Knowing the password does not buy a way past the wait.
    r = client.post(
        "/login", data={"username": "ana", "password": "hunter2!", "next": "/"}
    )
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1


def test_a_few_typos_do_not_lock_the_operator_out(tmp_path):
    """The single operator must always be able to get back in: the free
    attempts absorb ordinary mistyping."""
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    for _ in range(LOGIN_FREE_ATTEMPTS):
        client.post("/login", data={"username": "ana", "password": "no", "next": "/"})
    assert (
        client.post(
            "/login", data={"username": "ana", "password": "hunter2!", "next": "/"}
        ).status_code
        == 303
    )


def test_a_missing_user_still_pays_for_a_password_check(tmp_path, monkeypatch):
    """Same error message, same work: an early return for an unknown
    account leaks the user list to a stopwatch."""
    seen: list[str] = []
    real = auth_module.verify_password
    monkeypatch.setattr(
        auth_module,
        "verify_password",
        lambda pw, stored: (seen.append(stored), real(pw, stored))[1],
    )
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    client.post("/login", data={"username": "ghost", "password": "nope", "next": "/"})
    assert len(seen) == 1 and seen[0].startswith("scrypt$")

    with Session() as s:
        s.scalar(select(User).filter_by(username="ana")).disabled = True
        s.commit()
    client.post("/login", data={"username": "ana", "password": "nope", "next": "/"})
    assert len(seen) == 2  # disabled accounts are not a shortcut either


def test_failed_logins_are_audited(tmp_path):
    """A denied *action* is not audited — nothing happened — but a failed
    sign-in is the only trace a guessing run leaves at all."""
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    client.post("/login", data={"username": "ana", "password": "nope", "next": "/"})
    client.post("/login", data={"username": "ghost", "password": "nope", "next": "/"})
    with Session() as s:
        rows = s.scalars(select(AuditLog).order_by(AuditLog.id)).all()
        assert [r.status_code for r in rows] == [401, 401]
        assert [r.username for r in rows] == ["(failed:ana)", "(failed:ghost)"]
        assert all(r.user_id is None for r in rows)


def test_a_hostile_username_cannot_forge_an_audit_actor():
    assert failed_actor(" ana ") == "(failed:ana)"
    assert "\n" not in failed_actor("ana\nroot")
    assert len(failed_actor("x" * 5000)) == FAILED_ACTOR_MAX + len("(failed:)")


def test_throttled_attempts_do_not_grow_the_audit_log(tmp_path):
    """Otherwise the answer to hammering is unbounded table growth."""
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    for _ in range(LOGIN_FREE_ATTEMPTS + 6):
        client.post("/login", data={"username": "ana", "password": "no", "next": "/"})
    with Session() as s:
        assert len(s.scalars(select(AuditLog)).all()) == LOGIN_FREE_ATTEMPTS + 1


# -- session lifecycle (CLD-65) --------------------------------------------


def test_purge_deletes_only_expired_sessions(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    login(client, "ana")
    with Session() as s:
        s.add(
            WebSession(
                token="stale",
                user_id=s.scalar(select(User.id)),
                created_at=datetime(2020, 1, 1),
                expires_at=datetime(2020, 1, 15),
            )
        )
        s.commit()
        assert purge_expired_sessions(s) == 1
        s.commit()
        rows = s.scalars(select(WebSession)).all()
        assert len(rows) == 1 and rows[0].token != "stale"


def test_expired_sessions_do_not_accumulate(tmp_path):
    """Rows are only created by signing in, so signing in is where the
    dead ones get cleared: the table cannot grow unattended."""
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    for _ in range(3):
        login(client, "ana")
        with Session() as s:
            for row in s.scalars(select(WebSession)):
                row.expires_at = datetime(2020, 1, 1)
            s.commit()
    assert client.get("/").status_code == 303  # the expired cookie is rejected
    login(client, "ana")
    with Session() as s:
        rows = s.scalars(select(WebSession)).all()
        assert len(rows) == 1
        assert rows[0].expires_at > datetime(2026, 8, 6)


def test_logout_clears_expired_rows_too(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "ana", "admin")
    login(client, "ana")
    with Session() as s:
        s.add(
            WebSession(
                token="stale",
                user_id=s.scalar(select(User.id)),
                created_at=datetime(2020, 1, 1),
                expires_at=datetime(2020, 1, 15),
            )
        )
        s.commit()
    client.post("/logout")
    with Session() as s:
        assert s.scalars(select(WebSession)).all() == []


def test_enabling_auth_takes_effect_on_a_running_console(tmp_path):
    """auth_enabled is cached, and only ever in the "on" direction: a
    cached "off" would leave the console open after `siteloom users add`,
    which is the one moment staleness would matter."""
    client, Session = make_env(tmp_path)
    assert client.get("/").status_code == 200  # open mode, answer cached
    add_user(Session, "ana", "admin")
    assert client.get("/").status_code == 303
