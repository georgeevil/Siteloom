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
from siteloom.web.app import create_app
from siteloom.web.auth import hash_password, required_role, verify_password

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


def test_admin_can_reconfigure(tmp_path):
    client, Session = make_env(tmp_path)
    add_user(Session, "root", "admin")
    login(client, "root")
    assert (
        client.post("/classes/detection", json={"classes": ["person"]}).status_code
        == 200
    )


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
