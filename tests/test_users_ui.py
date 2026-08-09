"""The /users screen and the audit viewer (CLD-31).

The screen exists because everything *after* the first account had no
home: an admin could not see who existed, change a role, disable someone
who left or cut a live session without a shell on the host. Creating an
account stays in the shell, deliberately, and the screen says so.

The properties worth holding are the ones that are cheap to break:

* An admin cannot lock the console by removing their own admin or
  disabling their own account. The recovery is a shell, so the screen
  must not be the way in.
* Revoking a session actually invalidates the cookie — not "marks it
  revoked", which is a different and useless thing.
* The audit viewer reads the denormalized `AuditLog.username`, so it
  still names an operator whose account has been deleted.
* None of it needs an account to exist: with an empty users table the
  console is in open single-operator mode and every page still renders.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import select
from typer.testing import CliRunner

from siteloom.cli_users import users_app
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
from siteloom.web.auth import hash_password
from siteloom.web.users_routes import guard_self_change, guard_self_disable

runner = CliRunner()
TS = datetime(2026, 8, 6, 9, 0, 0)


@pytest.fixture
def env(tmp_path):
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/u.db", media_dir=str(tmp_path / "m")
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
    return {
        "app": create_app(config),
        "Session": Session,
        "config": config,
        "tmp_path": tmp_path,
    }


def add_user(env, username, role):
    with env["Session"]() as s:
        s.add(
            User(
                username=username,
                password_hash=hash_password("hunter2!"),
                role=role,
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        s.commit()
        return s.scalar(select(User.id).filter_by(username=username))


def client_for(env, username, role=None):
    """A separate browser per operator — cookies do not leak between them."""
    if role is not None:
        add_user(env, username, role)
    client = TestClient(env["app"], follow_redirects=False)
    r = client.post(
        "/login", data={"username": username, "password": "hunter2!", "next": "/"}
    )
    assert r.status_code == 303, r.text[:200]
    return client


def role_of(env, username) -> str:
    with env["Session"]() as s:
        return s.scalar(select(User.role).filter_by(username=username))


# -- the screen ------------------------------------------------------------


def test_open_mode_renders_the_screen_with_no_accounts(env):
    """A feature that needs a login to function is a bug: with an empty
    users table the console is open, and this page explains that rather
    than showing an empty list of colleagues."""
    client = TestClient(env["app"], follow_redirects=False)
    r = client.get("/users")
    assert r.status_code == 200
    assert "open single-operator mode" in r.text
    assert "siteloom users add" in r.text


def test_the_screen_lists_accounts_with_role_state_and_last_sign_in(env):
    admin = client_for(env, "ada", "admin")
    add_user(env, "vera", "view")
    client_for(env, "nina", "restricted")  # signs in, so it has a session row
    with env["Session"]() as s:
        s.scalar(select(User).filter_by(username="vera")).disabled = True
        s.commit()

    body = admin.get("/users").text
    for name in ("ada", "vera", "nina"):
        assert name in body
    # Operator vocabulary on screen, stored values in the form.
    assert "Viewer" in body and "Writer" in body and "Admin" in body
    assert 'value="restricted"' in body and 'value="edit"' in body
    assert "disabled" in body
    # Last sign-in comes from WebSession.created_at, so the two who have
    # signed in show a date and the one who never has does not.
    assert body.count("never / expired") == 1


def test_only_an_admin_reaches_the_screen(env):
    add_user(env, "ada", "admin")  # auth on
    for name, role in (("vera", "view"), ("eddy", "edit"), ("nina", "restricted")):
        client = client_for(env, name, role)
        assert client.get("/users").status_code == 403, name
        assert client.get("/audit").status_code == 403, name


def test_an_admin_changes_someone_elses_role(env):
    admin = client_for(env, "ada", "admin")
    vera = add_user(env, "vera", "view")
    r = admin.post(f"/users/{vera}/role", data={"role": "edit"})
    assert r.status_code == 303
    assert role_of(env, "vera") == "edit"
    # And the fourth rung is settable from the screen, not only the CLI.
    admin.post(f"/users/{vera}/role", data={"role": "restricted"})
    assert role_of(env, "vera") == "restricted"


def test_an_unknown_role_is_refused(env):
    """The screen shows labels; the column stores values. A label posted
    back by hand must not land in it."""
    admin = client_for(env, "ada", "admin")
    vera = add_user(env, "vera", "view")
    assert admin.post(f"/users/{vera}/role", data={"role": "Viewer"}).status_code == 400
    assert role_of(env, "vera") == "view"


def test_role_changes_are_audited_by_the_existing_middleware(env):
    """No audit call site was added: the POST goes through the one
    middleware that already writes a row for every mutation."""
    admin = client_for(env, "ada", "admin")
    vera = add_user(env, "vera", "view")
    admin.post(f"/users/{vera}/role", data={"role": "edit"})
    admin.post(f"/users/{vera}/disabled", data={"disabled": "1"})
    with env["Session"]() as s:
        actions = {(r.username, r.path) for r in s.scalars(select(AuditLog))}
    assert ("ada", f"/users/{vera}/role") in actions
    assert ("ada", f"/users/{vera}/disabled") in actions


# -- the self-lock guards --------------------------------------------------


def test_an_admin_cannot_remove_their_own_admin(env):
    admin = client_for(env, "ada", "admin")
    with env["Session"]() as s:
        me = s.scalar(select(User.id).filter_by(username="ada"))
    r = admin.post(f"/users/{me}/role", data={"role": "view"})
    assert r.status_code == 400
    assert "cannot remove your own admin" in r.text
    assert role_of(env, "ada") == "admin"


def test_an_admin_cannot_disable_their_own_account(env):
    admin = client_for(env, "ada", "admin")
    with env["Session"]() as s:
        me = s.scalar(select(User.id).filter_by(username="ada"))
    r = admin.post(f"/users/{me}/disabled", data={"disabled": "1"})
    assert r.status_code == 400
    assert "cannot disable your own account" in r.text
    with env["Session"]() as s:
        assert s.get(User, me).disabled is False


def test_a_refused_self_change_is_not_audited(env):
    """Denied requests are not audited — nothing happened."""
    admin = client_for(env, "ada", "admin")
    with env["Session"]() as s:
        me = s.scalar(select(User.id).filter_by(username="ada"))
    admin.post(f"/users/{me}/role", data={"role": "view"})
    with env["Session"]() as s:
        paths = [r.path for r in s.scalars(select(AuditLog))]
    assert f"/users/{me}/role" not in paths


def test_the_guards_are_about_self_not_about_admins():
    """An admin may demote or disable a colleague: they remain, able to
    reverse it. Only the last power to undo from inside is protected."""
    me = User(id=1, username="ada", password_hash="x", role="admin", created_at=TS)
    them = User(id=2, username="bo", password_hash="x", role="admin", created_at=TS)
    guard_self_change(me, them, "view")  # no raise
    guard_self_disable(me, them, True)
    guard_self_change(me, me, "admin")  # staying admin is not a change
    guard_self_disable(me, me, False)  # re-enabling yourself is harmless
    with pytest.raises(ValueError):
        guard_self_change(me, me, "edit")
    with pytest.raises(ValueError):
        guard_self_disable(me, me, True)


# -- sessions --------------------------------------------------------------


def test_revoking_sessions_invalidates_a_live_cookie(env):
    admin = client_for(env, "ada", "admin")
    vera = client_for(env, "vera", "view")
    assert vera.get("/").status_code == 200
    with env["Session"]() as s:
        vera_id = s.scalar(select(User.id).filter_by(username="vera"))
    assert admin.post(f"/users/{vera_id}/revoke").status_code == 303
    # The cookie still exists in the browser; the row behind it does not.
    assert vera.get("/").status_code == 303
    with env["Session"]() as s:
        assert s.scalars(select(WebSession).filter_by(user_id=vera_id)).all() == []
    # And the account is untouched: signing in again works.
    assert client_for(env, "vera").get("/").status_code == 200


def test_disabling_cuts_the_session_and_re_enabling_lets_them_back(env):
    admin = client_for(env, "ada", "admin")
    vera = client_for(env, "vera", "view")
    with env["Session"]() as s:
        vera_id = s.scalar(select(User.id).filter_by(username="vera"))
    admin.post(f"/users/{vera_id}/disabled", data={"disabled": "1"})
    assert vera.get("/").status_code == 303
    with env["Session"]() as s:
        assert s.scalars(select(WebSession).filter_by(user_id=vera_id)).all() == []
    # A disabled account cannot sign in at all.
    assert (
        vera.post(
            "/login", data={"username": "vera", "password": "hunter2!", "next": "/"}
        ).status_code
        == 401
    )
    admin.post(f"/users/{vera_id}/disabled", data={"disabled": "0"})
    assert client_for(env, "vera").get("/").status_code == 200


def test_an_expired_session_does_not_count_as_live(env):
    """The count beside Revoke is what an admin acts on, so it has to mean
    "browsers that still work", not "rows that still exist"."""
    admin = client_for(env, "ada", "admin")
    client_for(env, "vera", "view")
    with env["Session"]() as s:
        for row in s.scalars(select(WebSession)):
            if row.user_id != s.scalar(select(User.id).filter_by(username="ada")):
                row.expires_at = datetime.now(timezone.utc).replace(
                    tzinfo=None
                ) - timedelta(days=1)
        s.commit()
    body = admin.get("/users").text
    assert "<b>1</b> live session" in body  # the admin's own, not vera's
    # vera's row still dates her last sign-in, though: the row that
    # recorded it is expired, not deleted.
    assert "never / expired" not in body


# -- the audit viewer ------------------------------------------------------


def test_the_audit_viewer_shows_actions_and_filters_them(env):
    admin = client_for(env, "ada", "admin")
    vera = add_user(env, "vera", "view")
    admin.post(f"/users/{vera}/role", data={"role": "edit"})

    body = admin.get("/audit").text
    assert "ada" in body and f"/users/{vera}/role" in body
    assert "/login" in body

    only_login = admin.get("/audit", params={"path": "/login"}).text
    assert "/login" in only_login
    assert f"/users/{vera}/role" not in only_login

    other = admin.get("/audit", params={"actor": "vera"}).text
    assert "No audited action matches this filter" in other


def test_the_audit_viewer_still_names_a_deleted_user(env):
    """`AuditLog.username` is denormalized precisely so the trail survives
    the account. A viewer that joined to `users` would blank exactly the
    history someone is looking for."""
    admin = client_for(env, "ada", "admin")
    eddy = client_for(env, "eddy", "edit")
    assert eddy.post("/events/1/review", data={"reviewed": "1"}).status_code == 303
    with env["Session"]() as s:
        s.delete(s.scalar(select(User).filter_by(username="eddy")))
        s.commit()

    filtered = admin.get("/audit", params={"actor": "eddy"})
    assert filtered.status_code == 200
    # The action is still there, and still has a name on it.
    assert "eddy" in filtered.text
    assert "/events/1/review" in filtered.text


def test_a_failed_sign_in_has_no_account_to_join_to(env):
    admin = client_for(env, "ada", "admin")
    admin.post("/login", data={"username": "ghost", "password": "no", "next": "/"})
    body = admin.get("/audit").text
    assert "(failed:ghost)" in body


def test_audit_filter_text_is_a_literal_not_a_pattern(env):
    """`%` typed into the path filter must match a percent sign, not
    everything — the LIKE-escaping rule from CLD-67."""
    admin = client_for(env, "ada", "admin")
    r = admin.get("/audit", params={"path": "%"})
    assert r.status_code == 200
    assert "No audited action matches this filter" in r.text


# -- the shell half --------------------------------------------------------


def test_the_cli_can_recover_a_console_whose_admin_demoted_themselves(env):
    """The screen's self-guards are only safe because this exists."""
    add_user(env, "ada", "admin")
    config_file = env["tmp_path"] / "site.yaml"
    config_file.write_text(yaml.safe_dump(env["config"].model_dump(mode="json")))

    with env["Session"]() as s:  # a shell already demoted them
        s.scalar(select(User).filter_by(username="ada")).role = "view"
        s.commit()
    result = runner.invoke(
        users_app, ["role", "ada", "admin", "--config", str(config_file)]
    )
    assert result.exit_code == 0, result.output
    assert role_of(env, "ada") == "admin"

    assert (
        runner.invoke(
            users_app, ["role", "ada", "wizard", "--config", str(config_file)]
        ).exit_code
        != 0
    )
    assert role_of(env, "ada") == "admin"

    runner.invoke(users_app, ["disable", "ada", "--config", str(config_file)])
    with env["Session"]() as s:
        assert s.scalar(select(User).filter_by(username="ada")).disabled
    runner.invoke(users_app, ["enable", "ada", "--config", str(config_file)])
    with env["Session"]() as s:
        assert not s.scalar(select(User).filter_by(username="ada")).disabled


def test_the_cli_can_create_a_restricted_account(env):
    config_file = env["tmp_path"] / "site.yaml"
    config_file.write_text(yaml.safe_dump(env["config"].model_dump(mode="json")))
    result = runner.invoke(
        users_app,
        [
            "add",
            "nina",
            "--role",
            "restricted",
            "--password",
            "hunter2!",
            "--config",
            str(config_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert role_of(env, "nina") == "restricted"
    listing = runner.invoke(users_app, ["list", "--config", str(config_file)]).output
    assert "restricted (Restricted)" in listing
