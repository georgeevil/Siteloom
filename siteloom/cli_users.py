"""User management for the operator console (CLD auth milestone).

Creating the first user is the switch that turns authentication on —
until then the console runs in the open single-operator mode. That makes
`users add` the enrolment step and the reason these commands live in the
CLI: there is no web path to create the first account, by design.

Everything *after* the first account now also has a screen (`/users`),
and these commands are deliberately not retired by it. The screen refuses
to let an admin remove their own admin or disable their own account,
because a console with no reachable admin is recoverable only from a
shell — so the shell has to be able to do it. `users role` and
`users enable` are that recovery path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
from sqlalchemy import select

from siteloom.web.auth import ROLE_SUMMARIES, ROLES, hash_password, role_label

#: Built from the ladder rather than typed out, so a rung added to ROLES
#: cannot go unmentioned in the help text of the commands that set it.
ROLE_HELP = "one of " + ", ".join(ROLES)

users_app = typer.Typer(
    help="Operator accounts (restricted / view / edit / admin)."
)

CONFIG_OPT = typer.Option("site.yaml", "--config", "-c", help="Site config YAML")


def _check_role(role: str) -> str:
    if role not in ROLES:
        raise typer.BadParameter(f"role must be one of {', '.join(ROLES)}")
    return role


def _session(config: Path):
    from siteloom.config import load_config
    from siteloom.store import get_session, init_db, make_engine

    cfg = load_config(config)
    engine = make_engine(cfg.storage.db_url)
    init_db(engine)
    return get_session(engine)


@users_app.command("add")
def add_user(
    username: str,
    role: str = typer.Option("view", help=ROLE_HELP),
    password: str = typer.Option(
        ..., prompt=True, hide_input=True, confirmation_prompt=True
    ),
    config: Path = CONFIG_OPT,
):
    """Create an account. The first account enables authentication."""
    from siteloom.store import User

    _check_role(role)
    Session = _session(config)
    with Session() as session:
        if session.scalar(select(User).filter_by(username=username)):
            typer.echo(f"user {username!r} already exists", err=True)
            raise typer.Exit(1)
        first = session.scalar(select(User.id).limit(1)) is None
        session.add(
            User(
                username=username,
                password_hash=hash_password(password),
                role=role,
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        session.commit()
    typer.echo(f"created {username} ({role})")
    if first:
        typer.echo(
            "authentication is now ON — every console page requires sign-in"
        )


@users_app.command("list")
def list_users(config: Path = CONFIG_OPT):
    """List accounts. An empty list means open access."""
    from siteloom.store import User

    Session = _session(config)
    with Session() as session:
        users = session.scalars(select(User).order_by(User.username)).all()
    if not users:
        typer.echo("no users — authentication is off (open access)")
        return
    for u in users:
        flags = " (disabled)" if u.disabled else ""
        # Stored value and operator-facing label side by side: the screen
        # says "Viewer", the database says "view", and a command whose
        # output cannot be pasted back into `users role` is a trap.
        typer.echo(f"{u.username}\t{u.role} ({role_label(u.role)}){flags}")


@users_app.command("role")
def set_role(username: str, role: str, config: Path = CONFIG_OPT):
    """Set an account's role — the shell half of the /users screen.

    Deliberately without the screen's self-guards: this *is* the recovery
    path for a console whose last admin demoted themselves, and a shell
    prompt is already the "you can break this" boundary.
    """
    from siteloom.store import User

    _check_role(role)
    Session = _session(config)
    with Session() as session:
        user = session.scalar(select(User).filter_by(username=username))
        if user is None:
            typer.echo(f"no such user {username!r}", err=True)
            raise typer.Exit(1)
        was = user.role
        user.role = role
        session.commit()
    typer.echo(f"{username}: {was} -> {role} ({role_label(role)})")


@users_app.command("enable")
def enable_user(username: str, config: Path = CONFIG_OPT):
    """Re-enable a disabled account. Sessions stay revoked — signing in
    again is what mints a new one."""
    from siteloom.store import User

    Session = _session(config)
    with Session() as session:
        user = session.scalar(select(User).filter_by(username=username))
        if user is None:
            typer.echo(f"no such user {username!r}", err=True)
            raise typer.Exit(1)
        user.disabled = False
        session.commit()
    typer.echo(f"enabled {username}")


@users_app.command("passwd")
def change_password(
    username: str,
    password: str = typer.Option(
        ..., prompt=True, hide_input=True, confirmation_prompt=True
    ),
    config: Path = CONFIG_OPT,
):
    """Reset an account's password and revoke its sessions."""
    from siteloom.store import User, WebSession

    Session = _session(config)
    with Session() as session:
        user = session.scalar(select(User).filter_by(username=username))
        if user is None:
            typer.echo(f"no such user {username!r}", err=True)
            raise typer.Exit(1)
        user.password_hash = hash_password(password)
        for row in session.scalars(select(WebSession).filter_by(user_id=user.id)):
            session.delete(row)
        session.commit()
    typer.echo(f"password updated for {username}; sessions revoked")


@users_app.command("disable")
def disable_user(username: str, config: Path = CONFIG_OPT):
    """Disable an account (kept for the audit trail, cannot sign in)."""
    from siteloom.store import User, WebSession

    Session = _session(config)
    with Session() as session:
        user = session.scalar(select(User).filter_by(username=username))
        if user is None:
            typer.echo(f"no such user {username!r}", err=True)
            raise typer.Exit(1)
        user.disabled = True
        for row in session.scalars(select(WebSession).filter_by(user_id=user.id)):
            session.delete(row)
        session.commit()
    typer.echo(f"disabled {username}")
