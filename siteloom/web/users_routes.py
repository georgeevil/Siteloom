"""Operator accounts and the audit trail (CLD-31).

Account *creation* stays CLI-only, deliberately: there is no web path to
the first account, and a web-creatable account is a web-creatable admin.
What this adds is everything after the first account — seeing who exists,
changing a role, disabling someone who left, revoking a live session, and
reading back who did what — none of which had a screen, and all of which
an admin needed a shell for.

Four decisions shape it:

* **The screen says what it cannot do.** `siteloom users add` is printed
  on the page rather than left as a missing button, because "there is no
  Add button" and "the Add button is somewhere else" look identical to an
  operator hunting for one.
* **An admin cannot remove their own admin or disable their own
  account.** A console with no reachable admin is recovered only from a
  shell, and the two self-inflicted ways to get there are the ones a
  screen makes easy. The shell keeps both powers (`siteloom users role`,
  `siteloom users enable`) — that is the recovery path, not an oversight.
* **Nothing here writes an audit row.** Every mutation below is a POST
  that goes through the one middleware in `create_app`, which is what
  already audits it. A second call site would double-count and could
  drift from the first.
* **The audit viewer reads `AuditLog.username`, never a join.** That
  column is denormalized on purpose: the row must still say who acted
  after the account is renamed or deleted, and a join to `users` would
  quietly turn a departed operator's actions into blanks.

The fourth rung, `restricted`, is enforced in `web/auth.py` rather than
here — a prefix floor beside `ADMIN_PREFIXES`, checked in the same single
middleware. Two consequences are worth naming rather than discovering:

* `restricted` sits *below* `view`, so it grants no mutation. It can read
  the triage queue, an event, the live view; it cannot record a verdict
  or clear an event, because mutations need `edit` and a ladder cannot
  give a lower rung a power a higher rung lacks. "Triage" is therefore
  read-only triage.
* The sidebar is global (`web/nav.py`), so a restricted operator still
  sees Identities and Training data listed and gets a 403 on clicking
  them. Filtering the sidebar per role is a nav.py change, not this one.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import delete, func, select

from siteloom.store import AuditLog, User, WebSession
from siteloom.web import auth, nav

#: Rows the audit viewer renders. The table is append-only and grows with
#: every mutation, so the page is a window on the recent past; the
#: filters are how you reach further back.
AUDIT_LIMIT = 200


class UserActionError(ValueError):
    """A submission the screen refuses, with a reason to show."""


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def guard_self_change(actor: User | None, target: User, role: str) -> None:
    """Refuse the two changes that can lock the console (CLD-31).

    Both are the same failure: an admin removing the last power to undo
    it from inside the console. Refusing only for *self* is deliberate —
    an admin may demote or disable a colleague, because they remain to
    reverse it, and the self-guards make "no enabled admin left" reachable
    only from a shell, which is where the recovery lives too.
    """
    if actor is None or actor.id != target.id:
        return
    if role != "admin":
        raise UserActionError(
            "You cannot remove your own admin. A console with no reachable "
            "admin can only be recovered from a shell — ask another admin, "
            "or run `siteloom users role <name> <role>` on the host."
        )


def guard_self_disable(actor: User | None, target: User, disabled: bool) -> None:
    if disabled and actor is not None and actor.id == target.id:
        raise UserActionError(
            "You cannot disable your own account. Sign out instead, or have "
            "another admin disable it — `siteloom users enable <name>` is "
            "the only way back from disabling the last admin."
        )


def register(app, templates, Session, config) -> None:
    # app.py owns the LIKE-escaping helper (CLD-67): the operator's filter
    # text is a literal substring, never a pattern, and borrowing the
    # builder rather than re-spelling it here means `%` typed into the
    # path filter cannot come to mean "any run". Imported inside
    # register() because create_app imports this module, not the reverse.
    from siteloom.web.app import _contains

    nav.add("/users", "Operators", "OP")

    def _ctx(**extra) -> dict:
        return {
            "site_name": config.site_name or config.site_id,
            "roles": list(auth.ROLES),
            "role_labels": auth.ROLE_LABELS,
            "role_summaries": auth.ROLE_SUMMARIES,
            **extra,
        }

    def _rows(session) -> list[dict]:
        """Every account, with what the sessions table knows about it.

        Last sign-in is derived from `WebSession.created_at` rather than
        stamped on the User row: the session table already records it, and
        a denormalized copy is one more thing that can disagree with the
        sessions an admin is looking at on the same line. The cost is that
        it is only as long-lived as the rows — an expired session is
        purged at the next sign-in, and revoking below deletes them — so
        the column reads "never" for an operator whose sessions are all
        gone, which the page says rather than implying they never came.
        """
        now = _now()
        last = dict(
            session.execute(
                select(WebSession.user_id, func.max(WebSession.created_at)).group_by(
                    WebSession.user_id
                )
            ).all()
        )
        live = dict(
            session.execute(
                select(WebSession.user_id, func.count())
                .where(WebSession.expires_at >= now)
                .group_by(WebSession.user_id)
            ).all()
        )
        return [
            {
                "user": u,
                "label": auth.role_label(u.role),
                "last_seen": last.get(u.id),
                "sessions": live.get(u.id, 0),
            }
            for u in session.scalars(select(User).order_by(User.username)).all()
        ]

    def _page(request: Request, error: str | None = None, status: int = 200):
        with Session() as session:
            rows = _rows(session)
        actor = getattr(request.state, "user", None)
        return templates.TemplateResponse(
            request,
            "users.html",
            _ctx(
                rows=rows,
                # The screen must be honest about the mode it is in: with
                # no accounts the console is in open single-operator mode
                # and this page is a description of a switch nobody has
                # thrown, not an empty list of colleagues.
                auth_on=bool(rows),
                actor_id=actor.id if actor else None,
                error=error,
            ),
            status_code=status,
        )

    @app.get("/users", response_class=HTMLResponse)
    def users_page(request: Request):
        return _page(request)

    def _target(session, user_id: int) -> User:
        user = session.get(User, user_id)
        if user is None:
            raise HTTPException(404)
        return user

    @app.post("/users/{user_id}/role")
    def set_user_role(request: Request, user_id: int, role: str = Form(...)):
        """Move an account up or down the ladder.

        Stored values are `restricted`/`view`/`edit`/`admin` — the screen
        shows Viewer/Writer/Admin, and this validates against `auth.ROLES`
        so a label posted back by hand cannot land in the column.
        """
        if role not in auth.ROLES:
            raise HTTPException(400, f"role must be one of {', '.join(auth.ROLES)}")
        with Session() as session:
            user = _target(session, user_id)
            try:
                guard_self_change(getattr(request.state, "user", None), user, role)
            except UserActionError as exc:
                return _page(request, error=str(exc), status=400)
            user.role = role
            session.commit()
        return RedirectResponse("/users", status_code=303)

    @app.post("/users/{user_id}/disabled")
    def set_user_disabled(request: Request, user_id: int, disabled: str = Form(...)):
        """Disable an account, or let it back in.

        Disabling revokes the account's live sessions in the same
        transaction. Anything else would leave a departed operator's
        browser working until its cookie expired, which is the whole
        thing the button is for; `resolve_user` also refuses a disabled
        account, so this is belt and braces on purpose.
        """
        off = disabled == "1"
        with Session() as session:
            user = _target(session, user_id)
            try:
                guard_self_disable(getattr(request.state, "user", None), user, off)
            except UserActionError as exc:
                return _page(request, error=str(exc), status=400)
            user.disabled = off
            if off:
                session.execute(delete(WebSession).where(WebSession.user_id == user.id))
            session.commit()
        return RedirectResponse("/users", status_code=303)

    @app.post("/users/{user_id}/revoke")
    def revoke_user_sessions(request: Request, user_id: int):
        """Sign an account out everywhere without changing what it may do.

        The cookie carries only an opaque token whose server-side row this
        deletes, so the next request from that browser resolves to nobody
        — which is what makes a shared or forgotten session recoverable
        without a password reset. Revoking your own is allowed: it logs
        you out, and signing back in is the whole cost.
        """
        with Session() as session:
            user = _target(session, user_id)
            session.execute(delete(WebSession).where(WebSession.user_id == user.id))
            session.commit()
        return RedirectResponse("/users", status_code=303)

    @app.get("/audit", response_class=HTMLResponse)
    def audit_page(request: Request, actor: str = "", path: str = ""):
        """Who changed what, when.

        Both filters are substring matches on the row's own columns.
        `actor` reads `AuditLog.username` — the denormalized copy, never a
        join to `users` — because the point of that column is that the log
        still names an operator whose account was renamed or deleted, and
        a join would erase exactly the history an audit trail exists for.
        The same column is what carries `(open)` for mutations made before
        auth was enabled and `(failed:name)` for a sign-in that never
        succeeded, neither of which has a User row to join to at all.
        """
        actor, path = actor.strip()[:200], path.strip()[:200]
        with Session() as session:
            q = select(AuditLog)
            if actor:
                q = q.filter(_contains(AuditLog.username, actor))
            if path:
                q = q.filter(_contains(AuditLog.path, path))
            total = session.scalar(select(func.count()).select_from(q.subquery())) or 0
            rows = (
                session.scalars(q.order_by(AuditLog.id.desc()).limit(AUDIT_LIMIT))
                .unique()
                .all()
            )
            # The actor picker is built from the log, not from the user
            # table, for the same reason the filter is: the names worth
            # offering include the ones with no account behind them.
            actors = sorted(
                {a for (a,) in session.execute(select(AuditLog.username).distinct())}
            )
        return templates.TemplateResponse(
            request,
            "audit.html",
            _ctx(
                rows=rows,
                actors=actors,
                total=total,
                shown=len(rows),
                limit=AUDIT_LIMIT,
                filters={"actor": actor, "path": path},
                filtered=bool(actor or path),
            ),
        )
