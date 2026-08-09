"""Incidents (CLD-96) — the record the triage rail's Escalate opens.

The triage rail's Escalate action had never been built, because nobody had
answered "escalate to whom". The answer, decided 2026-08-09: to a record,
not a person. For a property manager there is nobody on the other end
immediately, and somebody later — an insurer, the police, a tenant dispute
weeks on.

That makes an incident its own object rather than a fifth Event status.
One incident links many events across cameras, carries the operator's
prose, has an open/closed lifecycle, and exports as something a reader
outside the console can use. A per-event flag can express none of that,
which is the whole reason to build it (see `store/models.Incident`).

Four decisions shape this module:

* **Adding to an existing incident is the common case**, so it is the
  default and not a second step behind "create". The rail's control is
  one select with the most recent open incident already chosen and
  "+ new incident" as the last option — the daily interaction is pick
  nothing and click once.
* **The rail gets its options through a Jinja global**, the same way the
  sidebar does, because `_rail_context` lives in app.py and a screen that
  has to be wired into another module's context builder is a screen that
  renders wrong the day someone forgets.
* **Deleting an incident detaches its events, never deletes them.** The
  events are the record; the incident is an interpretation of them.
* **The export is one self-contained HTML file** — see `export_html`.
"""

from __future__ import annotations

import base64
import html
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from siteloom.store import Camera, Event, EventIdentity
from siteloom.store.models import Incident, IncidentNote
from siteloom.web import nav

log = logging.getLogger(__name__)

#: How many open incidents the triage rail offers. The rail is a narrow
#: column and the answer an operator wants is nearly always the one they
#: opened minutes ago; a longer list would be scrolling, not choosing.
RAIL_CHOICES = 8

#: Cap on a single embedded crop. A bundle is meant to be mailed, and one
#: pathological full-frame JPEG must not be what makes it unmailable.
MAX_EMBED_BYTES = 2_000_000


def _now() -> datetime:
    """Naive UTC — the tz-free convention every stored timestamp uses."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def actor(request: Request) -> str:
    """Who is doing this, for the record.

    Denormalized into the row rather than joined to a user id, like
    `AuditLog.username`: an incident must still say who judged it after
    the account is renamed or deleted. "(open)" is the honest answer
    before the first `siteloom users add` — the console is genuinely
    single-operator then, and inventing a name would be worse.
    """
    user = getattr(request.state, "user", None)
    return user.username if user is not None else "(open)"


def default_title(event: Event, camera: Camera | None) -> str:
    """A title for an incident opened straight off one event.

    Never blank: the list screen and every export are read by someone who
    was not there, and "Incident 7" tells them nothing. The operator can
    type a better one, and usually should.
    """
    where = camera.name if camera is not None and camera.name else event.camera_id
    return f"{event.class_name} at {where}, {event.first_seen.strftime('%Y-%m-%d %H:%M')}"


def crop_bytes(raw: str | None, media_root: Path) -> bytes | None:
    """Read a stored crop for embedding, confined to media_dir.

    Reuses `/media`'s own path reading (`_media_candidates`) and repeats
    its containment check rather than trusting the column: the export
    turns a database string into bytes inside a file that gets mailed
    out, which is exactly the wrong place to widen what is readable.
    """
    from siteloom.web.app import _media_candidates

    if not raw:
        return None
    for candidate in _media_candidates(raw, media_root):
        full = candidate.resolve()
        if full.is_relative_to(media_root) and full.is_file():
            try:
                data = full.read_bytes()
            except OSError as exc:  # unreadable file must not 500 an export
                log.warning("incident export: cannot read crop %s (%s)", full, exc)
                return None
            return data if len(data) <= MAX_EMBED_BYTES else None
    return None


def export_html(bundle: dict) -> str:
    """The downloadable bundle: one self-contained HTML file.

    Format choice. The artifact is handed to an insurer, a police officer
    or a tenant's solicitor — people with a browser and nothing else — so
    the requirements are: opens with a double click, survives being
    forwarded as an attachment, needs no server, and prints to PDF. A
    single HTML file with the crops inlined as `data:` URIs meets all
    four. A zip does not: it arrives as a folder of loose JPEGs whose
    filenames are the only thing tying them to a timeline, and half of
    the recipients above cannot open one on the device they read mail on.
    JSON and CSV lose the images entirely, and a PDF would put a rendering
    dependency in the request path for a document nobody wants to edit.

    Written here rather than as a Jinja template on purpose: this file
    leaves the console and must not inherit base.html, its stylesheet
    link or any other reference to a host the reader cannot reach.
    """
    esc = html.escape
    inc = bundle["incident"]
    parts: list[str] = []
    parts.append(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>Incident {inc['id']} — {esc(inc['title'])}</title>\n"
        "<style>\n"
        "  body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;"
        " color: #14171c; background: #fff; margin: 0; padding: 32px; line-height: 1.55; }\n"
        "  main { max-width: 900px; margin: 0 auto; }\n"
        "  h1 { font-size: 20px; margin: 0 0 4px; }\n"
        "  h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .07em;"
        " color: #5a6472; margin: 28px 0 8px; }\n"
        "  .sub { color: #5a6472; font-size: 13px; margin: 0 0 18px; }\n"
        "  dl.facts { display: grid; grid-template-columns: 150px 1fr; gap: 4px 14px;"
        " font-size: 13px; margin: 0 0 8px; }\n"
        "  dt { color: #5a6472; }\n"
        "  dd { margin: 0; }\n"
        "  table { border-collapse: collapse; width: 100%; font-size: 13px; }\n"
        "  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #e3e6ea;"
        " vertical-align: top; }\n"
        "  th { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;"
        " color: #5a6472; }\n"
        "  img { width: 130px; height: auto; border: 1px solid #e3e6ea; border-radius: 4px;"
        " display: block; }\n"
        "  .nocrop { color: #8a93a0; font-size: 12px; }\n"
        "  .note { border-left: 3px solid #c9ced6; padding: 2px 0 2px 12px; margin: 0 0 14px; }\n"
        "  .note .who { color: #5a6472; font-size: 12px; }\n"
        "  .note.close { border-left-color: #2f7d4f; }\n"
        "  .note.reopen { border-left-color: #b06a12; }\n"
        "  .note p { margin: 4px 0 0; white-space: pre-wrap; }\n"
        "  footer { margin-top: 32px; border-top: 1px solid #e3e6ea; padding-top: 10px;"
        " color: #5a6472; font-size: 12px; }\n"
        "  @media print { body { padding: 0; } }\n"
        "</style>\n</head>\n<body>\n<main>\n"
    )
    parts.append(f"<h1>{esc(inc['title'])}</h1>\n")
    parts.append(
        f"<p class=\"sub\">Incident {inc['id']} · {esc(inc['status'])} · "
        f"{esc(bundle['site_name'])}</p>\n"
    )
    parts.append("<dl class=\"facts\">")
    for label, value in bundle["facts"]:
        parts.append(f"<dt>{esc(label)}</dt><dd>{esc(value)}</dd>")
    parts.append("</dl>\n")

    parts.append("<h2>Timeline</h2>\n")
    if bundle["events"]:
        parts.append(
            "<table>\n<tr><th>Time (UTC)</th><th>Camera</th><th>What</th>"
            "<th>Identified as</th><th>Crop</th></tr>\n"
        )
        for row in bundle["events"]:
            if row["crop"]:
                encoded = base64.b64encode(row["crop"]).decode("ascii")
                cell = f'<img src="data:image/jpeg;base64,{encoded}" alt="">'
            else:
                cell = '<span class="nocrop">no crop stored</span>'
            parts.append(
                "<tr>"
                f"<td>{esc(row['when'])}</td>"
                f"<td>{esc(row['camera'])}</td>"
                f"<td>{esc(row['what'])}</td>"
                f"<td>{esc(row['identities'])}</td>"
                f"<td>{cell}</td>"
                "</tr>\n"
            )
        parts.append("</table>\n")
    else:
        parts.append("<p class=\"nocrop\">No events are attached to this incident.</p>\n")

    parts.append("<h2>Notes</h2>\n")
    if bundle["notes"]:
        for note in bundle["notes"]:
            parts.append(
                f"<div class=\"note {esc(note['kind'])}\">"
                f"<span class=\"who\">{esc(note['at'])} · {esc(note['author'])}"
                f"{' · ' + esc(note['kind']) if note['kind'] != 'note' else ''}</span>"
                f"<p>{esc(note['body'])}</p></div>\n"
            )
    else:
        parts.append("<p class=\"nocrop\">No notes were written.</p>\n")

    parts.append(
        "<footer>Exported from Siteloom on "
        f"{esc(bundle['exported_at'])} UTC. Crops are embedded in this file; "
        "it needs no network and no other file to read.</footer>\n"
        "</main>\n</body>\n</html>\n"
    )
    return "".join(parts)


def register(app, templates, Session, config) -> None:  # noqa: C901 — route table
    # app.py owns the events list's query-param vocabulary and the media
    # path reading; borrowing them rather than re-spelling them here is
    # what keeps a rename there from leaving this page linking at nothing.
    from siteloom.web.app import _safe_next

    nav.add("/incidents", "Incidents", "IN", after="/")

    media_root = Path(config.storage.media_dir).expanduser().resolve()
    site_name = config.site_name or config.site_id

    # -- shared queries ----------------------------------------------------

    def _spans(session, incident_ids: list[int]) -> dict[int, dict]:
        """Window, camera list and event count per incident, in one pass.

        Computed from the member events rather than stored on the row: the
        window of an incident is whatever its events say it is, and a
        cached copy would be wrong the moment one is attached or removed.
        """
        if not incident_ids:
            return {}
        rows = session.execute(
            select(
                Event.incident_id,
                func.min(Event.first_seen),
                func.max(Event.last_seen),
                func.count(Event.id),
            )
            .where(Event.incident_id.in_(incident_ids))
            .group_by(Event.incident_id)
        ).all()
        out = {
            i: {"start": None, "end": None, "event_count": 0, "cameras": []}
            for i in incident_ids
        }
        for incident_id, start, end, count in rows:
            out[incident_id] = {
                "start": start,
                "end": end,
                "event_count": count,
                "cameras": [],
            }
        cams = session.execute(
            select(Event.incident_id, Camera.id, Camera.name)
            .join(Camera, Event.camera_id == Camera.id, isouter=True)
            .where(Event.incident_id.in_(incident_ids))
            .distinct()
        ).all()
        for incident_id, camera_id, name in cams:
            names = out[incident_id]["cameras"]
            label = name or camera_id or "—"
            if label not in names:
                names.append(label)
        for value in out.values():
            value["cameras"].sort()
        return out

    def _rows(session, incidents: list[Incident]) -> list[dict]:
        spans = _spans(session, [i.id for i in incidents])
        note_counts = dict(
            session.execute(
                select(IncidentNote.incident_id, func.count(IncidentNote.id))
                .where(IncidentNote.incident_id.in_([i.id for i in incidents] or [0]))
                .group_by(IncidentNote.incident_id)
            ).all()
        )
        return [
            {
                "incident": incident,
                "notes": note_counts.get(incident.id, 0),
                **spans.get(
                    incident.id,
                    {"start": None, "end": None, "event_count": 0, "cameras": []},
                ),
            }
            for incident in incidents
        ]

    # -- the rail's escalate control ---------------------------------------

    def escalate_options(event_id: int) -> dict:
        """What the triage rail needs to offer Escalate on one event.

        A Jinja global, like `nav_items`, because the rail's context is
        built in app.py (`_rail_context`) and is rendered from two places.
        Threading this through there would make the control depend on a
        module that must not know incidents exist; a global keeps the
        seam one-directional.

        Returns plain dicts, not ORM rows: the rail renders after the
        index page's session has closed, and a lazy attribute there is a
        DetachedInstanceError rather than a page.
        """
        with Session() as session:
            event = session.get(Event, event_id)
            current = None
            if event is not None and event.incident_id is not None:
                incident = session.get(Incident, event.incident_id)
                if incident is not None:
                    current = {
                        "id": incident.id,
                        "title": incident.title,
                        "status": incident.status,
                    }
            recent = (
                session.scalars(
                    select(Incident)
                    .where(Incident.status == "open")
                    .order_by(Incident.opened_at.desc())
                    .limit(RAIL_CHOICES)
                )
                .unique()
                .all()
            )
            counts = _spans(session, [i.id for i in recent])
            return {
                "current": current,
                "recent": [
                    {
                        "id": i.id,
                        "title": i.title,
                        "events": counts.get(i.id, {}).get("event_count", 0),
                    }
                    for i in recent
                ],
            }

    templates.env.globals["escalate_options"] = escalate_options

    # -- screens -----------------------------------------------------------

    @app.get("/incidents", response_class=HTMLResponse)
    def incidents_page(request: Request):
        with Session() as session:
            everything = (
                session.scalars(
                    select(Incident).order_by(Incident.opened_at.desc())
                )
                .unique()
                .all()
            )
            open_rows = _rows(session, [i for i in everything if i.is_open])
            closed_rows = _rows(session, [i for i in everything if not i.is_open])
        # Two emptinesses, the /noise and /bookings rule (CLD-26): "nothing
        # has ever been escalated" is a fact about this install and needs
        # to say how one is opened; "nothing is open" is a fact about right
        # now, and is the good news it looks like.
        return templates.TemplateResponse(
            request,
            "incidents.html",
            {
                "site_name": site_name,
                "sections": [
                    {
                        "key": "open",
                        "title": "Open",
                        "empty": "Nothing is open. Every incident has been closed.",
                        "rows": open_rows,
                    },
                    {
                        "key": "closed",
                        "title": "Closed",
                        "empty": "No incident has been closed yet.",
                        "rows": closed_rows,
                    },
                ],
                "total": len(everything),
                "open_count": len(open_rows),
                "closed_count": len(closed_rows),
                "empty": not everything,
            },
        )

    def _detail(session, incident_id: int) -> tuple[Incident, dict]:
        incident = session.get(Incident, incident_id)
        if incident is None:
            raise HTTPException(404)
        events = (
            session.scalars(
                select(Event)
                .options(
                    selectinload(Event.camera),
                    selectinload(Event.identities).selectinload(EventIdentity.identity),
                )
                .where(Event.incident_id == incident_id)
                # Time order, not insertion order: the record is read as a
                # chronology, and events are attached in whatever order the
                # operator worked through the queue.
                .order_by(Event.first_seen)
            )
            .unique()
            .all()
        )
        notes = (
            session.scalars(
                select(IncidentNote)
                .where(IncidentNote.incident_id == incident_id)
                .order_by(IncidentNote.at, IncidentNote.id)
            )
            .unique()
            .all()
        )
        span = _spans(session, [incident_id]).get(
            incident_id, {"start": None, "end": None, "event_count": 0, "cameras": []}
        )
        return incident, {"events": events, "notes": notes, **span}

    @app.get("/incidents/{incident_id}", response_class=HTMLResponse)
    def incident_detail(request: Request, incident_id: int):
        with Session() as session:
            incident, detail = _detail(session, incident_id)
        return templates.TemplateResponse(
            request,
            "incident.html",
            {"site_name": site_name, "incident": incident, **detail},
        )

    @app.get("/incidents/{incident_id}/export")
    def incident_export(incident_id: int):
        """Download the incident as one self-contained HTML file.

        A GET so it can be linked and bookmarked, and `view` work: reading
        the record out is reading, and the operator who may look at an
        incident on screen may hand it to the insurer who asked for it.
        """
        with Session() as session:
            incident, detail = _detail(session, incident_id)
            window = "—"
            if detail["start"] is not None:
                window = (
                    f"{detail['start'].strftime('%Y-%m-%d %H:%M:%S')} → "
                    f"{detail['end'].strftime('%Y-%m-%d %H:%M:%S')}"
                )
            facts = [
                ("Status", incident.status),
                ("Opened", f"{incident.opened_at.strftime('%Y-%m-%d %H:%M:%S')} UTC"),
                ("Opened by", incident.opened_by or "—"),
            ]
            if incident.closed_at is not None:
                facts.append(
                    (
                        "Closed",
                        f"{incident.closed_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
                    )
                )
                facts.append(("Closed by", incident.closed_by or "—"))
            facts.append(("Window", window))
            facts.append(("Cameras", ", ".join(detail["cameras"]) or "—"))
            facts.append(("Events", str(detail["event_count"])))
            bundle = {
                "site_name": site_name,
                "exported_at": _now().strftime("%Y-%m-%d %H:%M:%S"),
                "incident": {
                    "id": incident.id,
                    "title": incident.title,
                    "status": incident.status,
                },
                "facts": facts,
                "events": [
                    {
                        "when": e.first_seen.strftime("%Y-%m-%d %H:%M:%S"),
                        "camera": (e.camera.name if e.camera else None) or e.camera_id,
                        "what": f"{e.class_name} · event {e.id} · "
                        f"{e.detection_count} detection"
                        f"{'' if e.detection_count == 1 else 's'}",
                        "identities": ", ".join(
                            link.identity.display_name
                            for link in e.active_identities
                            if link.identity is not None
                        )
                        or "—",
                        "crop": crop_bytes(e.best_crop_path, media_root),
                    }
                    for e in detail["events"]
                ],
                "notes": [
                    {
                        "at": n.at.strftime("%Y-%m-%d %H:%M:%S"),
                        "author": n.author or "—",
                        "kind": n.kind,
                        "body": n.body,
                    }
                    for n in detail["notes"]
                ],
            }
        return Response(
            export_html(bundle),
            media_type="text/html; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="incident-{incident_id}.html"'
                )
            },
        )

    # -- escalate ----------------------------------------------------------

    @app.post("/events/{event_id}/escalate")
    def escalate_event(
        request: Request,
        event_id: int,
        incident_id: str = Form("new"),
        title: str = Form(""),
        next_url: str = Form("/"),
    ):
        """Attach this event to an incident, opening one if asked.

        `incident_id` is an existing open incident's id or "new". Adding
        to an existing one is the common case — the second event of a
        night is the same night — so it is the default the rail submits,
        not a path behind a create form.

        Attaching to a *closed* incident is refused rather than silently
        reopening it: a closed record has been handed to someone, and an
        event appearing in it afterwards without a trace is the one thing
        that would make the export untrustworthy. Reopen, then attach.
        """
        with Session() as session:
            event = session.get(Event, event_id)
            if event is None:
                raise HTTPException(404)
            if incident_id == "new":
                camera = session.get(Camera, event.camera_id)
                incident = Incident(
                    title=title.strip() or default_title(event, camera),
                    status="open",
                    opened_at=_now(),
                    opened_by=actor(request),
                )
                session.add(incident)
                session.flush()
            else:
                if not incident_id.isdigit():
                    raise HTTPException(400, "incident_id must be an integer or 'new'")
                incident = session.get(Incident, int(incident_id))
                if incident is None:
                    raise HTTPException(404, f"no incident {incident_id}")
                if not incident.is_open:
                    raise HTTPException(
                        409,
                        "That incident is closed. Reopen it before attaching "
                        "another event to it.",
                    )
            event.incident_id = incident.id
            session.commit()
            target = incident.id
        # Back to where the operator was — the filtered queue and their
        # place in it — like every other rail action, with the incident
        # itself as the fallback rather than a fixed path.
        return RedirectResponse(
            _safe_next(next_url, event_id) if next_url else f"/incidents/{target}",
            status_code=303,
        )

    @app.post("/incidents/{incident_id}/events/{event_id}/remove")
    def remove_event(incident_id: int, event_id: int, next_url: str = Form("")):
        """Detach an event escalated by mistake. The event is untouched."""
        with Session() as session:
            event = session.get(Event, event_id)
            if event is None or event.incident_id != incident_id:
                raise HTTPException(404)
            event.incident_id = None
            session.commit()
        return RedirectResponse(
            _safe_next(next_url, event_id) if next_url else f"/incidents/{incident_id}",
            status_code=303,
        )

    # -- notes and lifecycle ----------------------------------------------

    def _append_note(session, request, incident_id: int, body: str, kind: str) -> None:
        text = body.strip()
        if not text:
            return
        session.add(
            IncidentNote(
                incident_id=incident_id,
                at=_now(),
                author=actor(request),
                kind=kind,
                body=text,
            )
        )

    @app.post("/incidents/{incident_id}/note")
    def add_note(request: Request, incident_id: int, body: str = Form("")):
        with Session() as session:
            incident = session.get(Incident, incident_id)
            if incident is None:
                raise HTTPException(404)
            _append_note(session, request, incident_id, body, "note")
            session.commit()
        return RedirectResponse(f"/incidents/{incident_id}", status_code=303)

    @app.post("/incidents/{incident_id}/status")
    def set_status(
        request: Request,
        incident_id: int,
        status: str = Form(...),
        body: str = Form(""),
    ):
        """Close an incident, or reopen it.

        The closing note is the point of closing — "how did this end" is
        the question a reader six weeks later actually has — so it shares
        this endpoint rather than needing a second click, and lands in the
        same chronology as every other note with its kind recorded.
        """
        if status not in ("open", "closed"):
            raise HTTPException(400, "status must be open or closed")
        with Session() as session:
            incident = session.get(Incident, incident_id)
            if incident is None:
                raise HTTPException(404)
            if status == "closed":
                incident.status = "closed"
                incident.closed_at = _now()
                incident.closed_by = actor(request)
                _append_note(session, request, incident_id, body, "close")
            else:
                # Reopening clears the closure rather than keeping a stale
                # one: the row says whether it is closed now, and the note
                # trail says it was closed once and by whom.
                incident.status = "open"
                incident.closed_at = None
                incident.closed_by = None
                _append_note(session, request, incident_id, body, "reopen")
            session.commit()
        return RedirectResponse(f"/incidents/{incident_id}", status_code=303)

    @app.post("/incidents/{incident_id}/delete")
    def delete_incident(incident_id: int):
        """Withdraw the judgement, keeping everything it judged.

        The events are the record and the incident is an interpretation of
        them, so the member events are detached explicitly here — not left
        to a cascade whose default could change under us — and only the
        incident and its notes go.
        """
        with Session() as session:
            incident = session.get(Incident, incident_id)
            if incident is None:
                raise HTTPException(404)
            for event in session.scalars(
                select(Event).where(Event.incident_id == incident_id)
            ):
                event.incident_id = None
            session.delete(incident)
            session.commit()
        return RedirectResponse("/incidents", status_code=303)
