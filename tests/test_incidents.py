"""Incidents (CLD-96) — what Escalate on the triage rail opens.

The decision this screen implements is that escalation produces a
*record*, not a flag and not a notification: for a property manager there
is nobody on the other end immediately, and somebody later. So the
properties worth holding are the ones that separate a record from a
status:

* one incident spans events on more than one camera — a per-event flag
  could not say that, and saying it is the whole reason to build this,
* adding an event to an existing open incident is the common case and is
  one interaction, not a second step behind "create",
* deleting an incident leaves its events untouched — the events are the
  record, the incident is an interpretation of them,
* the column arrives additively on a database that predates it,
* the bundle is self-contained: readable with no console and no network,
* the empty states distinguish "none yet" from "none open" (CLD-26),
* and judging that something happened is `edit` work, not `admin`.
"""

from __future__ import annotations

import base64
import re
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import select, text

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import (
    Camera,
    Event,
    EventIdentity,
    Identity,
    User,
    get_session,
    init_db,
    make_engine,
)
from siteloom.store.models import Incident, IncidentNote
from siteloom.web.app import create_app
from siteloom.web.auth import hash_password, required_role
from siteloom.web.incidents_routes import crop_bytes, export_html

NOW = datetime.now(timezone.utc).replace(tzinfo=None)


def make_env(tmp_path):
    media = tmp_path / "m"
    media.mkdir(exist_ok=True)
    config = SiteConfig(
        site_id="t",
        site_name="Kai",
        cameras=[
            CameraConfig(id="cam1", adapter="file", source="x"),
            CameraConfig(id="cam2", adapter="file", source="y"),
        ],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/i.db", media_dir=str(media)
        ),
    )
    config.identity.enabled = False
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(Camera(id="cam1", site_id="t", name="Gate"))
        s.add(Camera(id="cam2", site_id="t", name="Front door"))
        s.commit()
    return TestClient(create_app(config), follow_redirects=False), Session, media


def add_event(Session, camera_id="cam1", minutes=0, class_name="car", crop=None):
    with Session() as s:
        event = Event(
            camera_id=camera_id,
            track_id=1,
            class_name=class_name,
            first_seen=NOW + timedelta(minutes=minutes),
            last_seen=NOW + timedelta(minutes=minutes, seconds=20),
            detection_count=4,
            best_confidence=0.8,
            best_crop_path=crop,
        )
        s.add(event)
        s.commit()
        return event.id


def add_user(Session, username, role):
    with Session() as s:
        s.add(
            User(
                username=username,
                password_hash=hash_password("hunter2!"),
                role=role,
                created_at=NOW,
            )
        )
        s.commit()


def login(client, username):
    r = client.post(
        "/login",
        data={"username": username, "password": "hunter2!", "next": "/incidents"},
    )
    assert r.status_code == 303, r.text[:200]
    return client


def escalate(client, event_id, incident_id="new", title="", next_url="/"):
    return client.post(
        f"/events/{event_id}/escalate",
        data={
            "incident_id": incident_id,
            "title": title,
            "next_url": next_url,
        },
    )


# -- the reason it is an object and not a fifth Event status ---------------


def test_one_incident_spans_events_on_two_cameras(tmp_path):
    """The property a per-event flag could not express, and therefore the
    reason this table exists: a vehicle at the gate and someone at the
    door are one occurrence."""
    client, Session, _ = make_env(tmp_path)
    gate = add_event(Session, "cam1", minutes=0, class_name="car")
    door = add_event(Session, "cam2", minutes=8, class_name="person")

    assert escalate(client, gate, title="Break-in, unit C").status_code == 303
    with Session() as s:
        incident_id = s.scalars(select(Incident)).one().id
    assert escalate(client, door, incident_id=str(incident_id)).status_code == 303

    with Session() as s:
        members = s.scalars(
            select(Event).where(Event.incident_id == incident_id).order_by(Event.id)
        ).all()
        assert [e.camera_id for e in members] == ["cam1", "cam2"]

    page = client.get("/incidents").text
    assert "Break-in, unit C" in page
    # Cameras involved and the event count, on the row.
    assert "Front door" in page and "Gate" in page

    detail = client.get(f"/incidents/{incident_id}").text
    # Time order, not attachment order.
    assert detail.index(f"event {gate}") < detail.index(f"event {door}")


def test_escalating_does_not_touch_the_events_review_status(tmp_path):
    """`Event` gains an incident_id and nothing else — CLD-20's status
    derivation is deliberately left alone."""
    client, Session, _ = make_env(tmp_path)
    event_id = add_event(Session)
    escalate(client, event_id)
    with Session() as s:
        event = s.get(Event, event_id)
        assert event.review_status == "new"
        assert event.reviewed_at is None


# -- adding to an existing incident is the common case ---------------------


def test_the_rail_offers_open_incidents_before_the_create_path(tmp_path):
    """Adding to an incident already open is the daily interaction, so it
    is the default selection; "+ new incident" is the last option in the
    same control rather than a separate step."""
    client, Session, _ = make_env(tmp_path)
    first = add_event(Session, minutes=0)
    escalate(client, first, title="Night of the 9th")
    second = add_event(Session, minutes=30)

    rail = client.get(f"/events/{second}/rail").text
    form = rail.split(f'action="/events/{second}/escalate"', 1)[1]
    options = re.findall(r'<option value="([^"]+)"', form.split("</form>", 1)[0])
    # The open incident is the pre-selected first option — the daily
    # interaction is choosing nothing and clicking once — and creating a
    # new incident is the fallback at the end of the same control.
    assert options[0].isdigit()
    assert options[-1] == "new"
    assert "Night of the 9th" in rail


def test_a_new_incident_without_a_title_still_names_itself(tmp_path):
    """The list and every export are read by someone who was not there;
    "Incident 7" tells them nothing."""
    client, Session, _ = make_env(tmp_path)
    event_id = add_event(Session, "cam1", class_name="car")
    escalate(client, event_id, title="   ")
    with Session() as s:
        title = s.scalars(select(Incident)).one().title
    assert "car" in title and "Gate" in title


def test_the_rail_of_an_escalated_event_shows_the_incident_not_the_form(tmp_path):
    client, Session, _ = make_env(tmp_path)
    event_id = add_event(Session)
    escalate(client, event_id, title="Parcel theft")
    rail = client.get(f"/events/{event_id}/rail").text
    assert "Parcel theft" in rail
    assert "Remove from incident" in rail
    assert "+ new incident" not in rail


def test_escalate_returns_the_operator_to_where_they_were(tmp_path):
    """Escalating is done in runs, from the filtered queue — a fixed
    redirect would drop the filters and their place in it."""
    client, Session, _ = make_env(tmp_path)
    event_id = add_event(Session)
    r = escalate(client, event_id, next_url="/?needs_review=1&selected=1")
    assert r.headers["location"] == "/?needs_review=1&selected=1"
    # And an off-site target is refused like every other rail redirect.
    other = add_event(Session, minutes=5)
    r = escalate(client, other, next_url="https://evil.example/")
    assert r.headers["location"] == f"/events/{other}"


def test_attaching_to_a_closed_incident_is_refused_not_silently_reopened(tmp_path):
    """A closed record may already have been handed to someone; an event
    appearing in it afterwards without a trace is what would make the
    export untrustworthy."""
    client, Session, _ = make_env(tmp_path)
    first = add_event(Session)
    escalate(client, first, title="Settled")
    with Session() as s:
        incident_id = s.scalars(select(Incident)).one().id
    client.post(f"/incidents/{incident_id}/status", data={"status": "closed", "body": ""})

    later = add_event(Session, minutes=60)
    r = escalate(client, later, incident_id=str(incident_id))
    assert r.status_code == 409
    with Session() as s:
        assert s.get(Event, later).incident_id is None


def test_escalating_to_a_missing_incident_is_a_404(tmp_path):
    client, Session, _ = make_env(tmp_path)
    event_id = add_event(Session)
    assert escalate(client, event_id, incident_id="999").status_code == 404
    assert escalate(client, 999).status_code == 404


def test_an_event_can_be_detached_without_being_deleted(tmp_path):
    client, Session, _ = make_env(tmp_path)
    event_id = add_event(Session)
    escalate(client, event_id, title="Mistake")
    with Session() as s:
        incident_id = s.scalars(select(Incident)).one().id
    r = client.post(f"/incidents/{incident_id}/events/{event_id}/remove")
    assert r.status_code == 303
    with Session() as s:
        assert s.get(Event, event_id).incident_id is None
        assert s.get(Event, event_id) is not None


# -- deleting an incident keeps its events --------------------------------


def test_deleting_an_incident_leaves_its_events_intact(tmp_path):
    """The events are the record; the incident is an interpretation of
    them. Withdrawing the interpretation must not destroy the evidence."""
    client, Session, _ = make_env(tmp_path)
    gate = add_event(Session, "cam1", minutes=0)
    door = add_event(Session, "cam2", minutes=5)
    escalate(client, gate, title="Wrongly escalated")
    with Session() as s:
        incident_id = s.scalars(select(Incident)).one().id
    escalate(client, door, incident_id=str(incident_id))
    client.post(f"/incidents/{incident_id}/note", data={"body": "on reflection, nothing"})

    r = client.post(f"/incidents/{incident_id}/delete")
    assert r.status_code == 303

    with Session() as s:
        assert s.scalars(select(Incident)).all() == []
        # The notes were part of the interpretation and go with it.
        assert s.scalars(select(IncidentNote)).all() == []
        events = s.scalars(select(Event).order_by(Event.id)).all()
        assert [e.id for e in events] == [gate, door]
        assert [e.incident_id for e in events] == [None, None]
        assert [e.camera_id for e in events] == ["cam1", "cam2"]


def test_deleting_a_missing_incident_is_a_404(tmp_path):
    client, _, _ = make_env(tmp_path)
    assert client.post("/incidents/999/delete").status_code == 404


# -- migration over a pre-column database ---------------------------------


def test_the_incident_column_is_added_to_an_existing_database(tmp_path):
    """Additive only (`_ensure_columns`). A nullable column arrives NULL
    everywhere, which is exactly right: no event had been escalated,
    because escalation did not exist."""
    db = tmp_path / "old.db"
    engine = make_engine(f"sqlite:///{db}")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE events (id INTEGER NOT NULL PRIMARY KEY, "
                "camera_id VARCHAR NOT NULL, track_id INTEGER, class_name VARCHAR "
                "NOT NULL, first_seen DATETIME NOT NULL, last_seen DATETIME NOT NULL, "
                "detection_count INTEGER, best_confidence FLOAT)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO events (camera_id, class_name, first_seen, last_seen, "
                "detection_count, best_confidence) VALUES ('cam1', 'car', "
                "'2026-08-01 10:00:00', '2026-08-01 10:00:20', 4, 0.8)"
            )
        )

    init_db(engine)

    with get_session(engine)() as s:
        row = s.scalars(select(Event)).one()
        assert row.class_name == "car"
        assert row.incident_id is None
        # And the new table exists alongside it.
        s.add(
            Incident(title="After the migration", opened_at=NOW, opened_by="op")
        )
        s.commit()
        row.incident_id = s.scalars(select(Incident)).one().id
        s.commit()
        assert s.scalars(select(Event)).one().incident_id is not None


# -- lifecycle -------------------------------------------------------------


def test_close_records_who_and_the_closing_note_then_reopen_clears_it(tmp_path):
    client, Session, _ = make_env(tmp_path)
    event_id = add_event(Session)
    escalate(client, event_id, title="Loud party, unit B")
    with Session() as s:
        incident_id = s.scalars(select(Incident)).one().id

    r = client.post(
        f"/incidents/{incident_id}/status",
        data={"status": "closed", "body": "Guests left at 01:10; no damage."},
    )
    assert r.status_code == 303
    with Session() as s:
        incident = s.get(Incident, incident_id)
        assert incident.status == "closed"
        assert incident.closed_at is not None
        assert incident.closed_by == "(open)"
        notes = s.scalars(select(IncidentNote)).all()
        assert [n.kind for n in notes] == ["close"]
        assert "01:10" in notes[0].body

    r = client.post(
        f"/incidents/{incident_id}/status",
        data={"status": "open", "body": "Tenant disputed it."},
    )
    assert r.status_code == 303
    with Session() as s:
        incident = s.get(Incident, incident_id)
        assert incident.status == "open"
        # The row says whether it is closed now; the notes say it was.
        assert incident.closed_at is None and incident.closed_by is None
        kinds = [n.kind for n in s.scalars(select(IncidentNote)).all()]
        assert kinds == ["close", "reopen"]


def test_closing_without_a_note_is_allowed_and_writes_no_empty_note(tmp_path):
    client, Session, _ = make_env(tmp_path)
    event_id = add_event(Session)
    escalate(client, event_id, title="Nothing to say")
    with Session() as s:
        incident_id = s.scalars(select(Incident)).one().id
    client.post(f"/incidents/{incident_id}/status", data={"status": "closed", "body": "  "})
    with Session() as s:
        assert s.get(Incident, incident_id).status == "closed"
        assert s.scalars(select(IncidentNote)).all() == []


def test_an_unknown_status_is_refused(tmp_path):
    client, Session, _ = make_env(tmp_path)
    event_id = add_event(Session)
    escalate(client, event_id)
    with Session() as s:
        incident_id = s.scalars(select(Incident)).one().id
    r = client.post(f"/incidents/{incident_id}/status", data={"status": "archived"})
    assert r.status_code == 400


def test_notes_are_appended_in_order_with_their_author(tmp_path):
    client, Session, _ = make_env(tmp_path)
    event_id = add_event(Session)
    escalate(client, event_id, title="Ongoing")
    with Session() as s:
        incident_id = s.scalars(select(Incident)).one().id
    client.post(f"/incidents/{incident_id}/note", data={"body": "Called the tenant."})
    client.post(f"/incidents/{incident_id}/note", data={"body": "No answer."})
    body = client.get(f"/incidents/{incident_id}").text
    assert body.index("Called the tenant.") < body.index("No answer.")
    with Session() as s:
        assert [n.author for n in s.scalars(select(IncidentNote)).all()] == [
            "(open)",
            "(open)",
        ]


# -- export ----------------------------------------------------------------


def test_the_export_is_one_self_contained_file(tmp_path):
    """The bundle is handed to an insurer or the police, who have a
    browser and nothing else: no network, no second file, no console."""
    client, Session, media = make_env(tmp_path)
    (media / "cam1").mkdir()
    crop = media / "cam1" / "c1.jpg"
    crop.write_bytes(b"\xff\xd8\xff\xe0jpeg-bytes")
    gate = add_event(Session, "cam1", minutes=0, class_name="car", crop="cam1/c1.jpg")
    door = add_event(Session, "cam2", minutes=6, class_name="person")
    with Session() as s:
        identity = Identity(
            identifier_key="face",
            class_name="person",
            label="Ana Okonjo",
            first_seen=NOW,
            last_seen=NOW,
        )
        s.add(identity)
        s.flush()
        s.add(
            EventIdentity(
                event_id=door, identity_id=identity.id, identifier_key="face",
                similarity=0.5,
            )
        )
        s.commit()

    escalate(client, gate, title="Break-in, unit C")
    with Session() as s:
        incident_id = s.scalars(select(Incident)).one().id
    escalate(client, door, incident_id=str(incident_id))
    client.post(
        f"/incidents/{incident_id}/note", data={"body": "Police reference KX-2213."}
    )

    r = client.get(f"/incidents/{incident_id}/export")
    assert r.status_code == 200
    assert f'filename="incident-{incident_id}.html"' in r.headers["content-disposition"]
    body = r.text

    # Timeline, notes and crops — all three, in one file.
    assert "Break-in, unit C" in body
    assert f"event {gate}" in body and f"event {door}" in body
    assert body.index(f"event {gate}") < body.index(f"event {door}")
    assert "Gate" in body and "Front door" in body
    assert "Ana Okonjo" in body
    assert "Police reference KX-2213." in body
    encoded = base64.b64encode(b"\xff\xd8\xff\xe0jpeg-bytes").decode()
    assert f"data:image/jpeg;base64,{encoded}" in body

    # Nothing it cannot read on a machine with no route anywhere.
    assert not re.search(r'(?:src|href)\s*=\s*["\'](?:https?:)?//', body)
    assert "/static/" not in body and "/media/" not in body


def test_the_export_does_not_read_outside_media_dir(tmp_path):
    """A column value becomes bytes in a file that gets mailed out — the
    wrong place to widen what is readable."""
    _, _, media = make_env(tmp_path)
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"not yours")
    root = media.resolve()
    assert crop_bytes("../secret.jpg", root) is None
    assert crop_bytes(str(outside), root) is None
    assert crop_bytes(None, root) is None


def test_the_export_escapes_operator_prose(tmp_path):
    """Notes and titles are free text typed by a person and read in a
    browser somewhere else entirely."""
    bundle = {
        "site_name": "Kai",
        "exported_at": "2026-08-09 21:00:00",
        "incident": {"id": 1, "title": "<script>alert(1)</script>", "status": "open"},
        "facts": [("Status", "open")],
        "events": [],
        "notes": [
            {"at": "2026-08-09 20:00:00", "author": "op", "kind": "note",
             "body": "<img onerror=alert(2)>"},
        ],
    }
    out = export_html(bundle)
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out
    assert "<img onerror" not in out


def test_an_incident_with_no_events_still_exports(tmp_path):
    client, Session, _ = make_env(tmp_path)
    event_id = add_event(Session)
    escalate(client, event_id, title="Detached later")
    with Session() as s:
        incident_id = s.scalars(select(Incident)).one().id
    client.post(f"/incidents/{incident_id}/events/{event_id}/remove")
    r = client.get(f"/incidents/{incident_id}/export")
    assert r.status_code == 200
    assert "No events are attached" in r.text


def test_exporting_a_missing_incident_is_a_404(tmp_path):
    client, _, _ = make_env(tmp_path)
    assert client.get("/incidents/999/export").status_code == 404
    assert client.get("/incidents/999").status_code == 404


# -- empty states (the CLD-26 rule) ---------------------------------------


def test_no_incidents_yet_is_not_the_same_as_none_open(tmp_path):
    client, Session, _ = make_env(tmp_path)
    body = client.get("/incidents").text
    assert "No incidents have been opened" in body
    assert "Escalate" in body
    # The misleading reading: an install where nothing was ever escalated
    # must not read as a site that has been worked to zero.
    assert "Nothing is open." not in body

    event_id = add_event(Session)
    escalate(client, event_id, title="Something")
    with Session() as s:
        incident_id = s.scalars(select(Incident)).one().id
    client.post(f"/incidents/{incident_id}/status", data={"status": "closed"})

    body = client.get("/incidents").text
    assert "No incidents have been opened" not in body
    assert "Nothing is open." in body
    assert "Something" in body


def test_an_open_incident_hides_both_empty_states(tmp_path):
    client, Session, _ = make_env(tmp_path)
    escalate(client, add_event(Session), title="Live one")
    body = client.get("/incidents").text
    assert "No incidents have been opened" not in body
    assert "Nothing is open." not in body
    assert "No incident has been closed yet." in body


# -- roles -----------------------------------------------------------------


def test_opening_and_closing_an_incident_is_edit_work_not_admin():
    """Judging that something happened is exactly what an editor is for,
    so /incidents stays out of ADMIN_PREFIXES."""
    assert required_role("GET", "/incidents") == "view"
    assert required_role("GET", "/incidents/1/export") == "view"
    assert required_role("POST", "/incidents/1/note") == "edit"
    assert required_role("POST", "/incidents/1/status") == "edit"
    assert required_role("POST", "/incidents/1/delete") == "edit"
    assert required_role("POST", "/events/1/escalate") == "edit"


def test_a_viewer_can_read_an_incident_but_not_open_one(tmp_path):
    client, Session, _ = make_env(tmp_path)
    event_id = add_event(Session)
    add_user(Session, "vera", "view")
    login(client, "vera")
    assert client.get("/incidents").status_code == 200
    assert escalate(client, event_id).status_code == 403
    with Session() as s:
        assert s.scalars(select(Incident)).all() == []


def test_an_editor_can_open_annotate_and_close(tmp_path):
    client, Session, _ = make_env(tmp_path)
    event_id = add_event(Session)
    add_user(Session, "eddy", "edit")
    login(client, "eddy")
    assert escalate(client, event_id, title="Editor's call").status_code == 303
    with Session() as s:
        incident = s.scalars(select(Incident)).one()
        assert incident.opened_by == "eddy"
        incident_id = incident.id
    assert (
        client.post(f"/incidents/{incident_id}/note", data={"body": "Seen."}).status_code
        == 303
    )
    assert (
        client.post(
            f"/incidents/{incident_id}/status", data={"status": "closed", "body": "done"}
        ).status_code
        == 303
    )
    with Session() as s:
        assert s.get(Incident, incident_id).closed_by == "eddy"


def test_incidents_are_reachable_from_the_sidebar(tmp_path):
    client, _, _ = make_env(tmp_path)
    body = client.get("/incidents").text
    assert 'href="/incidents"' in body
    assert body.count("sl-nav-item active") == 1
