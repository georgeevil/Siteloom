"""Recognition without identity, on every screen (CLD-111).

CLD-31 closed the screens that list people and CLD-103 closed the picker
that listed them inside a screen `restricted` keeps. Three surfaces still
printed a name one at a time — the triage queue's identity column (fifty
a page, which is the directory reassembled by scrolling), an incident
title an operator typed, a booking summary an iCal feed carried — and all
three were permitted by the rules as they stood.

The decision: a restricted operator sees *whether* the system recognised
someone, never *who*. So the test that matters is not "these three
templates were fixed" — that passes the day it is written and rots on the
next screen, which is exactly how the three arose. It is a **walk**:
every registered GET route, requested as `restricted`, asserted to carry
none of the seeded strings. Same idiom as `test_nav.py` walking every
sidebar entry and `test_static_assets.py` walking every template.

Three things keep the walk honest:

* The same walk runs at `view`, where every seeded string must appear
  somewhere. A substitution applied to everybody would pass the first
  test and fail this one.
* Open single-operator mode (no User rows) walks too. Auth turns on with
  the first `User` row and everything must keep working before that.
* Recognised-vs-unrecognised is asserted to stay legible on the queue.
  If an operator cannot tell those apart, the rung has been made useless
  and the change has failed — which is a thing a leak test cannot see.

A path parameter this table does not know is a failure, not a skip: a new
route with a new parameter is exactly the screen this walk exists to
cover before anyone remembers it.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import (
    Annotation,
    Booking,
    Camera,
    Event,
    EventIdentity,
    Identity,
    LibraryItem,
    LibrarySource,
    User,
    get_session,
    init_db,
    make_engine,
)
from siteloom.store.models import Incident
from siteloom.web import auth, redaction
from siteloom.web.app import create_app

NOW = datetime.now(timezone.utc).replace(tzinfo=None)
TS = NOW - timedelta(hours=1)

#: Deliberately distinctive: every assertion below is a substring search
#: over a whole HTML page, so a seeded value that could occur by accident
#: (a class name, a camera id, a date) would make the walk lie.
IDENTITY_LABEL = "Dorotea Vasilenko"
PLATE = "KX19-ZTP"
INCIDENT_TITLE = "Vasilenko forced the side gate"
BOOKING_SUMMARY = "Reserved by Marguerite Achterberg"
#: The Takeout importer's guess at who a face is. Reached through
#: `/api/items/{id}/annotations`, which is the library's read API under a
#: URL that does not begin with `/library` — found by this walk.
PROPOSED_NAME = "Ilse Brandtsdottir"

SECRETS = {
    "identity label": IDENTITY_LABEL,
    "plate": PLATE,
    "incident title": INCIDENT_TITLE,
    "booking summary": BOOKING_SUMMARY,
    "proposed name": PROPOSED_NAME,
}

#: One value per path parameter in the console. Chosen to *exist* in the
#: fixture wherever possible: a route that 404s proves nothing about what
#: it renders.
PATH_PARAMS = {
    "event_id": "1",
    "identity_id": "1",
    "incident_id": "1",
    "item_id": "1",
    "booking_id": "1",
    "run_id": "1",
    "user_id": "1",
    "class_id": "1",
    "read_id": "1",
    "link_id": "1",
    "camera_id": "cam1",
    # /detector/proposals/{camera_id}/{stamp} — no proposal exists in the
    # walk's fixture, so the route answers 404, which discloses nothing.
    "stamp": "20260101-000000",
    # /plates/p/{plate} — the canonical (normalized) spelling, since the
    # route 307s anything else and a redirect body proves nothing.
    "plate": "KX19ZTP",
    # The event crop, which stays visible by decision — the walk still
    # reads its bytes, because a name could only be in them by accident.
    "path": "cam1/2026-08-06/crop.jpg",
}

#: Query strings a route needs to answer at all. `/api/jobs/stream` is an
#: SSE endpoint that streams until the client disconnects; `updates=1`
#: takes one snapshot and returns, which is what the /jobs page's own
#: tests use.
ROUTE_QUERY = {"/api/jobs/stream": "updates=1"}

#: Renders reachable only through a query parameter, so no route path
#: names them: the index page renders the triage rail inline for a deep
#: link, which is a third spelling of the rail template.
EXTRA_URLS = ("/?selected=1", "/?selected=2", "/?unmatched=1")

_PARAM = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(?::[^}]+)?\}")


def fill(path: str) -> str:
    """A route's declared path with its parameters filled in."""
    url = path
    for match in _PARAM.finditer(path):
        name = match.group(1)
        assert name in PATH_PARAMS, (
            f"{path} takes a path parameter {name!r} this walk has no value "
            "for — add one to PATH_PARAMS so the route is covered"
        )
        url = url.replace(match.group(0), PATH_PARAMS[name])
    return url


def get_urls(app) -> list[str]:
    """Every registered GET route, as a URL that can actually be sent."""
    urls: list[str] = []
    for route in app.routes:
        methods = getattr(route, "methods", None) or set()
        path = getattr(route, "path", None)
        if path is None or "GET" not in methods:
            continue
        query = ROUTE_QUERY.get(path)
        url = fill(path)
        urls.append(f"{url}?{query}" if query else url)
    urls.extend(EXTRA_URLS)
    return sorted(set(urls))


@pytest.fixture
def env(tmp_path):
    """A console holding one of everything that carries a name."""
    media = tmp_path / "m"
    event_crop = media / "cam1" / "2026-08-06" / "crop.jpg"
    gallery_crop = media / "library" / "crops" / "face.jpg"
    for path in (event_crop, gallery_crop):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\xff\xd8\xff\xd9")
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/d.db", media_dir=str(media)
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(Camera(id="cam1", site_id="t", name="Cam One"))
        s.add(
            Incident(
                title=INCIDENT_TITLE,
                status="open",
                opened_at=TS,
                opened_by="ed",
            )
        )
        s.add(
            Identity(
                identifier_key="vehicle",
                class_name="car",
                label=IDENTITY_LABEL,
                plate=PLATE,
                first_seen=TS,
                last_seen=TS,
            )
        )
        # Matched, escalated, with a crop: the row the queue prints a name
        # on, and the rail prints a name, a plate and an incident title on.
        s.add(
            Event(
                camera_id="cam1",
                track_id=1,
                class_name="car",
                first_seen=TS,
                last_seen=TS,
                detection_count=4,
                best_confidence=0.9,
                incident_id=1,
                best_crop_path=str(event_crop),
            )
        )
        # Unmatched, so "recognised vs not" has both halves on one screen.
        s.add(
            Event(
                camera_id="cam1",
                track_id=2,
                class_name="person",
                first_seen=TS,
                last_seen=TS,
                detection_count=3,
                best_confidence=0.8,
            )
        )
        s.flush()
        s.add(
            EventIdentity(
                event_id=1,
                identity_id=1,
                identifier_key="vehicle",
                similarity=0.88,
                matched_by="visual",
            )
        )
        # Manual and current, so the row renders in the Current section
        # with its edit form — whose input value ships the summary too.
        s.add(
            Booking(
                uid="siteloom-manual:walk",
                summary=BOOKING_SUMMARY,
                start=NOW - timedelta(hours=2),
                end=NOW + timedelta(hours=2),
                source="manual",
            )
        )
        s.add(
            LibrarySource(
                name="takeout", path=str(tmp_path / "src"), kind="takeout", added_at=TS
            )
        )
        s.flush()
        s.add(
            LibraryItem(
                source_id=1,
                path=str(tmp_path / "src" / "photo.jpg"),
                kind="image",
                status="indexed",
                mtime=TS,
                thumb_path=str(gallery_crop),
            )
        )
        s.flush()
        s.add(
            Annotation(
                item_id=1,
                bbox=json.dumps([0.1, 0.1, 0.4, 0.4]),
                class_name="face",
                identity_id=1,
                proposed_name=PROPOSED_NAME,
                proposal_basis="takeout:people",
                source="import",
                created_at=TS,
            )
        )
        s.commit()
    return {"app": create_app(config), "Session": Session}


def signed_in(env, username: str, role: str) -> TestClient:
    with env["Session"]() as s:
        s.add(
            User(
                username=username,
                password_hash=auth.hash_password("hunter2!"),
                role=role,
                created_at=NOW,
            )
        )
        s.commit()
    client = TestClient(env["app"], follow_redirects=False)
    r = client.post(
        "/login", data={"username": username, "password": "hunter2!", "next": "/"}
    )
    assert r.status_code == 303, r.text[:200]
    return client


def leaks(client, urls) -> dict[str, list[str]]:
    """Which seeded strings each URL's response body carries."""
    found: dict[str, list[str]] = {}
    for url in urls:
        body = client.get(url).content
        for name, value in SECRETS.items():
            if value.encode() in body:
                found.setdefault(name, []).append(url)
    return found


# -- the walk --------------------------------------------------------------


def test_the_walk_actually_covers_the_console(env):
    """A walk that quietly stopped enumerating would pass everything."""
    urls = get_urls(env["app"])
    assert len(urls) > 20, urls
    for expected in ("/", "/events/1", "/events/1/rail", "/bookings", "/?selected=1"):
        assert expected in urls, expected


def test_no_screen_hands_a_restricted_operator_a_name(env):
    """The enforcement. Every GET route, as `restricted`, carrying none of
    the seeded strings — a name, a plate, an incident title, a booking
    summary, a proposed name — in its response body.

    Asserted on the body rather than on a control, because hiding a
    `<select>` in a template leaves every name in the payload, which is
    the same leak with a stylesheet in front of it (CLD-103).
    """
    client = signed_in(env, "nina", "restricted")
    found = leaks(client, get_urls(env["app"]))
    assert not found, f"restricted was handed: {found}"


def test_the_same_walk_at_view_still_shows_everything(env):
    """So the walk above cannot pass by substituting for everybody.

    `view` is the floor this substitution sits under, and nothing at or
    above it changes: an operator who may open /identities may read a
    name wherever the console prints one.
    """
    client = signed_in(env, "vera", "view")
    found = leaks(client, get_urls(env["app"]))
    missing = [name for name in SECRETS if name not in found]
    assert not missing, f"a viewer could not see: {missing}"


def test_open_single_operator_mode_is_untouched(env):
    """No User rows means no rungs at all. A feature that needs an account
    to work is a bug, and open mode is most of the suite."""
    client = TestClient(env["app"], follow_redirects=False)
    found = leaks(client, get_urls(env["app"]))
    missing = [name for name in SECRETS if name not in found]
    assert not missing, f"open mode lost: {missing}"


def test_edit_and_admin_are_unchanged_too(env):
    """The floor is a floor: closing it below `view` costs the rungs above
    it nothing."""
    for username, role in (("ed", "edit"), ("ada", "admin")):
        client = signed_in(env, username, role)
        found = leaks(client, get_urls(env["app"]))
        missing = [name for name in SECRETS if name not in found]
        assert not missing, (role, missing)


# -- the signal that has to survive ----------------------------------------


def test_recognised_and_unrecognised_stay_legible_on_the_queue(env):
    """The rung is only staffable if this distinction survives.

    "Should I wake someone" is answered by recognised-vs-not, which is a
    boolean and survives losing the name. If an operator cannot tell the
    two apart after this change, the substitution has made the screen
    useless rather than safe.
    """
    body = signed_in(env, "nina", "restricted").get("/").text
    # The matched row: recognised, named by its kind, not by its name.
    assert body.count("Known vehicle") == 1, body.count("Known vehicle")
    assert IDENTITY_LABEL not in body
    # The unmatched row, still saying so — in the cell and in its class.
    assert 'class="r-id c-identity unmatched"' in body


def test_the_rail_still_says_the_event_was_recognised(env):
    """Withholding the claim itself would leave nothing to judge. The
    identifier, the similarity and the verdict buttons stay; only who it
    was is replaced."""
    client = signed_in(env, "nina", "restricted")
    for url in ("/events/1", "/events/1/rail", "/?selected=1"):
        body = client.get(url).text
        assert "Known vehicle" in body, url
        assert "sim 0.88" in body or "0.88" in body, url


def test_two_withheld_claims_are_not_styled_apart(env):
    """A named identity and an unknown-bucket one both read `Known
    vehicle` — so styling one of them "no name yet" would say which of
    the two the system can name. Smaller than a name, still not this
    rung's, and free either way: the class goes through the same gate.
    """
    with env["Session"]() as s:
        s.add(
            Identity(
                identifier_key="vehicle",
                class_name="car",
                first_seen=TS,
                last_seen=TS,
            )
        )
        s.flush()
        s.add(
            EventIdentity(
                event_id=1,
                identity_id=2,
                identifier_key="vehicle",
                similarity=0.71,
                matched_by="visual",
            )
        )
        s.commit()
    restricted = signed_in(env, "nina", "restricted").get("/events/1/rail").text
    assert restricted.count("Known vehicle") == 2
    assert "id-name unmatched" not in restricted
    # A viewer, who may read both names, still gets the distinction.
    viewer = signed_in(env, "vera", "view").get("/events/1/rail").text
    assert "id-name unmatched" in viewer
    assert IDENTITY_LABEL in viewer


def test_a_withheld_booking_still_says_when_and_from_where(env):
    """Restricted keeps what tells them an arrival was expected: the
    window, the provenance badge and both event counts."""
    body = signed_in(env, "nina", "restricted").get("/bookings").text
    assert BOOKING_SUMMARY not in body
    assert "Booking 1" in body
    assert "manual" in body
    assert "Guest-stamped" in body


def test_a_withheld_incident_is_still_identified_by_its_number(env):
    """Free text is replaced, not filtered — but the rail must still say
    which incident this event is in, or escalation becomes unreadable."""
    body = signed_in(env, "nina", "restricted").get("/events/1/rail").text
    assert INCIDENT_TITLE not in body
    assert "Incident 1" in body


# -- the helper ------------------------------------------------------------


def test_the_floor_is_the_identities_floor_not_a_second_rung():
    """Asked of `required_role`, so moving that floor moves both or
    neither — a copy of the rule is wrong the first time a prefix does."""
    floor = auth.required_role("GET", redaction.NAMING_FLOOR_PATH)
    for role in auth.ROLES:
        user = User(username="x", password_hash="x", role=role, created_at=NOW)
        assert redaction.may_name(user) is auth.has_role(user, floor), role
    assert redaction.may_name(None) is True


@pytest.mark.parametrize(
    "identifier_key,class_name,expected",
    [
        ("face", "person", "Known person"),
        ("person", "person", "Known person"),
        ("vehicle", "car", "Known vehicle"),
        # An identifier the registry auto-added for an unseen class: the
        # noun falls through to the class, so adding a class to
        # detection.classes needs no change here (NFR3).
        ("dog", "dog", "Known dog"),
        # A vehicle class under some other identifier is still a vehicle.
        ("appearance", "truck", "Known vehicle"),
        ("", "", "Known subject"),
    ],
)
def test_what_a_withheld_identity_is_announced_as(
    identifier_key, class_name, expected
):
    identity = Identity(
        identifier_key=identifier_key,
        class_name=class_name,
        first_seen=TS,
        last_seen=TS,
    )
    assert redaction.recognised_as(identity) == expected


def test_an_unlabeled_identity_is_recognised_too():
    """An identity exists because sightings were grouped — that grouping
    is the recognition being reported, whether or not it has a name. The
    unknown bucket's own label ("unknown-face-3") is still a row id, and
    a row id read fifty times is a directory."""
    identity = Identity(
        identifier_key="face", class_name="person", first_seen=TS, last_seen=TS
    )
    identity.id = 3
    assert identity.display_name == "unknown-face-3"
    assert redaction.name_for(identity, allowed=False) == "Known person"
    assert redaction.name_for(identity, allowed=True) == "unknown-face-3"


def test_a_plate_is_withheld_with_the_name():
    identity = Identity(
        identifier_key="vehicle",
        class_name="car",
        label=IDENTITY_LABEL,
        plate=PLATE,
        first_seen=TS,
        last_seen=TS,
    )
    assert redaction.plate_for(identity, allowed=True) == PLATE
    # None rather than "": the template prints no separator either, so a
    # withheld plate does not leave a bare dot behind.
    assert redaction.plate_for(identity, allowed=False) is None


def test_free_text_is_replaced_wholesale_but_emptiness_is_not():
    assert redaction.text_for("anything at all", "Incident 4", allowed=False) == (
        "Incident 4"
    )
    assert redaction.text_for("anything at all", "Incident 4", allowed=True) == (
        "anything at all"
    )
    # Nothing to withhold — substituting here would invent a description
    # where the screen means to show its own "—".
    assert redaction.text_for("", "Booking 4", allowed=False) == ""
    assert redaction.text_for(None, "Booking 4", allowed=False) is None


def test_a_render_with_no_request_withholds(env):
    """The Jinja globals resolve the viewer themselves. A context with no
    request is a template rendered outside the console's request path, and
    withholding is the safe direction — a rendering mistake must not be
    able to become a disclosure."""

    class _Ctx(dict):
        pass

    identity = Identity(
        identifier_key="face",
        class_name="person",
        label=IDENTITY_LABEL,
        first_seen=TS,
        last_seen=TS,
    )
    assert redaction._may_name_in(_Ctx()) is False
    assert redaction.name_for(identity, redaction._may_name_in(_Ctx())) == (
        "Known person"
    )


# -- the library's read API (found by the walk) ----------------------------


def test_the_library_json_api_sits_on_the_library_floor(env):
    """`/api/items/{id}/annotations` is the training corpus under a URL
    that does not start with `/library`, so the prefix that closed the
    screen did not close its data. Nothing here can be usefully
    substituted — an annotation editor without names edits nothing — so
    it takes the floor rather than the substitution.
    """
    assert auth.required_role("GET", "/api/items/1/annotations") == "view"
    assert signed_in(env, "nina", "restricted").get(
        "/api/items/1/annotations"
    ).status_code == 403
    assert signed_in(env, "vera", "view").get(
        "/api/items/1/annotations"
    ).status_code == 200
