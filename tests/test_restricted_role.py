"""The fourth rung: `restricted` (CLD-31).

`restricted` sits below `view` and exists for one deployment: a
night-shift operator who should judge whether an event matters without
browsing everyone's face. That is biometric data minimisation (NFR5), so
the thing to hold is not "a role string was added" but the two halves of
the claim — the triage surface stays reachable, and identities, the face
gallery and the training corpus do not.

Two properties get their own tests because they are the ones that would
rot quietly:

* **The mechanism is a prefix floor in one middleware.** Every assertion
  below goes through `required_role`/the live app, never a decorator, so
  a route added under a covered prefix is gated without anyone deciding
  to gate it.
* **Adding a rung at the bottom renumbered the other three.** Nothing
  stores or compares the integer, so no data migrates — but that is a
  claim about every comparison in the codebase, and the ladder matrix
  here is what keeps it true.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

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
from siteloom.web import auth, nav
from siteloom.web.app import create_app
from siteloom.web.auth import ROLES, has_role, is_gallery_media, required_role

TS = datetime(2026, 8, 6, 9, 0, 0)

#: The identity this event matched — the rail must keep showing it, or
#: there is nothing on the screen to judge.
ON_THE_EVENT = "Bo Truck"
#: Someone else entirely, who has never been on this camera. The picker
#: used to list them on every event detail (CLD-103).
SOMEONE_ELSE = "Aleks Corolla"
ELSEWHERE_PLATE = "XYZ-9021"


@pytest.fixture
def env(tmp_path):
    """A console with one event, and both kinds of crop on disk."""
    media = tmp_path / "m"
    # Event crops: media_dir/<camera-id>/<date>/. What triage looks at.
    event_crop = media / "cam1" / "2026-08-06" / "crop.jpg"
    # The face gallery and training corpus: media_dir/library/.
    gallery_crop = media / "library" / "crops" / "face.jpg"
    thumb = media / "library" / "thumbs" / "photo.jpg"
    for path in (event_crop, gallery_crop, thumb):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\xff\xd8\xff\xd9")
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(db_url=f"sqlite:///{tmp_path}/r.db", media_dir=str(media)),
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
                class_name="person",
                first_seen=TS,
                last_seen=TS,
                detection_count=3,
                best_crop_path=str(event_crop),
            )
        )
        # Two labelled identities: one this event matched, one that has
        # nothing to do with it. The picker is the only thing on an event
        # screen that ever mentioned the second.
        s.add(
            Identity(
                identifier_key="vehicle",
                class_name="truck",
                label=ON_THE_EVENT,
                first_seen=TS,
                last_seen=TS,
            )
        )
        s.add(
            Identity(
                identifier_key="vehicle",
                class_name="car",
                label=SOMEONE_ELSE,
                plate=ELSEWHERE_PLATE,
                first_seen=TS,
                last_seen=TS,
            )
        )
        s.flush()
        matched = s.scalar(select(Identity).filter_by(label=ON_THE_EVENT))
        s.add(
            EventIdentity(
                event_id=1,
                identity_id=matched.id,
                identifier_key="vehicle",
                similarity=0.91,
                matched_by="visual",
            )
        )
        s.commit()
    app = create_app(config)
    return {
        "app": app,
        "Session": Session,
        "media": media,
        "event_crop": event_crop,
        "gallery_crop": gallery_crop,
    }


def add_user(Session, username, role):
    with Session() as s:
        s.add(
            User(
                username=username,
                password_hash=auth.hash_password("hunter2!"),
                role=role,
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        s.commit()


def signed_in(env, username, role):
    add_user(env["Session"], username, role)
    client = TestClient(env["app"], follow_redirects=False)
    r = client.post(
        "/login", data={"username": username, "password": "hunter2!", "next": "/"}
    )
    assert r.status_code == 303, r.text[:200]
    return client


# -- the ladder itself -----------------------------------------------------


def test_restricted_is_the_bottom_rung_and_the_order_is_unchanged():
    assert list(ROLES) == ["restricted", "view", "edit", "admin"]
    assert sorted(ROLES, key=ROLES.get) == ["restricted", "view", "edit", "admin"]
    assert ROLES["restricted"] < ROLES["view"] < ROLES["edit"] < ROLES["admin"]


def test_renumbering_left_every_existing_comparison_saying_the_same_thing():
    """The rungs moved from 0/1/2 to 1/2/3; who may do what did not.

    Walked as a matrix rather than spot-checked, because "no comparison
    hardcodes an integer" is a claim about all of them at once.
    """
    reach = {
        "restricted": {"restricted"},
        "view": {"restricted", "view"},
        "edit": {"restricted", "view", "edit"},
        "admin": {"restricted", "view", "edit", "admin"},
    }
    for role, allowed in reach.items():
        user = User(username=role, password_hash="x", role=role, created_at=TS)
        for floor in ROLES:
            assert has_role(user, floor) is (floor in allowed), (role, floor)


def test_an_unrecognised_role_string_grants_nothing():
    """`restricted` is a real rung at 0 now, so "unknown" can no longer be
    spelled as 0 — a row written by a newer build must still be inert."""
    user = User(username="x", password_hash="x", role="wizard", created_at=TS)
    assert not any(has_role(user, floor) for floor in ROLES)


# -- the prefix floors -----------------------------------------------------


def test_the_triage_surface_sits_on_the_bottom_floor():
    for path in ("/", "/events/12", "/events/12/rail", "/live", "/noise", "/stats"):
        assert required_role("GET", path) == "restricted", path


def test_people_and_training_material_sit_on_the_view_floor():
    for path in (
        "/identities",
        "/identities/4",
        "/training",
        "/train",
        "/train/status",
        "/classes",
        "/library",
        "/library/items/3",
        "/search",
    ):
        assert required_role("GET", path) == "view", path


def test_the_account_and_audit_screens_are_admin_reads():
    """Not merely admin *writes*: the account list and the audit trail are
    reads a viewer may not make at all, which is why they need a floor of
    their own rather than reusing ADMIN_PREFIXES."""
    assert required_role("GET", "/users") == "admin"
    assert required_role("GET", "/audit") == "admin"
    assert required_role("POST", "/users/3/role") == "admin"


def test_a_restricted_operator_can_work_the_queue(env):
    client = signed_in(env, "nina", "restricted")
    for path in ("/", "/events/1", "/noise", "/stats", "/jobs", "/bookings"):
        assert client.get(path).status_code == 200, path


def test_a_restricted_operator_cannot_reach_people_or_training_data(env):
    client = signed_in(env, "nina", "restricted")
    for path in (
        "/identities",
        "/identities/1",
        "/training",
        "/train",
        "/classes",
        "/library",
        "/search?q=okonjo",
    ):
        assert client.get(path).status_code == 403, path


def test_the_same_paths_stay_open_to_a_viewer(env):
    """The floor is a floor: renumbering must not have quietly cost the
    rung above it anything."""
    client = signed_in(env, "vera", "view")
    for path in ("/", "/identities", "/training", "/classes", "/search?q=x"):
        assert client.get(path).status_code == 200, path


def test_restricted_grants_no_mutation(env):
    """The consequence of a strict ladder, asserted rather than assumed: a
    rung below `view` cannot be given a power `view` lacks, so a restricted
    operator reads the queue and records nothing. Triage is read-only."""
    client = signed_in(env, "nina", "restricted")
    assert client.post("/events/1/review", data={"reviewed": "1"}).status_code == 403


# -- /media: one route, two kinds of image ---------------------------------


def test_event_crops_are_reachable_but_gallery_crops_are_not(env):
    """The hard case. Both are served by `/media/{path}`, so the prefix
    cannot separate them — the directory the crop was written to can.
    """
    client = signed_in(env, "nina", "restricted")
    assert client.get("/media/cam1/2026-08-06/crop.jpg").status_code == 200
    assert client.get("/media/library/crops/face.jpg").status_code == 403
    assert client.get("/media/library/thumbs/photo.jpg").status_code == 403


def test_a_viewer_still_gets_both(env):
    client = signed_in(env, "vera", "view")
    assert client.get("/media/cam1/2026-08-06/crop.jpg").status_code == 200
    assert client.get("/media/library/crops/face.jpg").status_code == 200


def test_the_gallery_gate_holds_for_the_absolute_path_spelling(env):
    """`_media_candidates` serves an absolute stored path as well as one
    anchored on media_dir, so the same file has two URLs and only one of
    them starts with `/media/library`. A gate on the URL prefix would look
    like it worked and be one spelling away from open.
    """
    absolute = f"/media/{env['gallery_crop']}"
    assert signed_in(env, "vera", "view").get(absolute).status_code == 200
    assert signed_in(env, "nina", "restricted").get(absolute).status_code == 403


def test_the_gallery_segment_is_matched_case_insensitively():
    """The primary target is a case-insensitive filesystem, where
    /media/Library/... serves exactly what /media/library/... does."""
    assert is_gallery_media("/media/library/crops/a.jpg")
    assert is_gallery_media("/media/Library/Crops/a.jpg")
    assert is_gallery_media("/media//srv/media/library/faces/a.jpg")
    assert not is_gallery_media("/media/cam1/2026-08-06/a.jpg")
    # "library" has to be a whole segment, not a substring of one: a
    # camera called "library-yard" is not the face gallery.
    assert not is_gallery_media("/media/library-yard/2026-08-06/a.jpg")


# -- the identity picker: the directory by another name (CLD-103) ----------
#
# The screens holding people are closed to `restricted`; the triage rail
# is deliberately open, and it carried the same list as a dropdown. These
# assert on the response *body*, not on the control: hiding the <select>
# in the template would leave every name in the payload, which is the
# same leak with a stylesheet in front of it.

#: Every URL that renders the picker. The full page and the fragment are
#: two spellings of one screen, and the index renders the rail inline for
#: a deep link, so a fix in one of the three is a fix in none.
PICKER_URLS = ("/events/1", "/events/1/rail", "/?selected=1")


def test_a_restricted_operator_is_not_handed_every_identity(env):
    client = signed_in(env, "nina", "restricted")
    for url in PICKER_URLS:
        body = client.get(url).text
        assert SOMEONE_ELSE not in body, url
        assert ELSEWHERE_PLATE not in body, url


def test_the_events_own_identity_is_still_shown(env):
    """The rung withholds the list of everyone else, not the event.

    A triage screen that will not say who the system thinks this was
    cannot be triaged, so the claim in front of the operator stays — with
    its verdict buttons, which is the judgement `restricted` exists for.
    """
    client = signed_in(env, "nina", "restricted")
    for url in PICKER_URLS:
        assert ON_THE_EVENT in client.get(url).text, url


def test_the_picker_works_normally_at_view_and_above(env):
    """The floor is a floor: closing it below `view` must cost the rungs
    above it nothing, on all three URLs."""
    for username, role in (("vera", "view"), ("ed", "edit"), ("ada", "admin")):
        client = signed_in(env, username, role)
        for url in PICKER_URLS:
            body = client.get(url).text
            assert SOMEONE_ELSE in body, (role, url)
            assert ON_THE_EVENT in body, (role, url)


def test_the_pickers_floor_is_the_identities_floor(env):
    """Not a second literal rung sitting next to the first.

    The candidate list *is* `/identities` in a dropdown, so it is held to
    that screen's own floor — asked of `required_role`, so moving the
    floor moves both or neither. Walked as a matrix for the same reason
    the ladder above is: this is a claim about every rung.
    """
    floor = required_role("GET", "/identities")
    for role in ROLES:
        client = signed_in(env, f"op-{role}", role)
        allowed = has_role(
            User(username="x", password_hash="x", role=role, created_at=TS), floor
        )
        assert (SOMEONE_ELSE in client.get("/events/1/rail").text) is allowed, role


def test_open_mode_still_offers_the_whole_picker(env):
    """No User rows means no rungs; correcting a claim must not start
    needing an account to be possible."""
    body = TestClient(env["app"]).get("/events/1/rail").text
    assert SOMEONE_ELSE in body and ON_THE_EVENT in body


# -- the sidebar -----------------------------------------------------------


def test_the_sidebar_lists_only_what_the_viewer_may_open(env):
    """An entry that answers 403 describes a console the operator does
    not have. The filter is `required_role` — the same function the
    middleware refuses with — so the two cannot disagree."""
    client = signed_in(env, "nina", "restricted")
    body = client.get("/").text
    for label in ("Identities", "Media library", "Models", "Operators"):
        assert f">{label}</span>" not in body, label
    # What it does keep: the queue it exists to work.
    for label in ("Events", "Live view", "Noise"):
        assert f">{label}</span>" in body, label


def test_every_entry_a_role_is_shown_actually_opens(env):
    """tests/test_nav.py holds this for the whole sidebar in open mode;
    filtering it per role means holding it once per role, or the fix
    trades a 403 on click for a 404 on click."""
    for username, role in (
        ("nina", "restricted"),
        ("vera", "view"),
        ("ed", "edit"),
        ("ada", "admin"),
    ):
        client = signed_in(env, username, role)
        user = User(username=username, password_hash="x", role=role, created_at=TS)
        shown = nav.items(user)
        assert shown, role
        for item in shown:
            assert client.get(item.href).status_code == 200, (role, item.href)


def test_an_admin_still_sees_the_whole_sidebar(env):
    admin = User(username="ada", password_hash="x", role="admin", created_at=TS)
    assert nav.items(admin) == nav.items()


def test_the_sidebar_names_the_rung_the_way_the_console_does(env):
    """`edit` is a stored value; "Writer" is what the operator is called
    everywhere else (auth.ROLE_LABELS)."""
    body = signed_in(env, "ed", "edit").get("/").text
    assert '<div class="sl-op-role">Writer</div>' in body
    assert '<div class="sl-op-role">edit</div>' not in body


# -- open mode -------------------------------------------------------------


def test_open_mode_is_untouched_by_the_new_floor(env):
    """With no User rows the gate is off entirely, and the floors decide
    nothing. Everything the console has stays reachable."""
    client = TestClient(env["app"], follow_redirects=False)
    for path in ("/", "/identities", "/training", "/users", "/audit"):
        assert client.get(path).status_code == 200, path
    assert client.get("/media/library/crops/face.jpg").status_code == 200
