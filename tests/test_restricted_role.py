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

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import (
    Camera,
    Event,
    User,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web import auth
from siteloom.web.app import create_app
from siteloom.web.auth import ROLES, has_role, is_gallery_media, required_role

TS = datetime(2026, 8, 6, 9, 0, 0)


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


# -- open mode -------------------------------------------------------------


def test_open_mode_is_untouched_by_the_new_floor(env):
    """With no User rows the gate is off entirely, and the floors decide
    nothing. Everything the console has stays reachable."""
    client = TestClient(env["app"], follow_redirects=False)
    for path in ("/", "/identities", "/training", "/users", "/audit"):
        assert client.get(path).status_code == 200, path
    assert client.get("/media/library/crops/face.jpg").status_code == 200
