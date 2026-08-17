"""Link / unlink / reassign an identity on an event (CLD-36).

Before this the console could only grade a claim ("wrong") — never
correct it. A wrong name kept rendering in the events list forever, the
polluted gallery kept re-attracting the same wrong match, and the right
name was unsayable.

The rules these tests hold:

* Correcting a claim edits the vector store as well as the database.
  A name the matcher cannot see is not a name (identity/enroll.py), and
  vectors left in the wrong gallery re-create the error on the next
  frame. Unlink strips what the event taught; reassign moves it.
* An unlinked row is kept, not deleted — the record of what the system
  got wrong is the data the accuracy work reads — but it stops counting
  as a claim everywhere: the events list, review status, the unmatched
  chip, the identity's own event list, and ingest's link lookup.
* A verdict still touches nothing. Judging and editing are separate.

The embedder is stubbed (colour-square crops -> fixed unit vectors);
tests must not require model weights.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from siteloom.config import CameraConfig, IdentityConfig, SiteConfig, StorageConfig
from siteloom.identity import VectorStore, get_shared_store
from siteloom.store import (
    Camera,
    Detection,
    Event,
    EventIdentity,
    Identity,
    User,
    get_session,
    init_db,
    make_engine,
)
from siteloom.store.models import status_clause, unmatched_clause
from siteloom.web.app import create_app
from siteloom.web.auth import hash_password, required_role


def _unit(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / np.linalg.norm(v)


#: Crop colour (BGR) -> the embedding the stub produces for it.
COLOURS = {
    "blue": ((255, 0, 0), _unit(1.0, 0.0, 0.0, 0.0)),
    "green": ((0, 255, 0), _unit(0.0, 1.0, 0.0, 0.0)),
    "red": ((0, 0, 255), _unit(0.0, 0.0, 1.0, 0.0)),
}


class StubEmbedder:
    """Embeds a solid-colour crop as its colour's unit vector."""

    dim = 4

    def embed(self, bgr):
        mean = bgr.reshape(-1, 3).mean(axis=0)
        best, best_dist = None, 1e9
        for _, (colour, vector) in COLOURS.items():
            dist = float(np.abs(mean - np.array(colour, dtype=np.float32)).sum())
            if dist < best_dist:
                best, best_dist = vector, dist
        return best


@pytest.fixture
def edit_env(tmp_path, monkeypatch):
    """One event the resolver attributed to the wrong vehicle.

    The event's crops are red; the identity it was linked to ("Aleks
    Corolla") has the red vector the event taught it plus a green one
    from another visit, so a correction that removes too much is as
    visible as one that removes too little. A second identity ("Bo
    Truck") is the reassignment target.
    """
    monkeypatch.setattr(
        "siteloom.identity.embedders.build_embedder",
        lambda algo, device="mps", projection_path=None: StubEmbedder(),
    )
    config = SiteConfig(
        site_id="test-site",
        site_name="Test Site",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/edit.db", media_dir=str(tmp_path / "media")
        ),
        identity=IdentityConfig(vector_db_path=str(tmp_path / "vectors")),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)

    crops = tmp_path / "crops"
    crops.mkdir()
    crop_paths = {}
    for name, (colour, _) in COLOURS.items():
        square = np.full((16, 16, 3), colour, dtype=np.uint8)
        path = crops / f"{name}.jpg"
        cv2.imwrite(str(path), square)
        crop_paths[name] = str(path)

    with Session() as session:
        session.add(Camera(id="cam1", site_id="test-site", name="Cam One"))
        event = Event(
            camera_id="cam1",
            track_id=7,
            class_name="car",
            first_seen=datetime(2026, 8, 7, 14, 16, 0),
            last_seen=datetime(2026, 8, 7, 14, 16, 30),
            detection_count=2,
            best_crop_path=crop_paths["red"],
            best_confidence=0.9,
        )
        session.add(event)
        session.flush()
        for i in range(2):
            session.add(
                Detection(
                    event_id=event.id,
                    timestamp=datetime(2026, 8, 7, 14, 16, i * 10),
                    class_name="car",
                    confidence=0.9,
                    bbox="[1, 2, 3, 4]",
                    zones='["driveway"]',
                    crop_path=crop_paths["red"],
                )
            )
        wrong = Identity(
            identifier_key="vehicle",
            class_name="car",
            label="Aleks Corolla",
            first_seen=datetime(2026, 8, 1),
            last_seen=datetime(2026, 8, 7, 14, 16, 30),
            appearance_count=5,
            vector_count=2,
            best_crop_path=crop_paths["green"],
        )
        other = Identity(
            identifier_key="vehicle",
            class_name="truck",
            label="Bo Truck",
            first_seen=datetime(2026, 8, 2),
            last_seen=datetime(2026, 8, 3),
            appearance_count=1,
            vector_count=0,
        )
        session.add_all([wrong, other])
        session.flush()
        link = EventIdentity(
            event_id=event.id,
            identity_id=wrong.id,
            identifier_key="vehicle",
            similarity=0.83,
            matched_by="visual",
            hit_count=2,
        )
        session.add(link)
        session.commit()
        ids = SimpleNamespace(
            event=event.id, wrong=wrong.id, other=other.id, link=link.id
        )

    # Seeded the way live matching writes them (CLD-84): the red vector
    # records the crop this event contributed it from.
    store = VectorStore(config.identity.vector_db_path)
    try:
        store.add(
            "vehicle", COLOURS["red"][1], ids.wrong, crop_path=crop_paths["red"]
        )
        store.add(
            "vehicle", COLOURS["green"][1], ids.wrong, crop_path=crop_paths["green"]
        )
    finally:
        store.close()

    return SimpleNamespace(
        client=TestClient(create_app(config)),
        Session=Session,
        config=config,
        ids=ids,
        crop_paths=crop_paths,
    )


def _store(env):
    return get_shared_store(env.config.identity.vector_db_path)


def _owner(env, vector):
    hit = _store(env).best_match("vehicle", vector)
    assert hit is not None
    return hit.identity_id, hit.score


# -- attach ---------------------------------------------------------------


def test_attach_links_an_identity_and_teaches_it_the_event(edit_env):
    """The missing half of the verdict buttons: saying who it was, and
    making the system able to see that person next time."""
    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity",
        data={"identity_id": edit_env.ids.other, "enroll": "1"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == f"/identities/{edit_env.ids.other}"

    with edit_env.Session() as session:
        link = session.scalar(
            select(EventIdentity).filter_by(
                event_id=edit_env.ids.event, identity_id=edit_env.ids.other
            )
        )
        # Provenance of a correction matters as much as of a match.
        assert link.matched_by == "human"
        assert link.verdict == "confirmed"
        assert link.similarity == 0.0
        assert link.unlinked_at is None
        target = session.get(Identity, edit_env.ids.other)
        assert target.vector_count == 1
        assert target.appearance_count == 2
    # The event's crop now answers as the attached identity.
    owner, score = _owner(edit_env, COLOURS["red"][1])
    assert owner == edit_env.ids.other and score > 0.99


def test_attach_without_learning_records_the_link_but_no_vector(edit_env):
    """A crop can be too poor to want in a gallery; the link is still
    worth recording, and it must not quietly enroll anyway."""
    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity",
        data={"identity_id": edit_env.ids.other, "enroll": "0"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    with edit_env.Session() as session:
        assert session.get(Identity, edit_env.ids.other).vector_count == 0
        assert session.scalar(
            select(EventIdentity).filter_by(
                event_id=edit_env.ids.event, identity_id=edit_env.ids.other
            )
        ) is not None


def test_an_unchecked_learn_box_does_not_enroll(edit_env):
    """The form submits a hidden "0" ahead of the checkbox, because an
    unchecked box sends nothing at all — without it, clearing the box
    would still enroll. Pinned here so the template pattern cannot be
    "simplified" away."""
    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity",
        # What the browser sends with the box cleared: the hidden field
        # alone.
        data={"identity_id": str(edit_env.ids.other), "enroll": "0"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    with edit_env.Session() as session:
        assert session.get(Identity, edit_env.ids.other).vector_count == 0

    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity",
        # ... and with it ticked: both fields, the checkbox last.
        data={"identity_id": "new", "identifier": "vehicle", "enroll": ["0", "1"]},
        follow_redirects=False,
    )
    assert r.status_code == 303
    new_id = int(r.headers["location"].split("/identities/")[1])
    with edit_env.Session() as session:
        assert session.get(Identity, new_id).vector_count == 1


def test_attaching_the_same_identity_twice_changes_nothing(edit_env):
    """A double-submit or a refreshed form must not enroll the same crop
    again or count a second visit — the gallery would fill with copies of
    one image and the identity's appearance count would drift."""
    for _ in range(2):
        assert edit_env.client.post(
            f"/events/{edit_env.ids.event}/identity",
            data={"identity_id": edit_env.ids.other, "enroll": "1"},
        ).status_code == 200
    with edit_env.Session() as session:
        target = session.get(Identity, edit_env.ids.other)
        assert target.vector_count == 1
        assert target.appearance_count == 2  # 1 seeded + this one visit
        assert len(
            session.scalars(
                select(EventIdentity).filter_by(
                    event_id=edit_env.ids.event, identity_id=edit_env.ids.other
                )
            ).all()
        ) == 1


def test_attach_can_mint_a_named_identity_the_resolver_never_had(edit_env):
    """When the resolver folded two people into one row, the right target
    does not exist yet — so "new identity" is part of the correction, not
    a convenience."""
    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity",
        data={
            "identity_id": "new",
            "identifier": "vehicle",
            "label": "Dana Van",
            "enroll": "1",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    new_id = int(r.headers["location"].split("/identities/")[1])
    with edit_env.Session() as session:
        fresh = session.get(Identity, new_id)
        assert fresh.label == "Dana Van"
        assert fresh.identifier_key == "vehicle"
        assert fresh.class_name == "car"  # taken from the event
        assert fresh.vector_count == 1  # named AND visible to matching
        assert fresh.best_crop_path == edit_env.crop_paths["red"]


def test_attach_rejects_a_nonexistent_target_and_identifier(edit_env):
    assert edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity", data={"identity_id": "999"}
    ).status_code == 404
    assert edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity", data={"identity_id": "abc"}
    ).status_code == 400
    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity",
        data={"identity_id": "new", "identifier": "nonsense"},
    )
    assert r.status_code == 400 and "nonsense" in r.json()["detail"]
    assert edit_env.client.post(
        "/events/999/identity", data={"identity_id": str(edit_env.ids.other)}
    ).status_code == 404


@pytest.mark.parametrize("surface", ["/rail", ""])
def test_the_rail_offers_the_correction_actions(edit_env, surface):
    """The endpoints are only half the feature — CLD-36 was filed because
    the rail had verdict buttons and nothing else. Both surfaces owe the
    same actions: an operator on the full page is correcting the same
    claim as one on the rail."""
    event, link = edit_env.ids.event, edit_env.ids.link
    page = edit_env.client.get(f"/events/{event}{surface}").text
    assert f'action="/events/{event}/identity"' in page  # attach
    assert f'action="/events/{event}/identity/{link}/unlink"' in page
    assert f'action="/events/{event}/identity/{link}/reassign"' in page
    assert "Bo Truck" in page  # the picker lists candidates
    # Reassigning to the identity already linked is not a correction, so
    # the picker for this claim leaves it out.
    reassign_form = page.split(f"/identity/{link}/reassign")[1].split("</form>")[0]
    assert "Bo Truck" in reassign_form
    assert "Aleks Corolla" not in reassign_form


# -- unlink ---------------------------------------------------------------


def test_unlink_keeps_the_record_and_strips_what_it_taught(edit_env):
    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity/{edit_env.ids.link}/unlink",
        follow_redirects=False,
    )
    assert r.status_code == 303

    with edit_env.Session() as session:
        link = session.get(EventIdentity, edit_env.ids.link)
        # Negatives are data: what was claimed, how strongly, by what.
        assert link.identity_id == edit_env.ids.wrong
        assert link.similarity == 0.83
        assert link.matched_by == "visual"
        assert link.unlinked_at is not None
        assert link.verdict == "wrong"
        assert not link.is_active
        wrong = session.get(Identity, edit_env.ids.wrong)
        assert wrong.vector_count == 1  # the event's crop is gone
        assert wrong.appearance_count == 4
    # The other visit's vector survives — unlink removes what THIS event
    # contributed, not the gallery.
    owner, score = _owner(edit_env, COLOURS["green"][1])
    assert owner == edit_env.ids.wrong and score > 0.99
    _, red_score = _owner(edit_env, COLOURS["red"][1])
    assert red_score < 0.5


def test_an_unlinked_claim_stops_being_a_claim_everywhere(edit_env):
    """Half a correction is worse than none: if the name still renders in
    the events list, or the event still sits `flagged`, the operator has
    no way to tell the fix took."""
    edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity/{edit_env.ids.link}/unlink"
    )
    with edit_env.Session() as session:
        event = session.get(Event, edit_env.ids.event)
        assert event.active_identities == []
        # Not "flagged" by the wrong verdict it carries: that claim was
        # corrected, so nothing is outstanding.
        assert event.review_status == "new"
        # ... and the SQL form agrees, or triage paging would disagree
        # with the badge it renders.
        assert session.scalar(
            select(Event.id).where(Event.id == event.id, status_clause("new"))
        ) == event.id
        # The event reads as unmatched again, which is where an operator
        # looking for events to name will find it.
        assert session.scalar(
            select(Event.id).where(Event.id == event.id, unmatched_clause())
        ) == event.id
    # The identity's own page no longer counts this as one of its visits.
    body = edit_env.client.get(f"/identities/{edit_env.ids.wrong}").text
    assert f"/events/{edit_env.ids.event}" not in body
    # The events list stops printing the wrong name on the row.
    assert "Aleks Corolla" not in edit_env.client.get("/").text
    # The rail still says it happened.
    rail = edit_env.client.get(f"/events/{edit_env.ids.event}/rail").text
    assert "Unlinked from" in rail


def test_unlink_reverts_a_plate_this_claim_taught(edit_env):
    """A mis-learned plate wins outright over visual similarity (PRD
    §6.4), so it poisons every future sighting of that number."""
    with edit_env.Session() as session:
        session.get(Identity, edit_env.ids.wrong).plate = "XYZ789"
        session.get(EventIdentity, edit_env.ids.link).learned_plate = True
        session.commit()
    edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity/{edit_env.ids.link}/unlink"
    )
    with edit_env.Session() as session:
        assert session.get(Identity, edit_env.ids.wrong).plate is None


def test_re_attaching_an_unlinked_identity_revives_the_same_row(edit_env):
    """An operator changing their mind about one pairing is one decision;
    stacking a second row for it would double-count the visit and leave
    the events page showing a name that is also marked wrong."""
    edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity/{edit_env.ids.link}/unlink"
    )
    edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity",
        data={"identity_id": edit_env.ids.wrong, "enroll": "1"},
    )
    with edit_env.Session() as session:
        links = session.scalars(
            select(EventIdentity).filter_by(event_id=edit_env.ids.event)
        ).all()
        assert len(links) == 1
        assert links[0].id == edit_env.ids.link
        assert links[0].unlinked_at is None
        assert links[0].verdict == "confirmed"
        assert links[0].matched_by == "visual"  # how it was originally made


def test_unlink_rejects_a_link_from_another_event(edit_env):
    assert edit_env.client.post(
        f"/events/999/identity/{edit_env.ids.link}/unlink"
    ).status_code == 404
    assert edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity/999/unlink"
    ).status_code == 404


# -- reassign -------------------------------------------------------------


def test_reassign_moves_the_vectors_to_the_right_identity(edit_env):
    """The whole point of doing this as one action: the evidence moves.
    Unlink-then-attach would delete the event's vector and re-embed it;
    reassign carries the original across, so the wrong identity stops
    matching it and the right one starts."""
    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity/{edit_env.ids.link}/reassign",
        data={"identity_id": edit_env.ids.other},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == f"/identities/{edit_env.ids.other}"

    with edit_env.Session() as session:
        old = session.get(EventIdentity, edit_env.ids.link)
        assert old.unlinked_at is not None and old.verdict == "wrong"
        new = session.scalar(
            select(EventIdentity).filter_by(
                event_id=edit_env.ids.event, identity_id=edit_env.ids.other
            )
        )
        assert new.matched_by == "human" and new.verdict == "confirmed"
        assert session.get(Identity, edit_env.ids.wrong).vector_count == 1
        assert session.get(Identity, edit_env.ids.other).vector_count == 1
        # The events list now prints the corrected name.
        assert session.get(Event, edit_env.ids.event).active_identities == [new]
    owner, score = _owner(edit_env, COLOURS["red"][1])
    assert owner == edit_env.ids.other and score > 0.99


def test_reassign_enrolls_when_there_was_nothing_to_move(edit_env):
    """A legacy vector with no provenance, or an event whose crops never
    entered a gallery, leaves nothing to carry across — and a claim the
    matcher cannot see is the failure this feature exists to prevent, so
    the crop is enrolled fresh instead."""
    _store(edit_env).delete_identity("vehicle", edit_env.ids.wrong)
    with edit_env.Session() as session:
        session.get(Identity, edit_env.ids.wrong).vector_count = 0
        session.commit()

    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity/{edit_env.ids.link}/reassign",
        data={"identity_id": edit_env.ids.other},
        follow_redirects=False,
    )
    assert r.status_code == 303
    with edit_env.Session() as session:
        assert session.get(Identity, edit_env.ids.other).vector_count == 1
    owner, score = _owner(edit_env, COLOURS["red"][1])
    assert owner == edit_env.ids.other and score > 0.99


def _person_identity(env, key: str, label: str) -> int:
    with env.Session() as session:
        identity = Identity(
            identifier_key=key,
            class_name="person",
            label=label,
            first_seen=datetime(2026, 8, 1),
            last_seen=datetime(2026, 8, 1),
        )
        session.add(identity)
        session.commit()
        return identity.id


def test_cross_kind_association_is_refused(edit_env):
    """A person identity may not claim a car event — attach or reassign.

    This is the "association mess" gate: one cross-kind link poisons the
    gallery for every future match of that identity, and unwinding it is
    identity surgery. The rule is by identifier applicability, not by
    the Identity row's own class — Bo Truck (class "truck") claiming a
    "car" event stays legal because the vehicle identifier spans both.
    """
    person_id = _person_identity(edit_env, "person", "Dana")

    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity",
        data={"identity_id": person_id, "enroll": "0"},
    )
    assert r.status_code == 400
    assert "cross-kind" in r.json()["detail"]

    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity/{edit_env.ids.link}/reassign",
        data={"identity_id": person_id},
    )
    assert r.status_code == 400
    # Nothing moved: the refusal happened before any store edit.
    store = _store(edit_env)
    assert store.count_identity("vehicle", edit_env.ids.wrong) == 2
    assert store.count_identity("person", person_id) == 0

    # Minting a NEW identity under an inapplicable identifier is the
    # same wrong link one step earlier.
    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity",
        data={"identity_id": "new", "identifier": "face", "label": "Who"},
    )
    assert r.status_code == 400
    assert "does not apply" in r.json()["detail"]


@pytest.mark.parametrize("surface", ["/rail", ""])
def test_the_picker_offers_only_compatible_identities(edit_env, surface):
    """The offer side of the same rule: a person identity is not listed
    on a car event, and neither is the face identifier — the form must
    not offer what the POST refuses.

    Parametrized over both surfaces because testing only the rail is the
    gap this bug shipped through (CLD-135). The rail filtered its picker
    and was held to it here; the full page built its own picker and was
    held to nothing, so `/events/{id}` offered every identity in the
    store and the operator learned about it by having the POST refuse
    the choice it had just been offered.
    """
    _person_identity(edit_env, "person", "Dana")
    page = edit_env.client.get(f"/events/{edit_env.ids.event}{surface}").text
    assert "Bo Truck" in page
    assert "Dana" not in page
    assert '<option value="vehicle">' in page
    assert '<option value="face">' not in page


def test_reassign_across_compatible_identifiers_never_moves_the_vector(edit_env):
    """face -> person is the legitimate cross-identifier reassign (both
    consume person events). A face embedding is not a person embedding —
    different pipeline, different dimensionality, different collection —
    so the vector is stripped from the source and the target is enrolled
    with its own embedder, never moved raw between collections."""
    face_id = _person_identity(edit_env, "face", "Dana Face")
    person_id = _person_identity(edit_env, "person", "Dana")
    with edit_env.Session() as session:
        event = Event(
            camera_id="cam1",
            track_id=8,
            class_name="person",
            first_seen=datetime(2026, 8, 7, 15, 0, 0),
            last_seen=datetime(2026, 8, 7, 15, 0, 30),
            detection_count=1,
            best_crop_path=edit_env.crop_paths["red"],
            best_confidence=0.9,
        )
        session.add(event)
        session.flush()
        link = EventIdentity(
            event_id=event.id,
            identity_id=face_id,
            identifier_key="face",
            similarity=0.5,
            matched_by="visual",
        )
        session.add(link)
        session.commit()
        event_id, link_id = event.id, link.id
    store = _store(edit_env)
    store.add("face", COLOURS["red"][1], face_id, crop_path=edit_env.crop_paths["red"])

    r = edit_env.client.post(
        f"/events/{event_id}/identity/{link_id}/reassign",
        data={"identity_id": person_id},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert store.count_identity("face", face_id) == 0  # stripped
    assert store.count_identity("person", face_id) == 0  # never moved raw
    assert store.count_identity("person", person_id) == 1  # re-enrolled
    with edit_env.Session() as session:
        assert session.get(Identity, person_id).vector_count == 1


def test_reassign_to_a_new_identity_and_to_the_same_one(edit_env):
    same = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity/{edit_env.ids.link}/reassign",
        data={"identity_id": edit_env.ids.wrong},
    )
    assert same.status_code == 400

    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity/{edit_env.ids.link}/reassign",
        data={"identity_id": "new", "identifier": "vehicle"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    new_id = int(r.headers["location"].split("/identities/")[1])
    with edit_env.Session() as session:
        fresh = session.get(Identity, new_id)
        assert fresh.label is None  # unlabeled bucket, like any unknown
        assert fresh.vector_count == 1
    owner, _ = _owner(edit_env, COLOURS["red"][1])
    assert owner == new_id


# -- boundaries -----------------------------------------------------------


def test_a_verdict_still_edits_nothing(edit_env):
    """Judging and editing stay separate: "wrong" records an opinion for
    the accuracy numbers, and only unlink/reassign touch the gallery."""
    edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity/{edit_env.ids.link}/verdict",
        data={"verdict": "wrong"},
    )
    with edit_env.Session() as session:
        link = session.get(EventIdentity, edit_env.ids.link)
        assert link.verdict == "wrong"
        assert link.unlinked_at is None
        assert session.get(Identity, edit_env.ids.wrong).vector_count == 2
    owner, score = _owner(edit_env, COLOURS["red"][1])
    assert owner == edit_env.ids.wrong and score > 0.99


def test_edits_refuse_when_another_process_holds_the_store(edit_env):
    """Rows moved without their vectors is exactly the failure mode these
    endpoints exist to fix, so a locked store refuses rather than doing
    half the correction — the same contract merge and split follow."""
    holder = VectorStore(edit_env.config.identity.vector_db_path)
    try:
        unlink = edit_env.client.post(
            f"/events/{edit_env.ids.event}/identity/{edit_env.ids.link}/unlink"
        )
        reassign = edit_env.client.post(
            f"/events/{edit_env.ids.event}/identity/{edit_env.ids.link}/reassign",
            data={"identity_id": edit_env.ids.other},
        )
        enrolling = edit_env.client.post(
            f"/events/{edit_env.ids.event}/identity",
            data={"identity_id": edit_env.ids.other, "enroll": "1"},
        )
        # Recording who someone was needs no vector work, so it still
        # works while a backfill runs — with learning off, as asked.
        plain = edit_env.client.post(
            f"/events/{edit_env.ids.event}/identity",
            data={"identity_id": edit_env.ids.other, "enroll": "0"},
        )
    finally:
        holder.close()

    for response in (unlink, reassign, enrolling):
        assert response.status_code == 503
        assert "locked by another process" in response.json()["detail"]
    assert plain.status_code == 200  # followed the redirect

    with edit_env.Session() as session:
        assert session.get(EventIdentity, edit_env.ids.link).unlinked_at is None
        assert session.get(Identity, edit_env.ids.wrong).vector_count == 2


def test_identity_edits_require_the_edit_rung(edit_env):
    """Enforcement lives in the one auth middleware, not on the routes."""
    event, link = edit_env.ids.event, edit_env.ids.link
    for path in (
        f"/events/{event}/identity",
        f"/events/{event}/identity/{link}/unlink",
        f"/events/{event}/identity/{link}/reassign",
    ):
        assert required_role("POST", path) == "edit"

    with edit_env.Session() as session:
        session.add(
            User(
                username="vera",
                password_hash=hash_password("hunter2!"),
                role="view",
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        session.commit()
    assert edit_env.client.post(
        "/login", data={"username": "vera", "password": "hunter2!", "next": "/"},
        follow_redirects=False,
    ).status_code == 303
    r = edit_env.client.post(
        f"/events/{event}/identity/{link}/unlink"
    )
    assert r.status_code == 403
    with edit_env.Session() as session:
        assert session.get(EventIdentity, link).unlinked_at is None


# -- one standing claim per pairing (CLD-133) ------------------------------


def _active_links(session, event_id, identity_id):
    return session.scalars(
        select(EventIdentity)
        .where(
            EventIdentity.event_id == event_id,
            EventIdentity.identity_id == identity_id,
            EventIdentity.unlinked_at.is_(None),
        )
        .order_by(EventIdentity.id)
    ).all()


def test_attach_reaffirms_the_standing_claim_not_a_newer_repudiated_one(edit_env):
    """A pair can legitimately hold one live claim and several
    repudiated ones — an operator may unlink, re-attach, and unlink
    again. Attach looked up "the newest row for this pair" in any state,
    so it picked the repudiated one and cleared its `unlinked_at`,
    producing a second standing claim (and now an IntegrityError). The
    live claim is what a re-affirmation is about.
    """
    with edit_env.Session() as session:
        session.add(
            EventIdentity(
                event_id=edit_env.ids.event,
                identity_id=edit_env.ids.wrong,
                identifier_key="vehicle",
                similarity=0.4,
                matched_by="visual",
                verdict="wrong",
                unlinked_at=datetime(2026, 8, 7, 16, 0, 0),
            )
        )
        session.commit()

    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity",
        data={"identity_id": edit_env.ids.wrong, "enroll": "1"},
        follow_redirects=False,
    )
    assert r.status_code == 303

    with edit_env.Session() as session:
        live = _active_links(session, edit_env.ids.event, edit_env.ids.wrong)
        assert len(live) == 1
        assert live[0].id == edit_env.ids.link  # the claim that stood
        assert live[0].verdict == "confirmed"
        assert live[0].matched_by == "visual"  # how it was originally made
        # The repudiation is untouched — negatives are data.
        repudiated = session.scalars(
            select(EventIdentity).where(EventIdentity.unlinked_at.is_not(None))
        ).all()
        assert len(repudiated) == 1 and repudiated[0].verdict == "wrong"
        # Re-affirming a claim that already stands is not a new sighting.
        wrong = session.get(Identity, edit_env.ids.wrong)
        assert wrong.appearance_count == 5
        assert wrong.vector_count == 2


def test_attach_folds_into_a_claim_another_writer_just_made(edit_env):
    """The same pairing attached twice at once: one request wins, and the
    other must land on that row rather than stack a second one. Counting
    the visit again, or enrolling the same crop again, would let a
    double-submit inflate both the appearance count and the gallery."""
    with edit_env.Session() as session:
        # What the winning request left behind, committed before ours.
        session.add(
            EventIdentity(
                event_id=edit_env.ids.event,
                identity_id=edit_env.ids.other,
                identifier_key="vehicle",
                similarity=0.0,
                matched_by="human",
                verdict="confirmed",
                verdict_at=datetime(2026, 8, 7, 16, 0, 0),
            )
        )
        session.get(Identity, edit_env.ids.other).appearance_count = 2
        session.commit()

    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity",
        data={"identity_id": edit_env.ids.other, "enroll": "1"},
        follow_redirects=False,
    )
    assert r.status_code == 303

    with edit_env.Session() as session:
        assert len(_active_links(session, edit_env.ids.event, edit_env.ids.other)) == 1
        target = session.get(Identity, edit_env.ids.other)
        assert target.appearance_count == 2  # the visit is counted once
        assert target.vector_count == 0  # and this crop is not enrolled again


def test_an_attach_that_loses_the_race_leaves_the_survivor_uncounted(
    edit_env, monkeypatch
):
    """The race the previous test cannot reach through the client: the
    request looks, finds nothing, and only then does the other writer
    commit. The insert now collides and folds into their row.

    `hit_count` counts frames that evidenced the claim, and a manual
    attach contributes none — its own row is discarded and its
    `appearance_count` increment withheld — so the surviving row must not
    gain one either. The two counters move together or Σ hit_count stops
    matching appearance_count, which is the invariant the merge fold is
    argued from.
    """
    from siteloom.store import claims

    real_active_claim = claims.active_claim
    raced = []

    def race_then_miss(session, event_id, identity_id):
        if raced:
            return real_active_claim(session, event_id, identity_id)
        raced.append(True)
        # The other request commits its claim — and its bookkeeping —
        # after our lookup read, which is what makes this a race.
        with edit_env.Session() as other:
            other.add(
                EventIdentity(
                    event_id=edit_env.ids.event,
                    identity_id=edit_env.ids.other,
                    identifier_key="vehicle",
                    similarity=0.0,
                    hit_count=8,
                    matched_by=None,
                    verdict="confirmed",
                )
            )
            other.get(Identity, edit_env.ids.other).appearance_count = 2
            other.commit()
        return None

    monkeypatch.setattr(claims, "active_claim", race_then_miss)

    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity",
        data={"identity_id": edit_env.ids.other, "enroll": "1"},
        follow_redirects=False,
    )
    assert r.status_code == 303  # the collision is folded, not a 500

    with edit_env.Session() as session:
        live = _active_links(session, edit_env.ids.event, edit_env.ids.other)
        assert len(live) == 1
        assert live[0].hit_count == 8  # no frame was invented
        assert live[0].matched_by == "human"  # the evidence still arrived
        target = session.get(Identity, edit_env.ids.other)
        assert target.appearance_count == 2  # counted by the winner, once
        assert target.vector_count == 0


# -- the pickers on the full page (CLD-135) --------------------------------
#
# `/events/{id}` grew its own copies of the rail's pickers, unfiltered,
# and its missed form had no attribution at all. What follows holds the
# page to the rail's rules, and holds the miss form to being a record of
# *which* identifier failed rather than a bare flag.


def _event_of_class(env, class_name: str, track_id: int) -> int:
    with env.Session() as session:
        event = Event(
            camera_id="cam1",
            track_id=track_id,
            class_name=class_name,
            first_seen=datetime(2026, 8, 7, 15, 0, 0),
            last_seen=datetime(2026, 8, 7, 15, 0, 30),
            detection_count=1,
            best_crop_path=env.crop_paths["red"],
            best_confidence=0.9,
        )
        session.add(event)
        session.commit()
        return event.id


def _options(page: str) -> list[str]:
    """The identity ids a picker on this page is offering."""
    import re

    return re.findall(r'<option value="(\d+)"', page)


def test_two_identities_with_one_name_are_told_apart(edit_env):
    """A face identity and a person identity for the same human are the
    normal shape of this store, and both are legitimate targets on a
    person event — so a picker that prints only the label offers the
    operator the same word twice and no way to choose between them."""
    _person_identity(edit_env, "face", "Klara")
    _person_identity(edit_env, "person", "Klara")
    event_id = _event_of_class(edit_env, "person", track_id=21)

    page = edit_env.client.get(f"/events/{event_id}").text

    assert "Klara · face" in page
    assert "Klara · person" in page


def test_a_restricted_viewer_is_offered_nothing_to_pick(edit_env):
    """The label format has to compose with the naming floor, and the
    guarantee is one level above the label: `_identity_candidates`
    returns nothing at all below the floor, so the suffix cannot leak a
    kind or a plate because there is no option to hang it on.

    Asserted as an empty candidate list rather than as a substituted
    string — "Known person" here would pass for the wrong reason, by
    describing a redaction that never has to happen.
    """
    _person_identity(edit_env, "face", "Klara")
    event_id = _event_of_class(edit_env, "person", track_id=22)

    with edit_env.Session() as session:
        for username, role in (("vera", "view"), ("rick", "restricted")):
            session.add(
                User(
                    username=username,
                    password_hash=hash_password("hunter2!"),
                    role=role,
                    created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )
        session.commit()

    def _login(username):
        assert edit_env.client.post(
            "/login",
            data={"username": username, "password": "hunter2!", "next": "/"},
            follow_redirects=False,
        ).status_code == 303

    _login("vera")
    allowed = edit_env.client.get(f"/events/{event_id}").text
    assert "Klara · face" in allowed
    assert _options(allowed)  # the control: there is something to offer

    _login("rick")
    withheld = edit_env.client.get(f"/events/{event_id}")
    assert withheld.status_code == 200  # the screen is kept, not refused
    assert _options(withheld.text) == []
    assert "Klara" not in withheld.text


# -- attributing a miss ----------------------------------------------------


def _mark_missed(env, event_id, **data):
    return env.client.post(
        f"/events/{event_id}/missed", data=data, follow_redirects=False
    )


def _misses(env, event_id):
    with env.Session() as session:
        return session.scalars(
            select(EventIdentity)
            .filter_by(event_id=event_id, identity_id=None)
            .order_by(EventIdentity.id)
        ).all()


def test_a_missed_vehicle_is_recorded_as_a_missed_vehicle(edit_env):
    """Per-identifier recall (CLD-17) is the whole reason a miss is a row
    rather than a flag. Filing every miss under `face` because that is
    the form's default answers "how often does face ID miss" with events
    that never had a face in them."""
    r = _mark_missed(edit_env, edit_env.ids.event, missed="1", identifier="vehicle")

    assert r.status_code == 303
    misses = _misses(edit_env, edit_env.ids.event)
    assert len(misses) == 1
    assert misses[0].identifier_key == "vehicle"
    assert misses[0].verdict == "missed"


@pytest.mark.parametrize(
    "data",
    [
        {"missed": "1", "identifier": "face"},  # offered by nothing, posted anyway
        {"missed": "1"},  # the old wire shape, defaulting to face
    ],
)
def test_a_miss_attributed_to_an_impossible_identifier_is_refused(edit_env, data):
    """The accept side of the offer rule. A defaulted `face` on a car
    event used to be filed silently, which is worse than an error: it
    corrupts the recall numbers of an identifier that was never run."""
    r = _mark_missed(edit_env, edit_env.ids.event, **data)

    assert r.status_code == 400
    assert "vehicle" in r.json()["detail"]  # names what would be accepted
    assert _misses(edit_env, edit_env.ids.event) == []


def test_the_page_shows_a_miss_and_offers_to_retract_it(edit_env):
    """A mark an operator cannot see is a mark they cannot correct.

    The assertion is scoped to the miss form rather than to the page,
    because the word "vehicle" appears all over this page in the pickers
    — a page-level `in` would pass against the clear-all button that
    exists today and prove nothing about a per-miss retract.
    """
    import re

    _mark_missed(edit_env, edit_env.ids.event, missed="1", identifier="vehicle")

    page = edit_env.client.get(f"/events/{edit_env.ids.event}").text

    forms = re.findall(
        r'<form[^>]*action="/events/\d+/missed"[^>]*>(.*?)</form>', page, re.S
    )
    assert forms, "the page offers no missed form at all"
    retracts = [form for form in forms if 'value="0"' in form]
    assert retracts, "the miss is shown with no way to take it back"
    # It has to say *which* miss it retracts, or it is the clear-all
    # button wearing a per-row disguise.
    assert any(
        'name="identifier"' in form and "vehicle" in form for form in retracts
    )


def test_retracting_one_miss_leaves_the_others_and_the_flag(edit_env):
    """`Event.missed_identity` mirrors "any miss rows exist" and this
    endpoint is its single writer, so a per-identifier retract has to
    recompute the flag from what survives. Setting it False while a miss
    row remains would put every triage query that reads it — the flagged
    bucket, the status SQL — at odds with the table it summarises."""
    event_id = _event_of_class(edit_env, "person", track_id=23)
    for key in ("face", "person"):
        assert _mark_missed(
            edit_env, event_id, missed="1", identifier=key
        ).status_code == 303
    assert len(_misses(edit_env, event_id)) == 2

    assert _mark_missed(
        edit_env, event_id, missed="0", identifier="face"
    ).status_code == 303

    remaining = _misses(edit_env, event_id)
    assert [m.identifier_key for m in remaining] == ["person"]
    with edit_env.Session() as session:
        event = session.get(Event, event_id)
        assert event.missed_identity is True  # a miss still stands
        assert event.missed_at is not None

    assert _mark_missed(
        edit_env, event_id, missed="0", identifier="person"
    ).status_code == 303

    assert _misses(edit_env, event_id) == []
    with edit_env.Session() as session:
        event = session.get(Event, event_id)
        assert event.missed_identity is False
        assert event.missed_at is None


def test_clearing_without_an_identifier_still_removes_every_miss(edit_env):
    """The rail's "Clear missed marks" button posts no identifier and is
    unchanged by this work, so the old shape has to keep meaning what it
    always meant."""
    event_id = _event_of_class(edit_env, "person", track_id=24)
    for key in ("face", "person"):
        _mark_missed(edit_env, event_id, missed="1", identifier=key)

    assert _mark_missed(edit_env, event_id, missed="0").status_code == 303

    assert _misses(edit_env, event_id) == []
    with edit_env.Session() as session:
        assert session.get(Event, event_id).missed_identity is False


def test_an_auto_added_class_still_gets_an_identifier_to_pick(edit_env):
    """Adding a class to `detection.classes` is meant to be the only step
    (the registry auto-adds an identifier keyed by the class name), so a
    class nobody configured must not land the operator on a page whose
    every select is empty."""
    event_id = _event_of_class(edit_env, "deer", track_id=25)

    page = edit_env.client.get(f"/events/{event_id}")

    assert page.status_code == 200
    assert '<option value="deer">' in page.text
    # ... and it is not offered the identifiers that do not consume deer.
    assert '<option value="vehicle">' not in page.text
    assert '<option value="face">' not in page.text


def test_a_blank_identifier_is_unset_not_a_face_miss(edit_env):
    """An empty submission is a form that said nothing, not a form that
    said "face".

    Collapsing the two costs the distinction the retract path depends on
    — there, absent means "clear every miss" and explicit means "clear
    this one" — and on a person event, where face *is* compatible, the
    collapse is silent: a blank field files a face miss and the operator
    is never told they attributed anything.
    """
    event_id = _event_of_class(edit_env, "person", track_id=26)

    r = _mark_missed(edit_env, event_id, missed="1", identifier="")

    assert r.status_code == 400
    assert _misses(edit_env, event_id) == []


def test_an_unknown_identifier_is_refused_as_unknown(edit_env):
    """Two different mistakes deserve two different answers: an
    identifier that does not exist is a typo, and one that exists but
    does not consume this class is a misunderstanding about the event.
    Reporting both as the latter sends an operator looking for the wrong
    problem — and the attach POST next door already tells them apart.
    """
    unknown = _mark_missed(
        edit_env, edit_env.ids.event, missed="1", identifier="nonsense"
    )
    assert unknown.status_code == 400
    assert "unknown" in unknown.json()["detail"].lower()

    incompatible = _mark_missed(
        edit_env, edit_env.ids.event, missed="1", identifier="face"
    )
    assert incompatible.status_code == 400
    # A real identifier aimed at the wrong class is not an unknown one.
    assert "unknown" not in incompatible.json()["detail"].lower()
    assert _misses(edit_env, edit_env.ids.event) == []

    # The trap in telling them apart: an auto-added class is a legitimate
    # identifier key that is *not* in `identity.identifiers`, so an
    # unknown check written against that mapping alone would refuse the
    # only key such an event can ever have.
    deer = _event_of_class(edit_env, "deer", track_id=27)
    assert _mark_missed(
        edit_env, deer, missed="1", identifier="deer"
    ).status_code == 303
    assert [m.identifier_key for m in _misses(edit_env, deer)] == ["deer"]


def test_a_consumed_class_is_not_an_identifier(edit_env):
    """`vehicle` owns the car class, so a bare "car" key names a pipeline
    that can never exist — the registry only mints a class-keyed
    identifier when nothing configured consumes the class. Accepting it
    minted identities and filed recall stats under a phantom, which is
    the exact corruption this issue set out to stop (CLD-135 review).
    """
    r = _mark_missed(edit_env, edit_env.ids.event, missed="1", identifier="car")
    assert r.status_code == 400
    assert "unknown" in r.json()["detail"].lower()
    assert _misses(edit_env, edit_env.ids.event) == []

    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity",
        data={"identity_id": "new", "identifier": "car"},
        follow_redirects=False,
    )
    assert r.status_code == 400
    with edit_env.Session() as session:
        assert (
            session.query(Identity).filter_by(identifier_key="car").count() == 0
        )


def test_minting_without_naming_an_identifier_is_refused(edit_env):
    """An omitted identifier on an identity_id=new attach must not fall
    back to "face": FastAPI substitutes a Form default before the
    handler sees the omission, so the guard only works if there is no
    default to substitute. Reassign funnels through the same
    _resolve_target, so one pin covers both endpoints' mint path.
    """
    with edit_env.Session() as session:
        before = session.query(Identity).count()

    r = edit_env.client.post(
        f"/events/{edit_env.ids.event}/identity",
        data={"identity_id": "new"},
        follow_redirects=False,
    )

    assert r.status_code == 400
    assert "no identifier" in r.json()["detail"].lower()
    with edit_env.Session() as session:
        assert session.query(Identity).count() == before
