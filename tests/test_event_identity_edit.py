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


def test_the_rail_offers_the_correction_actions(edit_env):
    """The endpoints are only half the feature — CLD-36 was filed because
    the rail had verdict buttons and nothing else."""
    event, link = edit_env.ids.event, edit_env.ids.link
    rail = edit_env.client.get(f"/events/{event}/rail").text
    assert f'action="/events/{event}/identity"' in rail  # attach
    assert f'action="/events/{event}/identity/{link}/unlink"' in rail
    assert f'action="/events/{event}/identity/{link}/reassign"' in rail
    assert "Bo Truck" in rail  # the picker lists candidates
    # Reassigning to the identity already linked is not a correction, so
    # the picker for this claim leaves it out.
    reassign_form = rail.split(f"/identity/{link}/reassign")[1].split("</form>")[0]
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


def test_the_picker_offers_only_compatible_identities(edit_env):
    """The offer side of the same rule: a person identity is not listed
    on a car event's rail, and neither is the face identifier — the form
    must not offer what the POST refuses."""
    _person_identity(edit_env, "person", "Dana")
    rail = edit_env.client.get(f"/events/{edit_env.ids.event}/rail").text
    assert "Bo Truck" in rail
    assert "Dana" not in rail
    assert '<option value="vehicle">' in rail
    assert '<option value="face">' not in rail


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
