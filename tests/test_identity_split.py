"""Identity split (web console) — vectors move with the annotations.

Split exists to fix a polluted gallery, so the split endpoint must do
more than re-point Annotation rows: it enrolls the fresh identity from
the verified crops it takes (a label without vectors is a name the
system cannot see), removes exactly those crops' vectors from the
source, and refuses with an actionable 503 when another process holds
the embedded vector store — the same shared-client rules merge follows.

The sharpest constraint is what split must NOT do. An identity's
gallery is the union of two disjoint populations: vectors added by live
camera matching (identity/resolver.py adds them directly and never
creates an Annotation) and vectors enrolled from verified library
annotations. Only the second is reconstructible, so rebuilding a
gallery from annotations would silently delete everything the cameras
learned. `test_split_keeps_live_matched_vectors_it_cannot_rebuild`
pins that; it is the reason the source's gallery is edited surgically
rather than dropped and rebuilt.

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
    Annotation,
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
from siteloom.web.app import create_app
from siteloom.web.auth import hash_password, required_role


def _unit(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / np.linalg.norm(v)


#: Crop colour (BGR) -> the embedding the stub produces for it. The
#: first two are the vehicle the source identity is really about; the
#: red one is the polluted sample the operator splits off.
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
def split_env(tmp_path, monkeypatch):
    """One vehicle identity whose gallery absorbed a second vehicle:
    two blue/green annotations that belong, one red one that doesn't,
    all with stored crops and enrolled vectors — plus an event link
    and the source's best crop pointing at the polluted sample."""
    monkeypatch.setattr(
        "siteloom.identity.embedders.build_embedder",
        lambda algo, device="mps", projection_path=None: StubEmbedder(),
    )
    config = SiteConfig(
        site_id="test-site",
        site_name="Test Site",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/split.db", media_dir=str(tmp_path / "media")
        ),
        identity=IdentityConfig(vector_db_path=str(tmp_path / "vectors")),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)

    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    crop_paths = {}
    for name, (colour, _) in COLOURS.items():
        square = np.full((16, 16, 3), colour, dtype=np.uint8)
        path = crops_dir / f"{name}.jpg"
        cv2.imwrite(str(path), square)
        crop_paths[name] = str(path)

    store = VectorStore(config.identity.vector_db_path)
    try:
        for name in COLOURS:
            store.add("vehicle", COLOURS[name][1], identity_id=1)
    finally:
        store.close()

    with Session() as session:
        session.add(Camera(id="cam1", site_id="test-site", name="Cam One"))
        event = Event(
            camera_id="cam1",
            track_id=7,
            class_name="car",
            first_seen=datetime(2026, 8, 6, 14, 16, 0),
            last_seen=datetime(2026, 8, 6, 14, 16, 30),
            detection_count=1,
        )
        session.add(event)
        session.flush()
        source = Identity(
            identifier_key="vehicle",
            class_name="car",
            label="Aleks Corolla",
            first_seen=datetime(2026, 8, 6, 14, 16, 0),
            last_seen=datetime(2026, 8, 6, 19, 32, 0),
            appearance_count=10,
            vector_count=3,
            best_crop_path=crop_paths["red"],  # the polluted sample won
        )
        session.add(source)
        session.flush()
        session.add(
            EventIdentity(
                event_id=event.id,
                identity_id=source.id,
                identifier_key="vehicle",
                similarity=0.9,
            )
        )
        lib_source = LibrarySource(
            path=str(tmp_path / "library"), added_at=datetime(2026, 8, 1)
        )
        session.add(lib_source)
        session.flush()
        item = LibraryItem(
            source_id=lib_source.id,
            path=str(tmp_path / "library" / "photo.jpg"),
            kind="image",
            mtime=datetime(2026, 8, 1),
        )
        session.add(item)
        session.flush()
        annotation_ids = {}
        for name in ("blue", "green", "red"):
            annotation = Annotation(
                item_id=item.id,
                bbox="[0.1, 0.1, 0.5, 0.5]",
                class_name="car",
                identity_id=source.id,
                crop_path=crop_paths[name],
                # Verified and enrolled: each contributed the vector
                # seeded above, which is what makes them removable.
                verified=True,
                enrolled=True,
                created_at=datetime(2026, 8, 1),
            )
            session.add(annotation)
            session.flush()
            annotation_ids[name] = annotation.id
        session.commit()
        source_id = source.id

    client = TestClient(create_app(config))
    return SimpleNamespace(
        client=client,
        Session=Session,
        config=config,
        source_id=source_id,
        annotation_ids=annotation_ids,
        crop_paths=crop_paths,
    )


def _vector_owner(config, vector) -> tuple[int, float]:
    store = get_shared_store(config.identity.vector_db_path)
    hit = store.best_match("vehicle", vector)
    assert hit is not None
    return hit.identity_id, hit.score


def test_split_moves_annotations_and_their_vectors(split_env):
    red_id = split_env.annotation_ids["red"]
    r = split_env.client.post(
        f"/identities/{split_env.source_id}/split",
        data={"annotation_ids": str(red_id)},
        follow_redirects=False,
    )
    assert r.status_code == 303
    location = r.headers["location"]
    new_id = int(location.split("/identities/")[1].split("?")[0])
    assert location == f"/identities/{new_id}?split_from={split_env.source_id}"

    with split_env.Session() as session:
        assert session.get(Annotation, red_id).identity_id == new_id
        for name in ("blue", "green"):
            annotation = session.get(Annotation, split_env.annotation_ids[name])
            assert annotation.identity_id == split_env.source_id
        fresh = session.get(Identity, new_id)
        assert fresh.identifier_key == "vehicle"
        assert fresh.label is None  # starts in the unknown bucket
        assert fresh.vector_count == 1  # enrolled, not empty
        assert fresh.appearance_count == 1
        # The source's best crop was the split-off sample, so it moves
        # and the source falls back to a crop it still owns.
        assert fresh.best_crop_path == split_env.crop_paths["red"]
        source = session.get(Identity, split_env.source_id)
        assert source.vector_count == 2  # lost exactly the moved crop's
        assert source.appearance_count == 10 - 1
        assert source.best_crop_path in (
            split_env.crop_paths["blue"],
            split_env.crop_paths["green"],
        )
        # Historical event claims stay with the source — annotations
        # carry no event linkage to reattribute them from.
        link = session.get(EventIdentity, 1)
        assert link.identity_id == split_env.source_id

    # The red embedding now answers as the new identity; blue and green
    # still answer as the source.
    owner, score = _vector_owner(split_env.config, COLOURS["red"][1])
    assert owner == new_id and score > 0.99
    for name in ("blue", "green"):
        owner, score = _vector_owner(split_env.config, COLOURS[name][1])
        assert owner == split_env.source_id and score > 0.99


def test_split_keeps_live_matched_vectors_it_cannot_rebuild(split_env):
    """The gallery is two disjoint populations and only one of them is
    reconstructible: live camera matching adds vectors directly with no
    Annotation row (identity/resolver.py), so rebuilding a gallery from
    annotations would delete everything the cameras learned. Split must
    remove the moved crops' vectors and nothing else.
    """
    store = get_shared_store(split_env.config.identity.vector_db_path)
    # Vectors the cameras contributed: no annotation covers them.
    live = [_unit(1.0, 1.0, 0.0, 0.0), _unit(0.0, 1.0, 1.0, 0.0)]
    for vector in live:
        store.add("vehicle", vector, identity_id=split_env.source_id)
    with split_env.Session() as session:
        session.get(Identity, split_env.source_id).vector_count = 5
        session.commit()

    r = split_env.client.post(
        f"/identities/{split_env.source_id}/split",
        data={"annotation_ids": str(split_env.annotation_ids["red"])},
        follow_redirects=False,
    )
    assert r.status_code == 303

    with split_env.Session() as session:
        source = session.get(Identity, split_env.source_id)
        # 5 seeded - 1 moved: the two live vectors and the two annotation
        # vectors that stayed are all still there.
        assert source.vector_count == 4
    for vector in live:
        owner, score = _vector_owner(split_env.config, vector)
        assert owner == split_env.source_id and score > 0.99


def test_split_enrolls_only_verified_crops_and_leaves_the_rest_sweepable(split_env):
    """Only verified annotations are ground truth, so an unreviewed
    auto-assignment must not enter the live gallery — and must not be
    flagged `enrolled` either, or the confirmation a human gives it
    later never reaches the vector store (`enroll_verified` sweeps on
    enrolled=False), which is a label the system cannot see.
    """
    with split_env.Session() as session:
        item_id = session.get(
            Annotation, split_env.annotation_ids["red"]
        ).item_id
        guess = Annotation(
            item_id=item_id,
            bbox="[0.2, 0.2, 0.6, 0.6]",
            class_name="car",
            identity_id=split_env.source_id,
            crop_path=split_env.crop_paths["green"],
            source="auto",
            verified=False,
            created_at=datetime(2026, 8, 2),
        )
        session.add(guess)
        session.commit()
        guess_id = guess.id

    r = split_env.client.post(
        f"/identities/{split_env.source_id}/split",
        data={
            "annotation_ids": f"{split_env.annotation_ids['red']},{guess_id}"
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    new_id = int(r.headers["location"].split("/identities/")[1].split("?")[0])

    with split_env.Session() as session:
        guess = session.get(Annotation, guess_id)
        assert guess.identity_id == new_id  # the row still moves
        assert not guess.enrolled  # ... but contributed no vector
        # Only the verified crop was enrolled.
        assert session.get(Identity, new_id).vector_count == 1


def test_a_crop_skipped_by_split_is_still_enrolled_when_verified_later(split_env):
    """The `enrolled` flag is what retires an annotation from
    `enroll_verified`. Split must set it only on crops whose vector
    actually landed — otherwise a human confirming one of the crops
    split passed over (unverified, or over the identifier's cap) sees
    the sweep skip it forever, and the name never reaches the store.
    """
    with split_env.Session() as session:
        item_id = session.get(
            Annotation, split_env.annotation_ids["red"]
        ).item_id
        face_identity = Identity(
            identifier_key="face",
            class_name="person",
            label="Dana",
            first_seen=datetime(2026, 8, 6),
            last_seen=datetime(2026, 8, 6),
        )
        session.add(face_identity)
        session.flush()
        pending = Annotation(
            item_id=item_id,
            bbox="[0.2, 0.2, 0.6, 0.6]",
            class_name="face",
            identity_id=face_identity.id,
            crop_path=split_env.crop_paths["blue"],
            proposed_name="Dana",
            source="auto",
            verified=False,  # split would pass this over
            created_at=datetime(2026, 8, 2),
        )
        session.add(pending)
        session.commit()
        pending_id, face_id = pending.id, face_identity.id

    # The human confirms it after the split.
    from siteloom.identity.enroll import enroll_verified

    vectors = get_shared_store(split_env.config.identity.vector_db_path)
    with split_env.Session() as session:
        session.get(Annotation, pending_id).verified = True
        session.commit()
        stats = enroll_verified(session, vectors, StubEmbedder())

    assert stats.enrolled == 1
    with split_env.Session() as session:
        assert session.get(Annotation, pending_id).enrolled
        assert session.get(Identity, face_id).vector_count == 1


def test_split_does_not_touch_another_identitys_matching_vectors(split_env):
    """`delete_duplicates_of` filters by identity before comparing
    vectors. Two vehicles photographed in the same colour produce
    near-identical embeddings, and deleting by similarity alone would
    strip a bystander identity's gallery."""
    store = get_shared_store(split_env.config.identity.vector_db_path)
    with split_env.Session() as session:
        bystander = Identity(
            identifier_key="vehicle",
            class_name="car",
            label="Someone Else",
            first_seen=datetime(2026, 8, 6),
            last_seen=datetime(2026, 8, 6),
            vector_count=1,
        )
        session.add(bystander)
        session.commit()
        bystander_id = bystander.id
    # The exact vector split is about to remove from the source.
    store.add("vehicle", COLOURS["red"][1], identity_id=bystander_id)

    r = split_env.client.post(
        f"/identities/{split_env.source_id}/split",
        data={"annotation_ids": str(split_env.annotation_ids["red"])},
        follow_redirects=False,
    )
    assert r.status_code == 303

    assert store.count_identity("vehicle", bystander_id) == 1
    with split_env.Session() as session:
        assert session.get(Identity, bystander_id).vector_count == 1


def test_split_rejects_invalid_and_empty_annotation_ids(split_env):
    # Empty selection.
    r = split_env.client.post(
        f"/identities/{split_env.source_id}/split", data={"annotation_ids": ""}
    )
    assert r.status_code == 400
    assert "select at least one" in r.json()["detail"]

    # A malformed token must 400 loudly, not silently split a subset.
    red_id = split_env.annotation_ids["red"]
    r = split_env.client.post(
        f"/identities/{split_env.source_id}/split",
        data={"annotation_ids": f"{red_id},abc"},
    )
    assert r.status_code == 400
    assert "abc" in r.json()["detail"]

    # Unknown annotation ids are named, not skipped.
    r = split_env.client.post(
        f"/identities/{split_env.source_id}/split",
        data={"annotation_ids": f"{red_id},999"},
    )
    assert r.status_code == 400
    assert "999" in r.json()["detail"]

    # Annotations belonging to a different identity cannot be pulled
    # through an endpoint that only names the source in its URL.
    with split_env.Session() as session:
        other = Identity(
            identifier_key="vehicle",
            class_name="car",
            first_seen=datetime(2026, 8, 6),
            last_seen=datetime(2026, 8, 6),
        )
        session.add(other)
        session.flush()
        foreign = Annotation(
            item_id=1,
            bbox="[0.1, 0.1, 0.5, 0.5]",
            class_name="car",
            identity_id=other.id,
            created_at=datetime(2026, 8, 1),
        )
        session.add(foreign)
        session.commit()
        foreign_id = foreign.id
    r = split_env.client.post(
        f"/identities/{split_env.source_id}/split",
        data={"annotation_ids": str(foreign_id)},
    )
    assert r.status_code == 400
    assert str(foreign_id) in r.json()["detail"]

    # None of the rejections changed anything.
    with split_env.Session() as session:
        assert session.get(Annotation, red_id).identity_id == split_env.source_id
        assert session.get(Identity, split_env.source_id).vector_count == 3

    assert split_env.client.post(
        "/identities/999/split", data={"annotation_ids": str(red_id)}
    ).status_code == 404


def test_split_succeeds_when_the_shared_store_is_already_open(split_env):
    """The recognition API and face enrollment keep the shared client
    open for the process lifetime; split must reuse it rather than
    opening a second one (which raises RuntimeError -> a bare 500)."""
    get_shared_store(split_env.config.identity.vector_db_path)

    r = split_env.client.post(
        f"/identities/{split_env.source_id}/split",
        data={"annotation_ids": str(split_env.annotation_ids["red"])},
        follow_redirects=False,
    )
    assert r.status_code == 303
    with split_env.Session() as session:
        assert session.get(Identity, split_env.source_id).vector_count == 2


def test_split_says_whats_wrong_when_another_process_holds_the_store(split_env):
    """A concurrent backfill/index/frigate job holds the flock; split
    must refuse with an actionable 503 — and must not move the rows,
    because rows without their vectors is the very bug split had."""
    holder = VectorStore(split_env.config.identity.vector_db_path)
    try:
        r = split_env.client.post(
            f"/identities/{split_env.source_id}/split",
            data={"annotation_ids": str(split_env.annotation_ids["red"])},
        )
    finally:
        holder.close()
    assert r.status_code == 503
    assert "locked by another process" in r.json()["detail"]

    with split_env.Session() as session:
        red = session.get(Annotation, split_env.annotation_ids["red"])
        assert red.identity_id == split_env.source_id
        source = session.get(Identity, split_env.source_id)
        assert source.vector_count == 3
        # No half-made identity left behind.
        others = session.scalars(
            select(Identity).filter(Identity.id != split_env.source_id)
        ).all()
        assert others == []


def test_split_requires_edit_role(split_env):
    """Enforcement lives in the one auth middleware, not on the route:
    a POST outside the admin prefixes needs the edit rung."""
    assert required_role("POST", f"/identities/{split_env.source_id}/split") == "edit"

    with split_env.Session() as session:
        session.add(
            User(
                username="vera",
                password_hash=hash_password("hunter2!"),
                role="view",
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        session.commit()
    r = split_env.client.post(
        "/login", data={"username": "vera", "password": "hunter2!", "next": "/"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    r = split_env.client.post(
        f"/identities/{split_env.source_id}/split",
        data={"annotation_ids": str(split_env.annotation_ids["red"])},
    )
    assert r.status_code == 403
    with split_env.Session() as session:
        red = session.get(Annotation, split_env.annotation_ids["red"])
        assert red.identity_id == split_env.source_id
