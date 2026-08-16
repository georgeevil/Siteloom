"""Vector payload provenance (CLD-84).

An identity's gallery is several disjoint populations — live camera
matches, crops enrolled from verified annotations, operator corrections,
API enrollments — and before provenance was recorded they were
indistinguishable once written. That is what forced identity split to
reason numerically about vectors (`delete_duplicates_of`) instead of by
origin, and what left live-matched vectors unreachable by any surgical
edit.

These tests pin the write side (every writer stamps what it knows) and
the read side (the by-origin queries find exactly their own points and
nothing else, including nothing belonging to a bystander identity).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from siteloom.config import IdentifierConfig, IdentityConfig
from siteloom.identity import IdentityResolver, VectorStore
from siteloom.identity.vectors import (
    SOURCE_API,
    SOURCE_ENROLLED,
    SOURCE_LIVE,
    SOURCE_MANUAL,
)
from siteloom.store import Annotation, Identity, get_session, init_db, make_engine


def _unit(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "vectors")
    try:
        yield s
    finally:
        s.close()


def _payloads(store: VectorStore, collection: str, identity_id: int) -> list[dict]:
    return [dict(p.payload or {}) for p in store._scroll_identity(collection, identity_id)]


def test_add_records_who_wrote_it_and_what_from(store):
    store.add("face", _unit(1, 0, 0), 1)
    store.add(
        "face",
        _unit(0, 1, 0),
        1,
        source=SOURCE_ENROLLED,
        annotation_id=42,
        crop_path="/media/crops/a.jpg",
    )
    by_source = {p["source"]: p for p in _payloads(store, "face", 1)}
    # The default is the live path — the population with no on-disk
    # record, so it must be marked without every caller remembering to.
    assert by_source[SOURCE_LIVE] == {"identity_id": 1, "source": SOURCE_LIVE}
    assert by_source[SOURCE_ENROLLED] == {
        "identity_id": 1,
        "source": SOURCE_ENROLLED,
        "annotation_id": 42,
        "crop_path": "/media/crops/a.jpg",
    }


def test_delete_by_annotations_is_exact_and_leaves_the_rest(store):
    store.add("face", _unit(1, 0, 0), 1, source=SOURCE_ENROLLED, annotation_id=7)
    store.add("face", _unit(0, 1, 0), 1, source=SOURCE_ENROLLED, annotation_id=8)
    store.add("face", _unit(0, 0, 1), 1)  # live: no annotation to name it

    assert store.delete_by_annotations("face", 1, [7]) == 1
    remaining = _payloads(store, "face", 1)
    assert {p.get("annotation_id") for p in remaining} == {None, 8}
    # An annotation that contributed nothing to this identity deletes
    # nothing — the count is the caller's evidence, not an estimate.
    assert store.delete_by_annotations("face", 1, [999]) == 0
    assert store.delete_by_annotations("face", 1, []) == 0


def test_by_origin_queries_never_reach_another_identitys_gallery(store):
    """Two people photographed in the same session can share an
    annotation id space and a crop path (the same file can be enrolled
    twice). The identity filter is the safety property; the payload
    predicate only narrows within one gallery."""
    store.add("face", _unit(1, 0, 0), 1, source=SOURCE_ENROLLED, annotation_id=7,
              crop_path="/crops/x.jpg")
    store.add("face", _unit(1, 0, 0), 2, source=SOURCE_ENROLLED, annotation_id=7,
              crop_path="/crops/x.jpg")

    assert store.delete_by_annotations("face", 1, [7]) == 1
    assert store.count_identity("face", 2) == 1
    assert store.delete_by_crops("face", 2, ["/crops/x.jpg"]) == 1
    assert store.count_identity("face", 2) == 0


def test_move_by_crops_repoints_a_reassigned_events_vectors(store):
    """A reassignment is not a deletion: the crop really is the other
    identity, so its vector moves and starts matching for the right one."""
    store.add("vehicle", _unit(1, 0, 0), 1, crop_path="/crops/visit.jpg")
    store.add("vehicle", _unit(0, 1, 0), 1, crop_path="/crops/other.jpg")

    assert store.move_by_crops("vehicle", 1, 2, ["/crops/visit.jpg"]) == 1
    assert store.count_identity("vehicle", 1) == 1
    hit = store.best_match("vehicle", _unit(1, 0, 0))
    assert hit is not None and hit.identity_id == 2
    # Provenance survives the move — the vector is still traceable to the
    # crop it came from, so a second correction can undo the first.
    moved = _payloads(store, "vehicle", 2)
    assert moved == [
        {"identity_id": 2, "source": SOURCE_LIVE, "crop_path": "/crops/visit.jpg"}
    ]
    assert store.move_by_crops("vehicle", 1, 2, ["/crops/missing.jpg"]) == 0


def test_points_written_before_provenance_are_untouched_by_origin_queries(store):
    """Existing installations have provenance-free points. They must not
    be swept up by a by-origin delete — silently deleting whatever has no
    marker would destroy exactly the live vectors this feature exists to
    protect. They stay reachable through the numeric pass."""
    store.ensure_collection("face", dim=3)
    from qdrant_client.models import PointStruct

    legacy = _unit(1, 0, 0)
    store._client.upsert(
        collection_name="face",
        points=[
            PointStruct(
                id="00000000-0000-4000-8000-000000000001",
                vector=legacy.tolist(),
                payload={"identity_id": 1},
            )
        ],
    )
    assert store.delete_by_annotations("face", 1, [7]) == 0
    assert store.delete_by_crops("face", 1, ["/crops/x.jpg"]) == 0
    assert store.count_identity("face", 1) == 1
    assert store.delete_duplicates_of("face", 1, [legacy]) == 1


class _Recorder:
    """A vector store stand-in that records provenance kwargs."""

    def __init__(self):
        self.calls: list[dict] = []

    def add(self, collection, vector, identity_id, **kwargs):
        self.calls.append({"collection": collection, "identity_id": identity_id, **kwargs})


def test_the_resolver_stamps_the_crop_a_live_vector_came_from(tmp_path):
    """Live matching creates no Annotation row, so the crop file is the
    only handle a later correction has on the vector it added."""
    engine = make_engine(f"sqlite:///{tmp_path}/prov.db")
    init_db(engine)
    Session = get_session(engine)
    vectors = VectorStore(tmp_path / "vectors")
    try:
        resolver = IdentityResolver(
            IdentityConfig(
                identifiers={
                    "vehicle": IdentifierConfig(
                        applies_to=["car"], algo="generic", threshold=0.8
                    )
                }
            ),
            vectors,
        )
        with Session() as session:
            resolution = resolver.resolve(
                session,
                identifier_key="vehicle",
                class_name="car",
                vector=list(_unit(1.0, 0.0, 0.0)),
                plate=None,
                timestamp=datetime(2026, 8, 7, 9, 0, 0),
                crop_path="/media/crops/e1.jpg",
                camera_id="cam1",
            )
            session.commit()
            identity_id = resolution.identity.id
        assert _payloads(vectors, "vehicle", identity_id) == [
            {
                "identity_id": identity_id,
                "source": SOURCE_LIVE,
                "crop_path": "/media/crops/e1.jpg",
            }
        ]
    finally:
        vectors.close()


def test_a_promoted_pending_sighting_keeps_its_crop(tmp_path):
    """Unknowns wait in the pending pool before minting an identity
    (CLD-41). The promoted vectors are as much this identity's gallery as
    the one that triggered promotion, so they must arrive traceable too —
    otherwise the first sightings of every new person are anonymous."""
    engine = make_engine(f"sqlite:///{tmp_path}/pending.db")
    init_db(engine)
    Session = get_session(engine)
    vectors = VectorStore(tmp_path / "vectors")
    try:
        resolver = IdentityResolver(
            IdentityConfig(
                identifiers={
                    "person": IdentifierConfig(
                        applies_to=["person"],
                        algo="generic",
                        threshold=0.8,
                        min_sightings=2,
                    )
                    # Both frames below sit above learn_min_quality and
                    # below immediate_quality (CLD-139): poor enough to
                    # go through the pending pool, good enough to be
                    # worth storing. What is under test is the crop
                    # provenance of what gets stored, not which frames
                    # the learn gates admit.
                }
            ),
            vectors,
        )
        vector = list(_unit(1.0, 0.0, 0.0))
        with Session() as session:
            first = resolver.resolve(
                session,
                identifier_key="person",
                class_name="person",
                vector=vector,
                plate=None,
                timestamp=datetime(2026, 8, 7, 9, 0, 0),
                crop_path="/media/crops/first.jpg",
                camera_id="cam1",
                quality=0.7,
            )
            assert first.pending and first.identity is None
            second = resolver.resolve(
                session,
                identifier_key="person",
                class_name="person",
                vector=vector,
                plate=None,
                timestamp=datetime(2026, 8, 7, 9, 0, 30),
                crop_path="/media/crops/second.jpg",
                camera_id="cam1",
                quality=0.7,
            )
            session.commit()
            identity_id = second.identity.id
        crops = {p.get("crop_path") for p in _payloads(vectors, "person", identity_id)}
        assert crops == {"/media/crops/first.jpg", "/media/crops/second.jpg"}
    finally:
        vectors.close()


def test_enrollment_stamps_the_annotation_it_came_from(tmp_path):
    from siteloom.identity.enroll import enroll_embedded

    identity = Identity(
        identifier_key="face",
        class_name="person",
        label="Dana",
        first_seen=datetime(2026, 8, 7),
        last_seen=datetime(2026, 8, 7),
    )
    identity.id = 5
    identity.vector_count = 0
    annotation = Annotation(
        item_id=1,
        bbox="[0, 0, 1, 1]",
        class_name="face",
        crop_path="/crops/dana.jpg",
        verified=True,
        created_at=datetime(2026, 8, 7),
    )
    annotation.id = 11
    recorder = _Recorder()
    assert enroll_embedded(recorder, identity, [(annotation, _unit(1, 0, 0))]) == 1
    assert recorder.calls == [
        {
            "collection": "face",
            "identity_id": 5,
            "source": SOURCE_ENROLLED,
            "annotation_id": 11,
            "crop_path": "/crops/dana.jpg",
        }
    ]


def test_source_markers_are_distinct_populations():
    """The four writers must stay distinguishable — the whole point is
    that "which of these did a camera teach us?" is answerable."""
    assert len({SOURCE_LIVE, SOURCE_ENROLLED, SOURCE_MANUAL, SOURCE_API}) == 4
