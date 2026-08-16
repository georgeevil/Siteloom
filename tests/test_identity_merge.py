"""Identity merge (web console) — vectors, links and stats move together.

The merge endpoint must reuse the process-wide vector-store client:
embedded Qdrant takes an exclusive flock on the storage directory, and
a second client on the same path raises RuntimeError even when opened
by the same process. These tests pin both the shared-client behavior
and the actionable error surfaced when ANOTHER process holds the lock
(the production 500 that prompted this work: a `backfill-unifi` job
was running while an operator merged two vehicles).
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

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
    get_session,
    init_db,
    make_engine,
)
from siteloom.web.app import create_app


def _unit(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def merge_env(tmp_path):
    """Two vehicle identities with vectors, an event link and a library
    annotation on the source — everything merge is supposed to move."""
    config = SiteConfig(
        site_id="test-site",
        site_name="Test Site",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/merge.db", media_dir=str(tmp_path / "media")
        ),
        identity=IdentityConfig(vector_db_path=str(tmp_path / "vectors")),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)

    source_vecs = [_unit(1.0, 0.0, 0.0, 0.0), _unit(0.0, 1.0, 0.0, 0.0)]
    target_vecs = [_unit(0.0, 0.0, 1.0, 0.0)]
    store = VectorStore(config.identity.vector_db_path)
    try:
        for v in source_vecs:
            store.add("vehicle", v, identity_id=1)
        for v in target_vecs:
            store.add("vehicle", v, identity_id=2)
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
            plate="ABC-123",
            first_seen=datetime(2026, 8, 6, 14, 16, 0),
            last_seen=datetime(2026, 8, 6, 19, 32, 0),
            appearance_count=16,
            vector_count=2,
            best_crop_path="crops/source.jpg",
        )
        target = Identity(
            identifier_key="vehicle",
            class_name="car",
            label="George Tacoma",
            first_seen=datetime(2026, 8, 5, 23, 38, 0),
            last_seen=datetime(2026, 8, 6, 18, 0, 0),
            appearance_count=35,
            vector_count=1,
        )
        session.add_all([source, target])
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
        session.add(
            Annotation(
                item_id=item.id,
                bbox="[0.1, 0.1, 0.5, 0.5]",
                class_name="car",
                identity_id=source.id,
                created_at=datetime(2026, 8, 1),
            )
        )
        session.commit()
        source_id, target_id = source.id, target.id

    client = TestClient(create_app(config))
    return SimpleNamespace(
        client=client,
        Session=Session,
        config=config,
        source_id=source_id,
        target_id=target_id,
        source_vecs=source_vecs,
    )


def test_embedded_qdrant_rejects_a_second_client_on_the_same_path(tmp_path):
    """The platform constraint the shared store exists for: one client
    per path, even within a single process."""
    path = tmp_path / "vectors"
    holder = VectorStore(path)
    try:
        with pytest.raises(RuntimeError, match="another instance"):
            VectorStore(path)
    finally:
        holder.close()


def test_merge_moves_vectors_links_and_stats(merge_env):
    """`appearance_count` is summed, deliberately (CLD-133 spec §0(c)).

    The column counts *identified frames*, not visits: the resolver
    increments it once per resolved frame, and the invariant that keeps
    it honest is `appearance_count ≈ Σ hit_count` over the identity's
    links. Because the collision fold below *sums* hit_count, adding the
    two counters keeps that invariant exact. Recomputing it from the
    surviving rows would report the number of links where the system
    means the number of frames, and would diverge from every other
    writer of the column.
    """
    r = merge_env.client.post(
        f"/identities/{merge_env.source_id}/merge",
        data={"target_id": merge_env.target_id},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == f"/identities/{merge_env.target_id}"

    with merge_env.Session() as session:
        assert session.get(Identity, merge_env.source_id) is None
        target = session.get(Identity, merge_env.target_id)
        assert target.appearance_count == 35 + 16
        assert target.vector_count == 1 + 2  # the moved vectors are counted
        assert target.first_seen == datetime(2026, 8, 5, 23, 38, 0)
        assert target.last_seen == datetime(2026, 8, 6, 19, 32, 0)
        assert target.label == "George Tacoma"  # kept, not overwritten
        assert target.plate == "ABC-123"  # learned from the source
        assert target.best_crop_path == "crops/source.jpg"
        link = session.get(EventIdentity, 1)
        assert link.identity_id == merge_env.target_id
        annotation = session.get(Annotation, 1)
        assert annotation.identity_id == merge_env.target_id

    # Every source embedding now carries the target's id in its payload.
    vectors = get_shared_store(merge_env.config.identity.vector_db_path)
    for vec in merge_env.source_vecs:
        hit = vectors.best_match("vehicle", vec)
        assert hit is not None
        assert hit.identity_id == merge_env.target_id
        assert hit.score > 0.99


def test_merge_succeeds_when_the_shared_store_is_already_open(merge_env):
    """Regression: the recognition API and face enrollment open the
    store lazily and keep it open for the process lifetime. Merge must
    reuse that client instead of opening a second one — opening a
    second client raised RuntimeError and turned the request into a
    bare 500."""
    get_shared_store(merge_env.config.identity.vector_db_path)

    r = merge_env.client.post(
        f"/identities/{merge_env.source_id}/merge",
        data={"target_id": merge_env.target_id},
        follow_redirects=False,
    )
    assert r.status_code == 303
    with merge_env.Session() as session:
        assert session.get(Identity, merge_env.source_id) is None


def test_merge_says_whats_wrong_when_another_process_holds_the_store(merge_env):
    """A concurrent backfill/index/frigate job holds the flock; merge
    must refuse with an actionable 503 rather than a bare 500 — and it
    must not merge halfway (vectors left pointing at a deleted row)."""
    holder = VectorStore(merge_env.config.identity.vector_db_path)
    try:
        r = merge_env.client.post(
            f"/identities/{merge_env.source_id}/merge",
            data={"target_id": merge_env.target_id},
            follow_redirects=False,
        )
    finally:
        holder.close()
    assert r.status_code == 503
    assert "locked by another process" in r.json()["detail"]

    with merge_env.Session() as session:
        assert session.get(Identity, merge_env.source_id) is not None
        assert session.get(Identity, merge_env.target_id).vector_count == 1


def test_merge_guards(merge_env):
    r = merge_env.client.post(
        f"/identities/{merge_env.source_id}/merge",
        data={"target_id": merge_env.source_id},
    )
    assert r.status_code == 400

    with merge_env.Session() as session:
        session.add(
            Identity(
                identifier_key="face",
                class_name="face",
                first_seen=datetime(2026, 8, 6),
                last_seen=datetime(2026, 8, 6),
            )
        )
        session.commit()
    r = merge_env.client.post(
        f"/identities/{merge_env.source_id}/merge", data={"target_id": 3}
    )
    assert r.status_code == 400

    assert (
        merge_env.client.post(
            "/identities/999/merge", data={"target_id": merge_env.target_id}
        ).status_code
        == 404
    )


# -- colliding claims (CLD-133) -------------------------------------------
#
# Merge used to re-point the source's links with a blind bulk UPDATE. When
# an event already carried a claim on the target, that stacked a second
# active row for one (event, identity) pair instead of collapsing them —
# event 30 rendering four claims for two identities is what filed the
# issue. The links now go through the same fold ingest uses.


def _event(session, track_id: int) -> int:
    event = Event(
        camera_id="cam1",
        track_id=track_id,
        class_name="car",
        first_seen=datetime(2026, 8, 6, 15, 0, 0),
        last_seen=datetime(2026, 8, 6, 15, 0, 30),
        detection_count=1,
    )
    session.add(event)
    session.flush()
    return event.id


def _claims_on(session, event_id: int, identity_id: int):
    return session.scalars(
        select(EventIdentity)
        .where(
            EventIdentity.event_id == event_id,
            EventIdentity.identity_id == identity_id,
            EventIdentity.unlinked_at.is_(None),
        )
        .order_by(EventIdentity.id)
    ).all()


def _merge(env):
    r = env.client.post(
        f"/identities/{env.source_id}/merge",
        data={"target_id": env.target_id},
        follow_redirects=False,
    )
    assert r.status_code == 303
    return r


def test_merge_folds_a_claim_the_event_already_carries_on_the_target(merge_env):
    """The headline acceptance criterion: two identities claiming one
    event leave one claim behind, carrying the evidence of both.

    Stacking them instead showed the operator the same identity twice on
    one event, double-counted it in every accuracy query that reads these
    rows, and made "how many claims does this event have" unanswerable.
    """
    with merge_env.Session() as session:
        source_link = session.get(EventIdentity, 1)
        source_link.hit_count = 16
        source_link.similarity = 0.9
        source_link.matched_by = "plate"
        source_link.learned_plate = True
        source_link.verdict = "confirmed"
        source_link.verdict_at = datetime(2026, 8, 6, 20, 0, 0)
        session.add(
            EventIdentity(
                event_id=1,
                identity_id=merge_env.target_id,
                identifier_key="vehicle",
                similarity=0.5,
                hit_count=35,
                matched_by="visual",
                verdict="wrong",
                verdict_at=datetime(2026, 8, 6, 18, 0, 0),
            )
        )
        session.commit()

    _merge(merge_env)

    with merge_env.Session() as session:
        claims = _claims_on(session, 1, merge_env.target_id)
        assert len(claims) == 1
        kept = claims[0]
        assert kept.hit_count == 16 + 35  # evidence adds up
        assert kept.similarity == 0.9  # the best frame either row saw
        assert kept.matched_by == "plate"  # a plate read outranks visual
        assert kept.learned_plate is True
        # A human verdict is never erased, and an identification outranks
        # a repudiation of the same pairing.
        assert kept.verdict == "confirmed"
        assert kept.verdict_at == datetime(2026, 8, 6, 20, 0, 0)
        assert session.get(EventIdentity, 1) is None  # folded in, not stacked
        # ... and the counter stays in step with Σ hit_count (§0(c)).
        assert session.get(Identity, merge_env.target_id).appearance_count == 35 + 16


def test_merge_carries_repudiated_claims_across_without_folding_them(merge_env):
    """An unlinked row is the record of a claim that was wrong, and there
    may be several for one pairing (CLD-36) — so they follow the merge
    untouched rather than collapsing into the live claim.

    They must move, not be left behind: deleting the source identity
    would null their identity_id, turning "this was claimed and it was
    wrong" into "nothing was claimed", which is a different fact.
    """
    with merge_env.Session() as session:
        session.add_all(
            [
                EventIdentity(
                    event_id=1,
                    identity_id=merge_env.target_id,
                    identifier_key="vehicle",
                    similarity=0.6,
                    hit_count=3,
                ),
                EventIdentity(
                    event_id=1,
                    identity_id=merge_env.source_id,
                    identifier_key="vehicle",
                    similarity=0.71,
                    hit_count=2,
                    matched_by="visual",
                    verdict="wrong",
                    verdict_at=datetime(2026, 8, 6, 17, 0, 0),
                    unlinked_at=datetime(2026, 8, 6, 17, 0, 0),
                ),
            ]
        )
        session.commit()

    _merge(merge_env)

    with merge_env.Session() as session:
        repudiated = session.scalars(
            select(EventIdentity).where(EventIdentity.unlinked_at.is_not(None))
        ).all()
        assert len(repudiated) == 1
        assert repudiated[0].identity_id == merge_env.target_id  # moved, not nulled
        assert repudiated[0].verdict == "wrong"  # and still a negative
        assert repudiated[0].similarity == 0.71  # untouched by the fold
        assert len(_claims_on(session, 1, merge_env.target_id)) == 1


def test_merge_folds_every_colliding_event_and_moves_the_rest(merge_env):
    """Three events, three shapes: two where both identities claimed the
    event, one where only the source did. The loop must fold the first
    two and move the third, leaving one standing claim per event."""
    with merge_env.Session() as session:
        session.add(
            EventIdentity(
                event_id=1,
                identity_id=merge_env.target_id,
                identifier_key="vehicle",
                similarity=0.4,
                hit_count=2,
            )
        )
        colliding = _event(session, track_id=8)
        source_only = _event(session, track_id=9)
        session.add_all(
            [
                EventIdentity(
                    event_id=colliding,
                    identity_id=merge_env.source_id,
                    identifier_key="vehicle",
                    similarity=0.8,
                    hit_count=5,
                ),
                EventIdentity(
                    event_id=colliding,
                    identity_id=merge_env.target_id,
                    identifier_key="vehicle",
                    similarity=0.6,
                    hit_count=7,
                ),
                EventIdentity(
                    event_id=source_only,
                    identity_id=merge_env.source_id,
                    identifier_key="vehicle",
                    similarity=0.75,
                    hit_count=4,
                ),
            ]
        )
        session.commit()

    _merge(merge_env)  # a 500 here is the IntegrityError the index raises

    with merge_env.Session() as session:
        claims = session.scalars(
            select(EventIdentity)
            .where(EventIdentity.unlinked_at.is_(None))
            .order_by(EventIdentity.event_id)
        ).all()
        assert len(claims) == 3
        assert {c.identity_id for c in claims} == {merge_env.target_id}
        by_event = {c.event_id: c for c in claims}
        assert by_event[1].hit_count == 1 + 2
        assert by_event[colliding].hit_count == 5 + 7
        assert by_event[colliding].similarity == 0.8
        assert by_event[source_only].hit_count == 4  # moved, not folded