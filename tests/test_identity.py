"""Identity framework tests — stub embedders, real Qdrant local mode."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from siteloom.config import IdentifierConfig, IdentityConfig
from siteloom.identity.registry import IdentifierRegistry
from siteloom.identity.resolver import IdentityResolver
from siteloom.identity.vectors import VectorStore
from siteloom.store import Identity, get_session, init_db, make_engine

TS = datetime(2026, 8, 5, 12, 0, 0)


def unit(v: list[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    return arr / np.linalg.norm(arr)


@pytest.fixture
def vectors(tmp_path):
    store = VectorStore(tmp_path / "vectors")
    yield store
    store.close()


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/identity.db")
    init_db(engine)
    with get_session(engine)() as s:
        yield s


@pytest.fixture
def resolver(vectors):
    return IdentityResolver(IdentityConfig(), vectors)


def test_vector_store_roundtrip(vectors):
    v = unit([1.0, 0.0, 0.0, 0.0])
    vectors.add("face", v, identity_id=1)
    hit = vectors.best_match("face", v)
    assert hit is not None
    assert hit.identity_id == 1
    assert hit.score > 0.99


def test_search_missing_collection_is_empty(vectors):
    assert vectors.search("nope", unit([1.0, 0.0])) == []


def test_resolver_quarantines_then_promotes(resolver, session):
    """person requires two consistent sightings (CLD-41): the first crop
    is parked, the second mints the identity with both appearances and
    both vectors, the third is a plain match."""
    v = unit([0.3, 0.4, 0.5, 0.1])
    kw = dict(
        identifier_key="person", class_name="person",
        vector=v.tolist(), plate=None, threshold=0.8,
    )
    first = resolver.resolve(session, timestamp=TS, **kw)
    assert first.identity is None
    assert first.pending
    assert session.query(Identity).count() == 0

    second = resolver.resolve(session, timestamp=TS, **kw)
    assert second.is_new
    assert second.identity.label is None  # unknown bucket
    assert second.identity.appearance_count == 2  # parked sighting counts
    assert second.identity.vector_count == 2  # pool vector moved over
    assert second.identity.first_seen == TS

    third = resolver.resolve(session, timestamp=TS, **kw)
    assert not third.is_new
    assert third.identity.id == second.identity.id
    assert third.similarity > 0.99
    assert third.identity.appearance_count == 3


def test_resolver_high_quality_mints_immediately(resolver, session):
    """Detection confidence at or above immediate_quality bypasses the
    sighting gate — a crisp crop is trusted on its own."""
    first = resolver.resolve(
        session, identifier_key="person", class_name="person",
        vector=unit([0.3, 0.4, 0.5, 0.1]).tolist(), plate=None,
        timestamp=TS, threshold=0.8, quality=0.9,
    )
    assert first.is_new
    assert first.identity is not None


def test_resolver_below_threshold_stays_pending(resolver, session):
    """Two one-off unknowns mint nothing — sub-threshold embeddings park
    in the pending pool instead of becoming vector-space magnets."""
    a = resolver.resolve(
        session, identifier_key="person", class_name="person",
        vector=unit([1.0, 0.0, 0.0, 0.0]).tolist(), plate=None,
        timestamp=TS, threshold=0.8,
    )
    b = resolver.resolve(
        session, identifier_key="person", class_name="person",
        vector=unit([0.0, 1.0, 0.0, 0.0]).tolist(), plate=None,
        timestamp=TS, threshold=0.8,
    )
    assert a.pending and b.pending
    assert session.query(Identity).count() == 0


def test_resolver_pending_expires(resolver, session):
    """A parked sighting older than pending_ttl_s no longer counts
    toward promotion — consistency has a time horizon."""
    from datetime import timedelta

    v = unit([0.3, 0.4, 0.5, 0.1])
    kw = dict(
        identifier_key="person", class_name="person",
        vector=v.tolist(), plate=None, threshold=0.8,
    )
    resolver.resolve(session, timestamp=TS, **kw)
    late = resolver.resolve(session, timestamp=TS + timedelta(hours=2), **kw)
    assert late.pending  # the stale sighting was pruned, count restarts
    assert session.query(Identity).count() == 0


def test_resolver_auto_added_class_mints_immediately(resolver, session):
    """Identifiers without config (auto-added classes) keep the
    pre-CLD-41 behavior: min_sightings defaults to 1."""
    res = resolver.resolve(
        session, identifier_key="deer", class_name="deer",
        vector=unit([1.0, 0.0, 0.0, 0.0]).tolist(), plate=None,
        timestamp=TS, threshold=0.8,
    )
    assert res.is_new
    assert res.identity is not None


def _mint(resolver, session, vector, camera=None, ts=TS):
    """Create a person identity directly via the quality bypass."""
    res = resolver.resolve(
        session, identifier_key="person", class_name="person",
        vector=vector.tolist(), plate=None, timestamp=ts,
        threshold=0.8, quality=0.99, camera_id=camera,
    )
    assert res.identity is not None
    return res.identity


def test_resolver_margin_leaves_near_ties_unresolved(resolver, session):
    """A query sitting between two known identities inside min_margin is
    an ambiguous read: no match, and nothing is learned from it."""
    a = unit([1.0, 0.3, 0.0, 0.0])
    b = unit([0.3, 1.0, 0.0, 0.0])
    ia = _mint(resolver, session, a)
    ib = _mint(resolver, session, b)
    assert ia.id != ib.id

    q = unit(a + b)  # exactly equidistant, similarity ~0.88 to both
    res = resolver.resolve(
        session, identifier_key="person", class_name="person",
        vector=q.tolist(), plate=None, timestamp=TS, threshold=0.8,
    )
    assert res.identity is None
    assert res.ambiguous
    assert not res.pending  # ambiguity must not seed a duplicate
    assert session.query(Identity).count() == 2


def test_resolver_recency_breaks_near_ties(resolver, session):
    """Within the margin band, an identity seen on the same camera
    moments ago wins the tie; on a different camera it stays ambiguous."""
    a = unit([1.0, 0.3, 0.0, 0.0])
    b = unit([0.3, 1.0, 0.0, 0.0])
    ia = _mint(resolver, session, a, camera="gate")
    _mint(resolver, session, b, camera="yard")

    q = unit(a + b)
    kw = dict(
        identifier_key="person", class_name="person",
        vector=q.tolist(), plate=None, timestamp=TS, threshold=0.8,
    )
    # On a camera where neither was seen recently: still ambiguous (and
    # checked first — a successful match below enrolls q into a gallery,
    # which would make any later read of q unambiguous).
    elsewhere = resolver.resolve(session, camera_id="lobby", **kw)
    assert elsewhere.identity is None
    assert elsewhere.ambiguous

    on_gate = resolver.resolve(session, camera_id="gate", **kw)
    assert on_gate.identity is not None
    assert on_gate.identity.id == ia.id


def test_plate_beats_visual_similarity(resolver, session):
    """A matching plate wins even when the visual embedding differs
    (repaint, night shot) — PRD §6.4's 'matched by plate OR signature'."""
    first = resolver.resolve(
        session, identifier_key="vehicle", class_name="car",
        vector=unit([1.0, 0.0, 0.0, 0.0]).tolist(), plate="ABC123",
        timestamp=TS, threshold=0.99,
    )
    second = resolver.resolve(
        session, identifier_key="vehicle", class_name="car",
        vector=unit([0.0, 1.0, 0.0, 0.0]).tolist(), plate="ABC123",
        timestamp=TS, threshold=0.99,
    )
    assert not second.is_new
    assert second.identity.id == first.identity.id
    assert second.similarity == 1.0
    assert second.matched_by == "plate"


def test_visual_match_learns_plate(resolver, session):
    """Vehicle first seen without a readable plate gains one later —
    both paths write to the same identity record."""
    v = unit([0.5, 0.5, 0.5, 0.5])
    first = resolver.resolve(
        session, identifier_key="vehicle", class_name="motorcycle",
        vector=v.tolist(), plate=None, timestamp=TS, threshold=0.8,
    )
    assert first.identity.plate is None
    second = resolver.resolve(
        session, identifier_key="vehicle", class_name="motorcycle",
        vector=v.tolist(), plate="XYZ789", timestamp=TS, threshold=0.8,
    )
    assert second.identity.id == first.identity.id
    assert second.identity.plate == "XYZ789"
    assert second.matched_by == "visual"
    assert second.learned_plate is True


def test_registry_dynamic_class():
    cfg = IdentityConfig(auto_add_classes=True)
    registry = IdentifierRegistry(cfg)
    # "deer" is not configured anywhere — a generic identifier appears.
    matches = registry.identifiers_for("deer")
    assert len(matches) == 1
    key, ident = matches[0]
    assert key == "deer"
    assert ident.algo == "generic"
    # ... and it is remembered, not re-created.
    assert registry.identifiers_for("deer")[0][1] is ident


def test_registry_excluded_class_not_added():
    cfg = IdentityConfig(auto_add_classes=True, auto_add_exclude=["bird"])
    assert IdentifierRegistry(cfg).identifiers_for("bird") == []


def test_registry_person_gets_face_and_appearance():
    registry = IdentifierRegistry(IdentityConfig())
    keys = {k for k, _ in registry.identifiers_for("person")}
    assert keys == {"face", "person"}


def test_registry_no_auto_add_when_disabled():
    cfg = IdentityConfig(auto_add_classes=False)
    assert IdentifierRegistry(cfg).identifiers_for("deer") == []
