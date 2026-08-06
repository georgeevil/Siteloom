"""Identity framework tests — stub embedders, real Qdrant local mode."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from siteloom.config import IdentifierConfig, IdentityConfig
from siteloom.identity.registry import IdentifierRegistry
from siteloom.identity.resolver import IdentityResolver
from siteloom.identity.vectors import VectorStore
from siteloom.store import get_session, init_db, make_engine

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


def test_resolver_creates_then_matches(resolver, session):
    v = unit([0.3, 0.4, 0.5, 0.1])
    first = resolver.resolve(
        session, identifier_key="person", class_name="person",
        vector=v.tolist(), plate=None, timestamp=TS, threshold=0.8,
    )
    assert first.is_new
    assert first.identity.label is None  # unknown bucket

    again = resolver.resolve(
        session, identifier_key="person", class_name="person",
        vector=v.tolist(), plate=None, timestamp=TS, threshold=0.8,
    )
    assert not again.is_new
    assert again.identity.id == first.identity.id
    assert again.similarity > 0.99
    assert again.identity.appearance_count == 2


def test_resolver_below_threshold_creates_new(resolver, session):
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
    assert b.is_new
    assert b.identity.id != a.identity.id


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
