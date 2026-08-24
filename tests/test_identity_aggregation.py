"""Per-identity score aggregation (CLD-152): max vs mean-of-top-k.

Max — the historical rule — lets one lucky near-duplicate define a
gallery's score, which is how a 70-vector gallery of a parking spot
matched an unrelated car at 0.91. mean_top_k makes a match need
corroboration from several vectors. Default stays max, byte-identical.

Vectors are hand-built so their cosine to the probe is exact: a vector
`s·e0 + √(1−s²)·e_k` scores precisely `s` against the probe `e0`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from siteloom.config import IdentityConfig
from siteloom.identity.resolver import IdentityResolver
from siteloom.identity.vectors import VectorStore
from siteloom.store import Identity, get_session, init_db, make_engine

DIM = 64
TS = __import__("datetime").datetime(2026, 8, 23, 12, 0, 0)


def _axis(i: int) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float32)
    vec[i] = 1.0
    return vec


def _scoring(sim: float, axis: int) -> np.ndarray:
    """A unit vector whose cosine against the probe (_axis(0)) is `sim`."""
    return (sim * _axis(0) + math.sqrt(1 - sim * sim) * _axis(axis)).astype(
        np.float32
    )


PROBE = _axis(0)


@pytest.fixture
def vectors(tmp_path):
    store = VectorStore(tmp_path / "vectors")
    yield store
    store.close()


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/agg.db")
    init_db(engine)
    with get_session(engine)() as s:
        yield s


def _gallery(vectors, identity_id: int, sims_axes: list[tuple[float, int]]):
    for sim, axis in sims_axes:
        vectors.add("person", _scoring(sim, axis), identity_id)


def test_mean_top_k_needs_corroboration(vectors):
    """One outlier (0.95) over a weak gallery loses to two consistent
    0.9s — the exact inversion the knob exists for."""
    _gallery(vectors, 1, [(0.95, 1), (0.50, 2)])  # the outlier identity
    _gallery(vectors, 2, [(0.90, 3), (0.89, 4)])  # the corroborated one

    by_max = vectors.search_identities("person", PROBE)
    assert [hit.identity_id for hit in by_max] == [1, 2]
    assert by_max[0].score == pytest.approx(0.95, abs=1e-3)

    by_mean = vectors.search_identities(
        "person", PROBE, aggregation="mean_top_k", top_k=2
    )
    assert [hit.identity_id for hit in by_mean] == [2, 1]
    assert by_mean[0].score == pytest.approx((0.90 + 0.89) / 2, abs=1e-3)
    assert by_mean[1].score == pytest.approx((0.95 + 0.50) / 2, abs=1e-3)


def test_a_small_gallery_averages_what_it_has(vectors):
    """top_k above a gallery's size divides by what exists — a
    two-vector identity is not punished for not having three."""
    _gallery(vectors, 1, [(0.92, 1)])
    _gallery(vectors, 2, [(0.95, 2), (0.50, 3)])
    ranked = vectors.search_identities(
        "person", PROBE, aggregation="mean_top_k", top_k=3
    )
    assert [hit.identity_id for hit in ranked] == [1, 2]
    assert ranked[0].score == pytest.approx(0.92, abs=1e-3)


def test_default_ranking_is_unchanged(vectors):
    _gallery(vectors, 1, [(0.95, 1), (0.50, 2)])
    _gallery(vectors, 2, [(0.90, 3), (0.89, 4)])
    default = vectors.search_identities("person", PROBE)
    explicit = vectors.search_identities(
        "person", PROBE, aggregation="max", top_k=5
    )
    assert [(h.identity_id, round(h.score, 6)) for h in default] == [
        (h.identity_id, round(h.score, 6)) for h in explicit
    ]


def test_crowded_gallery_grouped_path_still_aggregates(vectors):
    """A 40-vector gallery saturates the flat window (CLD-139's case);
    the grouped fallback must aggregate the same way and still surface
    the runner-up."""
    _gallery(vectors, 1, [(0.93, 5)] * 40)
    _gallery(vectors, 2, [(0.90, 6), (0.88, 7)])
    ranked = vectors.search_identities(
        "person", PROBE, aggregation="mean_top_k", top_k=2
    )
    assert [hit.identity_id for hit in ranked] == [1, 2]
    assert ranked[0].score == pytest.approx(0.93, abs=1e-3)
    assert ranked[1].score == pytest.approx(0.89, abs=1e-3)


def test_resolver_reads_the_knob_off_the_identifier(vectors, session):
    """The same frame matches a different identity under mean_top_k —
    through `resolve()`, not just the store call."""
    outlier = Identity(
        identifier_key="person", class_name="person", first_seen=TS, last_seen=TS
    )
    steady = Identity(
        identifier_key="person", class_name="person", first_seen=TS, last_seen=TS
    )
    session.add_all([outlier, steady])
    session.flush()
    _gallery(vectors, outlier.id, [(0.95, 1), (0.50, 2)])
    _gallery(vectors, steady.id, [(0.90, 3), (0.89, 4)])
    # The row counter is what _may_learn checks against max_vectors —
    # it must agree with the store or the resolves below accrete the
    # probe into the winner and rig each other.
    outlier.vector_count = steady.vector_count = 2

    def resolve_with(**overrides):
        cfg = IdentityConfig()
        for field, value in overrides.items():
            setattr(cfg.identifiers["person"], field, value)
        resolver = IdentityResolver(cfg, vectors)
        return resolver.resolve(
            session,
            identifier_key="person",
            class_name="person",
            vector=PROBE.tolist(),
            plate=None,
            timestamp=TS,
            threshold=0.6,
            # Both galleries are at this cap, so the first resolve
            # cannot teach its winner the probe and rig the second.
            max_vectors=2,
        )

    assert resolve_with(min_margin=0.0).identity.id == outlier.id
    assert (
        resolve_with(
            min_margin=0.0, score_aggregation="mean_top_k", score_top_k=2
        ).identity.id
        == steady.id
    )
