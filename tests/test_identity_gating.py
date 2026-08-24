"""What a gallery will accept, and who it lets a match be contested by (CLD-139).

Three mechanisms, all of which used to be absent:

1. **The candidate window is a contest between identities**, not between
   raw vectors. A flat top-k lets one gallery holding k near-duplicates
   fill the window, which collapses the ranking to a single identity and
   skips `min_margin` — the check that exists precisely for a crowded
   neighbourhood.
2. **Accretion is gated.** Every matching frame used to add its
   embedding, so one visit could fill a gallery and one wrong match
   became a permanent attractor that recruited more wrong vectors on
   every later visit. A quality floor and a per-event budget bound that,
   with a deliberate exemption for an identity's *first* vector.
3. **A "wrong" verdict stops further learning** on that pairing. It does
   not edit the gallery — that is unlink/reassign's job — it stops
   adding to it.

Mechanisms only: no test here asserts an accuracy number. Whether the
gates improve identification is measured on the corpus replay, which
needs model weights and gigabytes of crops, and can never run in CI.

Stub-free by construction — the vectors are hand-built unit vectors and
the store is real Qdrant in local mode, as in `test_identity.py`, whose
fixtures this reuses.
"""

from __future__ import annotations

import pytest

from siteloom.config import IdentityConfig
from siteloom.identity.resolver import _EVENT_BUDGET_MEMORY, IdentityResolver
from siteloom.identity.vectors import VectorStore
from siteloom.store import EventIdentity, Identity, get_session, init_db, make_engine

# The vector and minting helpers this file is built on. The two fixtures
# are re-declared below rather than imported: a fixture imported into a
# module is shadowed by every test that names it, which ruff reads as a
# redefinition.
from test_identity import TS, _mint, unit


@pytest.fixture
def vectors(tmp_path):
    store = VectorStore(tmp_path / "vectors")
    yield store
    store.close()


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/gating.db")
    init_db(engine)
    with get_session(engine)() as s:
        yield s


def _resolver(vectors, identifier: str = "person", **overrides) -> IdentityResolver:
    """A resolver whose named identifier carries the given gate values."""
    cfg = IdentityConfig()
    for field, value in overrides.items():
        setattr(cfg.identifiers[identifier], field, value)
    return IdentityResolver(cfg, vectors)


def _identity(session, key: str = "person", cls: str = "person") -> Identity:
    """An Identity row with no vectors — the gallery is filled directly."""
    row = Identity(identifier_key=key, class_name=cls, first_seen=TS, last_seen=TS)
    session.add(row)
    session.flush()
    return row


def _resolve(resolver, session, vector, *, key="person", cls="person",
             threshold=0.8, **kw):
    return resolver.resolve(
        session,
        identifier_key=key,
        class_name=cls,
        vector=vector.tolist(),
        plate=None,
        timestamp=TS,
        threshold=threshold,
        **kw,
    )


# -- 1-3. the candidate window --------------------------------------------

#: A query, a gallery of 40 near-duplicates that all beat the runner-up,
#: and a runner-up close enough to be inside any sane margin.
QUERY = unit([1.0, 0.0, 0.0, 0.0])
CROWD = [unit([1.0, 0.05 + i * 0.0001, 0.0, 0.0]) for i in range(40)]
RUNNER_UP = unit([1.0, 0.10, 0.0, 0.0])


def test_a_stuffed_gallery_cannot_hide_the_runner_up(vectors, session):
    """The headline: an identity holding 40 near-duplicate vectors used to
    monopolise the candidate window, so the margin check never saw the
    second identity and handed out the match unopposed.

    Both identities are above threshold and within `min_margin` of each
    other, which is the definition of an ambiguous read — the frame must
    go unresolved rather than be awarded to whoever happened to own more
    vectors.
    """
    crowd = _identity(session)
    quiet = _identity(session)
    for vector in CROWD:
        vectors.add("person", vector, crowd.id)
    vectors.add("person", RUNNER_UP, quiet.id)

    # The bug this prevents, stated as a fact about the old call: a flat
    # top-k window comes back holding one identity.
    flat = vectors.search("person", QUERY, limit=10)
    assert {hit.identity_id for hit in flat} == {crowd.id}

    ranked = vectors.search_identities("person", QUERY)
    assert [hit.identity_id for hit in ranked] == [crowd.id, quiet.id]
    assert ranked[0].score > ranked[1].score

    res = _resolve(_resolver(vectors), session, QUERY)
    assert res.identity is None
    assert res.ambiguous


def test_the_second_search_runs_only_when_the_window_is_monopolised(
    vectors, session, monkeypatch
):
    """Grouped search is ~20x the price of a flat one and the collection
    only grows, so it is paid in the pathological case and nowhere else.

    An unsaturated window *is* the whole collection — there is no hidden
    identity in it — so grouping it again could not change the answer.
    """
    grouped: list[str] = []
    real = vectors._client.query_points_groups

    def counting(*args, **kwargs):
        grouped.append(kwargs.get("collection_name"))
        return real(*args, **kwargs)

    monkeypatch.setattr(vectors._client, "query_points_groups", counting)

    for vector in CROWD[:5]:
        vectors.add("solo", vector, 1)
    vectors.search_identities("solo", QUERY)
    assert grouped == []  # one identity, and the window saw all of it

    for i, vector in enumerate(CROWD[:16]):
        vectors.add("healthy", vector, 1 if i % 2 else 2)
    assert len(vectors.search_identities("healthy", QUERY)) == 2
    assert grouped == []  # two galleries, neither crowding the other out

    for vector in CROWD:
        vectors.add("stuffed", vector, 1)
    vectors.add("stuffed", RUNNER_UP, 2)
    assert len(vectors.search_identities("stuffed", QUERY)) == 2
    assert grouped == ["stuffed"]  # the one case that needed it


def test_search_identities_reports_one_ranked_hit_per_identity(vectors):
    """One hit per identity — the best score it can offer — best first.
    Three galleries of three vectors each collapse to three hits."""
    for i, vector in enumerate(CROWD[:9]):
        vectors.add("person", vector, 1 + i % 3)
    hits = vectors.search_identities("person", QUERY)

    assert {hit.identity_id for hit in hits} == {1, 2, 3}
    assert len(hits) == 3  # not one per vector
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_identities_on_a_missing_collection_is_empty(vectors):
    assert vectors.search_identities("nope", QUERY) == []


# -- 4-6. the quality floor ------------------------------------------------


def test_a_frame_below_the_quality_floor_matches_but_is_not_stored(vectors, session):
    """The floor separates matching from learning. A crop too poor to
    trust in a gallery is still good enough to say who this probably is —
    refusing the *match* would lose the sighting, refusing the *vector*
    only declines to learn from it."""
    resolver = _resolver(vectors, learn_min_quality=0.6)
    identity = _mint(resolver, session, QUERY)
    assert identity.vector_count == 1

    res = _resolve(resolver, session, QUERY, quality=0.5)

    assert res.identity is not None and res.identity.id == identity.id
    assert not res.is_new
    assert identity.vector_count == 1  # matched, learned nothing


def test_an_identity_with_no_vectors_learns_below_the_floor(vectors, session):
    """The exemption that keeps the floor from backfiring: an identity
    with no vectors is invisible to matching, so gating its *first*
    vector turns a quality problem into identity churn — the next frame
    fails to match and mints another identity, and so on.

    The founding crop has already passed the minting gates; the floor is
    about accretion, and accretion starts at the second vector.
    """
    resolver = _resolver(vectors, "vehicle", learn_min_quality=0.6)
    kw = dict(key="vehicle", cls="car", threshold=0.82, quality=0.5)

    first = _resolve(resolver, session, QUERY, **kw)
    assert first.is_new
    assert first.identity.vector_count == 1  # stored despite the low score

    second = _resolve(resolver, session, QUERY, **kw)
    assert not second.is_new
    assert second.identity.id == first.identity.id  # found, not re-minted
    assert session.query(Identity).count() == 1
    assert second.identity.vector_count == 1  # and now the floor applies


def test_a_frame_reporting_no_quality_is_not_treated_as_zero(vectors, session):
    """Absent is absent, exactly as for the plate floors: a caller that
    never measured quality has not reported a bad one."""
    resolver = _resolver(vectors, learn_min_quality=0.6)
    identity = _mint(resolver, session, QUERY)

    _resolve(resolver, session, QUERY, quality=None)

    assert identity.vector_count == 2


# -- 7-8. the per-event budget --------------------------------------------


def test_one_event_may_contribute_only_so_many_vectors(vectors, session):
    """The cap that makes a single visit unable to flood a gallery. A
    30-frame visit used to contribute 30 vectors, which is how one event
    became most of what an identity looks like."""
    resolver = _resolver(vectors, learn_max_per_event=3)
    identity = _mint(resolver, session, QUERY)
    founding = identity.vector_count

    for _ in range(6):
        _resolve(resolver, session, QUERY, quality=0.9, event_id=11)
    assert identity.vector_count - founding == 3

    # A different visit is a different budget — the cap bounds one event,
    # not the identity (that is max_vectors_per_identity).
    for _ in range(6):
        _resolve(resolver, session, QUERY, quality=0.9, event_id=12)
    assert identity.vector_count - founding == 6


def test_the_per_identity_cap_still_wins_when_it_is_lower(vectors, session):
    resolver = _resolver(vectors, learn_max_per_event=10)
    identity = _mint(resolver, session, QUERY)

    for _ in range(5):
        _resolve(resolver, session, QUERY, quality=0.9, event_id=21, max_vectors=2)

    assert identity.vector_count == 2


def test_a_mint_from_quarantine_cannot_found_a_gallery_from_one_visit(
    vectors, session
):
    """The same flooding on the minting path: the quarantine pool holds
    one visit's frames by construction, so promoting all of them founds a
    gallery in which one visit is the whole identity."""
    resolver = _resolver(vectors, min_sightings=6, learn_max_per_event=3)
    kw = dict(quality=0.7, event_id=31)  # under immediate_quality, over the floor

    for _ in range(5):
        assert _resolve(resolver, session, QUERY, **kw).pending

    minted = _resolve(resolver, session, QUERY, **kw)
    assert minted.is_new
    # Exactly the cap: five sightings were waiting and three were kept.
    # "At most" would also be satisfied by a promotion loop that stored
    # nothing, which is the opposite failure — an identity founded with
    # an empty gallery is invisible to matching.
    assert minted.identity.vector_count == 3
    # Every sighting still counts as an appearance — they happened.
    assert minted.identity.appearance_count == 6

    stored = minted.identity.vector_count
    for _ in range(3):
        _resolve(resolver, session, QUERY, **kw)
    assert minted.identity.vector_count == stored  # the visit's budget is spent


def test_the_budget_memory_is_bounded_and_forgets_conservatively(vectors, session):
    """The budgets live in memory, so the dict that holds them has to be
    bounded — a process running for weeks would otherwise keep one entry
    per (event, identity) it ever saw.

    Eviction is oldest-first because that is the entry least likely to
    matter: an event that far back receives no further frames. What it
    costs when the guess is wrong is exactly one more budget — a visit
    seen again after its counter was dropped starts a fresh one rather
    than an unlimited one, and `max_vectors_per_identity` still caps the
    total. That is the same conservative degradation a restart produces.
    """
    resolver = _resolver(vectors, learn_max_per_event=3)
    identity = _mint(resolver, session, QUERY)
    founding = identity.vector_count

    for _ in range(6):
        _resolve(resolver, session, QUERY, quality=0.9, event_id=71)
    assert identity.vector_count - founding == 3  # this visit is spent

    # Enough other visits go by to push it out of the window.
    for event_id in range(1000, 1000 + _EVENT_BUDGET_MEMORY):
        resolver._note_learned("person", event_id, identity.id)
    assert len(resolver._event_learned) == _EVENT_BUDGET_MEMORY  # bounded
    assert ("person", 71, identity.id) not in resolver._event_learned  # oldest out

    for _ in range(6):
        _resolve(resolver, session, QUERY, quality=0.9, event_id=71)
    assert identity.vector_count - founding == 6  # one fresh budget, not six
    assert len(resolver._event_learned) == _EVENT_BUDGET_MEMORY


# -- 9. a verdict stops further learning ----------------------------------


def _verdict(session, event_id: int, identity: Identity, verdict: str, **kw) -> None:
    session.add(
        EventIdentity(
            event_id=event_id,
            identity_id=identity.id,
            identifier_key=identity.identifier_key,
            similarity=0.9,
            verdict=verdict,
            **kw,
        )
    )
    session.flush()


def test_a_wrong_verdict_stops_the_gallery_growing_for_that_event(vectors, session):
    """One wrong match used to recruit more wrong vectors on every later
    frame of the same visit. The verdict does not edit the gallery — that
    is what unlink and reassign are for — it stops adding to it."""
    resolver = _resolver(vectors)
    identity = _mint(resolver, session, QUERY)
    before = identity.vector_count
    _verdict(session, 41, identity, "wrong")

    res = _resolve(resolver, session, QUERY, quality=0.9, event_id=41)

    # Resolution itself is untouched: the operator judged the claim, they
    # did not tell the matcher to stop seeing.
    assert res.identity is not None and res.identity.id == identity.id
    assert identity.vector_count == before

    # ... and it is that pairing that stopped, not the identity.
    _resolve(resolver, session, QUERY, quality=0.9, event_id=42)
    assert identity.vector_count == before + 1


def test_a_wrong_and_detached_claim_blocks_too(vectors, session):
    """The strongest repudiation there is: "wrong, and detached". A later
    frame never revives an unlinked claim (CLD-36) — it opens a fresh row
    — so a gate keyed only to the standing claim would miss exactly the
    case an operator felt most strongly about."""
    resolver = _resolver(vectors)
    identity = _mint(resolver, session, QUERY)
    before = identity.vector_count
    _verdict(session, 43, identity, "wrong", unlinked_at=TS)

    _resolve(resolver, session, QUERY, quality=0.9, event_id=43)

    assert identity.vector_count == before


def test_a_confirmed_verdict_does_not_block(vectors, session):
    resolver = _resolver(vectors)
    identity = _mint(resolver, session, QUERY)
    before = identity.vector_count
    _verdict(session, 44, identity, "confirmed")

    _resolve(resolver, session, QUERY, quality=0.9, event_id=44)

    assert identity.vector_count == before + 1


# -- 10. both gates are config, and 0 is off ------------------------------


def test_a_zero_quality_floor_restores_unfloored_learning(vectors, session):
    """The rollback lever: config only, no redeploy. Each knob switches
    off independently, because a site may want an unlimited count above a
    strict floor, or the reverse."""
    resolver = _resolver(vectors, learn_min_quality=0.0)
    identity = _mint(resolver, session, QUERY)

    _resolve(resolver, session, QUERY, quality=0.01)

    assert identity.vector_count == 2


def test_a_zero_event_cap_restores_unbounded_accretion(vectors, session):
    resolver = _resolver(vectors, learn_max_per_event=0)
    identity = _mint(resolver, session, QUERY)
    founding = identity.vector_count

    for _ in range(6):
        _resolve(resolver, session, QUERY, quality=0.9, event_id=51)

    assert identity.vector_count - founding == 6


@pytest.mark.parametrize("gate", ["learn_min_quality", "learn_max_per_event"])
def test_neither_gate_is_reachable_without_the_other(vectors, session, gate):
    """Switching one off leaves the other in force — they are two rules,
    not one switch."""
    resolver = _resolver(vectors, **{gate: 0})
    identity = _mint(resolver, session, QUERY)
    founding = identity.vector_count

    for _ in range(6):
        _resolve(resolver, session, QUERY, quality=0.5, event_id=61)

    grown = identity.vector_count - founding
    if gate == "learn_min_quality":
        assert grown == 3  # the floor is off, the per-event cap is not
    else:
        assert grown == 0  # the cap is off, but every frame is under the floor


# -- 4. the mint budget -----------------------------------------------------
#
# `learn_max_per_event` bounds what a visit teaches ONE identity, but a
# mint is a new identity with a fresh budget — so a visit whose frames
# kept missing could convert gallery flooding into identity flooding
# (one 73-second live visit minted 50 identities). `mint_max_per_event`
# is the bound on that: per (identifier, event), refused frames park in
# the pending pool instead of minting.


def _distinct(i: int):
    import numpy as np

    vec = np.zeros(64, dtype=np.float32)
    vec[i] = 1.0
    return vec


def test_the_mint_budget_bounds_identity_flooding(vectors, session):
    resolver = _resolver(vectors, min_sightings=1, mint_max_per_event=2)
    outcomes = [
        _resolve(resolver, session, _distinct(i), event_id=71) for i in range(5)
    ]
    assert [r.identity is not None for r in outcomes] == [
        True, True, False, False, False,
    ]
    assert all(r.pending for r in outcomes[2:])
    assert session.query(Identity).count() == 2


def test_a_plate_mint_is_exempt_from_the_budget(vectors, session):
    """A plate is exact evidence — the budget must never park it."""
    resolver = _resolver(vectors, "vehicle", min_sightings=1, mint_max_per_event=1)
    first = resolver.resolve(
        session, identifier_key="vehicle", class_name="car",
        vector=_distinct(1).tolist(), plate=None, timestamp=TS,
        threshold=0.8, event_id=72,
    )
    assert first.identity is not None  # spends the budget
    plated = resolver.resolve(
        session, identifier_key="vehicle", class_name="car",
        vector=_distinct(2).tolist(), plate="AAA111", timestamp=TS,
        threshold=0.8, event_id=72,
    )
    assert plated.identity is not None and plated.is_new
    assert plated.identity.plate == "AAA111"


def test_a_zero_budget_restores_unbounded_minting(vectors, session):
    resolver = _resolver(vectors, min_sightings=1, mint_max_per_event=0)
    for i in range(5):
        assert _resolve(
            resolver, session, _distinct(i), event_id=73
        ).identity is not None
    assert session.query(Identity).count() == 5


def test_budget_parked_frames_can_found_a_later_events_mint(vectors, session):
    """A refused frame is still a real sighting: it waits in the pool
    and corroborates a mint on the next visit, it is not evidence
    destroyed."""
    resolver = _resolver(vectors, min_sightings=2, mint_max_per_event=1)
    # Visit 81: one pair promotes (spending the budget), then a new
    # subject's frame arrives and is parked by the budget.
    assert _resolve(resolver, session, _distinct(1), event_id=81).pending
    minted = _resolve(resolver, session, _distinct(1), event_id=81)
    assert minted.identity is not None and minted.is_new
    parked = _resolve(resolver, session, _distinct(2), event_id=81)
    assert parked.pending and parked.identity is None
    # Visit 82: the same subject returns; the parked sighting counts.
    promoted = _resolve(resolver, session, _distinct(2), event_id=82)
    assert promoted.identity is not None and promoted.is_new
    assert session.query(Identity).count() == 2


def test_an_auto_added_class_gets_the_default_mint_budget(vectors, session):
    """The identifiers nobody tuned are exactly where flooding protection
    must not silently sit out: an identifier key with no config entry
    (auto-added classes) runs under the field-default budget, not 0."""
    resolver = _resolver(vectors)  # config has no "dog" identifier
    outcomes = [
        resolver.resolve(
            session, identifier_key="dog", class_name="dog",
            vector=_distinct(i).tolist(), plate=None, timestamp=TS,
            threshold=0.8, event_id=91,
        )
        for i in range(5)
    ]
    minted = [r for r in outcomes if r.identity is not None]
    from siteloom.config import IdentifierConfig

    assert len(minted) == IdentifierConfig.model_fields["mint_max_per_event"].default
    assert all(r.pending for r in outcomes[len(minted):])


def test_budget_refusals_of_an_ungated_identifier_skip_the_pool(vectors, session):
    """min_sightings=1 identifiers have no promotion path, so parking
    their refused frames would be a write nothing ever reads — the pool
    must stay untouched."""
    resolver = _resolver(vectors, min_sightings=1, mint_max_per_event=1)
    for i in range(4):
        _resolve(resolver, session, _distinct(i), event_id=92)
    assert vectors.search_labeled("person-pending", _distinct(2), limit=5) == []
