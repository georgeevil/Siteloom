"""One standing claim per (event, identity) — the fold and the index (CLD-133).

Event 30 rendered four claims for two identities because "these two rows
are the same claim" was implemented twice: ingest's de-fragmentation
folded them, and the identity-merge endpoint re-pointed rows with a blind
UPDATE. These tests hold the three pieces that replace that convention
with a structure:

1. `fold_claim` is the single set of collapse rules — summed evidence,
   strongest verdict, strongest match mode — and it refuses every fold
   that would lose data instead of collapsing it.
2. A partial unique index makes a second *active* claim for a pair
   unrepresentable, while leaving the two shapes that are legitimately
   repeatable alone: repudiated (unlinked) rows, which are evidence and
   may stack (CLD-36), and NULL-identity miss rows, which are per
   identifier.
3. `init_db` repairs a database that already stacked duplicates *before*
   it creates the index — in that order, because the index cannot be
   created while duplicates exist.

No model weights, no cameras: everything here is store-level.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import func, inspect, select, text
from sqlalchemy.exc import IntegrityError

from siteloom.store import (
    Camera,
    Event,
    EventIdentity,
    Identity,
    get_session,
    init_db,
    make_engine,
)
from siteloom.store.claims import (
    active_claim,
    fold_claim,
    insert_claim,
    link_claim,
    record_sighting,
    repair_duplicate_claims,
)
from siteloom.store.models import ACTIVE_CLAIM_INDEX

TS = datetime(2026, 8, 14, 9, 0, 0)
KEEP_AT = datetime(2026, 8, 14, 10, 0, 0)
LOSE_AT = datetime(2026, 8, 14, 11, 0, 0)


def _claim(**overrides) -> EventIdentity:
    """An EventIdentity with every column set explicitly.

    Column defaults are applied by the INSERT, so a detached row would
    otherwise carry None where the fold arithmetic expects a number.
    """
    fields = dict(
        event_id=1,
        identity_id=2,
        identifier_key="vehicle",
        similarity=0.5,
        hit_count=1,
        matched_by="visual",
        learned_plate=False,
        verdict=None,
        verdict_at=None,
        unlinked_at=None,
    )
    fields.update(overrides)
    return EventIdentity(**fields)


# -- 1. fold rules ---------------------------------------------------------


def test_fold_sums_evidence_and_keeps_the_strongest_of_each_field():
    """The fold is a collapse, not a pick: hit counts add up (the
    resolver's appearance_count is kept in step by Σ hit_count), the
    similarity is the best frame either row saw, and a plate read outranks
    visual similarity outright (PRD §6.4)."""
    keeper = _claim(hit_count=1190, similarity=0.91, matched_by="visual")
    loser = _claim(hit_count=3, similarity=0.998, matched_by="plate", learned_plate=True)

    fold_claim(keeper, loser)

    assert keeper.hit_count == 1193
    assert keeper.similarity == 0.998
    assert keeper.matched_by == "plate"
    assert keeper.learned_plate is True
    assert keeper.unlinked_at is None


def test_fold_does_not_demote_a_plate_match_or_a_better_similarity():
    keeper = _claim(hit_count=4, similarity=0.99, matched_by="plate", learned_plate=True)
    loser = _claim(hit_count=1, similarity=0.4, matched_by="visual", learned_plate=False)

    fold_claim(keeper, loser)

    assert (keeper.similarity, keeper.matched_by, keeper.learned_plate) == (
        0.99,
        "plate",
        True,
    )
    assert keeper.hit_count == 5


def test_fold_fills_a_null_matched_by_and_identifier_key_from_the_loser():
    """A row whose first frame *minted* the identity has no match mode —
    there was nothing to match against — so it picks up the mode of the
    frames that later re-matched it."""
    keeper = _claim(matched_by=None, identifier_key=None)
    loser = _claim(matched_by="visual", identifier_key="vehicle")

    fold_claim(keeper, loser)

    assert keeper.matched_by == "visual"
    assert keeper.identifier_key == "vehicle"


def test_fold_never_moves_a_row_between_the_human_and_visual_buckets():
    """`siteloom/stats.py` splits accuracy by this column, so promoting
    "human" over "visual" (or back) would silently move claims between
    two reported buckets. They rank equal; the keeper's stands."""
    keeper = _claim(matched_by="visual")
    fold_claim(keeper, _claim(matched_by="human"))
    assert keeper.matched_by == "visual"

    keeper = _claim(matched_by="human")
    fold_claim(keeper, _claim(matched_by="visual"))
    assert keeper.matched_by == "human"


def test_record_sighting_applies_the_same_rules_to_one_new_frame():
    """The per-frame increment ingest and the Frigate consumer both do:
    the fold's rules for a single incoming observation."""
    link = _claim(hit_count=2, similarity=0.7, matched_by="visual")

    record_sighting(link, similarity=0.4, matched_by="plate", learned_plate=True)

    assert link.hit_count == 3
    assert link.similarity == 0.7  # max, not last
    assert link.matched_by == "plate"
    assert link.learned_plate is True


# -- 2. verdict precedence -------------------------------------------------


@pytest.mark.parametrize(
    ("keeper_verdict", "loser_verdict", "expected", "from_loser"),
    [
        (None, "confirmed", "confirmed", True),
        (None, "wrong", "wrong", True),
        ("wrong", None, "wrong", False),
        ("confirmed", None, "confirmed", False),
        ("wrong", "confirmed", "confirmed", True),
        ("confirmed", "wrong", "confirmed", False),
    ],
)
def test_fold_keeps_the_strongest_verdict_with_its_timestamp(
    keeper_verdict, loser_verdict, expected, from_loser
):
    """A human verdict always outranks no verdict — folding must never
    erase a review, which would also drop the event out of the `flagged`
    triage bucket — and an affirmative identification outranks a
    repudiation of the same pairing (CLD-143 spec §0(d)).

    The verdict and its timestamp move together: a verdict stamped at a
    time belonging to the other row is a falsified record.
    """
    keeper = _claim(
        verdict=keeper_verdict, verdict_at=KEEP_AT if keeper_verdict else None
    )
    loser = _claim(verdict=loser_verdict, verdict_at=LOSE_AT if loser_verdict else None)

    fold_claim(keeper, loser)

    assert keeper.verdict == expected
    assert keeper.verdict_at == (LOSE_AT if from_loser else KEEP_AT if expected else None)


# -- 3. fold guards --------------------------------------------------------


def test_fold_refuses_rows_from_different_events():
    with pytest.raises(ValueError):
        fold_claim(_claim(event_id=1), _claim(event_id=2))


@pytest.mark.parametrize("side", ["keeper", "loser"])
def test_fold_refuses_an_unlinked_row_on_either_side(side):
    """An unlinked row is a closed record of what the system got wrong
    (CLD-36), not a standing claim; folding it in would launder a
    repudiation back into a live claim, or destroy it."""
    rows = {"keeper": _claim(), "loser": _claim()}
    rows[side].unlinked_at = TS
    with pytest.raises(ValueError):
        fold_claim(rows["keeper"], rows["loser"])


def test_fold_refuses_a_row_folded_into_itself():
    row = _claim(hit_count=5)
    with pytest.raises(ValueError):
        fold_claim(row, row)
    assert row.hit_count == 5  # refused before it doubled anything


def test_fold_allows_rows_on_different_identities():
    """The merge endpoint folds a source-identity row into a
    target-identity row — same event, deliberately different ids."""
    keeper = _claim(identity_id=2, hit_count=1)
    fold_claim(keeper, _claim(identity_id=3, hit_count=2))
    assert keeper.hit_count == 3


# -- 4. the index on a fresh database --------------------------------------


@pytest.fixture
def claims_db(tmp_path):
    """A fresh store with one event and two identities to claim it."""
    engine = make_engine(f"sqlite:///{tmp_path}/claims.db")
    init_db(engine)
    Session = get_session(engine)
    with Session() as session:
        session.add(Camera(id="cam1", site_id="t", name="Cam One"))
        event = Event(
            camera_id="cam1",
            track_id=1,
            class_name="car",
            first_seen=TS,
            last_seen=TS,
            detection_count=1,
        )
        session.add(event)
        identities = [
            Identity(
                identifier_key="vehicle",
                class_name="car",
                first_seen=TS,
                last_seen=TS,
            )
            for _ in range(2)
        ]
        session.add_all(identities)
        session.commit()
        ids = SimpleNamespace(
            engine=engine,
            Session=Session,
            event=event.id,
            identity=identities[0].id,
            other=identities[1].id,
        )
    return ids


def _index(engine):
    return next(
        (
            i
            for i in inspect(engine).get_indexes("event_identities")
            if i["name"] == ACTIVE_CLAIM_INDEX
        ),
        None,
    )


def _active_rows(session, event_id, identity_id):
    return session.scalars(
        select(EventIdentity)
        .where(
            EventIdentity.event_id == event_id,
            EventIdentity.identity_id == identity_id,
            EventIdentity.unlinked_at.is_(None),
        )
        .order_by(EventIdentity.id)
    ).all()


def test_a_fresh_database_declares_the_partial_unique_index(claims_db):
    index = _index(claims_db.engine)
    assert index is not None
    assert index["unique"]
    assert index["column_names"] == ["event_id", "identity_id"]
    with claims_db.engine.connect() as conn:
        ddl = conn.execute(
            text("SELECT sql FROM sqlite_master WHERE name = :name"),
            {"name": ACTIVE_CLAIM_INDEX},
        ).scalar()
    # Partial, or it would forbid the two shapes that are legitimately
    # repeatable (see cases below).
    assert "unlinked_at IS NULL" in ddl


# -- 5-7. migrating a database that already stacked duplicates -------------

#: The current event_identities schema, minus the new index — what a
#: database written before this change looks like.
CURRENT_SCHEMA = """CREATE TABLE event_identities (
     id INTEGER PRIMARY KEY,
     event_id INTEGER NOT NULL,
     identity_id INTEGER,
     identifier_key VARCHAR,
     similarity FLOAT,
     hit_count INTEGER,
     matched_by VARCHAR,
     learned_plate BOOLEAN,
     verdict VARCHAR,
     verdict_at DATETIME,
     unlinked_at DATETIME)"""

_COLUMNS = (
    "id, event_id, identity_id, identifier_key, similarity, hit_count,"
    " matched_by, learned_plate, verdict, verdict_at, unlinked_at"
)

#: Event 30 as the field found it: two identities, four active claims,
#: plus two repudiated rows that are not duplicates of anything.
EVENT_30_ROWS = [
    # (30, 5): the big claim an operator confirmed, and a stray 3-hit row.
    (33, 30, 5, "vehicle", 0.91, 1190, "visual", 0, "confirmed", "2026-08-10", None),
    (42, 30, 5, "vehicle", 0.998, 3, "plate", 1, None, None, None),
    # (30, 9): a confirmed claim and a single-frame duplicate.
    (41, 30, 9, "vehicle", 0.983, 149, "visual", 0, "confirmed", "2026-08-10", None),
    (44, 30, 9, "vehicle", 0.9, 1, "visual", 0, None, None, None),
    # Repudiated claims on a pair that also has a live one: legal, and
    # not the migration's business (CLD-36).
    (32, 30, 5, "vehicle", 0.72, 2, "visual", 0, "wrong", "2026-08-09", "2026-08-09"),
    (43, 30, 5, "vehicle", 0.70, 1, "visual", 0, "wrong", "2026-08-09", "2026-08-09"),
]


def _seed(db, schema: str, columns: str, rows) -> None:
    con = sqlite3.connect(db)
    con.execute(schema)
    placeholders = ", ".join("?" * len(rows[0]))
    con.executemany(
        f"INSERT INTO event_identities ({columns}) VALUES ({placeholders})", rows
    )
    con.commit()
    con.close()


def test_migration_folds_duplicate_active_claims_then_indexes(tmp_path):
    """The headline repair: a database carrying event 30's four active
    claims for two identities comes back with two, carrying the summed
    evidence — and only then gains the index that keeps it that way."""
    db = tmp_path / "dirty.db"
    _seed(db, CURRENT_SCHEMA, _COLUMNS, EVENT_30_ROWS)

    engine = make_engine(f"sqlite:///{db}")
    init_db(engine)

    Session = get_session(engine)
    with Session() as session:
        first = _active_rows(session, 30, 5)
        assert len(first) == 1
        keeper = first[0]
        assert keeper.id == 33  # most evidence behind it, and the oldest
        assert keeper.hit_count == 1193
        assert keeper.similarity == 0.998
        assert keeper.verdict == "confirmed"  # a review is never erased
        assert keeper.matched_by == "plate"  # strongest mode, from the loser
        assert keeper.learned_plate is True
        assert session.get(EventIdentity, 42) is None

        second = _active_rows(session, 30, 9)
        assert len(second) == 1
        assert (second[0].id, second[0].hit_count) == (41, 150)
        assert second[0].matched_by == "visual"

        # The repudiated rows are untouched — they are evidence, and
        # multiple ones for a pair are legal.
        unlinked = session.scalars(
            select(EventIdentity).where(EventIdentity.unlinked_at.is_not(None))
        ).all()
        assert {row.id for row in unlinked} == {32, 43}

    assert _index(engine) is not None


def test_migration_adds_unlinked_at_before_it_repairs(tmp_path):
    """Ordering, from the other side: the repair asks which rows are
    standing claims, so the column that answers that has to exist before
    it runs. A database from before `unlinked_at` still repairs."""
    schema = """CREATE TABLE event_identities (
         id INTEGER PRIMARY KEY,
         event_id INTEGER NOT NULL,
         identity_id INTEGER,
         similarity FLOAT,
         hit_count INTEGER,
         verdict VARCHAR,
         verdict_at DATETIME)"""
    _seed(
        tmp_path / "old.db",
        schema,
        "id, event_id, identity_id, similarity, hit_count, verdict",
        [(1, 4, 2, 0.8, 6, "confirmed"), (2, 4, 2, 0.95, 1, None)],
    )

    engine = make_engine(f"sqlite:///{tmp_path}/old.db")
    init_db(engine)

    with get_session(engine)() as session:
        rows = _active_rows(session, 4, 2)
        assert len(rows) == 1
        assert (rows[0].id, rows[0].hit_count, rows[0].similarity) == (1, 7, 0.95)
        assert rows[0].verdict == "confirmed"
    assert _index(engine) is not None


def test_the_repair_is_idempotent(tmp_path):
    """Second boot over the same file: nothing to find, nothing written,
    no error from re-creating an index that exists."""
    db = tmp_path / "twice.db"
    _seed(db, CURRENT_SCHEMA, _COLUMNS, EVENT_30_ROWS)

    engine = make_engine(f"sqlite:///{db}")
    init_db(engine)
    Session = get_session(engine)
    with Session() as session:
        before = session.execute(
            select(EventIdentity.id, EventIdentity.hit_count).order_by(EventIdentity.id)
        ).all()

    init_db(engine)

    with Session() as session:
        after = session.execute(
            select(EventIdentity.id, EventIdentity.hit_count).order_by(EventIdentity.id)
        ).all()
    assert after == before
    with Session() as session:
        assert repair_duplicate_claims(session) == 0


def test_an_interrupted_rebuild_holding_duplicates_completes_and_repairs(tmp_path):
    """The trap in the rebuild path: creating the new table also creates
    its declared indexes, so the unique one would refuse the very rows
    being copied. The rebuild must land the rows first, the repair must
    fold them, and the index must exist at the end."""
    db = tmp_path / "interrupted.db"
    con = sqlite3.connect(db)
    con.execute(
        """CREATE TABLE event_identities_old (
             id INTEGER PRIMARY KEY,
             event_id INTEGER NOT NULL,
             identity_id INTEGER NOT NULL,
             similarity FLOAT,
             hit_count INTEGER,
             verdict VARCHAR,
             verdict_at DATETIME)"""
    )
    con.executemany(
        "INSERT INTO event_identities_old (id, event_id, identity_id, similarity,"
        " hit_count, verdict) VALUES (?, ?, ?, ?, ?, ?)",
        [(7, 1, 3, 0.9, 2, "confirmed"), (8, 1, 3, 0.5, 4, None)],
    )
    con.commit()
    con.close()

    engine = make_engine(f"sqlite:///{db}")
    init_db(engine)

    inspector = inspect(engine)
    assert "event_identities_old" not in inspector.get_table_names()
    with get_session(engine)() as session:
        rows = _active_rows(session, 1, 3)
        assert len(rows) == 1
        assert (rows[0].id, rows[0].hit_count, rows[0].similarity) == (8, 6, 0.9)
        assert rows[0].verdict == "confirmed"
    assert _index(engine) is not None


# -- 8-10. what the index does and does not forbid -------------------------


def test_the_database_refuses_a_second_active_claim_for_a_pair(claims_db):
    with claims_db.Session() as session:
        session.add(
            EventIdentity(
                event_id=claims_db.event,
                identity_id=claims_db.identity,
                identifier_key="vehicle",
                similarity=0.8,
            )
        )
        session.commit()
        session.add(
            EventIdentity(
                event_id=claims_db.event,
                identity_id=claims_db.identity,
                identifier_key="vehicle",
                similarity=0.6,
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_several_repudiated_claims_for_one_pair_stay_legal(claims_db):
    """An operator may say "not this one" about the same pairing more
    than once; each refusal is evidence, and deleting the older one would
    delete a negative (CLD-36)."""
    with claims_db.Session() as session:
        for i in range(3):
            session.add(
                EventIdentity(
                    event_id=claims_db.event,
                    identity_id=claims_db.identity,
                    identifier_key="vehicle",
                    similarity=0.5,
                    verdict="wrong",
                    unlinked_at=datetime(2026, 8, 14, 12, i),
                )
            )
        session.commit()
        assert session.scalar(
            select(func.count()).select_from(EventIdentity)
        ) == 3


def test_several_missed_identifiers_on_one_event_stay_legal(claims_db):
    """Miss rows carry a NULL identity_id and are outside a unique index
    by SQL's NULL semantics — which is what lets face and plate OCR each
    record a failure on the same event (CLD-16)."""
    with claims_db.Session() as session:
        for key in ("face", "vehicle", "person"):
            session.add(
                EventIdentity(
                    event_id=claims_db.event,
                    identity_id=None,
                    identifier_key=key,
                    verdict="missed",
                )
            )
        session.commit()
        misses = session.scalars(
            select(EventIdentity).where(EventIdentity.identity_id.is_(None))
        ).all()
        assert {m.identifier_key for m in misses} == {"face", "vehicle", "person"}


# -- 11-12. the writers ----------------------------------------------------


def test_a_racing_insert_folds_into_the_winner_and_keeps_the_transaction(claims_db):
    """Two writers, one pairing: the loser's insert violates the index,
    and the recovery folds its observation into the row that won instead
    of failing the request.

    The rollback is a savepoint, so the loser's own unrelated pending
    work survives — without that, ingest's long-lived transaction would
    be doomed by the failed statement.
    """
    winner_session = claims_db.Session()
    loser_session = claims_db.Session()
    try:
        assert active_claim(winner_session, claims_db.event, claims_db.identity) is None
        assert active_claim(loser_session, claims_db.event, claims_db.identity) is None

        row, created = insert_claim(
            winner_session,
            EventIdentity(
                event_id=claims_db.event,
                identity_id=claims_db.identity,
                identifier_key="vehicle",
                similarity=0.8,
                matched_by="visual",
            ),
        )
        assert created is True
        winner_session.commit()

        # Unrelated pending work in the losing transaction.
        loser_session.add(
            Identity(
                identifier_key="face",
                class_name="person",
                first_seen=TS,
                last_seen=TS,
            )
        )
        survivor, created = insert_claim(
            loser_session,
            EventIdentity(
                event_id=claims_db.event,
                identity_id=claims_db.identity,
                identifier_key="vehicle",
                similarity=0.95,
                matched_by="plate",
                learned_plate=True,
            ),
        )
        assert created is False
        assert survivor.id == row.id
        loser_session.commit()
    finally:
        winner_session.close()
        loser_session.close()

    with claims_db.Session() as session:
        rows = _active_rows(session, claims_db.event, claims_db.identity)
        assert len(rows) == 1
        assert rows[0].hit_count == 2
        assert rows[0].similarity == 0.95
        assert rows[0].matched_by == "plate"
        assert rows[0].learned_plate is True
        # The savepoint rolled back the conflicting INSERT and nothing else.
        assert session.scalar(select(func.count()).select_from(Identity)) == 3


def test_two_sessions_both_linking_a_pair_yield_one_active_claim(claims_db):
    """The same race through the composed helper the ingest-shaped
    writers call: whatever the interleaving, the pair ends with one
    standing claim carrying both sightings, and only one caller is told
    it created it (which is what keeps the once-per-pairing publish from
    firing twice)."""
    first = claims_db.Session()
    second = claims_db.Session()
    try:
        assert active_claim(first, claims_db.event, claims_db.identity) is None
        assert active_claim(second, claims_db.event, claims_db.identity) is None

        _, created_first = link_claim(
            first,
            event_id=claims_db.event,
            identity_id=claims_db.identity,
            identifier_key="vehicle",
            similarity=0.7,
            matched_by="visual",
        )
        first.commit()
        _, created_second = link_claim(
            second,
            event_id=claims_db.event,
            identity_id=claims_db.identity,
            identifier_key="vehicle",
            similarity=0.6,
            matched_by="visual",
        )
        second.commit()
    finally:
        first.close()
        second.close()

    assert (created_first, created_second) == (True, False)
    with claims_db.Session() as session:
        rows = _active_rows(session, claims_db.event, claims_db.identity)
        assert len(rows) == 1
        assert rows[0].hit_count == 2


def test_link_claim_increments_a_standing_claim_and_never_revives_a_repudiated_one(
    claims_db,
):
    """Today's behaviour, re-pinned through the helper: a later frame
    hit-counts the claim that stands, and a claim an operator unlinked is
    never brought back to life by one (CLD-36) — a fresh row records the
    new evidence instead."""
    with claims_db.Session() as session:
        link, created = link_claim(
            session,
            event_id=claims_db.event,
            identity_id=claims_db.identity,
            identifier_key="vehicle",
            similarity=0.6,
            matched_by=None,
        )
        session.commit()
        assert created is True

        again, created = link_claim(
            session,
            event_id=claims_db.event,
            identity_id=claims_db.identity,
            identifier_key="vehicle",
            similarity=0.9,
            matched_by="visual",
        )
        session.commit()
        assert created is False
        assert again.id == link.id
        assert (again.hit_count, again.similarity, again.matched_by) == (
            2,
            0.9,
            "visual",
        )

        again.unlinked_at = TS
        again.verdict = "wrong"
        session.commit()

        fresh, created = link_claim(
            session,
            event_id=claims_db.event,
            identity_id=claims_db.identity,
            identifier_key="vehicle",
            similarity=0.5,
            matched_by="visual",
        )
        session.commit()
        assert created is True
        assert fresh.id != link.id
        assert fresh.hit_count == 1
        # The repudiation is intact, and there is still one live claim.
        assert session.get(EventIdentity, link.id).unlinked_at == TS
        assert len(_active_rows(session, claims_db.event, claims_db.identity)) == 1
