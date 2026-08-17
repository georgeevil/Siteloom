"""Match provenance, plate revert, and attributable misses (CLD-16/17 follow-up).

Three properties under guard:
1. How a match was made (plate vs visual) is persisted at ingest — it
   cannot be reconstructed later, so losing it is permanent.
2. A wrong verdict reverts a plate that this very match taught the
   identity, because plate matches win outright and a mis-learned plate
   poisons every future sighting of that number.
3. A recorded miss names the identifier that failed, so per-identifier
   recall is computable.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import (
    Camera,
    Event,
    EventIdentity,
    Identity,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web.app import create_app

TS = datetime(2026, 8, 6, 9, 0, 0)


@pytest.fixture
def webenv(tmp_path):
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/w.db", media_dir=str(tmp_path / "m")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(Camera(id="cam1", site_id="t", name="Cam One"))
        event = Event(
            camera_id="cam1",
            track_id=1,
            class_name="car",
            first_seen=TS,
            last_seen=TS,
            detection_count=1,
        )
        s.add(event)
        # The identity knows a plate only because this event's match
        # taught it (learned_plate on the link).
        identity = Identity(
            identifier_key="vehicle",
            class_name="car",
            plate="KJ-4471",
            first_seen=TS,
            last_seen=TS,
        )
        s.add(identity)
        s.flush()
        s.add(
            EventIdentity(
                event_id=event.id,
                identity_id=identity.id,
                identifier_key="vehicle",
                similarity=0.9,
                matched_by="visual",
                learned_plate=True,
            )
        )
        s.commit()
    return SimpleNamespace(client=TestClient(create_app(config)), Session=Session)


# -- plate revert ----------------------------------------------------------


def test_wrong_verdict_reverts_a_learned_plate(webenv):
    webenv.client.post("/events/1/identity/1/verdict", data={"verdict": "wrong"})
    with webenv.Session() as s:
        assert s.get(Identity, 1).plate is None
        # The judgment itself is kept — negatives are data.
        assert s.get(EventIdentity, 1).verdict == "wrong"


def test_confirmed_verdict_leaves_the_plate_alone(webenv):
    webenv.client.post("/events/1/identity/1/verdict", data={"verdict": "confirmed"})
    with webenv.Session() as s:
        assert s.get(Identity, 1).plate == "KJ-4471"


def test_wrong_verdict_without_learned_plate_keeps_the_plate(webenv):
    """The revert is scoped to the evidence being repudiated: a plate the
    identity knew before this match is not this match's to take away."""
    with webenv.Session() as s:
        s.get(EventIdentity, 1).learned_plate = False
        s.commit()
    webenv.client.post("/events/1/identity/1/verdict", data={"verdict": "wrong"})
    with webenv.Session() as s:
        assert s.get(Identity, 1).plate == "KJ-4471"


# -- attributable misses ---------------------------------------------------


def test_missed_mark_records_which_identifier_failed(webenv):
    webenv.client.post(
        "/events/1/missed", data={"missed": "1", "identifier": "face"}
    )
    with webenv.Session() as s:
        miss = s.scalars(
            select(EventIdentity).filter_by(event_id=1, identity_id=None)
        ).one()
        assert miss.verdict == "missed"
        assert miss.identifier_key == "face"
        # The denormalized mirror stays in step.
        assert s.get(Event, 1).missed_identity is True


def test_two_identifiers_can_miss_the_same_event(webenv):
    """A car pulls up, someone gets out: the face identifier and plate OCR
    can both fail on one event, and they are different failures."""
    webenv.client.post("/events/1/missed", data={"missed": "1", "identifier": "face"})
    webenv.client.post(
        "/events/1/missed", data={"missed": "1", "identifier": "vehicle"}
    )
    with webenv.Session() as s:
        keys = {
            m.identifier_key
            for m in s.scalars(
                select(EventIdentity).filter_by(event_id=1, identity_id=None)
            )
        }
        assert keys == {"face", "vehicle"}


def test_marking_the_same_identifier_twice_is_idempotent(webenv):
    for _ in range(2):
        webenv.client.post(
            "/events/1/missed", data={"missed": "1", "identifier": "face"}
        )
    with webenv.Session() as s:
        misses = list(
            s.scalars(select(EventIdentity).filter_by(event_id=1, identity_id=None))
        )
        assert len(misses) == 1


def test_clearing_removes_miss_rows_and_the_mirror(webenv):
    webenv.client.post("/events/1/missed", data={"missed": "1", "identifier": "face"})
    webenv.client.post("/events/1/missed", data={"missed": "0"})
    with webenv.Session() as s:
        assert (
            s.scalars(
                select(EventIdentity).filter_by(event_id=1, identity_id=None)
            ).first()
            is None
        )
        event = s.get(Event, 1)
        assert event.missed_identity is False and event.missed_at is None


def test_a_missed_event_is_still_unmatched_in_triage(webenv):
    """A miss row is an operator saying nothing was matched — it must not
    make the event look matched to the Unmatched chip."""
    with webenv.Session() as s:
        # Detach the real link so only the miss row remains.
        s.delete(s.get(EventIdentity, 1))
        s.commit()
    webenv.client.post("/events/1/missed", data={"missed": "1", "identifier": "face"})
    page = webenv.client.get("/", params={"unmatched": 1}).text
    assert 'class="row rs-' in page  # the event is listed
    # And it reads flagged, not reviewing, despite the non-null verdict.
    assert "st-chip st-flagged" in page


# -- migration -------------------------------------------------------------


def test_old_databases_are_rebuilt_for_null_identity_rows(tmp_path):
    """A database from before this change has identity_id NOT NULL; init_db
    must rebuild the table, keep the rows, and accept miss rows after."""
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute(
        """CREATE TABLE event_identities (
             id INTEGER PRIMARY KEY,
             event_id INTEGER NOT NULL,
             identity_id INTEGER NOT NULL,
             similarity FLOAT,
             hit_count INTEGER,
             verdict VARCHAR,
             verdict_at DATETIME)"""
    )
    # The real pre-change schema carried these named indexes; a rename
    # keeps their names, which is exactly the collision that bit in the
    # field the first time this migration ran.
    con.execute(
        "CREATE INDEX ix_event_identities_event_id ON event_identities (event_id)"
    )
    con.execute(
        "CREATE INDEX ix_event_identities_identity_id ON event_identities (identity_id)"
    )
    con.execute(
        "INSERT INTO event_identities (id, event_id, identity_id, similarity,"
        " hit_count, verdict) VALUES (7, 1, 3, 0.9, 2, 'confirmed')"
    )
    con.commit()
    con.close()

    engine = make_engine(f"sqlite:///{db}")
    init_db(engine)

    cols = {c["name"]: c for c in inspect(engine).get_columns("event_identities")}
    assert cols["identity_id"]["nullable"] is True
    assert "matched_by" in cols and "learned_plate" in cols

    Session = get_session(engine)
    with Session() as s:
        old = s.get(EventIdentity, 7)
        assert (old.event_id, old.identity_id, old.verdict) == (1, 3, "confirmed")
        assert old.hit_count == 2
        assert old.learned_plate is False  # backfilled, not NULL
        s.add(EventIdentity(event_id=1, identity_id=None, verdict="missed"))
        s.commit()  # would raise IntegrityError without the rebuild


def test_interrupted_rebuild_is_completed_not_lost(tmp_path):
    """pysqlite autocommits DDL, so the rebuild is not atomic. A crash
    between rename and copy leaves the data in event_identities_old —
    the next init_db must finish the job, and the recovered table must
    end up with its declared indexes."""
    db = tmp_path / "interrupted.db"
    con = sqlite3.connect(db)
    # The state an interrupted run leaves behind: renamed old table with
    # the data (and the stale-named index), new table created empty.
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
    con.execute(
        "CREATE INDEX ix_event_identities_identity_id"
        " ON event_identities_old (identity_id)"
    )
    con.execute(
        "INSERT INTO event_identities_old (id, event_id, identity_id,"
        " similarity, hit_count) VALUES (7, 1, 3, 0.9, 2)"
    )
    con.commit()
    con.close()

    engine = make_engine(f"sqlite:///{db}")
    init_db(engine)

    inspector = inspect(engine)
    assert "event_identities_old" not in inspector.get_table_names()
    index_names = {i["name"] for i in inspector.get_indexes("event_identities")}
    assert "ix_event_identities_identity_id" in index_names
    Session = get_session(engine)
    with Session() as s:
        row = s.get(EventIdentity, 7)
        assert (row.event_id, row.identity_id) == (1, 3)


def test_rebuild_does_not_run_twice(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/fresh.db")
    init_db(engine)
    init_db(engine)  # idempotent — second pass must not touch the table
    assert (
        inspect(engine).get_columns("event_identities")[0]["name"] == "id"
    )


def test_a_database_without_plate_source_gains_it_as_unknown(tmp_path):
    """`Identity.plate_source` (CLD-134) is nullable with no default, so
    it is exactly the additive case the column pass covers — no migration
    code, and every existing plate arrives as *unknown provenance*.

    NULL is the honest answer there, and a real fourth state rather than
    a missing one: the database cannot say whether a plate predating the
    column was read at mint, learned later, or typed by someone. It is
    also the safe one — only "operator" locks a plate against re-learning,
    so a migrated row keeps behaving exactly as it did.
    """
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/legacy.db", media_dir=str(tmp_path / "m")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(
            Identity(
                identifier_key="vehicle",
                class_name="car",
                label="the gray van",
                plate="KJ4471",
                first_seen=TS,
                last_seen=TS,
            )
        )
        s.commit()
        identity_id = s.query(Identity).one().id

    # Roll the schema back to before the column existed, rather than
    # hand-listing every other column of a table three issues are adding
    # to.
    con = sqlite3.connect(tmp_path / "legacy.db")
    con.execute("ALTER TABLE identities DROP COLUMN plate_source")
    con.commit()
    con.close()
    assert "plate_source" not in {
        c["name"] for c in inspect(engine).get_columns("identities")
    }

    init_db(engine)

    assert "plate_source" in {
        c["name"] for c in inspect(engine).get_columns("identities")
    }
    with Session() as s:
        identity = s.get(Identity, identity_id)
        assert identity.plate == "KJ4471"  # the plate itself survives
        assert identity.plate_source is None

    page = TestClient(create_app(config)).get(f"/identities/{identity_id}").text
    assert "recorded before provenance was tracked" in page


def test_a_database_without_cover_locked_gains_it_as_false(tmp_path):
    """`Identity.cover_locked` (CLD-137) is a Boolean with a scalar
    default, which is the one shape the additive column pass emits a DDL
    DEFAULT for — so existing rows arrive `False`, not NULL.

    That matters beyond tidiness: the flag is read as "an operator chose
    this cover", and NULL would be falsy in Python but would make the
    column's meaning depend on which pass wrote the row. False is the
    honest value — nobody has chosen a cover on a database that had no
    way to say so.
    """
    db = tmp_path / "legacy.db"
    engine = make_engine(f"sqlite:///{db}")
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(
            Identity(
                identifier_key="vehicle",
                class_name="car",
                best_crop_path="cam1/crop.jpg",
                first_seen=TS,
                last_seen=TS,
            )
        )
        s.commit()
        identity_id = s.query(Identity).one().id

    con = sqlite3.connect(db)
    con.execute("ALTER TABLE identities DROP COLUMN cover_locked")
    con.commit()
    con.close()
    assert "cover_locked" not in {
        c["name"] for c in inspect(engine).get_columns("identities")
    }

    init_db(engine)

    with Session() as s:
        identity = s.get(Identity, identity_id)
        assert identity.cover_locked is False  # backfilled, not NULL
        assert identity.best_crop_path == "cam1/crop.jpg"  # untouched
