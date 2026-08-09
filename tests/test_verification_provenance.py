"""Who verified an annotation, and why the row has to remember (CLD-95).

`Annotation.verified` is written by two things that mean very different
concepts: a person clicking confirm, and the Takeout importer signing off
its own one-face-one-tag matches with nobody looking. `training/dataset.py`
reads both as ground truth, and `source` cannot separate them — it is the
provenance of the *box*, and no reviewer ever rewrites it, so a human
confirming an imported annotation stays "import" forever.

The tests here pin the four things that fixes:

* a human confirming an imported annotation ends up "human" — the exact
  transition the old schema lost;
* the one-time backfill attributes a pre-column database and then leaves
  it alone, because the `proposal_basis` inference is a migration and not
  a permanent query;
* human-only training data is an *option* and the default set is
  unchanged;
* the archive report reads the column when it is there and says when it
  is only inferring.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from siteloom.config import CameraConfig, IdentityConfig, SiteConfig, StorageConfig
from siteloom.store import (
    VERIFIED_BY_HUMAN,
    VERIFIED_BY_IMPORT,
    Annotation,
    LibraryItem,
    LibrarySource,
    get_session,
    init_db,
    make_engine,
)
from siteloom.training.dataset import (
    collect_face_samples,
    describe_provenance,
    sample_provenance,
)
from siteloom.web.app import create_app

TS = datetime(2026, 8, 9, 12, 0)


class StubEmbedder:
    """Stands in for SFace so the review endpoint needs no ONNX weights."""

    dim = 4

    def embed(self, bgr):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class StubStore:
    """A vector store that records adds and nothing else."""

    def __init__(self):
        self.added = []

    def add(self, *args, **kwargs):
        self.added.append((args, kwargs))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "siteloom.identity.embedders.FaceEmbedder",
        lambda *a, **k: StubEmbedder(),
    )
    monkeypatch.setattr(
        "siteloom.identity.get_shared_store", lambda *a, **k: StubStore()
    )
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="c", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/w.db", media_dir=str(tmp_path / "m")
        ),
        identity=IdentityConfig(vector_db_path=str(tmp_path / "vectors")),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    # A real file on disk: collect_face_samples drops annotations whose
    # source image has gone missing, so the readiness page would see none.
    import cv2

    cv2.imwrite(str(tmp_path / "a.jpg"), np.full((40, 40, 3), 80, dtype=np.uint8))
    with Session() as session:
        session.add(
            LibrarySource(id=1, path=str(tmp_path), name="archive", added_at=TS)
        )
        session.add(
            LibraryItem(
                id=1,
                source_id=1,
                path=str(tmp_path / "a.jpg"),
                kind="image",
                status="indexed",
                mtime=TS,
            )
        )
        session.commit()
    return SimpleNamespace(
        client=TestClient(create_app(config)),
        Session=Session,
        config=config,
        tmp_path=tmp_path,
    )


def add_annotation(env, **kwargs) -> int:
    """One face annotation, imported and auto-verified unless told otherwise."""
    fields = dict(
        item_id=1,
        bbox="[0,0,1,1]",
        class_name="face",
        confidence=0.9,
        proposed_name="Ana",
        proposal_basis="unambiguous",
        source="import",
        verified=True,
        verified_by=VERIFIED_BY_IMPORT,
        verified_at=TS,
        created_at=TS,
    )
    fields.update(kwargs)
    with env.Session() as session:
        annotation = Annotation(**fields)
        session.add(annotation)
        session.commit()
        return annotation.id


def review(env, **decision):
    response = env.client.post(
        "/api/training/review", json={"decisions": [decision]}
    )
    assert response.status_code == 200, response.text
    return response.json()


def get(env, annotation_id: int) -> Annotation:
    with env.Session() as session:
        return session.get(Annotation, annotation_id)


# -- the transition the old schema lost ------------------------------------


def test_a_human_confirming_an_imported_annotation_is_recorded_as_human(env):
    """The case `source` cannot express. The row arrives auto-verified by
    the importer and already `verified`, so nothing about `verified`,
    `source` or `proposal_basis` changes when a person confirms it — only
    `verified_by` does, and without it the sign-off is invisible."""
    annotation_id = add_annotation(env)
    assert get(env, annotation_id).verified_by == VERIFIED_BY_IMPORT

    review(env, id=annotation_id, action="confirm", name="Ana")

    row = get(env, annotation_id)
    assert row.verified is True
    assert row.verified_by == VERIFIED_BY_HUMAN
    assert row.verified_at is not None
    # Untouched, and that is the point: they describe the box, not the
    # review, so they cannot answer "did a person sign this off".
    assert row.source == "import"
    assert row.proposal_basis == "unambiguous"


def test_confirming_stamps_a_time_as_well_as_a_verifier(env):
    annotation_id = add_annotation(env, verified=False, verified_by=None,
                                   verified_at=None)
    review(env, id=annotation_id, action="confirm", name="Ana")
    row = get(env, annotation_id)
    assert row.verified_by == VERIFIED_BY_HUMAN
    assert row.verified_at is not None and row.verified_at != TS


def test_rejecting_takes_the_sign_off_off_with_it(env):
    """Rejection is a human act, but it is not a verification. `rejected`
    already carries the human part — no importer ever rejects — so the
    verifier is cleared rather than left behind on a row whose `verified`
    is False."""
    annotation_id = add_annotation(env)
    review(env, id=annotation_id, action="reject")

    row = get(env, annotation_id)
    assert row.rejected is True
    assert row.verified is False
    assert row.verified_by is None and row.verified_at is None


def test_unsetting_returns_the_row_to_unattributed(env):
    annotation_id = add_annotation(env)
    review(env, id=annotation_id, action="unset")

    row = get(env, annotation_id)
    assert (row.verified, row.rejected) == (False, False)
    assert row.verified_by is None


def test_classifying_a_crop_is_human_sign_off_too(env):
    """`classify` verifies the row on a different axis (a custom class),
    through the same column — so it records a verifier like any other."""
    with env.Session() as session:
        session.execute(
            text(
                "INSERT INTO custom_classes (name, parent_class, description, "
                "threshold, example_count, created_at) "
                "VALUES ('delivery-van', 'car', '', 0.85, 0, '2026-08-09')"
            )
        )
        session.commit()
    annotation_id = add_annotation(
        env,
        class_name="car",
        proposed_name=None,
        proposal_basis=None,
        source="auto",
        verified=False,
        verified_by=None,
        verified_at=None,
        crop_path=None,
    )
    review(env, id=annotation_id, action="classify", custom_class="delivery-van")

    row = get(env, annotation_id)
    assert row.custom_class == "delivery-van"
    assert row.verified_by == VERIFIED_BY_HUMAN


def test_saving_the_box_editor_does_not_relabel_the_importers_work(env):
    """The editor replaces an item's boxes wholesale, so it re-sends
    `verified` for rows nobody touched. Stamping on every save would turn
    every auto-verified row into human sign-off because somebody opened
    the item and pressed save."""
    annotation_id = add_annotation(env)
    response = env.client.post(
        "/api/items/1/annotations",
        json={
            "annotations": [
                {
                    "id": annotation_id,
                    "bbox": [0, 0, 1, 1],
                    "class_name": "face",
                    "verified": True,
                }
            ]
        },
    )
    assert response.status_code == 200
    assert get(env, annotation_id).verified_by == VERIFIED_BY_IMPORT


def test_ticking_verified_in_the_box_editor_is_an_explicit_human_act(env):
    annotation_id = add_annotation(
        env, verified=False, verified_by=None, verified_at=None
    )
    env.client.post(
        "/api/items/1/annotations",
        json={
            "annotations": [
                {
                    "id": annotation_id,
                    "bbox": [0, 0, 1, 1],
                    "class_name": "face",
                    "verified": True,
                }
            ]
        },
    )
    assert get(env, annotation_id).verified_by == VERIFIED_BY_HUMAN


def test_an_unknown_verifier_is_refused(env):
    """The vocabulary is closed. A typo that silently became a fourth
    verifier would split every count that reads the column."""
    with pytest.raises(ValueError):
        Annotation(item_id=1, bbox="[0,0,1,1]", class_name="face",
                   created_at=TS).mark_verified("operator")


# -- the one-time backfill -------------------------------------------------


OLD_ANNOTATIONS = """
CREATE TABLE annotations (
  id INTEGER NOT NULL PRIMARY KEY,
  item_id INTEGER,
  frame_index INTEGER,
  bbox TEXT NOT NULL,
  class_name VARCHAR NOT NULL,
  custom_class VARCHAR,
  confidence FLOAT,
  identity_id INTEGER,
  proposed_name VARCHAR,
  proposal_basis VARCHAR,
  source VARCHAR NOT NULL,
  verified BOOLEAN NOT NULL,
  rejected BOOLEAN NOT NULL,
  enrolled BOOLEAN NOT NULL,
  crop_path VARCHAR,
  created_at DATETIME NOT NULL
)
"""

#: (proposal_basis, verified, rejected) for a database written before the
#: column existed — the shape of the real archive in miniature.
PRE_COLUMN_ROWS = [
    ("unambiguous", 1, 0),  # importer pass 1: nobody looked
    ("unambiguous", 1, 0),
    ("matched", 1, 0),      # pass 2 proposal, confirmed by a person
    ("matched", 0, 1),      # thrown out by a person
    ("matched", 0, 0),      # still awaiting review
    (None, 1, 0),           # a hand-drawn box someone verified
]


def make_pre_column_db(tmp_path) -> Path:
    db = tmp_path / "old.db"
    engine = make_engine(f"sqlite:///{db}")
    with engine.begin() as conn:
        conn.execute(text(OLD_ANNOTATIONS))
        for basis, verified, rejected in PRE_COLUMN_ROWS:
            conn.execute(
                text(
                    "INSERT INTO annotations (item_id, frame_index, bbox, "
                    "class_name, confidence, proposal_basis, source, verified, "
                    "rejected, enrolled, created_at) VALUES (1, 0, '[0,0,1,1]', "
                    "'face', 0.9, :basis, 'import', :verified, :rejected, 0, "
                    "'2026-08-01 00:00:00')"
                ),
                {"basis": basis, "verified": verified, "rejected": rejected},
            )
    engine.dispose()
    return db


def test_the_backfill_attributes_a_pre_column_database(tmp_path):
    """`_ensure_columns` emits a DDL DEFAULT only for a non-nullable column
    with a scalar default, and `verified_by` is neither — it arrives NULL
    everywhere. The attribution is a separate, deliberate migration."""
    db = make_pre_column_db(tmp_path)
    engine = make_engine(f"sqlite:///{db}")
    init_db(engine)

    with get_session(engine)() as session:
        rows = session.scalars(select(Annotation).order_by(Annotation.id)).all()
        by_id = {r.id: r for r in rows}
    # Pass 1 is the only auto-verifying path there has ever been, so a
    # verified "unambiguous" row is the importer and everything else
    # verified was a person.
    assert by_id[1].verified_by == VERIFIED_BY_IMPORT
    assert by_id[2].verified_by == VERIFIED_BY_IMPORT
    assert by_id[3].verified_by == VERIFIED_BY_HUMAN
    assert by_id[6].verified_by == VERIFIED_BY_HUMAN
    # Nothing unverified acquires a verifier — including the rejected row,
    # whose human act is recorded by `rejected` itself.
    assert by_id[4].verified_by is None
    assert by_id[5].verified_by is None
    # The time was never recorded, and NULL says "unknown" where an
    # invented timestamp would say something false.
    assert all(r.verified_at is None for r in rows)


def test_the_backfill_records_rather_than_recomputes(tmp_path):
    """The `proposal_basis` inference is legitimate once, as a migration.
    Re-deriving it on every init_db would overwrite the recorded answer —
    so a person confirming an "unambiguous" row must not be demoted back
    to the importer the next time the database is opened."""
    db = make_pre_column_db(tmp_path)
    engine = make_engine(f"sqlite:///{db}")
    init_db(engine)

    Session = get_session(engine)
    with Session() as session:
        row = session.get(Annotation, 1)
        row.mark_verified(VERIFIED_BY_HUMAN, TS)
        session.commit()

    init_db(engine)  # a later run: serve, another CLI command, anything

    with Session() as session:
        assert session.get(Annotation, 1).verified_by == VERIFIED_BY_HUMAN


def test_a_fresh_database_needs_no_backfill(tmp_path):
    """create_all writes the column, so nothing was added and nothing is
    inferred — a new database has no past to reconstruct."""
    engine = make_engine(f"sqlite:///{tmp_path}/fresh.db")
    init_db(engine)
    with get_session(engine)() as session:
        session.add(
            Annotation(
                item_id=1,
                bbox="[0,0,1,1]",
                class_name="face",
                proposal_basis="unambiguous",
                verified=True,
                created_at=TS,
            )
        )
        session.commit()
    init_db(engine)
    with get_session(engine)() as session:
        # Written directly rather than through mark_verified, so it stays
        # unattributed: the migration is not a repair pass.
        assert session.scalars(select(Annotation)).one().verified_by is None


# -- what it unlocks: a human-only training set ----------------------------


@pytest.fixture
def dataset_env(tmp_path):
    """A library whose face annotations are half machine-verified."""
    image = tmp_path / "a.jpg"
    import cv2

    cv2.imwrite(str(image), np.full((40, 40, 3), 80, dtype=np.uint8))
    engine = make_engine(f"sqlite:///{tmp_path}/d.db")
    init_db(engine)
    Session = get_session(engine)
    with Session() as session:
        session.add(LibrarySource(id=1, path=str(tmp_path), name="a", added_at=TS))
        session.add(
            LibraryItem(
                id=1, source_id=1, path=str(image), kind="image",
                status="indexed", mtime=TS,
            )
        )
        session.flush()
        for person, verifier, n in (
            ("Ana", VERIFIED_BY_HUMAN, 2),
            ("Ana", VERIFIED_BY_IMPORT, 3),
            ("Bo", VERIFIED_BY_IMPORT, 2),
            ("Cal", None, 1),  # verified before the column existed
        ):
            for _ in range(n):
                session.add(
                    Annotation(
                        item_id=1,
                        bbox="[0,0,1,1]",
                        class_name="face",
                        proposed_name=person,
                        source="import",
                        verified=True,
                        verified_by=verifier,
                        verified_at=TS if verifier else None,
                        created_at=TS,
                    )
                )
        session.commit()
    return Session


def test_the_default_training_set_is_unchanged(dataset_env):
    """Adding a provenance dimension must not quietly change what counts
    as training data: every verified, non-rejected annotation still does."""
    with dataset_env() as session:
        samples = collect_face_samples(session)
    assert len(samples) == 8
    assert sample_provenance(samples) == {"import": 5, "human": 2, "unattributed": 1}


def test_human_only_is_an_option_not_the_default(dataset_env):
    with dataset_env() as session:
        samples = collect_face_samples(session, human_only=True)
    assert len(samples) == 2
    assert {s.person for s in samples} == {"Ana"}
    assert all(s.verified_by == VERIFIED_BY_HUMAN for s in samples)


def test_an_unattributed_row_is_not_evidence_of_human_sign_off(dataset_env):
    """"We do not know who verified this" is not "a person did"."""
    with dataset_env() as session:
        samples = collect_face_samples(session, human_only=True)
    assert "Cal" not in {s.person for s in samples}


def test_a_run_can_state_what_it_trained_on(dataset_env):
    with dataset_env() as session:
        samples = collect_face_samples(session)
    described = describe_provenance(samples)
    assert "5 import-verified" in described
    assert "2 human-verified" in described


def test_the_models_screen_shows_both_readings(env):
    """A page that printed only "8 verified samples" would let a floor
    cleared by machine agreement read as a floor cleared by review."""
    for verifier in [VERIFIED_BY_IMPORT] * 3 + [VERIFIED_BY_HUMAN]:
        add_annotation(env, verified_by=verifier)
    html = env.client.get("/train").text
    assert "human-confirmed" in html
    assert "without anyone looking" in html
    # The count, not just the caption: 3 of the 4 samples are the
    # importer's own.
    assert "3 of these samples" in html
    assert "--human-only" in html


def test_the_floor_is_reported_twice(dataset_env):
    """`train status` and /train both read this: 2 people clear a
    2-sample floor, but only one of them does on human sign-off."""
    from siteloom.web import train_routes

    with dataset_env() as session:
        readiness = train_routes.face_readiness(session, 2, 0.25)
    assert readiness.samples == 8 and readiness.ready_people == 2
    assert readiness.human_samples == 2 and readiness.human_ready_people == 1
    assert readiness.provenance["import"] == 5


# -- the archive report ----------------------------------------------------

SCRIPT = Path(__file__).parent.parent / "scripts" / "archive_report.py"

NEW_SCHEMA = """
CREATE TABLE library_sources (id INTEGER PRIMARY KEY, name TEXT, path TEXT,
  kind TEXT, added_at TEXT, last_indexed_at TEXT);
CREATE TABLE library_items (id INTEGER PRIMARY KEY, source_id INTEGER, path TEXT,
  kind TEXT, status TEXT, error TEXT, attempts INT, mtime TEXT, taken_at TEXT,
  width INT, height INT, duration_s REAL, thumb_path TEXT, indexed_at TEXT,
  reviewed INT);
CREATE TABLE annotations (id INTEGER PRIMARY KEY, item_id INTEGER, bbox TEXT,
  class_name TEXT, custom_class TEXT, confidence REAL, identity_id INTEGER,
  proposed_name TEXT, proposal_basis TEXT, source TEXT, verified INT,
  verified_by TEXT, verified_at TEXT, rejected INT, enrolled INT,
  crop_path TEXT, created_at TEXT);
CREATE TABLE identities (id INTEGER PRIMARY KEY, identifier_key TEXT,
  class_name TEXT, label TEXT, plate TEXT, first_seen TEXT, last_seen TEXT,
  appearance_count INT, vector_count INT, best_crop_path TEXT);
"""


def report(path: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def make_reported_db(tmp_path, rows) -> Path:
    path = tmp_path / "report.db"
    conn = sqlite3.connect(path)
    conn.executescript(NEW_SCHEMA)
    conn.execute(
        "INSERT INTO library_sources VALUES (1,'t','/x','takeout','2026',NULL)"
    )
    conn.execute(
        "INSERT INTO library_items (source_id,path,kind,status,attempts,mtime) "
        "VALUES (1,'/x/a.jpg','image','pending',0,'2026')"
    )
    for basis, verifier, verified, rejected, n in rows:
        for _ in range(n):
            conn.execute(
                "INSERT INTO annotations (item_id, bbox, class_name, identity_id, "
                "proposed_name, proposal_basis, source, verified, verified_by, "
                "rejected, enrolled, created_at) VALUES "
                "(1,'[0,0,1,1]','face',1,'Ana',?,'import',?,?,?,0,'2026')",
                (basis, verified, verifier, rejected),
            )
    conn.commit()
    conn.close()
    return path


def test_the_report_reads_the_column_instead_of_guessing(tmp_path):
    """The whole point of the column: the same split, but measured. The
    basis no longer decides it, so a row the importer proposed and a
    person confirmed counts as human even though its basis says
    "unambiguous"."""
    path = make_reported_db(
        tmp_path,
        [
            ("unambiguous", "import", 1, 0, 6),
            ("unambiguous", "human", 1, 0, 2),  # confirmed after import
            ("matched", None, 0, 1, 3),
        ],
    )
    out = report(path)
    assert "importer auto-verified         6  (75%)" in out
    assert "human-confirmed                2  (25%)" in out
    assert "human-rejected                 3" in out
    assert "Read from annotations.verified_by" in out
    # The inference disclaimer belongs only to databases that need it.
    assert "nothing stores *who* verified a row" not in out.replace("\n  ", " ")


def test_the_report_names_rows_it_cannot_attribute(tmp_path):
    """A database with the column but no migration behind it. Unknown is
    its own bucket — folding it into either side would invent a fact."""
    path = make_reported_db(
        tmp_path,
        [("unambiguous", "import", 1, 0, 2), ("matched", None, 1, 0, 2)],
    )
    out = report(path)
    assert "unattributed" in out
    assert "siteloom init-db" in out


def test_an_old_database_still_gets_the_inferred_split(tmp_path):
    """Degrading gracefully is the script's job — it runs against
    databases the package may not have written."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(NEW_SCHEMA.replace("verified_by TEXT, verified_at TEXT,", ""))
    conn.execute(
        "INSERT INTO library_sources VALUES (1,'t','/x','takeout','2026',NULL)"
    )
    conn.execute(
        "INSERT INTO library_items (source_id,path,kind,status,attempts,mtime) "
        "VALUES (1,'/x/a.jpg','image','pending',0,'2026')"
    )
    for basis, n in (("unambiguous", 6), ("matched", 2)):
        for _ in range(n):
            conn.execute(
                "INSERT INTO annotations (item_id, bbox, class_name, identity_id, "
                "proposed_name, proposal_basis, source, verified, rejected, "
                "enrolled, created_at) VALUES "
                "(1,'[0,0,1,1]','face',1,'Ana',?,'import',1,0,0,'2026')",
                (basis,),
            )
    conn.commit()
    conn.close()

    out = report(path)
    assert "importer auto-verified         6  (75%)" in out
    assert "nothing stores *who* verified a row" in out.replace("\n  ", " ")
    assert "Read from annotations.verified_by" not in out
