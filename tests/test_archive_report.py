"""scripts/archive_report.py — the claims it makes about a database.

The script is a diagnostic, so its failure mode is not crashing, it is
stating something untrue confidently. Both tests here are about that:
one pins the auto-verified / human-confirmed split, which was previously
reported as "no human verified anything" on a database where a person had
confirmed 968 faces; the other pins survival on a database predating half
the columns.

Run as a subprocess because that is how it is used — stdlib only, no
venv, against a database the package may not have written.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "archive_report.py"

CURRENT_SCHEMA = """
CREATE TABLE library_sources (id INTEGER PRIMARY KEY, name TEXT, path TEXT,
  kind TEXT, added_at TEXT, last_indexed_at TEXT);
CREATE TABLE library_items (id INTEGER PRIMARY KEY, source_id INTEGER, path TEXT,
  kind TEXT, status TEXT, error TEXT, attempts INT, mtime TEXT, taken_at TEXT,
  width INT, height INT, duration_s REAL, thumb_path TEXT, indexed_at TEXT,
  reviewed INT);
CREATE TABLE annotations (id INTEGER PRIMARY KEY, item_id INTEGER, bbox TEXT,
  class_name TEXT, custom_class TEXT, confidence REAL, identity_id INTEGER,
  proposed_name TEXT, proposal_basis TEXT, source TEXT, verified INT,
  rejected INT, enrolled INT, crop_path TEXT, created_at TEXT);
CREATE TABLE identities (id INTEGER PRIMARY KEY, identifier_key TEXT,
  class_name TEXT, label TEXT, plate TEXT, first_seen TEXT, last_seen TEXT,
  appearance_count INT, vector_count INT, best_crop_path TEXT);
"""

#: An archive.db from before review flags and retry bookkeeping existed.
OLD_SCHEMA = """
CREATE TABLE library_sources (id INTEGER PRIMARY KEY, name TEXT, path TEXT,
  kind TEXT, added_at TEXT, last_indexed_at TEXT);
CREATE TABLE library_items (id INTEGER PRIMARY KEY, source_id INTEGER, path TEXT,
  kind TEXT, status TEXT, mtime TEXT);
CREATE TABLE annotations (id INTEGER PRIMARY KEY, item_id INTEGER, bbox TEXT,
  class_name TEXT, confidence REAL, crop_path TEXT, created_at TEXT);
CREATE TABLE identities (id INTEGER PRIMARY KEY, identifier_key TEXT,
  class_name TEXT, label TEXT, first_seen TEXT, last_seen TEXT);
"""


def run(path: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def annotate(conn, n, basis, verified=0, rejected=0, identity=1):
    for _ in range(n):
        conn.execute(
            "INSERT INTO annotations (item_id, bbox, class_name, identity_id, "
            "proposed_name, proposal_basis, source, verified, rejected, enrolled, "
            "created_at) VALUES (1,'[0,0,1,1]','face',?,?,?,'import',?,?,0,'2026')",
            (identity, "Ana", basis, verified, rejected),
        )


@pytest.fixture
def current(tmp_path):
    path = tmp_path / "a.db"
    conn = sqlite3.connect(path)
    conn.executescript(CURRENT_SCHEMA)
    conn.execute(
        "INSERT INTO library_sources VALUES (1,'t','/x','takeout','2026',NULL)"
    )
    conn.execute(
        "INSERT INTO library_items (source_id,path,kind,status,attempts,mtime) "
        "VALUES (1,'/x/a.jpg','image','pending',0,'2026')"
    )
    conn.execute(
        "INSERT INTO identities (identifier_key,class_name,label,first_seen,"
        "last_seen,vector_count) VALUES ('face','person','Ana','2026','2026',6)"
    )
    # Pass 1 auto-verified by the importer; pass-2 matches a person
    # confirmed by hand, plus some they threw out.
    annotate(conn, 6, "unambiguous", verified=1)
    annotate(conn, 2, "matched", verified=1)
    annotate(conn, 3, "matched", rejected=1)
    annotate(conn, 9, "matched")  # proposed, still unreviewed
    conn.commit()
    conn.close()
    return path


def test_auto_verified_and_human_confirmed_are_reported_separately(current):
    """`source` stays "import" after a human confirms, so a split computed
    from it reported real review as zero. The basis is what distinguishes
    them: the importer auto-verifies pass 1 and nothing else."""
    out = run(current)
    assert "importer auto-verified         6  (75%)" in out
    assert "human-confirmed                2  (25%)" in out
    assert "human-rejected                 3" in out
    # The old, false claim must not come back.
    assert "No annotation in the training set was verified by a person" not in out


def test_the_split_admits_it_is_inferred(current):
    """Nothing records who verified a row; the split is derived from
    importer behaviour and stops being true if that changes."""
    out = run(current)
    assert "nothing stores *who* verified a row" in out.replace("\n  ", " ")


def test_rejected_annotations_stay_out_of_the_training_count(current):
    out = run(current)
    # 6 auto + 2 human = 8 verified; the 3 rejected are not training data.
    assert "8 samples total" in out


def test_an_old_schema_still_produces_a_report(tmp_path):
    """The databases worth pointing this at are the oldest ones."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.execute(
        "INSERT INTO library_sources VALUES (1,'t','/x','takeout','2026',NULL)"
    )
    conn.execute(
        "INSERT INTO library_items (source_id,path,kind,status,mtime) "
        "VALUES (1,'/x/a.jpg','image','pending','2026')"
    )
    conn.commit()
    conn.close()

    out = run(path)
    assert "attempts not in this database" in out
    assert "safe to paste" in out  # ran to the end
    assert "init-db" in out        # and said how to fix it
