#!/usr/bin/env python3
"""Read-only aggregate report over a Siteloom database.

Written for the access problem: the real archive lives on one machine and
the people reasoning about it often do not. This prints the *shape* of a
database — counts, distributions, gates — and deliberately never prints a
name, a path, or a box. The output is safe to paste into an issue.

Stdlib only, so it runs against any system Python 3 with no venv and no
install. Opens the file read-only; it cannot modify the database even if
something else holds it open.

    python3 scripts/archive_report.py ~/dev/Siteloom-data/archive.db

The one question it exists to answer directly is the CLD-10 gate: how many
people have cleared the 5-verified-face-sample floor that fine-tuning
needs. That is a labelling-volume fact, and it is invisible from anywhere
except this file.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# training/dataset.py counts only verified, non-rejected face annotations;
# config.py's default floor is 5 per person.
FACE_SAMPLE_FLOOR = 5


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        sys.exit(f"no such database: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r["name"] for r in rows}


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Which columns this database actually has.

    A long-lived archive predates whatever was added since it was
    written, and this script is most useful precisely on the oldest
    database anyone still has. Every optional column is checked rather
    than assumed.
    """
    try:
        return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


#: Printed once if anything was skipped. `init_db` runs `_ensure_columns`,
#: an additive ALTER pass over every model table, so this is the whole fix
#: — and it is safe, because additive is the only schema change the
#: project makes.
MIGRATE_HINT = (
    "Some columns are missing because this database predates them.\n"
    "  `siteloom init-db --config <your-archive.yaml>` adds them in place\n"
    "  (additive only — no data is rewritten), then re-run this report."
)

_skipped: list[str] = []


def note_missing(table: str, column: str) -> None:
    _skipped.append(f"{table}.{column}")
    print(f"      ({table}.{column} not in this database — section skipped)")


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def counts(conn: sqlite3.Connection, table: str, column: str) -> list[sqlite3.Row]:
    return conn.execute(
        f"SELECT {column} AS k, COUNT(*) AS n FROM {table} "
        f"GROUP BY {column} ORDER BY n DESC"
    ).fetchall()


def show_counts(conn: sqlite3.Connection, table: str, column: str, label: str) -> None:
    rows = counts(conn, table, column)
    total = sum(r["n"] for r in rows)
    print(f"  {label}: {total} total")
    for r in rows:
        key = r["k"] if r["k"] is not None else "(null)"
        print(f"      {str(key)[:40]:<40} {r['n']:>8}")


def report_sources(conn: sqlite3.Connection, present: set[str]) -> None:
    if "library_sources" not in present:
        return
    rule("Library sources")
    rows = conn.execute(
        "SELECT id, kind, added_at, last_indexed_at FROM library_sources ORDER BY id"
    ).fetchall()
    if not rows:
        print("  none registered")
        return
    # Path and name omitted on purpose — a directory layout is personal.
    for r in rows:
        print(
            f"  source {r['id']}: kind={r['kind']} added={r['added_at']} "
            f"last_indexed={r['last_indexed_at'] or 'never'}"
        )


def report_items(conn: sqlite3.Connection, present: set[str]) -> None:
    if "library_items" not in present:
        return
    rule("Library items")
    show_counts(conn, "library_items", "status", "by status")
    show_counts(conn, "library_items", "kind", "by kind")

    # `failed` is not `pending` — nothing picks a failed item up again, so
    # the two must never be read as one number.
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM library_items WHERE status='failed'"
    ).fetchone()
    if row["n"]:
        print(f"\n  {row['n']} failed — these are NOT retried without --retry-failed.")
        if "error" in columns(conn, "library_items"):
            errors = conn.execute(
                "SELECT substr(error, 1, 60) AS e, COUNT(*) AS n FROM library_items "
                "WHERE status='failed' AND error IS NOT NULL "
                "GROUP BY e ORDER BY n DESC LIMIT 10"
            ).fetchall()
            for r in errors:
                print(f"      {r['n']:>6}  {r['e']}")

    cols = columns(conn, "library_items")

    if "attempts" in cols:
        attempts = conn.execute(
            "SELECT attempts, COUNT(*) AS n FROM library_items "
            "GROUP BY attempts ORDER BY attempts"
        ).fetchall()
        if len(attempts) > 1:
            print("\n  attempts distribution:")
            for r in attempts:
                print(f"      tried {r['attempts']}x: {r['n']}")
    else:
        note_missing("library_items", "attempts")

    if "taken_at" in cols:
        span = conn.execute(
            "SELECT MIN(taken_at) AS lo, MAX(taken_at) AS hi, "
            "COUNT(taken_at) AS n FROM library_items"
        ).fetchone()
        if span["n"]:
            print(f"\n  capture times: {span['n']} known, {span['lo']} .. {span['hi']}")
    else:
        note_missing("library_items", "taken_at")


def report_annotations(conn: sqlite3.Connection, present: set[str]) -> None:
    if "annotations" not in present:
        return
    rule("Annotations")
    cols = columns(conn, "annotations")
    show_counts(conn, "annotations", "class_name", "by class")
    if "source" in cols:
        show_counts(conn, "annotations", "source", "by provenance")

    flags = [c for c in ("verified", "rejected", "enrolled") if c in cols]
    if flags:
        selects = ", ".join(f"SUM({c}) AS {c}" for c in flags)
        row = conn.execute(
            f"SELECT {selects}, COUNT(*) AS total FROM annotations"
        ).fetchone()
        summary = "  ".join(f"{c}={row[c] or 0}" for c in flags)
        print(f"\n  {summary}  of {row['total']}")
        for missing in ("verified", "rejected", "enrolled"):
            if missing not in cols:
                note_missing("annotations", missing)

    # Proposals: how many, on what basis, over how many distinct names.
    # The names themselves stay on the machine that owns them.
    if "proposed_name" in cols:
        prop = conn.execute(
            "SELECT COUNT(*) AS n, COUNT(DISTINCT proposed_name) AS names "
            "FROM annotations WHERE proposed_name IS NOT NULL"
        ).fetchone()
        if prop["n"]:
            print(
                f"  {prop['n']} proposals over {prop['names']} distinct names "
                f"(names not printed)"
            )
            if "proposal_basis" in cols:
                for r in counts(conn, "annotations", "proposal_basis"):
                    if r["k"] is not None:
                        print(f"      basis {r['k']:<20} {r['n']:>8}")
    else:
        note_missing("annotations", "proposed_name")


def report_training_gate(conn: sqlite3.Connection, present: set[str]) -> None:
    """The CLD-10 gate, as a distribution rather than a roster."""
    if "annotations" not in present:
        return
    rule(f"Face training readiness (floor: {FACE_SAMPLE_FLOOR} verified per person)")
    cols = columns(conn, "annotations")
    if not {"verified", "rejected"} <= cols:
        print("  cannot be computed: this database has no verified/rejected flags,")
        print("  and unverified annotations are never training data.")
        return

    # Verified, non-rejected, face-class annotations grouped per person.
    #
    # The two rows must not overlap. An annotation the importer verified
    # carries BOTH identity_id and proposed_name, so a proposals query
    # that does not exclude identity_id counts the same rows twice and
    # prints two identical lines — one of them labelled "unconfirmed"
    # about rows that are confirmed.
    queries = (
        (
            "confirmed identities",
            "identity_id",
            "identity_id IS NOT NULL",
        ),
        (
            "proposals only, no identity yet",
            "proposed_name",
            "proposed_name IS NOT NULL AND identity_id IS NULL",
        ),
    )
    for caption, column, where in queries:
        if column not in cols:
            note_missing("annotations", column)
            continue
        rows = conn.execute(
            f"SELECT {column} AS who, COUNT(*) AS n FROM annotations "
            f"WHERE {where} AND class_name='face' "
            f"AND verified=1 AND rejected=0 GROUP BY {column}"
        ).fetchall()
        if not rows:
            print(f"  {caption}: none")
            continue
        clearing = [r for r in rows if r["n"] >= FACE_SAMPLE_FLOOR]
        noun = "person" if len(rows) == 1 else "people"
        print(
            f"  {caption}: {len(rows)} {noun}, {len(clearing)} clear the floor, "
            f"{sum(r['n'] for r in rows)} samples total"
        )
        if clearing:
            sizes = sorted((r["n"] for r in clearing), reverse=True)
            print(f"      sample counts, largest first: {sizes[:20]}")

    # Who did the verifying. This is the number that decides whether the
    # floor means anything: `verified` is set both by a human clicking
    # confirm and by the Takeout importer's auto-verify pass, and
    # training/dataset.py cannot tell them apart. A floor cleared entirely
    # by machine agreement is a floor cleared by Google's people-tags
    # agreeing with a face detector — which may well be fine, but it is
    # not human sign-off and should never be mistaken for it.
    if "proposal_basis" in cols and "identity_id" in cols:
        rule("Who verified the training data")
        rows = conn.execute(
            "SELECT COALESCE(proposal_basis, '') AS basis, COUNT(*) AS n "
            "FROM annotations WHERE identity_id IS NOT NULL AND class_name='face' "
            "AND verified=1 AND rejected=0 GROUP BY basis"
        ).fetchall()
        by_basis = {r["basis"]: r["n"] for r in rows}
        total = sum(by_basis.values())
        # Pass 1 is the only thing the importer auto-verifies
        # (`verified=certain and self.auto_verify_unambiguous`). Anything
        # else carrying verified=1 was confirmed by a person in the UI,
        # which never rewrites `source`.
        auto = by_basis.get("unambiguous", 0)
        human = total - auto
        if total:
            print(f"      importer auto-verified  {auto:>8}  ({auto / total * 100:.0f}%)")
            print(f"      human-confirmed         {human:>8}  ({human / total * 100:.0f}%)")
        rejected = conn.execute(
            "SELECT COUNT(*) AS n FROM annotations "
            "WHERE class_name='face' AND rejected=1"
        ).fetchone()["n"]
        if rejected:
            print(f"      human-rejected          {rejected:>8}  (excluded from training)")
        if auto:
            print(
                f"\n  {auto} sample{'' if auto == 1 else 's'} cleared pass 1 — one face,\n"
                "  one Google people-tag — and were auto-verified without anyone\n"
                "  looking. training/dataset.py reads them as ground truth exactly\n"
                "  like the human-confirmed ones."
            )
        # The split above is inferred, not recorded, and saying so matters:
        # it holds only while pass 1 is the sole auto-verifying path.
        print(
            "\n  Note: nothing stores *who* verified a row. The split above is\n"
            "  inferred from proposal_basis — `source` stays \"import\" after a\n"
            "  human confirms, so it cannot answer this."
        )

    print(
        "\n  Only the confirmed-identity row is training data — training/dataset.py\n"
        "  reads verified, non-rejected annotations and never reads proposals."
    )


def report_identities(conn: sqlite3.Connection, present: set[str]) -> None:
    if "identities" not in present:
        return
    rule("Identities")
    cols = columns(conn, "identities")
    if "vector_count" not in cols:
        note_missing("identities", "vector_count")
        show_counts(conn, "identities", "identifier_key", "by identifier")
        return
    rows = conn.execute(
        "SELECT identifier_key, COUNT(*) AS n, "
        "SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END) AS labeled, "
        "SUM(CASE WHEN vector_count > 0 THEN 1 ELSE 0 END) AS with_vectors "
        "FROM identities GROUP BY identifier_key ORDER BY n DESC"
    ).fetchall()
    if not rows:
        print("  none")
        return
    print(f"  {'identifier':<16} {'total':>8} {'labeled':>8} {'vectors':>8}")
    for r in rows:
        print(
            f"  {r['identifier_key']:<16} {r['n']:>8} {r['labeled']:>8} "
            f"{r['with_vectors']:>8}"
        )
    # A label without vectors is a name the system cannot see.
    blind = conn.execute(
        "SELECT COUNT(*) AS n FROM identities "
        "WHERE label IS NOT NULL AND vector_count = 0"
    ).fetchone()
    if blind["n"]:
        print(
            f"\n  {blind['n']} labeled identities have no vectors — recognition\n"
            f"  cannot return them as subjects. `siteloom train enroll` fixes this."
        )


def report_events(conn: sqlite3.Connection, present: set[str]) -> None:
    if "events" not in present:
        return
    row = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
    if not row["n"]:
        return
    rule("Events")
    print(f"  {row['n']} events")
    show_counts(conn, "events", "class_name", "by class")
    if "event_identities" in present:
        verdicts = counts(conn, "event_identities", "verdict")
        print("  identity claims by verdict:")
        for r in verdicts:
            print(f"      {str(r['k'] or '(unjudged)'):<20} {r['n']:>8}")


def report_runs(conn: sqlite3.Connection, present: set[str]) -> None:
    if "operation_runs" not in present:
        return
    rule("Operation runs")
    rows = counts(conn, "operation_runs", "status")
    if not rows:
        print("  none recorded — no long-running job has ever heartbeated here.")
        return
    for r in rows:
        print(f"      {str(r['k']):<20} {r['n']:>8}")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <path-to.db>")
    path = Path(sys.argv[1]).expanduser()
    conn = connect(path)
    present = tables(conn)

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Siteloom database report: {path.name} ({size_mb:.1f} MB)")
    print(f"tables: {len(present)}")

    # One section failing must not take the others with it. A diagnostic
    # that dies on the first problem hides the rest — and the databases
    # worth reporting on are exactly the old ones most likely to surprise
    # it. Column drift is handled explicitly above; this is the backstop
    # for whatever was not anticipated.
    failures = 0
    for report in (
        report_sources,
        report_items,
        report_annotations,
        report_training_gate,
        report_identities,
        report_events,
        report_runs,
    ):
        try:
            report(conn, present)
        except sqlite3.Error as exc:
            failures += 1
            print(f"\n  [{report.__name__} could not run: {exc}]")

    if failures:
        print(
            f"\n{failures} section(s) failed. The rest of the report above is "
            "still valid."
        )

    if _skipped:
        print(f"\n{MIGRATE_HINT}")

    print("\nNo names, paths, or boxes are printed above — this output is")
    print("safe to paste into an issue or a chat.")


if __name__ == "__main__":
    main()
