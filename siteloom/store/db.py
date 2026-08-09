from __future__ import annotations

import logging

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from siteloom.store.models import Base

log = logging.getLogger(__name__)


def make_engine(db_url: str) -> Engine:
    engine = create_engine(db_url)
    if engine.dialect.name == "sqlite":
        # Per-camera ingest threads and the web UI write concurrently;
        # WAL lets readers proceed during a write, and the busy timeout
        # rides out the brief writer-writer contention that remains.
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    # The rebuild/heal must run BEFORE create_all: an interrupted rebuild
    # leaves only event_identities_old, whose stale-named indexes would
    # collide with the indexes create_all makes for the new table.
    _relax_event_identity_nullability(engine)
    Base.metadata.create_all(engine)
    added = _ensure_columns(engine)
    # Only when this run is the one that added the column: the backfill
    # below is a migration, not a repair pass (see its docstring).
    if ("annotations", "verified_by") in added:
        _backfill_verified_by(engine)


def _relax_event_identity_nullability(engine: Engine) -> None:
    """Rebuild event_identities if identity_id is still NOT NULL.

    Recorded misses are rows with a NULL identity_id (CLD-16's
    null-identity verdict rows), which the original schema forbade.
    SQLite cannot alter a column's nullability in place, so this is the
    one deliberate exception to the additive-only rule in
    `_ensure_columns`: a guarded, single-purpose rename-copy-swap. Runs
    before `_ensure_columns` so the rebuilt table already carries every
    current column and the additive pass has nothing left to do.

    Two hard-won subtleties:
    * `ALTER TABLE .. RENAME` keeps the old table's *index names*, which
      would collide with the new table's — the old indexes are dropped
      first (an index is derived data; nothing is lost).
    * pysqlite autocommits around DDL, so this sequence is NOT atomic. It
      is therefore written to be resumable instead: a leftover
      `event_identities_old` from an interrupted run is detected and the
      copy-and-drop is completed before anything else.
    """
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if "event_identities_old" in tables:
        # A previous rebuild was interrupted between rename and drop —
        # the data lives in the old table. Finish the job first.
        log.warning("resuming interrupted event_identities rebuild")
        _finish_event_identity_rebuild(engine, inspector)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
    if "event_identities" not in tables:
        return
    col = next(
        (
            c
            for c in inspector.get_columns("event_identities")
            if c["name"] == "identity_id"
        ),
        None,
    )
    if col is None or col["nullable"]:
        return
    if engine.dialect.name != "sqlite":
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE event_identities ALTER COLUMN identity_id DROP NOT NULL")
            )
        return

    log.info("migrating: rebuilding event_identities for nullable identity_id")
    with engine.begin() as conn:
        # Renaming a table does not rename its indexes, so the new
        # table's CREATE INDEX statements would collide. Drop them —
        # derived data, recreated with the new table.
        for index in inspector.get_indexes("event_identities"):
            if index["name"]:
                conn.execute(text(f'DROP INDEX IF EXISTS "{index["name"]}"'))
        conn.execute(text("ALTER TABLE event_identities RENAME TO event_identities_old"))
    _finish_event_identity_rebuild(engine, inspect(engine))


def _finish_event_identity_rebuild(engine: Engine, inspector) -> None:
    """Create the new event_identities, copy from _old, drop _old.

    Split out so an interrupted rebuild (pysqlite DDL autocommit means
    the steps are not atomic) can be completed on the next init_db.
    """
    table = Base.metadata.tables["event_identities"]
    old_cols = {c["name"] for c in inspector.get_columns("event_identities_old")}
    # Copy every column both schemas share; columns new to this release
    # get an explicit value because NOT NULL + no server default would
    # otherwise reject the insert.
    select_parts = []
    insert_names = []
    for column in table.columns:
        if column.name in old_cols:
            insert_names.append(column.name)
            select_parts.append(column.name)
        elif not column.nullable:
            insert_names.append(column.name)
            select_parts.append("0")
    with engine.begin() as conn:
        # Indexes that followed the rename onto _old still own the names
        # the new table wants — drop them by name before creating it.
        for index in table.indexes:
            conn.execute(text(f'DROP INDEX IF EXISTS "{index.name}"'))
        table.create(conn, checkfirst=True)
        # INSERT OR IGNORE keys on the primary key, so re-running after a
        # partial copy never duplicates a row.
        conn.execute(
            text(
                f"INSERT OR IGNORE INTO event_identities ({', '.join(insert_names)}) "
                f"SELECT {', '.join(select_parts)} FROM event_identities_old"
            )
        )
        conn.execute(text("DROP TABLE event_identities_old"))
        # Dropping _old also dropped the indexes that had followed the
        # rename; (re)create the ones the model declares.
        for index in table.indexes:
            index.create(conn, checkfirst=True)


def _ensure_columns(engine: Engine) -> set[tuple[str, str]]:
    """Add columns the models define but an existing database lacks.

    create_all() only creates missing tables; a column added to a model
    would otherwise break every query against a database created before
    it existed. This covers the additive case (new nullable/defaulted
    columns) — the only kind of schema change the project makes; anything
    beyond that warrants a real migration tool.

    Returns the (table, column) pairs it actually added, so a caller can
    run a one-time data migration in the same breath as the DDL. Note
    what an added column holds for existing rows: a DDL DEFAULT is
    emitted **only** for a non-nullable column with a scalar default, so
    a nullable column arrives NULL everywhere and any backfill is the
    caller's job.
    """
    added: set[tuple[str, str]] = set()
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                if column.primary_key or not (column.nullable or column.default is not None or column.server_default is not None):
                    log.warning(
                        "column %s.%s needs a real migration (not nullable, no default)",
                        table.name,
                        column.name,
                    )
                    continue
                ddl = f"ALTER TABLE {table.name} ADD COLUMN {column.name} {column.type.compile(engine.dialect)}"
                if column.nullable:
                    pass  # SQLite default for added columns is NULL
                elif column.default is not None and column.default.is_scalar:
                    ddl += f" DEFAULT {column.default.arg!r}"
                log.info("migrating: %s", ddl)
                conn.execute(text(ddl))
                added.add((table.name, column.name))
    return added


def _backfill_verified_by(engine: Engine) -> int:
    """Attribute already-verified annotations, once, at migration time.

    Before `verified_by` existed the verifier was not recorded, but for a
    database written by this project it is derivable: the Takeout
    importer auto-verifies pass 1 and only pass 1
    (`verified=certain and self.auto_verify_unambiguous` in
    library/takeout.py), so a verified row on basis "unambiguous" is the
    importer and a verified row on any other basis was confirmed by a
    person in the review UI.

    That inference is legitimate exactly here and nowhere else. It holds
    only over rows written before this column existed, and only while
    pass 1 was the sole auto-verifying path — both of which are facts
    about the past, which is why the result is *written down* rather than
    recomputed at read time. Every query from now on reads the column;
    nothing else in the codebase may infer a verifier from
    `proposal_basis` again.

    `verified_at` is left NULL: the time was never recorded, and NULL
    says "unknown" where any invented timestamp would say something
    false. Runs inside one transaction over rows that are `verified` with
    no verifier, so it is idempotent even if it is somehow reached twice.
    """
    with engine.begin() as conn:
        # `WHERE verified`, not `verified = 1`: SQLite stores the flag as
        # an int and Postgres as a boolean, and this has to run on both.
        result = conn.execute(
            text(
                "UPDATE annotations SET verified_by = "
                "CASE WHEN proposal_basis = 'unambiguous' THEN 'import' "
                "ELSE 'human' END "
                "WHERE verified AND verified_by IS NULL"
            )
        )
        count = result.rowcount or 0
    if count:
        log.info(
            "migrating: attributed %d verified annotations from proposal_basis "
            "(one-time; verified_by is read directly from now on)",
            count,
        )
    return count


def get_session(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
