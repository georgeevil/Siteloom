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
    Base.metadata.create_all(engine)
    _relax_event_identity_nullability(engine)
    _ensure_columns(engine)


def _relax_event_identity_nullability(engine: Engine) -> None:
    """Rebuild event_identities if identity_id is still NOT NULL.

    Recorded misses are rows with a NULL identity_id (CLD-16's
    null-identity verdict rows), which the original schema forbade.
    SQLite cannot alter a column's nullability in place, so this is the
    one deliberate exception to the additive-only rule in
    `_ensure_columns`: a guarded, single-purpose rename-copy-swap. Runs
    before `_ensure_columns` so the rebuilt table already carries every
    current column and the additive pass has nothing left to do.
    """
    inspector = inspect(engine)
    if "event_identities" not in inspector.get_table_names():
        return
    old_cols = {c["name"]: c for c in inspector.get_columns("event_identities")}
    col = old_cols.get("identity_id")
    if col is None or col["nullable"]:
        return
    if engine.dialect.name != "sqlite":
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE event_identities ALTER COLUMN identity_id DROP NOT NULL")
            )
        return

    table = Base.metadata.tables["event_identities"]
    new_names = [c.name for c in table.columns]
    # Copy every column both schemas share; columns new to this release
    # get an explicit value because NOT NULL + no server default would
    # otherwise reject the insert.
    select_parts = []
    insert_names = []
    for name in new_names:
        if name in old_cols:
            insert_names.append(name)
            select_parts.append(name)
        elif not table.columns[name].nullable:
            insert_names.append(name)
            select_parts.append("0")
    log.info("migrating: rebuilding event_identities for nullable identity_id")
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE event_identities RENAME TO event_identities_old"))
        table.create(conn)
        conn.execute(
            text(
                f"INSERT INTO event_identities ({', '.join(insert_names)}) "
                f"SELECT {', '.join(select_parts)} FROM event_identities_old"
            )
        )
        conn.execute(text("DROP TABLE event_identities_old"))


def _ensure_columns(engine: Engine) -> None:
    """Add columns the models define but an existing database lacks.

    create_all() only creates missing tables; a column added to a model
    would otherwise break every query against a database created before
    it existed. This covers the additive case (new nullable/defaulted
    columns) — the only kind of schema change the project makes; anything
    beyond that warrants a real migration tool.
    """
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


def get_session(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
