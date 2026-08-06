from __future__ import annotations

import logging

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from siteloom.store.models import Base

log = logging.getLogger(__name__)


def make_engine(db_url: str) -> Engine:
    return create_engine(db_url)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    _ensure_columns(engine)


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
