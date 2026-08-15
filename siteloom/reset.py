"""Factory reset: put an install back in its out-of-the-box state.

What a site has learned lives in three stores, and forgetting one of
them leaves a system that contradicts itself. The database holds the
rows, `media_dir` holds the JPEGs those rows point at, and the embedded
Qdrant directory holds the vectors that decide who a new sighting is.
Clear the rows only and the next unknown face still matches a gallery
whose Identity row no longer exists; clear the vectors only and every
crop on disk is orphaned. So a reset is defined over all three at once:
it clears them together or it refuses and clears nothing.

What it deliberately does not touch:

* **The config file.** Cameras, credentials, zones and thresholds *are*
  the install; they are not what it learned.
* **Library originals.** `library add` indexes a directory in place, so
  the photos stay wherever the operator keeps them. A reset removes the
  index, never the archive — and refuses outright if a registered source
  turns out to sit inside a directory it was about to clear.
* **Downloaded model weights** (`~/.cache/siteloom/models`). Fetched,
  not learned; re-downloading them is minutes of nothing.
* **Logs.** They describe what the machine did, which is the one thing
  worth keeping when the reason for the reset is that something went
  wrong.

Accounts are the single optional survivor (`keep_users`), and the reason
they *can* survive is structural rather than sentimental: every other
table references `users`, never the other way round, so holding them
back can never leave a dangling row behind. No other keep flag has that
property — keeping the library while dropping identities would strand
`annotations.identity_id`, which is the half-reset state this module
exists to prevent.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from siteloom.config import SiteConfig
from siteloom.store.models import Base, LibrarySource, OperationRun

# Accounts and their sessions: the one group `keep_users` holds back.
ACCOUNT_TABLES = ("users", "web_sessions")


class UnsafeReset(RuntimeError):
    """The reset was refused. Nothing has been removed."""


@dataclass(frozen=True)
class TableTarget:
    name: str
    rows: int


@dataclass(frozen=True)
class PathTarget:
    """A directory the reset clears.

    `contents_only` keeps the directory itself: `media_dir` is named in
    the config and other code creates subdirectories under it at import
    time, so it must still exist afterwards. The vector directory is
    recreated by Qdrant on the next open, so that one goes whole.
    """

    path: Path
    label: str
    files: int
    bytes: int
    contents_only: bool = True

    @property
    def exists(self) -> bool:
        return self.path.exists()


@dataclass(frozen=True)
class ResetPlan:
    tables: tuple[TableTarget, ...] = ()
    paths: tuple[PathTarget, ...] = ()
    kept: tuple[str, ...] = ()

    @property
    def rows(self) -> int:
        return sum(t.rows for t in self.tables)

    @property
    def files(self) -> int:
        return sum(p.files for p in self.paths)

    @property
    def bytes(self) -> int:
        return sum(p.bytes for p in self.paths)

    @property
    def is_empty(self) -> bool:
        """Nothing to do — the install is already out of the box."""
        return self.rows == 0 and self.files == 0


def _measure(path: Path) -> tuple[int, int]:
    """(files, bytes) under `path`, counting nothing for what is absent."""
    if not path.exists():
        return (0, 0)
    files = 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file() or child.is_symlink():
            files += 1
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return (files, total)


def _sqlite_file(db_url: str) -> Path | None:
    prefix = "sqlite:///"
    if not db_url.startswith(prefix) or db_url == prefix + ":memory:":
        return None
    return Path(db_url[len(prefix) :]).resolve()


def target_paths(config: SiteConfig) -> list[PathTarget]:
    """The directories a reset clears, measured."""
    targets: list[tuple[Path, str, bool]] = [
        (Path(config.storage.media_dir), "crops, thumbnails and clips", True),
        (Path(config.identity.vector_db_path), "identity vectors", False),
        (Path(config.training.output_dir), "trained models and exports", True),
    ]
    projection = Path(config.identity.face_projection_path)
    training_dir = Path(config.training.output_dir).resolve()
    if projection.exists() and training_dir not in projection.resolve().parents:
        targets.append((projection.parent, "face projection", True))

    out = []
    seen: set[Path] = set()
    for path, label, contents_only in targets:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        files, size = _measure(path)
        out.append(
            PathTarget(
                path=path,
                label=label,
                files=files,
                bytes=size,
                contents_only=contents_only,
            )
        )
    return out


def plan_reset(
    config: SiteConfig, session: Session, *, keep_users: bool = False
) -> ResetPlan:
    """What a reset would remove. Reads only; changes nothing."""
    keep = set(ACCOUNT_TABLES) if keep_users else set()
    tables = []
    for table in Base.metadata.sorted_tables:
        if table.name in keep:
            continue
        rows = session.execute(select(func.count()).select_from(table)).scalar_one()
        tables.append(TableTarget(name=table.name, rows=int(rows)))

    kept = ["the site config", "library originals", "downloaded model weights", "logs"]
    if keep_users:
        kept.insert(0, "operator accounts and sessions")

    return ResetPlan(
        tables=tuple(sorted(tables, key=lambda t: (-t.rows, t.name))),
        paths=tuple(target_paths(config)),
        kept=tuple(kept),
    )


def live_runs(session: Session) -> list[OperationRun]:
    """Operations that something is still working on.

    A `serve` or `run` process heartbeats one of these rows and holds the
    vector store open, so a reset underneath it would delete a directory
    from under a live Qdrant client and then watch the survivor write
    fresh rows into the database we just emptied.
    """
    running = (
        session.query(OperationRun).filter(OperationRun.status == "running").all()
    )
    return [r for r in running if not r.is_stale]


def _protected_paths(config: SiteConfig, session: Session) -> list[tuple[Path, str]]:
    """Things that must survive, with the reason to quote when refused."""
    protected: list[tuple[Path, str]] = []
    source_path = getattr(config, "_source_path", None)
    if source_path:
        protected.append((Path(source_path).resolve(), "the site config"))
    db_file = _sqlite_file(config.storage.db_url)
    if db_file:
        protected.append((db_file, "the database"))
    for (path,) in session.query(LibrarySource.path).all():
        if path:
            # Resolved, not merely absolute: on macOS the same directory
            # is reachable as both /var/... and /private/var/..., and a
            # containment check that compares the two spellings misses.
            protected.append((Path(path).expanduser().resolve(), "a library source archive"))
    return protected


def check_safe(config: SiteConfig, session: Session, plan: ResetPlan) -> None:
    """Refuse before anything is removed. Raises `UnsafeReset`.

    The library-source check is the one that earns its keep: nothing
    stops an operator from `library add`-ing a directory that happens to
    live under `media_dir`, and the difference between clearing an index
    and deleting somebody's photo archive is this comparison.
    """
    runs = live_runs(session)
    if runs:
        listed = ", ".join(f"{r.kind} (pid {r.pid})" for r in runs)
        raise UnsafeReset(
            f"{len(runs)} operation still running: {listed}. "
            "Stop the service and any running job first "
            "(`siteloom service stop`, `siteloom jobs cancel <id>`)."
        )

    protected = _protected_paths(config, session)
    for target in plan.paths:
        resolved = target.path.resolve()
        if resolved.parent == resolved or resolved == Path.home().resolve():
            raise UnsafeReset(
                f"refusing to clear {resolved} — that is a filesystem root "
                "or home directory, not a Siteloom directory."
            )
        for guarded, why in protected:
            if guarded == resolved or resolved in guarded.parents:
                raise UnsafeReset(
                    f"refusing to clear {resolved}: {guarded} ({why}) is "
                    "inside it. Move it out, or point the config elsewhere, "
                    "before resetting."
                )


def _clear_path(target: PathTarget) -> list[str]:
    """Remove a target, returning the failures rather than raising: one
    unreadable file must not leave the other two stores half-cleared."""
    problems: list[str] = []
    if not target.path.exists():
        return problems
    if target.contents_only:
        doomed = list(target.path.iterdir())
    else:
        doomed = [target.path]
    for path in doomed:
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            problems.append(f"{path}: {exc}")
    if target.contents_only:
        target.path.mkdir(parents=True, exist_ok=True)
    return problems


def perform_reset(
    config: SiteConfig, session: Session, plan: ResetPlan
) -> list[str]:
    """Apply `plan`. Returns the problems hit, empty when clean.

    Rows go first: while they are still there the crops on disk are at
    worst stale, but a database still pointing at deleted JPEGs is a
    console full of broken images.
    """
    check_safe(config, session, plan)

    doomed = {t.name for t in plan.tables}
    for table in reversed(Base.metadata.sorted_tables):
        if table.name in doomed:
            session.execute(delete(table))
    session.commit()
    # The caller's session still holds objects for rows that no longer
    # exist; anything it wrote back would resurrect them.
    session.expunge_all()

    # Reclaim the file and, since these tables use plain INTEGER primary
    # keys, restart ids at 1 — a reset install should not hand out event
    # 1260 as its first event.
    bind = session.get_bind()
    if bind is not None and bind.dialect.name == "sqlite":
        with bind.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.exec_driver_sql("VACUUM")

    problems: list[str] = []
    for target in plan.paths:
        problems.extend(_clear_path(target))
    return problems


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"
