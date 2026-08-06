"""CLI sub-apps: library indexing, Takeout import, classes, training."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import typer

library_app = typer.Typer(help="Index and label local media directories.")
takeout_app = typer.Typer(help="Import Google Photos Takeout archives.")
classes_app = typer.Typer(help="Manage custom sub-classes.")
train_app = typer.Typer(help="Train face models from verified annotations.")
jobs_app = typer.Typer(help="Inspect long-running operations.")

# 128 + SIGINT, the shell convention: an interrupted run is neither a
# success nor a crash, and wrapping scripts need to tell them apart.
INTERRUPTED_EXIT = 130

CONFIG_OPT = typer.Option("site.yaml", "--config", "-c", help="Site config YAML")
LOG_FILE_OPT = typer.Option(None, "--log-file", help="Also write logs to this file")
QUIET_OPT = typer.Option(
    False, "--quiet", "-q", help="No progress bar (log lines only; still tracked)"
)


def _setup(config_path, log_file: str | None = None, level: str = "INFO"):
    """Shared bootstrap: config, DB session factory, dispatcher, resolver."""
    from siteloom.config import load_config
    from siteloom.identity import IdentityResolver, VectorStore
    from siteloom.ingest import build_dispatcher
    from siteloom.library import LibraryIndexer
    from siteloom.progress import setup_logging
    from siteloom.store import get_session, init_db, make_engine

    setup_logging(level=level, log_file=log_file)
    config = load_config(config_path)
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    dispatcher = build_dispatcher(config)
    resolver = None
    if config.identity.enabled:
        resolver = IdentityResolver(
            config.identity, VectorStore(config.identity.vector_db_path)
        )
    indexer = LibraryIndexer(config, Session, dispatcher, resolver)
    return config, Session, indexer


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# -- library ---------------------------------------------------------------


@library_app.command("add")
def library_add(
    path: Path = typer.Argument(..., help="Directory of images/videos"),
    config: Path = CONFIG_OPT,
    name: str = typer.Option("", help="Display name for this source"),
):
    """Register a directory as a library source."""
    _cfg, _Session, indexer = _setup(config)
    source = indexer.add_source(path, name=name)
    typer.echo(f"source #{source.id}: {source.name} -> {source.path}")


@library_app.command("scan")
def library_scan(
    config: Path = CONFIG_OPT,
    source_id: int = typer.Option(None, help="Source to scan (default: all)"),
    limit: int = typer.Option(None, help="Max new files to register"),
):
    """Register files as pending. Cheap — no decoding or detection."""
    _cfg, Session, indexer = _setup(config)
    from siteloom.store import LibrarySource

    with Session() as session:
        ids = (
            [source_id]
            if source_id
            else [s.id for s in session.query(LibrarySource).all()]
        )
    if not ids:
        typer.echo("no sources registered; use `siteloom library add <path>`")
        raise typer.Exit(1)
    for sid in ids:
        result = indexer.scan(sid, limit=limit)
        typer.echo(
            f"source {sid}: +{result.added} new, {result.updated} changed, "
            f"{result.skipped} unchanged, {result.total_pending} pending"
        )


@library_app.command("index")
def library_index(
    config: Path = CONFIG_OPT,
    source_id: int = typer.Option(None, help="Only this source"),
    limit: int = typer.Option(None, help="Max items this run (default: config)"),
    all_pending: bool = typer.Option(False, "--all", help="Process everything pending"),
    no_identify: bool = typer.Option(False, help="Skip identification (faster pass)"),
    log_file: str = LOG_FILE_OPT,
    quiet: bool = QUIET_OPT,
):
    """Run detection over pending items. Resumable — stop and rerun freely."""
    cfg, Session, indexer = _setup(config, log_file)
    from siteloom.progress import ProgressReporter

    batch = limit or (10**9 if all_pending else cfg.library.batch_size)
    resume = f"siteloom library index --config {config}" + (
        " --all" if all_pending else ""
    )
    # Assigned inside the block, read after it: the reporter swallows the
    # Ctrl-C so the run can be recorded, which leaves this unset. `None`
    # is how the summary below knows the work never finished.
    result = None
    with ProgressReporter(
        Session,
        "library-index",
        target=f"source {source_id}" if source_id else "all sources",
        resume_command=resume,
        bar=not quiet,
    ) as progress:
        result = indexer.process(
            source_id=source_id,
            limit=batch,
            identify=not no_identify and cfg.library.identify_on_index,
            progress=progress,
        )
    if result is None:
        # Interrupted; the reporter already printed the resume command.
        raise typer.Exit(INTERRUPTED_EXIT)
    typer.echo(
        f"indexed {result.processed} items ({result.annotations} boxes), "
        f"{result.failed} failed, {result.remaining} still pending"
    )
    if result.remaining:
        typer.echo(f"continue with: {resume}")


@library_app.command("status")
def library_status(config: Path = CONFIG_OPT):
    """Show per-source indexing progress."""
    _cfg, Session, _indexer = _setup(config)
    from sqlalchemy import func, select

    from siteloom.store import Annotation, LibraryItem, LibrarySource

    with Session() as session:
        for source in session.scalars(select(LibrarySource)).all():
            counts = dict(
                session.execute(
                    select(LibraryItem.status, func.count())
                    .filter(LibraryItem.source_id == source.id)
                    .group_by(LibraryItem.status)
                ).all()
            )
            typer.echo(
                f"#{source.id} {source.name} [{source.kind}] "
                + ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
            )
        total = session.scalar(select(func.count()).select_from(Annotation)) or 0
        verified = (
            session.scalar(
                select(func.count())
                .select_from(Annotation)
                .filter(Annotation.verified.is_(True))
            )
            or 0
        )
        typer.echo(f"annotations: {total} total, {verified} verified")


# -- takeout ---------------------------------------------------------------


@takeout_app.command("import")
def takeout_import(
    path: Path = typer.Argument(..., help="Google Photos Takeout directory"),
    config: Path = CONFIG_OPT,
    limit: int = typer.Option(None, help="Max media files to consider"),
    batch_size: int = typer.Option(200, help="Commit every N items (resumable)"),
    include_derivatives: bool = typer.Option(
        False,
        help="Also import Google's '-edited' copies (near-duplicates of the original)",
    ),
    no_auto_verify: bool = typer.Option(
        False,
        help="Do not auto-verify unambiguous (1 face + 1 name) assignments",
    ),
    log_file: str = LOG_FILE_OPT,
    quiet: bool = QUIET_OPT,
):
    """Import a Takeout tree: people tags, face detection, name proposals.

    Resumable — Ctrl-C saves progress, and rerunning the same command
    continues where it stopped.
    """
    _cfg, Session, indexer = _setup(config, log_file)
    from siteloom.library.takeout import TakeoutImporter
    from siteloom.progress import ProgressReporter

    resume = f'siteloom takeout import "{path}" --config {config}'
    stats = timings = None
    with ProgressReporter(
        Session,
        "takeout-import",
        target=str(path),
        resume_command=resume,
        bar=not quiet,
    ) as progress:
        importer = TakeoutImporter(
            indexer,
            auto_verify_unambiguous=not no_auto_verify,
            progress=progress,
        )
        stats = importer.import_tree(
            path,
            limit=limit,
            batch_size=batch_size,
            include_derivatives=include_derivatives,
        )
        timings = progress.summary_lines()

    if stats is None:
        raise typer.Exit(INTERRUPTED_EXIT)
    _print_import_summary(stats, no_auto_verify, timings)


def _print_import_summary(stats, no_auto_verify: bool, timings: list[str]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Takeout import", show_header=False, title_justify="left")
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_column()
    named = stats.unambiguous + stats.matched
    total_faces = max(1, stats.faces_detected)
    table.add_row("media files seen", f"{stats.items_seen:,}", "")
    table.add_row("edited copies skipped", f"{stats.skipped_derivative:,}", "duplicates of originals")
    table.add_row("sidecars matched", f"{stats.sidecars_matched:,}", "")
    table.add_row(
        "people tags", f"{stats.people_tags:,}", f"{len(stats.people)} distinct people"
    )
    table.add_row("faces detected", f"{stats.faces_detected:,}", "")
    table.add_row(
        "  certain (1 face, 1 name)",
        f"{stats.unambiguous:,}",
        "[green]auto-verified[/]" if not no_auto_verify else "needs review",
    )
    table.add_row("  matched to gallery", f"{stats.matched:,}", "[yellow]needs review[/]")
    table.add_row("  unassigned", f"{stats.unresolved:,}", "label manually if wanted")
    table.add_row(
        "named coverage",
        f"{named / total_faces * 100:.0f}%",
        f"{named:,} of {stats.faces_detected:,} faces",
    )
    console.print(table)

    if timings:
        console.print("[dim]" + " · ".join(t.strip() for t in timings) + "[/]")
    top = sorted(stats.people.items(), key=lambda kv: -kv[1])[:10]
    if top:
        people = Table(title="Most-tagged people", title_justify="left")
        people.add_column("photos", justify="right")
        people.add_column("person")
        for name, n in top:
            people.add_row(f"{n:,}", name)
        console.print(people)
    console.print(
        "\nNext: review proposals at [bold]/training[/] "
        "(run [bold]siteloom serve[/]), then [bold]siteloom train face[/]."
    )


@takeout_app.command("status")
def takeout_status(config: Path = CONFIG_OPT):
    """How much of the imported archive is processed, named, and verified."""
    from rich.console import Console
    from rich.table import Table
    from sqlalchemy import func, select

    from siteloom.store import Annotation, ItemTag, LibraryItem, LibrarySource

    cfg, Session, _indexer = _setup(config, level="WARNING")
    console = Console()
    with Session() as session:
        sources = session.scalars(
            select(LibrarySource).filter_by(kind="takeout")
        ).all()
        if not sources:
            console.print(
                "No Takeout source imported yet. Run "
                "[bold]siteloom takeout import <path>[/]"
            )
            return
        for source in sources:
            items = session.scalar(
                select(func.count())
                .select_from(LibraryItem)
                .filter_by(source_id=source.id)
            )
            tagged = session.scalar(
                select(func.count(func.distinct(ItemTag.item_id)))
                .select_from(ItemTag)
                .join(LibraryItem, LibraryItem.id == ItemTag.item_id)
                .filter(ItemTag.kind == "person", LibraryItem.source_id == source.id)
            )
            console.print(
                f"[bold]{source.name}[/] — {items:,} items, {tagged:,} with people tags"
            )

        faces = session.scalar(
            select(func.count()).select_from(Annotation).filter_by(class_name="face")
        )
        by_basis = dict(
            session.execute(
                select(Annotation.proposal_basis, func.count())
                .filter(Annotation.class_name == "face")
                .group_by(Annotation.proposal_basis)
            ).all()
        )
        verified = session.scalar(
            select(func.count())
            .select_from(Annotation)
            .filter(
                Annotation.class_name == "face",
                Annotation.verified.is_(True),
                Annotation.rejected.is_(False),
            )
        )
        rejected = session.scalar(
            select(func.count())
            .select_from(Annotation)
            .filter(Annotation.class_name == "face", Annotation.rejected.is_(True))
        )
        ready = session.execute(
            select(Annotation.proposed_name, func.count())
            .filter(
                Annotation.class_name == "face",
                Annotation.verified.is_(True),
                Annotation.rejected.is_(False),
                Annotation.proposed_name.is_not(None),
            )
            .group_by(Annotation.proposed_name)
            .having(func.count() >= cfg.training.min_samples_per_person)
        ).all()
        pending_review = session.scalar(
            select(func.count())
            .select_from(Annotation)
            .filter(
                Annotation.class_name == "face",
                Annotation.verified.is_(False),
                Annotation.rejected.is_(False),
                Annotation.proposed_name.is_not(None),
            )
        )

    table = Table(show_header=False, title="Faces", title_justify="left")
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_row("detected", f"{faces:,}")
    for basis, n in sorted(by_basis.items(), key=lambda kv: -kv[1]):
        table.add_row(f"  {basis or 'unassigned'}", f"{n:,}")
    table.add_row("verified", f"{verified:,}")
    table.add_row("rejected", f"{rejected:,}")
    table.add_row("awaiting review", f"{pending_review:,}")
    console.print(table)
    console.print(
        f"[green]{len(ready)} people[/] have ≥{cfg.training.min_samples_per_person} "
        "verified samples — enough to include in a fine-tune."
    )
    if pending_review:
        console.print(
            f"[yellow]{pending_review:,} proposals await review[/] at /training "
            "(run [bold]siteloom serve[/])."
        )


@takeout_app.command("inspect")
def takeout_inspect(
    path: Path = typer.Argument(..., help="Takeout directory"),
    limit: int = typer.Option(5, help="Sidecars to print"),
):
    """Dry-run: show what the importer would read. No DB writes."""
    from siteloom.adapters.file import IMAGE_EXTS, VIDEO_EXTS
    from siteloom.library.takeout import find_sidecar, index_sidecars

    path = Path(path).expanduser()
    if not path.exists():
        typer.echo(f"path not found: {path}", err=True)
        raise typer.Exit(1)
    media = [
        p
        for p in sorted(path.rglob("*"))
        if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
    ]
    typer.echo(f"{len(media)} media files under {path}")
    shown = matched = 0
    people_total: dict[str, int] = {}
    indexes: dict[Path, dict] = {}
    for file in media:
        if file.parent not in indexes:
            indexes[file.parent] = index_sidecars(file.parent)
        sidecar = find_sidecar(file, indexes[file.parent])
        if sidecar is None:
            continue
        matched += 1
        for person in sidecar.people:
            people_total[person] = people_total.get(person, 0) + 1
        if shown < limit:
            shown += 1
            typer.echo(
                f"\n{file.name}\n  sidecar: {sidecar.path.name}"
                f"\n  taken:   {sidecar.taken_at}"
                f"\n  people:  {sidecar.people or '—'}"
            )
    typer.echo(f"\n{matched}/{len(media)} media files have a sidecar")
    typer.echo(f"{len(people_total)} distinct people tagged")
    for name, n in sorted(people_total.items(), key=lambda kv: -kv[1])[:15]:
        typer.echo(f"  {n:5d}  {name}")


# -- jobs ------------------------------------------------------------------


@jobs_app.command("list")
def jobs_list(
    config: Path = CONFIG_OPT,
    limit: int = typer.Option(15, help="How many runs to show"),
):
    """Recent long-running operations and their outcome."""
    from rich.console import Console
    from rich.table import Table
    from sqlalchemy import select

    from siteloom.progress import humanize
    from siteloom.store import OperationRun

    _cfg, Session, _indexer = _setup(config, level="WARNING")
    with Session() as session:
        runs = session.scalars(
            select(OperationRun).order_by(OperationRun.id.desc()).limit(limit)
        ).all()
        rows = [
            (
                r.id,
                r.kind,
                r.phase,
                "stale" if r.is_stale else r.status,
                r.current,
                r.total,
                r.percent,
                r.elapsed_s,
                r.eta_s,
                json.loads(r.counters or "{}"),
                r.started_at,
                r.resume_command,
                r.message,
            )
            for r in runs
        ]
    if not rows:
        typer.echo("no operations recorded yet")
        return

    console = Console()
    # Narrow terminals truncate every column into uselessness, so the
    # counters column is dropped below ~110 chars and shown per row
    # underneath instead.
    wide = console.width >= 110
    table = Table(title="Operations", title_justify="left", pad_edge=False)
    table.add_column("id", justify="right", no_wrap=True)
    table.add_column("kind", no_wrap=True)
    table.add_column("phase", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("progress", justify="right", no_wrap=True)
    table.add_column("elapsed", justify="right", no_wrap=True)
    table.add_column("eta", justify="right", no_wrap=True)
    if wide:
        table.add_column("counters")
    colours = {
        "complete": "green",
        "running": "cyan",
        "interrupted": "yellow",
        "failed": "red",
        "stale": "red",
    }
    for (
        rid, kind, phase, status, current, total, pct, elapsed, eta, counters,
        _started, _resume, _message,
    ) in rows:
        cells = [
            str(rid),
            kind.replace("takeout-import", "takeout").replace("library-index", "index"),
            phase or "—",
            f"[{colours.get(status, 'white')}]{status}[/]",
            f"{current:,}/{total:,} {pct:.0f}%" if total else f"{current:,}",
            humanize(elapsed),
            humanize(eta),
        ]
        if wide:
            cells.append(
                " ".join(f"{k}={v:,}" for k, v in list(counters.items())[:4]) or "—"
            )
        table.add_row(*cells)
    console.print(table)

    for row in rows:
        rid, status, counters, resume, message = row[0], row[3], row[9], row[11], row[12]
        if not wide and counters:
            console.print(
                f"[dim]#{rid}[/] "
                + " ".join(f"{k}={v:,}" for k, v in list(counters.items())[:5])
            )
        if status in ("interrupted", "stale") and resume:
            console.print(f"[yellow]#{rid} resume:[/] {resume}")
        elif status == "failed" and message:
            console.print(f"[red]#{rid} failed:[/] {message}")


@jobs_app.command("watch")
def jobs_watch(
    config: Path = CONFIG_OPT,
    interval: float = typer.Option(2.0, help="Refresh seconds"),
):
    """Live view of the currently running operation, from any terminal."""
    import time

    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from sqlalchemy import select

    from siteloom.progress import humanize
    from siteloom.store import OperationRun

    _cfg, Session, _indexer = _setup(config, level="WARNING")
    console = Console()

    def render() -> Table:
        with Session() as session:
            runs = session.scalars(
                select(OperationRun)
                .filter(OperationRun.status == "running")
                .order_by(OperationRun.id.desc())
            ).all()
            table = Table(title="Running operations", title_justify="left")
            for col in ("id", "kind", "phase", "progress", "rate", "elapsed", "eta", "counters"):
                table.add_column(col)
            for r in runs:
                table.add_row(
                    str(r.id),
                    r.kind,
                    r.phase or "—",
                    f"{r.current:,}/{r.total:,} ({r.percent:.0f}%)"
                    if r.total
                    else f"{r.current:,}",
                    f"{r.rate:.1f}/s",
                    humanize(r.elapsed_s),
                    humanize(r.eta_s),
                    " ".join(
                        f"{k}={v:,}"
                        for k, v in list(json.loads(r.counters or "{}").items())[:4]
                    )
                    or "—",
                )
            if not runs:
                table.add_row("—", "nothing running", "", "", "", "", "", "")
            return table

    with Live(render(), console=console, refresh_per_second=4) as live:
        try:
            while True:
                time.sleep(interval)
                live.update(render())
        except KeyboardInterrupt:
            pass


# -- classes ---------------------------------------------------------------


@classes_app.command("list")
def classes_list(config: Path = CONFIG_OPT):
    """Show detection classes, identifiers, and custom sub-classes."""
    cfg, Session, _indexer = _setup(config)
    from sqlalchemy import select

    from siteloom.store import CustomClass

    typer.echo("detection classes: " + ", ".join(cfg.detection.classes))
    typer.echo("\nidentifiers:")
    for key, ident in cfg.identity.identifiers.items():
        typer.echo(
            f"  {key:10s} {ident.algo:8s} thr={ident.threshold:.2f} "
            f"-> {', '.join(ident.applies_to)}"
        )
    with Session() as session:
        custom = session.scalars(select(CustomClass)).all()
    if custom:
        typer.echo("\ncustom sub-classes:")
        for c in custom:
            typer.echo(
                f"  {c.name:18s} refines {c.parent_class or 'any':10s} "
                f"{c.example_count} examples, thr={c.threshold:.2f}"
            )


@classes_app.command("add")
def classes_add(
    name: str = typer.Argument(..., help="Sub-class name, e.g. delivery-van"),
    config: Path = CONFIG_OPT,
    parent: str = typer.Option("", help="Detection class this refines"),
    threshold: float = typer.Option(0.85),
    description: str = typer.Option(""),
):
    """Define a custom sub-class."""
    _cfg, Session, _indexer = _setup(config)
    from sqlalchemy import select

    from siteloom.store import CustomClass

    slug = name.strip().lower().replace(" ", "-")
    with Session() as session:
        existing = session.scalar(select(CustomClass).filter_by(name=slug))
        if existing:
            typer.echo(f"{slug} already exists")
            raise typer.Exit(1)
        session.add(
            CustomClass(
                name=slug,
                parent_class=parent,
                description=description,
                threshold=threshold,
                created_at=_now(),
            )
        )
        session.commit()
    typer.echo(f"created custom class {slug}")


@classes_app.command("rebuild")
def classes_rebuild(config: Path = CONFIG_OPT):
    """Re-index custom-class examples from verified annotations."""
    cfg, Session, _indexer = _setup(config)
    import cv2

    from siteloom.identity import VectorStore
    from siteloom.identity.classes import CustomClassifier
    from siteloom.identity.embedders import GenericEmbedder

    embedder = GenericEmbedder(device=cfg.detection.device)

    def embed_crop(path: str):
        image = cv2.imread(path)
        return embedder.embed(image) if image is not None else None

    vectors = VectorStore(cfg.identity.vector_db_path)
    try:
        classifier = CustomClassifier(vectors)
        with Session() as session:
            count = classifier.rebuild(session, embed_crop)
    finally:
        vectors.close()
    typer.echo(f"indexed {count} custom-class examples")


# -- training --------------------------------------------------------------


@train_app.command("status")
def train_status(config: Path = CONFIG_OPT):
    """Show how much verified training data exists per person."""
    cfg, Session, _indexer = _setup(config)
    from siteloom.training.dataset import collect_face_samples
    from siteloom.training.face import person_coverage

    with Session() as session:
        samples = collect_face_samples(session)
    coverage = person_coverage(samples)
    if not coverage:
        typer.echo(
            "no verified face samples yet — import a Takeout archive and "
            "verify proposals at /training"
        )
        return
    ready = {p: n for p, n in coverage.items() if n >= cfg.training.min_samples_per_person}
    typer.echo(f"{len(samples)} verified face samples across {len(coverage)} people")
    typer.echo(f"{len(ready)} people meet the {cfg.training.min_samples_per_person}-sample minimum\n")
    for person, n in list(coverage.items())[:25]:
        mark = "✓" if n >= cfg.training.min_samples_per_person else " "
        typer.echo(f"  {mark} {n:4d}  {person}")


@train_app.command("enroll")
def train_enroll(config: Path = CONFIG_OPT, quiet: bool = QUIET_OPT):
    """Enroll verified faces into the recognition collection.

    Sweeps every verified, not-yet-enrolled face annotation and adds its
    embedding under the labeled identity. Run once after upgrading (or
    after bulk-verifying) so confirmed people are recognizable live and
    via the CompreFace-compatible API. Idempotent.
    """
    cfg, Session, indexer = _setup(config)
    from siteloom.identity import VectorStore
    from siteloom.identity.embedders import FaceEmbedder
    from siteloom.identity.enroll import enroll_verified
    from siteloom.progress import ProgressReporter

    face_cfg = cfg.identity.identifiers.get("face")
    max_vectors = face_cfg.max_vectors_per_identity if face_cfg else 20
    # Embedded Qdrant allows ONE client per path per machine — reuse the
    # store the shared bootstrap already opened rather than opening a
    # second one and deadlocking ourselves.
    if indexer.resolver is not None:
        vectors = indexer.resolver.vectors
        owns_vectors = False
    else:
        vectors = VectorStore(cfg.identity.vector_db_path)
        owns_vectors = True
    embedder = FaceEmbedder(
        projection_path=cfg.identity.face_projection_path or None
    )
    stats = None
    try:
        with ProgressReporter(
            Session,
            "face-enroll",
            resume_command=f"siteloom train enroll --config {config}",
            bar=not quiet,
        ) as progress:
            with Session() as session:
                with progress.phase("Enrolling verified faces"):
                    stats = enroll_verified(
                        session, vectors, embedder, max_vectors, progress=progress
                    )
    finally:
        if owns_vectors:
            vectors.close()
    if stats is None:
        raise typer.Exit(INTERRUPTED_EXIT)
    typer.echo(
        f"enrolled {stats.enrolled} embeddings across {stats.identities} people "
        f"({stats.skipped_cap} skipped: per-person cap reached or unusable crop)"
    )


@train_app.command("face")
def train_face(
    config: Path = CONFIG_OPT,
    apply_threshold: bool = typer.Option(
        False,
        "--apply-threshold",
        help="Write the newly-tuned face threshold back to the config",
    ),
):
    """Fine-tune the face embedding on verified samples.

    Learns a linear projection over SFace features so your people separate
    better. Only adopted if held-out AUC improves.
    """
    cfg, Session, _indexer = _setup(config)
    from siteloom.identity.embedders import FaceEmbedder
    from siteloom.store import TrainingRun
    from siteloom.training.dataset import collect_face_samples, split_by_person
    from siteloom.training.face import train_face_projection

    with Session() as session:
        samples = collect_face_samples(
            session, min_per_person=cfg.training.min_samples_per_person
        )
        if len(samples) < 4:
            typer.echo(
                f"only {len(samples)} verified samples meet the minimum of "
                f"{cfg.training.min_samples_per_person} per person — verify more "
                "training data at /training first",
                err=True,
            )
            raise typer.Exit(1)
        train, val = split_by_person(samples, cfg.training.val_fraction)
        run = TrainingRun(
            kind="face-embed",
            started_at=_now(),
            sample_count=len(samples),
            identity_count=len({s.person for s in samples}),
        )
        session.add(run)
        session.commit()
        run_id = run.id

    typer.echo(f"training on {len(train)} samples, validating on {len(val)}…")
    # Base embedder: never load an existing projection, or fine-tunes stack.
    embedder = FaceEmbedder(projection_path=None)
    result = train_face_projection(
        train,
        val,
        embedder,
        output_dir=cfg.training.output_dir,
        output_dim=cfg.training.embed_output_dim,
        epochs=cfg.training.embed_epochs,
        lr=cfg.training.embed_lr,
        threshold=cfg.identity.identifiers["face"].threshold,
    )

    with Session() as session:
        run = session.get(TrainingRun, run_id)
        run.finished_at = _now()
        run.status = "complete" if result.improved else "no-improvement"
        run.metrics = json.dumps(result.as_dict())
        run.artifact_path = result.projection_path
        run.notes = result.message
        session.commit()

    old_threshold = result.after.threshold
    new_threshold = result.after.best_threshold
    typer.echo(
        f"""
  people            {result.people}
  train / val       {result.train_samples} / {result.val_samples}
  AUC               {result.before.auc:.4f} -> {result.after.auc:.4f}
  same/diff margin  {result.before.margin:.4f} -> {result.after.margin:.4f}
  accuracy @{old_threshold:.2f}    {result.before.accuracy:.4f} -> {result.after.accuracy:.4f}
  accuracy @{new_threshold:.2f} (tuned) {result.after.best_accuracy:.4f}

{result.message}"""
    )
    if result.projection_path:
        typer.echo(f"projection: {result.projection_path}")

    # A projection reshapes the similarity distribution, so the previous
    # threshold is stale. Judging the new space at the old cutoff
    # understates it, and leaving the cutoff unchanged in production
    # throws away most of the gain.
    if result.improved and abs(new_threshold - old_threshold) > 0.01:
        if apply_threshold:
            from siteloom.config import save_config

            cfg.identity.identifiers["face"].threshold = round(new_threshold, 3)
            written = save_config(cfg)
            typer.echo(
                f"\nface threshold {old_threshold:.2f} -> {new_threshold:.2f} "
                f"written to {written}"
            )
        else:
            typer.echo(
                f"\n[!] The new embedding space wants a different threshold: "
                f"{old_threshold:.2f} -> {new_threshold:.2f} "
                f"(accuracy {result.after.accuracy:.4f} -> {result.after.best_accuracy:.4f}).\n"
                f"    Re-run with --apply-threshold to write it to your config, "
                f"or set identity.identifiers.face.threshold manually."
            )


@train_app.command("export-detector")
def train_export_detector(
    config: Path = CONFIG_OPT,
    output: Path = typer.Option(None, help="Dataset directory"),
):
    """Export verified face boxes as a YOLO dataset."""
    cfg, Session, _indexer = _setup(config)
    from siteloom.training.detector import export_yolo_dataset

    out = output or Path(cfg.training.output_dir) / "face-dataset"
    with Session() as session:
        result = export_yolo_dataset(session, out)
    typer.echo(
        f"{result.train_images} train / {result.val_images} val images, "
        f"{result.boxes} boxes\n{result.dataset_yaml}"
    )
    if result.message:
        typer.echo(result.message, err=True)


@train_app.command("detector")
def train_detector(
    config: Path = CONFIG_OPT,
    dataset: Path = typer.Option(None, help="dataset.yaml (default: export first)"),
    epochs: int = typer.Option(None),
):
    """Train a YOLO face detector on verified boxes.

    Improves face DETECTION on your own imagery. Identification stays with
    the embedding pipeline.
    """
    cfg, Session, _indexer = _setup(config)
    from siteloom.store import TrainingRun
    from siteloom.training.detector import export_yolo_dataset, train_face_detector

    if dataset is None:
        out = Path(cfg.training.output_dir) / "face-dataset"
        with Session() as session:
            export = export_yolo_dataset(session, out)
        if export.train_images == 0:
            typer.echo(export.message, err=True)
            raise typer.Exit(1)
        dataset = Path(export.dataset_yaml)
        typer.echo(
            f"exported {export.train_images} train / {export.val_images} val images"
        )

    with Session() as session:
        run = TrainingRun(kind="face-detect", started_at=_now())
        session.add(run)
        session.commit()
        run_id = run.id

    result = train_face_detector(
        dataset,
        base_model=cfg.training.detector_model,
        epochs=epochs or cfg.training.detector_epochs,
        imgsz=cfg.training.detector_imgsz,
        device=cfg.detection.device,
        project=Path(cfg.training.output_dir) / "detector",
    )
    with Session() as session:
        run = session.get(TrainingRun, run_id)
        run.finished_at = _now()
        run.status = "complete"
        run.metrics = json.dumps(result["metrics"])
        run.artifact_path = result["weights"]
        session.commit()
    typer.echo(f"weights: {result['weights']}\nmetrics: {result['metrics']}")
