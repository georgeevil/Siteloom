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

CONFIG_OPT = typer.Option("site.yaml", "--config", "-c", help="Site config YAML")


def _setup(config_path):
    """Shared bootstrap: config, DB session factory, dispatcher, resolver."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    from siteloom.config import load_config
    from siteloom.identity import IdentityResolver, VectorStore
    from siteloom.ingest import build_dispatcher
    from siteloom.library import LibraryIndexer
    from siteloom.store import get_session, init_db, make_engine

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
):
    """Run detection over pending items. Resumable — stop and rerun freely."""
    cfg, _Session, indexer = _setup(config)
    batch = limit or (10**9 if all_pending else cfg.library.batch_size)
    result = indexer.process(
        source_id=source_id,
        limit=batch,
        identify=not no_identify and cfg.library.identify_on_index,
    )
    typer.echo(
        f"indexed {result.processed} items ({result.annotations} boxes), "
        f"{result.failed} failed, {result.remaining} still pending"
    )


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
    no_auto_verify: bool = typer.Option(
        False,
        help="Do not auto-verify unambiguous (1 face + 1 name) assignments",
    ),
):
    """Import a Takeout tree: people tags, face detection, name proposals."""
    _cfg, _Session, indexer = _setup(config)
    from siteloom.library.takeout import TakeoutImporter

    importer = TakeoutImporter(
        indexer, auto_verify_unambiguous=not no_auto_verify
    )
    stats = importer.import_tree(path, limit=limit)
    typer.echo(
        f"""Takeout import complete
  media files seen      {stats.items_seen}
  sidecars matched      {stats.sidecars_matched}
  people tags imported  {stats.people_tags} ({len(stats.people)} distinct people)
  faces detected        {stats.faces_detected}
  unambiguous (1:1)     {stats.unambiguous}{'  [auto-verified]' if not no_auto_verify else ''}
  matched by gallery    {stats.matched}  [needs review]
  unassigned faces      {stats.unresolved}

Review at /training before training."""
    )
    top = sorted(stats.people.items(), key=lambda kv: -kv[1])[:10]
    if top:
        typer.echo("\nMost-tagged people:")
        for name, n in top:
            typer.echo(f"  {n:5d}  {name}")


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


@train_app.command("face")
def train_face(config: Path = CONFIG_OPT):
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

    typer.echo(
        f"""
  people            {result.people}
  train / val       {result.train_samples} / {result.val_samples}
  AUC               {result.before.auc:.4f} -> {result.after.auc:.4f}
  accuracy @{result.after.threshold:.2f}    {result.before.accuracy:.4f} -> {result.after.accuracy:.4f}
  same/diff margin  {result.before.margin:.4f} -> {result.after.margin:.4f}

{result.message}"""
    )
    if result.projection_path:
        typer.echo(f"projection: {result.projection_path}")


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
