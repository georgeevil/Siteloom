"""Siteloom CLI: init-db, run, serve, cameras."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from siteloom.cli_library import (
    classes_app,
    jobs_app,
    library_app,
    takeout_app,
    train_app,
)

app = typer.Typer(help="Siteloom — video & audio intelligence platform.")
app.add_typer(library_app, name="library")
app.add_typer(takeout_app, name="takeout")
app.add_typer(classes_app, name="classes")
app.add_typer(train_app, name="train")
app.add_typer(jobs_app, name="jobs")

CONFIG_OPT = typer.Option("site.yaml", "--config", "-c", help="Site config YAML")


@app.command()
def init_db(config: Path = CONFIG_OPT):
    """Create database tables."""
    from siteloom.config import load_config
    from siteloom.store import init_db as _init, make_engine

    cfg = load_config(config)
    _init(make_engine(cfg.storage.db_url))
    typer.echo(f"initialized {cfg.storage.db_url}")


@app.command()
def run(
    config: Path = CONFIG_OPT,
    max_frames: int | None = typer.Option(None, help="Stop after N frames per camera (debug)"),
):
    """Run ingestion over all configured cameras."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from siteloom.config import load_config
    from siteloom.ingest import IngestService

    service = IngestService(load_config(config))
    service.run(max_frames=max_frames)


@app.command()
def serve(
    config: Path = CONFIG_OPT,
    host: str = "127.0.0.1",
    port: int = 8000,
):
    """Serve the event-browser web UI."""
    import uvicorn

    from siteloom.config import load_config
    from siteloom.web.app import create_app

    uvicorn.run(create_app(load_config(config)), host=host, port=port)


@app.command()
def backfill(
    path: Path = typer.Argument(..., help="Directory or file of photos/videos"),
    config: Path = CONFIG_OPT,
    camera_id: str = typer.Option(
        "backfill", help="Camera id recorded on backfilled events"
    ),
    sample_fps: float = typer.Option(2.0, help="Frames/second sampled from videos"),
):
    """Backfill existing photos/video into the identity database (PRD §6.6).

    Runs the exact live pipeline (detection -> identity -> store) over a
    media archive; events carry the source file's mtime as timestamp.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from siteloom.config import CameraConfig, load_config
    from siteloom.ingest import IngestService

    cfg = load_config(config)
    cam = CameraConfig(
        id=camera_id,
        name=f"Backfill: {path}",
        adapter="file",
        source=str(path),
        sample_fps=sample_fps,
        modules=["detection", "identity", "audio"],
    )
    cfg.cameras = [cam]
    service = IngestService(cfg)
    count = service.run_camera(cam)
    typer.echo(f"backfill complete: {count} frames processed from {path}")


@app.command()
def sync_bookings(config: Path = CONFIG_OPT):
    """Sync guest bookings from the configured iCal feed (PRD §6.7)."""
    from siteloom.config import load_config
    from siteloom.guests import sync_bookings as _sync
    from siteloom.store import get_session, init_db, make_engine

    cfg = load_config(config)
    engine = make_engine(cfg.storage.db_url)
    init_db(engine)
    with get_session(engine)() as session:
        count = _sync(session, cfg.guests)
    typer.echo(f"synced {count} booking(s)")


@app.command()
def frigate(config: Path = CONFIG_OPT):
    """Consume Frigate events over MQTT and run recognition on them.

    Frigate keeps doing RTSP ingest + object detection; Siteloom plays
    the Double Take + CompreFace role: snapshot -> face/vehicle identity
    against the shared collection, results to siteloom/identity on MQTT
    and to configured webhooks.
    """
    from siteloom.config import load_config
    from siteloom.identity import IdentityResolver, VectorStore
    from siteloom.ingest import build_dispatcher
    from siteloom.integrations import MqttPublisher, WebhookNotifier
    from siteloom.integrations.frigate import FrigateConsumer
    from siteloom.progress import setup_logging
    from siteloom.store import get_session, init_db, make_engine

    setup_logging()
    cfg = load_config(config)
    if not cfg.integrations.frigate.enabled:
        typer.echo(
            "frigate integration is disabled — set integrations.frigate.enabled: "
            "true (and integrations.mqtt) in your config",
            err=True,
        )
        raise typer.Exit(1)
    engine = make_engine(cfg.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    dispatcher = build_dispatcher(cfg)
    resolver = None
    if cfg.identity.enabled:
        resolver = IdentityResolver(
            cfg.identity, VectorStore(cfg.identity.vector_db_path)
        )
    consumer = FrigateConsumer(
        cfg,
        Session,
        dispatcher,
        resolver,
        publisher=MqttPublisher(cfg.integrations.mqtt),
        notifier=WebhookNotifier(cfg.integrations.webhooks),
    )
    try:
        consumer.run()
    except KeyboardInterrupt:
        stats = consumer.stats
        typer.echo(
            f"\nstopped: {stats.received} received, {stats.processed} processed, "
            f"{stats.identities} identities, {stats.skipped} skipped "
            f"({stats.by_reason}), {stats.errors} errors"
        )


@app.command()
def cameras(config: Path = CONFIG_OPT):
    """List streams visible to each configured adapter (e.g. UniFi camera ids)."""
    from siteloom.config import load_config
    from siteloom.ingest import build_adapter

    cfg = load_config(config)
    seen: set[str] = set()
    for cam in cfg.cameras:
        key = f"{cam.adapter}:{cam.source if cam.adapter != 'unifi' else ''}"
        if key in seen:
            continue
        seen.add(key)
        adapter = build_adapter(cam, cfg)
        try:
            adapter.connect()
            for stream in adapter.list_streams():
                typer.echo(f"[{cam.adapter}] {stream.id}  {stream.name}")
        except Exception as exc:
            typer.echo(f"[{cam.adapter}] error: {exc}", err=True)
        finally:
            adapter.close()


if __name__ == "__main__":
    app()
