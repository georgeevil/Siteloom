"""Siteloom CLI: init-db, run, serve, doctor, cameras."""

from __future__ import annotations

import json
import logging
import os
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
    log_file: str = typer.Option(None, "--log-file", help="Also write logs here"),
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG|INFO|WARNING"),
):
    """Serve the event-browser web UI.

    Long-lived, so run it under a service manager (docs/operations.md).
    It stops on SIGINT/SIGTERM, and exposes /healthz and /readyz for
    whatever is watching it.
    """
    import uvicorn

    from siteloom.config import load_config
    from siteloom.progress import setup_logging
    from siteloom.web.app import create_app

    # The project's own logging, so a service-managed run writes the same
    # lines to the same file as everything else rather than uvicorn's.
    setup_logging(level=log_level, log_file=log_file)
    cfg = load_config(config)
    logging.getLogger(__name__).info(
        "serving %s on http://%s:%d (pid %d)", cfg.site_id, host, port, os.getpid()
    )
    uvicorn.run(
        create_app(cfg), host=host, port=port, log_level=log_level.lower()
    )


@app.command()
def doctor(
    config: Path = CONFIG_OPT,
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output"),
):
    """Check that this deployment is fit to run, and say what to fix.

    Exits non-zero if anything failed, so it works as a service
    ExecStartPre or a monitoring probe.
    """
    from siteloom.config import load_config
    from siteloom.health import FAIL, OK, WARN, run_checks

    try:
        cfg = load_config(config)
    except Exception as exc:
        typer.echo(f"config {config}: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(2)

    report = run_checks(cfg)
    if json_out:
        typer.echo(json.dumps(report.as_dict(), indent=2))
        raise typer.Exit(0 if report.ok else 1)

    from rich.console import Console
    from rich.table import Table

    console = Console()
    marks = {OK: "[green]ok[/]", WARN: "[yellow]warn[/]", FAIL: "[red]FAIL[/]"}
    console.print(f"[bold]{cfg.site_name or cfg.site_id}[/] — {config}\n")
    # A table rather than f-string padding: paths are long, and a wrapped
    # detail has to stay in its column to remain readable.
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(justify="right", no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(overflow="fold")
    for check in report.checks:
        table.add_row(marks[check.status], check.name, check.detail)
        if check.remedy and check.status != OK:
            table.add_row("", "", f"[dim]-> {check.remedy}[/]")
    console.print(table)
    failed, warned = len(report.failed), len(report.warned)
    console.print(
        f"\n{len(report.checks)} checks: "
        f"[red]{failed} failed[/], [yellow]{warned} warnings[/]"
        if failed or warned
        else f"\n{len(report.checks)} checks, all clear"
    )
    raise typer.Exit(1 if failed else 0)


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

    # First-time setup: a configured console must be listable before any
    # camera entry references it — this command is how those entries get
    # their ids in the first place.
    if "unifi:" not in seen and cfg.unifi.host:
        from siteloom.adapters.unifi import UniFiProtectAdapter

        adapter = UniFiProtectAdapter(unifi=cfg.unifi)
        try:
            adapter.connect()
            for stream in adapter.list_streams():
                typer.echo(f"[unifi] {stream.id}  {stream.name}")
        except Exception as exc:
            typer.echo(f"[unifi] error: {exc}", err=True)
        finally:
            adapter.close()


if __name__ == "__main__":
    app()
