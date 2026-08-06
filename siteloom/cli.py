"""Siteloom CLI: init-db, run, serve, cameras."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

app = typer.Typer(help="Siteloom — video & audio intelligence platform.")

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
