"""Siteloom CLI: init-db, run, serve, doctor, cameras."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import typer

from siteloom.cli_users import users_app
from siteloom.cli_identity import identity_app
from siteloom.cli_lab import lab_app
from siteloom.cli_service import service_app
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
app.add_typer(users_app, name="users")
app.add_typer(service_app, name="service")
app.add_typer(lab_app, name="lab")
app.add_typer(identity_app, name="identity")

events_app = typer.Typer(help="Event maintenance.")
app.add_typer(events_app, name="events")

CONFIG_OPT = typer.Option("site.yaml", "--config", "-c", help="Site config YAML")


@events_app.command()
def retag(config: Path = CONFIG_OPT):
    """Recompute every event's significance flag from the current rules.

    The gate normally flips at ingest time; this reapplies it to existing
    rows — after installing the feature on a database that predates it, or
    after changing the rules. Uses the same per-camera effective rules as
    ingest (EventConfig.for_camera), is idempotent, and never deletes
    anything.
    """
    from sqlalchemy import func

    from siteloom.config import load_config
    from siteloom.store import (
        Detection,
        Event,
        get_session,
        init_db as _init,
        make_engine,
    )

    cfg = load_config(config)
    engine = make_engine(cfg.storage.db_url)
    _init(engine)
    Session = get_session(engine)
    overrides = {cam.id: cfg.events.for_camera(cam) for cam in cfg.cameras}
    with Session() as session:
        before = (
            session.query(Event).filter(Event.significant.is_(True)).count()
        )
        changed = 0
        # Externally curated events (Frigate consumer sets external_id)
        # are pre-gated upstream and stay significant; events from cameras
        # no longer in the config are judged by the site-wide defaults.
        for event in (
            session.query(Event).filter(Event.external_id.is_(None)).yield_per(500)
        ):
            rules = overrides.get(event.camera_id, cfg.events)
            significant = (
                event.detection_count >= rules.min_detections
                and event.best_confidence >= rules.min_confidence
            )
            # Duration gates only when configured — out-of-order frame
            # timestamps can make last_seen precede first_seen.
            if significant and rules.min_duration_s > 0:
                duration = (event.last_seen - event.first_seen).total_seconds()
                significant = duration >= rules.min_duration_s
            if event.significant != significant:
                event.significant = significant
                changed += 1
        # Backfill confidence_sum for rows predating the column (CLD-40).
        # A zero sum alongside recorded detections can only mean the row
        # was written before the column existed; live sums are never
        # overwritten. Frigate-consumed events have no Detection rows and
        # accumulate their sums going forward instead.
        sums = dict(
            session.query(Detection.event_id, func.sum(Detection.confidence))
            .group_by(Detection.event_id)
            .all()
        )
        backfilled = 0
        for event in (
            session.query(Event)
            .filter(Event.confidence_sum == 0.0, Event.detection_count > 0)
            .yield_per(500)
        ):
            total_conf = sums.get(event.id)
            if total_conf:
                event.confidence_sum = total_conf
                backfilled += 1
        session.commit()
        after = session.query(Event).filter(Event.significant.is_(True)).count()
        total = session.query(Event).count()
    typer.echo(
        f"retagged {changed} of {total} events: "
        f"{before} -> {after} significant ({total - after} ephemeral); "
        f"backfilled mean confidence on {backfilled}"
    )


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
    ctx: typer.Context,
    config: Path = CONFIG_OPT,
    max_frames: int | None = typer.Option(None, help="Stop after N frames per camera (debug)"),
    log_file: str = typer.Option(None, "--log-file", help="Also write logs here"),
    quiet: bool = typer.Option(False, "--quiet", help="No progress bar"),
):
    """Run ingestion over all configured cameras.

    A soak is a long-running operation like any other, so it heartbeats an
    OperationRun row (CLD-15): `siteloom jobs` and /jobs can watch it from
    another terminal, and a crash at hour 3 shows up as stale rather than
    as a healthy-looking row. Unlike a bounded job it has nothing to skip
    on resume — see `IngestService.run` for that decision.
    """
    from siteloom.cli_library import INTERRUPTED_EXIT, _resume_command
    from siteloom.config import load_config
    from siteloom.ingest import IngestService
    from siteloom.progress import ProgressReporter, setup_logging

    setup_logging(level="INFO", log_file=log_file)
    cfg = load_config(config)
    service = IngestService(cfg)
    try:
        with ProgressReporter(
            service.Session,
            "run",
            target=cfg.site_id,
            resume_command=_resume_command(ctx),
            bar=not quiet,
        ) as progress:
            service.run(max_frames=max_frames, progress=progress)
    finally:
        # Release the embedded Qdrant client deterministically. Left to
        # the interpreter, QdrantClient.__del__ runs during shutdown and
        # dies on an import that can no longer resolve, printing a
        # traceback immediately below "Progress saved. Resume with:" —
        # which reads as though the save failed. It did not, but an
        # operator ending a soak has no way to tell.
        #
        # This is the "deliberate shutdown" that VectorStore.close()
        # documents, and it belongs here rather than in IngestService:
        # `run` owns its whole process, whereas the web process shares
        # one store with the recognition API and enrollment and must
        # never have it closed out from under them.
        if service.resolver is not None:
            service.resolver.vectors.close()
    if service.stopped:
        raise typer.Exit(INTERRUPTED_EXIT)


@app.command()
def serve(
    ctx: typer.Context,
    config: Path = CONFIG_OPT,
    host: str = typer.Option(None, "--host", help="Bind address (default: service.host)"),
    port: int = typer.Option(None, "--port", help="Bind port (default: service.port)"),
    log_file: str = typer.Option(None, "--log-file", help="Also write logs here"),
    log_level: str = typer.Option(
        None, "--log-level", help="DEBUG|INFO|WARNING (default: service.log_level)"
    ),
):
    """Serve the event-browser web UI.

    Long-lived, so run it under a service manager — `siteloom service
    install` generates one. It stops on SIGINT/SIGTERM, exposes /healthz
    and /readyz for whatever is watching it, and heartbeats an
    OperationRun row so `siteloom jobs` sees it like any other long
    operation.

    Bind address, port and log level fall back to the `service:` section
    of the config, so the generated unit and a hand-run server agree
    without repeating themselves.
    """
    from siteloom.cli_library import INTERRUPTED_EXIT, _resume_command
    from siteloom.config import load_config
    from siteloom.progress import setup_logging
    from siteloom.serve import serve_supervised

    cfg = load_config(config)
    host = host if host is not None else cfg.service.host
    port = port if port is not None else cfg.service.port
    log_level = log_level if log_level is not None else cfg.service.log_level
    # The project's own logging, so a service-managed run writes the same
    # lines to the same file as everything else rather than uvicorn's.
    setup_logging(level=log_level, log_file=log_file)
    stopped = serve_supervised(
        cfg,
        host=host,
        port=port,
        log_level=log_level,
        resume_command=_resume_command(ctx),
    )
    if stopped:
        # 128 + SIGINT, same as every other interruptible command: a
        # server the operator stopped is neither a success nor a crash,
        # and a supervisor told otherwise restarts it.
        raise typer.Exit(INTERRUPTED_EXIT)


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
def backfill_unifi(
    ctx: typer.Context,
    camera: list[str] = typer.Argument(
        None, help="Configured camera id(s) (adapter: unifi); several run side by side"
    ),
    config: Path = CONFIG_OPT,
    start: datetime = typer.Option(
        ..., help="Range start (local time unless an offset is given)"
    ),
    end: datetime = typer.Option(None, help="Range end (default: now)"),
    all_cameras: bool = typer.Option(
        False, "--all", help="Every configured UniFi camera, instead of naming them"
    ),
    parallel: int = typer.Option(
        None, help="Cameras processed at once (default: backfill.parallel in config)"
    ),
    pad_seconds: float = typer.Option(
        5.0, help="Padding added to each side of an NVR event window"
    ),
    chunk_minutes: float = typer.Option(
        None,
        help="Sweep the whole range in fixed chunks instead of NVR event windows",
    ),
    min_score: int = typer.Option(0, help="Skip NVR events scoring below this"),
    limit: int = typer.Option(None, help="Max clips processed this run, per camera"),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Re-queue clips that failed on an earlier run"
    ),
    scan_only: bool = typer.Option(
        False, "--scan-only", help="Register pending clips without processing them"
    ),
    log_file: str = typer.Option(None, "--log-file", help="Also write logs here"),
    quiet: bool = typer.Option(False, "--quiet", help="No progress bar"),
):
    """Index past UniFi Protect recordings (PRD §6.6).

    Asks the NVR for its motion/smart-detect events in the range, then
    downloads each recorded window and runs it through the exact live
    pipeline. Resumable — stop and rerun freely; processed windows are
    remembered by NVR event id. Several cameras run side by side, one
    job row each (CLD-317), bounded by --parallel.
    """
    from siteloom.backfill import ACTIVE_STATUSES, BackfillRequest, run_backfills
    from siteloom.cli_library import INTERRUPTED_EXIT, _resume_command
    from siteloom.config import load_config
    from siteloom.ingest import IngestService
    from siteloom.progress import ProgressReporter, setup_logging

    setup_logging(level="INFO", log_file=log_file)
    cfg = load_config(config)
    unifi = {c.id: c for c in cfg.cameras if c.adapter == "unifi"}
    wanted = list(dict.fromkeys(camera or []))
    if all_cameras and wanted:
        typer.echo("name cameras or pass --all, not both", err=True)
        raise typer.Exit(2)
    if all_cameras:
        wanted = list(unifi)
    if not wanted:
        typer.echo(
            f"name at least one camera (or --all); unifi cameras in {config}: "
            f"{list(unifi)}",
            err=True,
        )
        raise typer.Exit(2)
    cams_by_id = {c.id: c for c in cfg.cameras}
    for cam_id in wanted:
        if cam_id not in cams_by_id:
            typer.echo(
                f"no camera {cam_id!r} in {config}; known: {list(cams_by_id)}", err=True
            )
            raise typer.Exit(2)
    cams = [cams_by_id[cam_id] for cam_id in wanted]

    # Operators type wall time — the *site's* wall time (CLD-100), the
    # same frame the console's backfill form reads. An explicit offset in
    # the argument is honoured as given, and the resume commands the
    # console prints carry one. Never `.astimezone()` bare: that is the
    # server process's zone, wrong the day it differs from the site's.
    from siteloom.localtime import as_aware, site_zone

    zone = site_zone(cfg)
    start = as_aware(start, zone)
    end = as_aware(end, zone) if end else datetime.now(timezone.utc)
    request = BackfillRequest(
        start=start,
        end=end,
        pad_s=pad_seconds,
        chunk_minutes=chunk_minutes,
        min_score=min_score,
        limit=limit,
        retry_failed=retry_failed,
        scan_only=scan_only,
    )

    service = IngestService(cfg)
    # One resume line per camera, naming only that camera — a row on
    # /jobs is one camera's run, and its line must continue that run.
    resumes = {
        cam.id: _resume_command(ctx, camera=[cam.id], all_cameras=False) for cam in cams
    }
    # The reporters take no signals: with a worker per camera there is
    # one owner of Ctrl-C — the service's own handlers — and the runner
    # relays the stop to every row. A bar only when there is one camera;
    # two Rich bars in two threads fight over the terminal.
    with service._stop_signals(active=True):
        states = run_backfills(
            service,
            cams,
            request,
            parallel=cfg.backfill.parallel if parallel is None else parallel,
            make_reporter=lambda cam: ProgressReporter(
                service.Session,
                "backfill-unifi",
                target=cam.id,
                resume_command=resumes[cam.id],
                bar=not quiet and len(cams) == 1,
                signals=False,
            ),
            interrupt=lambda: service.stopped,
        )

    failed = interrupted = False
    for cam in cams:
        state = states[cam.id]
        label = f"[{cam.id}]" if len(cams) > 1 else ""
        status = state["status"]
        if status == "failed":
            failed = True
            typer.echo(f"{label} failed: {state.get('error', '')}".strip())
            continue
        if status in ACTIVE_STATUSES or status == "stopped":
            interrupted = True
            continue  # the reporter already printed the resume command
        if "added" in state:
            typer.echo(
                f"{label} scan: +{state['added']} new clip(s), {state['skipped']} "
                f"already covered, {state['pending']} pending".strip()
            )
        if scan_only:
            continue
        retried = f"retried {state['retried']}, " if state.get("retried") else ""
        typer.echo(
            f"{label} {retried}processed {state['processed']} clip(s) "
            f"({state['frames']} frames), {state['clip_failures']} failed, "
            f"{state['remaining']} still pending".strip()
        )
        if state["remaining"]:
            typer.echo(f"{label} continue with: {resumes[cam.id]}".strip())
        if state["failed_total"] and not retry_failed:
            typer.echo(
                f"{label} {state['failed_total']} clip(s) failed earlier and will "
                f"not be retried automatically; retry with: {resumes[cam.id]} "
                "--retry-failed".strip()
            )
    if interrupted:
        raise typer.Exit(INTERRUPTED_EXIT)
    if failed:
        raise typer.Exit(1)


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


@app.command()
def reset(
    config: Path = CONFIG_OPT,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be removed; change nothing"
    ),
    keep_users: bool = typer.Option(
        False, "--keep-users", help="Keep operator accounts and their sessions"
    ),
):
    """Erase everything this site has observed and learned.

    Events, detections, identities and their vectors, plate reads, the
    library index and its crops — the install is left as it came out of
    the box. The config, the library's original archives, downloaded
    model weights and the logs are kept (`siteloom/reset.py` says why).
    """
    from siteloom.cli_library import _light_setup
    from siteloom.reset import (
        UnsafeReset,
        check_safe,
        format_bytes,
        perform_reset,
        plan_reset,
    )

    cfg, Session = _light_setup(config, level="WARNING")
    with Session() as session:
        plan = plan_reset(cfg, session, keep_users=keep_users)
        try:
            check_safe(cfg, session, plan)
        except UnsafeReset as exc:
            typer.echo(f"refused: {exc}", err=True)
            raise typer.Exit(1) from None

        typer.echo(f"site: {cfg.site_name or cfg.site_id}")
        for table in plan.tables:
            if table.rows:
                typer.echo(f"  {table.rows:>7}  {table.name}")
        for target in plan.paths:
            if target.files:
                typer.echo(
                    f"  {target.files:>7}  files in {target.path} "
                    f"({format_bytes(target.bytes)} — {target.label})"
                )
        typer.echo(
            f"total: {plan.rows} rows, {plan.files} files "
            f"({format_bytes(plan.bytes)}); keeping {', '.join(plan.kept)}"
        )

        if plan.is_empty:
            typer.echo("already out of the box — nothing to remove")
            return
        if dry_run:
            typer.echo("dry run — nothing removed")
            return
        if not yes and not typer.confirm("permanently erase all of this?"):
            raise typer.Abort()

        problems = perform_reset(cfg, session, plan)

    for problem in problems:
        typer.echo(f"could not remove {problem}", err=True)
    if problems:
        typer.echo(f"reset incomplete: {len(problems)} path(s) left behind", err=True)
        raise typer.Exit(1)
    typer.echo("reset complete — the site is out of the box again")


if __name__ == "__main__":
    app()
