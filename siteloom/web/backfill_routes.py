"""The backfill console (CLD-93).

`siteloom backfill-unifi` was terminal-only, which left the console able
to tell an operator that a gap exists — `siteloom run` heartbeats an
OperationRun and /jobs shows it going stale (CLD-15) — and unable to help
them close it. It is also the only camera-derived path that produces
`NoiseEvent` rows at all: audio never reaches the detector on a live
stream, so /noise can only ever fill from downloaded NVR clips.

Two halves, and the read is the more valuable one:

* **What is already covered.** The resumability model is "the remaining
  clips are the ones still pending" (`BackfillClip`), and nothing
  surfaced it. Pending / done / failed and the covered range, per camera,
  is the whole state of a resumable sweep.
* **Starting one.** Camera, start, optional end, and the motion-window vs
  full-sweep choice. Full sweep is the non-default because it downloads
  every minute of the range rather than the minutes the NVR thinks
  something moved, and the form says so.

Deliberately scoped to `backfill-unifi`. Plain `siteloom backfill` over a
media archive has no per-file checkpointing (CLD-12) — an interrupted run
restarts from the top — so putting it behind the same button as the
resumable path would have the screen imply a guarantee it does not have.
The page says which one is missing and why, rather than quietly omitting
it.

The run happens in the *serving* process, one at a time, exactly like
/jobs/reindex: the pipeline keeps per-camera tracker state and the
embedded vector store is one client per path per machine, so a second
concurrent run is a failure rather than a slowdown.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from fastapi import Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select

from siteloom.store import BackfillClip
from siteloom.web import nav

log = logging.getLogger(__name__)

#: One UI-triggered backfill at a time, in the serve process, observed
#: through OperationRun like every other long job. `last` keeps the most
#: recent run's scan/process outcome so a re-run over an
#: already-covered range reads as the no-op it is instead of looking
#: like nothing happened at all.
_state: dict = {"thread": None, "last": None}

#: Chunk sizes offered for a full sweep. Deliberately a short list: the
#: number trades clip count against clip length, and every option here
#: downloads the entire range either way.
CHUNK_CHOICES = (5.0, 15.0, 30.0, 60.0)


def parse_range(start: str, end: str, zone=None) -> tuple[datetime, datetime]:
    """The window to scan, or a ValueError naming what is wrong.

    A `datetime-local` input has no offset, and operators think in wall
    time — the **site's** wall time (CLD-100), which is what `zone`
    carries; None is the unset-zone rung and reads as UTC. Interpreting
    in the server process's zone (`.astimezone()`, the old behaviour) is
    wrong the day the box serving the site sits in a different zone than
    the cameras. Returned aware; `UnifiBackfill.scan` converts. An
    open-ended range means "up to now", which is the common case after a
    crash: you know when ingest stopped, not when it should stop catching
    up.
    """
    from siteloom.localtime import as_aware

    text = (start or "").strip()
    if not text:
        raise ValueError("Enter a start time.")
    try:
        begins = as_aware(datetime.fromisoformat(text), zone)
    except ValueError:
        raise ValueError(f"{text!r} is not a date and time.") from None
    if (end or "").strip():
        try:
            finishes = as_aware(datetime.fromisoformat(end.strip()), zone)
        except ValueError:
            raise ValueError(f"{end.strip()!r} is not a date and time.") from None
    else:
        finishes = datetime.now(timezone.utc)
    if finishes <= begins:
        raise ValueError("The end of the range must be after its start.")
    return begins, finishes


def clip_rows(session, cameras) -> list[dict]:
    """Per-camera BackfillClip state: the read this screen exists for.

    Counts are scoped per camera and `failed` is reported separately from
    `pending`, following the library indexer's rule — nothing picks a
    failed clip up again without an explicit retry, so folding the two
    together would show a sweep as nearly done when part of it will never
    run.

    Cameras with clips but no configuration left (renamed, removed) still
    get a row, flagged. Their footage is in the database either way, and
    dropping them would make the counts on this page quietly disagree
    with the table.
    """
    grouped = session.execute(
        select(
            BackfillClip.camera_id,
            BackfillClip.status,
            func.count(),
            func.coalesce(func.sum(BackfillClip.frames), 0),
            func.min(BackfillClip.start),
            func.max(BackfillClip.end),
            func.max(BackfillClip.processed_at),
        ).group_by(BackfillClip.camera_id, BackfillClip.status)
    ).all()

    per_camera: dict[str, dict] = {}
    for camera_id, status, count, frames, first, last, processed in grouped:
        row = per_camera.setdefault(
            camera_id,
            {
                "id": camera_id,
                "name": camera_id,
                "configured": False,
                "pending": 0,
                "done": 0,
                "failed": 0,
                "total": 0,
                "frames": 0,
                "covered_from": None,
                "covered_to": None,
                "done_from": None,
                "done_to": None,
                "last_processed": None,
            },
        )
        row[status] = row.get(status, 0) + count
        row["total"] += count
        row["frames"] += frames or 0
        row["covered_from"] = min(filter(None, (row["covered_from"], first)))
        row["covered_to"] = max(filter(None, (row["covered_to"], last)))
        if processed is not None:
            row["last_processed"] = max(
                filter(None, (row["last_processed"], processed))
            )
        if status == "done":
            row["done_from"] = first
            row["done_to"] = last

    rows = []
    for cam in cameras:
        row = per_camera.pop(cam.id, None) or {
            "id": cam.id,
            "pending": 0,
            "done": 0,
            "failed": 0,
            "total": 0,
            "frames": 0,
            "covered_from": None,
            "covered_to": None,
            "done_from": None,
            "done_to": None,
            "last_processed": None,
        }
        row["name"] = cam.name or cam.id
        row["id"] = cam.id
        row["configured"] = True
        rows.append(row)
    # Whatever is left has clips but no camera in the config any more.
    rows.extend(sorted(per_camera.values(), key=lambda r: r["id"]))
    return rows


def register(app, templates, Session, config) -> None:
    nav.add("/backfill", "Backfill", "BF", after="/jobs")

    def unifi_cameras():
        return [c for c in config.cameras if c.adapter == "unifi"]

    def page_context(**kw) -> dict:
        cameras = unifi_cameras()
        with Session() as session:
            rows = clip_rows(session, cameras)
        outstanding = sum(r["pending"] for r in rows)
        failed = sum(r["failed"] for r in rows)
        clips = sum(r["total"] for r in rows)
        # Three different nothings, and an operator acts differently on
        # each: no camera can be backfilled at all / nothing has been
        # asked for yet / everything asked for is finished.
        if not cameras and not clips:
            empty_reason = "no-cameras"
        elif not clips:
            empty_reason = "no-clips"
        elif not outstanding and not failed:
            empty_reason = "all-done"
        else:
            empty_reason = None
        running = _state["thread"]
        return {
            "site_name": config.site_name or config.site_id,
            "cameras": cameras,
            "rows": rows,
            "clips": clips,
            "outstanding": outstanding,
            "failed": failed,
            "empty_reason": empty_reason,
            "chunk_choices": CHUNK_CHOICES,
            "running": running is not None and running.is_alive(),
            "last": _state["last"],
            "form": {},
            "error": None,
            **kw,
        }

    def page(request: Request, status_code: int = 200, **kw):
        return templates.TemplateResponse(
            request, "backfill.html", page_context(**kw), status_code=status_code
        )

    @app.get("/backfill")
    def backfill_page(request: Request):
        return page(request)

    @app.post("/backfill/start")
    def start_backfill(
        request: Request,
        camera: str = Form(""),
        start: str = Form(""),
        end: str = Form(""),
        mode: str = Form("events"),
        chunk_minutes: float = Form(15.0),
        retry_failed: str = Form("0"),
    ):
        """Scan the NVR for the range, then ingest what it registered.

        One at a time. The second start is refused rather than queued:
        two sweeps sharing this process would share the detector's
        per-camera tracker state and the embedded vector store, which is
        one client per path per machine — the failure would land inside
        the pipeline, far from the button that caused it.
        """
        form = {
            "camera": camera,
            "start": start,
            "end": end,
            "mode": mode,
            "chunk_minutes": chunk_minutes,
            "retry_failed": retry_failed == "1",
        }
        thread = _state["thread"]
        if thread is not None and thread.is_alive():
            return page(
                request,
                409,
                error="A backfill is already running. Wait for it to finish.",
                form=form,
            )
        cam = next((c for c in unifi_cameras() if c.id == camera), None)
        if cam is None:
            return page(
                request,
                400,
                error=(
                    f"{camera!r} is not a configured UniFi camera."
                    if camera
                    else "Choose a camera."
                ),
                form=form,
            )
        try:
            from siteloom.localtime import site_zone

            begins, finishes = parse_range(start, end, site_zone(config))
        except ValueError as exc:
            return page(request, 400, error=str(exc), form=form)

        sweep = mode == "sweep"
        chunk = max(1.0, min(float(chunk_minutes), 24 * 60)) if sweep else None
        wants_retry = retry_failed == "1"
        # Rebuilt from the whole invocation, not a couple of interesting
        # fields: an operator continuing this run in a shell must get the
        # sweep and the retry back, or the shell run means something else.
        resume = " ".join(
            [
                "siteloom backfill-unifi",
                cam.id,
                f"--start {begins.isoformat()}",
                f"--end {finishes.isoformat()}",
            ]
            + ([f"--chunk-minutes {chunk:g}"] if chunk else [])
            + (["--retry-failed"] if wants_retry else [])
        )
        _state["last"] = {
            "camera": cam.id,
            "sweep": sweep,
            "chunk_minutes": chunk,
            "retry_failed": wants_retry,
            "start": begins,
            "end": finishes,
            "status": "running",
            "resume": resume,
        }

        def work():
            from siteloom.backfill import UnifiBackfill
            from siteloom.ingest import IngestService
            from siteloom.progress import ProgressReporter

            last = _state["last"]
            try:
                service = IngestService(config)
                runner = UnifiBackfill(service, cam)
                with ProgressReporter(
                    Session,
                    "backfill-unifi",
                    target=cam.id,
                    resume_command=resume,
                    bar=False,
                ) as progress:
                    with progress.phase("Scanning NVR events"):
                        scan = runner.scan(
                            begins, finishes, chunk_minutes=chunk
                        )
                    # On the jobs board as well as this page: a sweep that
                    # registered nothing looks identical to a stalled one
                    # unless the counters say what it found.
                    progress.bump(
                        registered=scan.added, already_covered=scan.skipped
                    )
                    last.update(
                        added=scan.added,
                        skipped=scan.skipped,
                        pending=scan.pending,
                        status="processing",
                    )
                    result = runner.process(
                        retry_failed=wants_retry, progress=progress
                    )
                    last.update(
                        status="complete",
                        processed=result.processed,
                        frames=result.frames,
                        clip_failures=result.failed,
                        remaining=result.remaining,
                        failed_total=result.failed_total,
                        retried=result.retried,
                    )
            except Exception as exc:  # pragma: no cover — via OperationRun
                log.exception("backfill of %s failed", cam.id)
                last.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                # A run stopped by signal leaves `status` where the
                # reporter left it: the OperationRun row is the record of
                # how it ended, and inventing "complete" here would
                # contradict it.
                if last.get("status") in ("running", "processing"):
                    last["status"] = "stopped"
                _state["thread"] = None

        thread = threading.Thread(target=work, name="siteloom-backfill", daemon=True)
        _state["thread"] = thread
        thread.start()
        return RedirectResponse("/backfill", status_code=303)
