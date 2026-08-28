"""Backfill from an NVR's own recordings (PRD §6.6).

UniFi Protect records 24/7 (or on detections) and already knows *when*
something moved — its motion/smart-detect events. Backfilling therefore
follows the library indexer's two-phase shape:

* scan(): cheap. Ask the NVR for recording events in a time range and
  register each window as a pending BackfillClip, keyed by the NVR's
  event id so re-scanning the same range is a no-op.
* process(): expensive and bounded. Download each pending window as an
  MP4 and run it through the exact live pipeline
  (IngestService.process_source), stamping frames with historical time.

Clips are processed in chronological order so tracker state and the
event-linking gap behave the same as they would have live.

Several cameras at once (CLD-317) run through `run_backfills`: one
worker thread per camera over one shared `IngestService` — the shape
`IngestService.run` already has for live cameras — with one
`OperationRun` row each, bounded by a slot count. The two reasons a
second run used to be refused (per-camera tracker state, one embedded
vector store per process) forbid the *same* camera twice, not different
cameras side by side.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from siteloom.adapters.base import FrameSource
from siteloom.config import CameraConfig
from siteloom.ingest import IngestService, build_adapter
from siteloom.store import BackfillClip

log = logging.getLogger(__name__)


@dataclass
class ScanResult:
    added: int
    skipped: int  # already registered or covered by a wider window
    pending: int  # total pending for this camera after the scan


@dataclass
class BackfillResult:
    processed: int
    frames: int
    failed: int
    remaining: int
    failed_total: int  # terminal failures for this camera (library convention)
    retried: int


def _naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


@dataclass(frozen=True)
class BackfillRequest:
    """One range and one set of options, applied to each camera alike."""

    start: datetime
    end: datetime
    pad_s: float = 5.0
    chunk_minutes: float | None = None
    min_score: int = 0
    limit: int | None = None
    retry_failed: bool = False
    scan_only: bool = False


#: Per-camera state a UI can render while the run is going. Statuses in
#: ACTIVE mean "do not start this camera again"; everything else is over.
ACTIVE_STATUSES = ("waiting", "running", "processing")

#: How long a queued camera waits on the slot before ticking its heartbeat
#: — well inside the stale threshold, so "Waiting for a slot" on /jobs
#: never reads as a dead run.
SLOT_WAIT_S = 5.0


def new_state(cam: CameraConfig, request: BackfillRequest, resume: str) -> dict:
    return {
        "camera": cam.id,
        "name": cam.name or cam.id,
        "sweep": bool(request.chunk_minutes),
        "chunk_minutes": request.chunk_minutes,
        "retry_failed": request.retry_failed,
        "start": request.start,
        "end": request.end,
        "status": "waiting",
        "resume": resume,
    }


def run_backfills(
    service: IngestService,
    cams: list[CameraConfig],
    request: BackfillRequest,
    *,
    parallel: int,
    make_reporter: Callable[[CameraConfig], object],
    states: dict[str, dict] | None = None,
    interrupt: Callable[[], bool] | None = None,
) -> dict[str, dict]:
    """Scan and process `cams` side by side; returns per-camera state.

    One thread per camera, each under its own reporter from
    `make_reporter(cam)` (so /jobs shows one row per camera, with its own
    progress, cancel and resume command), and at most `parallel` of them
    holding a slot at once — `0` means all. A camera holds its slot from
    scan through process: the scan is an NVR API call too, and N of them
    at once is exactly the load the cap exists to bound.

    Blocks until every camera is done. `interrupt` is polled meanwhile;
    when it answers True every live reporter is asked to stop, which is
    how one Ctrl-C (owned by the caller — the reporters take
    `signals=False`) reaches every thread. A camera that fails records
    its error in its state and does not sink its siblings.
    """
    seen: set[str] = set()
    cams = [c for c in cams if not (c.id in seen or seen.add(c.id))]  # type: ignore[func-returns-value]
    if states is None:
        states = {c.id: new_state(c, request, "") for c in cams}
    slots = threading.BoundedSemaphore(parallel if parallel > 0 else max(1, len(cams)))
    reporters: dict[str, object] = {}
    threads = [
        threading.Thread(
            target=_run_one,
            args=(service, cam, request, slots, make_reporter, states[cam.id], reporters),
            name=f"backfill-{cam.id}",
            daemon=True,
        )
        for cam in cams
    ]
    for thread in threads:
        thread.start()
    while any(t.is_alive() for t in threads):
        if interrupt is not None and interrupt():
            for reporter in list(reporters.values()):
                reporter.interrupt_requested = True  # type: ignore[attr-defined]
        for thread in threads:
            thread.join(timeout=0.5)
    return states


def _run_one(service, cam, request, slots, make_reporter, state, reporters) -> None:
    from siteloom.progress import Interrupted

    try:
        with make_reporter(cam) as progress:
            reporters[cam.id] = progress
            with progress.phase("Waiting for a slot"):
                while not slots.acquire(timeout=SLOT_WAIT_S):
                    progress.heartbeat()  # queued is not dead
                    progress.check_interrupt()
            try:
                _backfill_camera(service, cam, request, progress, state)
            finally:
                slots.release()
    except Interrupted:
        pass  # the reporter recorded it; `status` says so below
    except Exception as exc:
        log.exception("backfill of %s failed", cam.id)
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        # A run stopped by signal leaves `status` where the reporter left
        # it: the OperationRun row is the record of how it ended, and
        # inventing "complete" here would contradict it.
        if state.get("status") in ACTIVE_STATUSES:
            state["status"] = "stopped"


def _backfill_camera(service, cam, request, progress, state) -> None:
    runner = UnifiBackfill(service, cam)
    state["status"] = "running"
    with progress.phase("Scanning NVR events"):
        scan = runner.scan(
            request.start,
            request.end,
            pad_s=request.pad_s,
            chunk_minutes=request.chunk_minutes,
            min_score=request.min_score,
        )
    # On the jobs board as well as the page: a sweep that registered
    # nothing looks identical to a stalled one unless the counters say
    # what it found.
    progress.bump(registered=scan.added, already_covered=scan.skipped)
    state.update(
        added=scan.added, skipped=scan.skipped, pending=scan.pending, status="processing"
    )
    if request.scan_only:
        state["status"] = "complete"
        return
    result = runner.process(
        limit=request.limit, retry_failed=request.retry_failed, progress=progress
    )
    state.update(
        status="complete",
        processed=result.processed,
        frames=result.frames,
        clip_failures=result.failed,
        remaining=result.remaining,
        failed_total=result.failed_total,
        retried=result.retried,
    )


class UnifiBackfill:
    """Scan + process NVR recordings for one configured camera."""

    def __init__(self, service: IngestService, cam: CameraConfig):
        if cam.adapter != "unifi":
            raise ValueError(
                f"camera {cam.id!r} uses adapter {cam.adapter!r}; NVR backfill "
                "needs a unifi camera"
            )
        self.service = service
        self.cam = cam
        self.Session = service.Session

    def scan(
        self,
        start: datetime,
        end: datetime,
        *,
        pad_s: float = 5.0,
        chunk_minutes: float | None = None,
        min_score: int = 0,
    ) -> ScanResult:
        """Register recording windows in [start, end] as pending clips.

        Event mode (default) registers one clip per NVR motion/smart
        event, padded by `pad_s` on both sides; windows fully contained
        in an already-registered one are skipped, since Protect fires
        motion and smart-detect events for the same activity.

        `chunk_minutes` switches to a full sweep in fixed chunks — for
        an always-recording camera or when the NVR's motion gating is
        not trusted. Chunk ids are deterministic, so re-scans still
        dedupe.
        """
        start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
        if chunk_minutes:
            windows = self._chunk_windows(start, end, chunk_minutes)
        else:
            windows = self._event_windows(start, end, pad_s, min_score)

        added = skipped = 0
        with self.Session() as session:
            known = set(
                session.scalars(
                    select(BackfillClip.external_id).filter_by(camera_id=self.cam.id)
                )
            )
            # Windows fully inside an accepted (or previously registered,
            # overlapping) window carry no new footage.
            covered = [
                (row.start, row.end)
                for row in session.scalars(
                    select(BackfillClip).filter(
                        BackfillClip.camera_id == self.cam.id,
                        BackfillClip.start < _naive_utc(end),
                        BackfillClip.end > _naive_utc(start),
                    )
                )
            ]
            for ext_id, kind, w_start, w_end in windows:
                w_start, w_end = _naive_utc(w_start), _naive_utc(w_end)
                if ext_id in known or any(
                    c0 <= w_start and w_end <= c1 for c0, c1 in covered
                ):
                    skipped += 1
                    continue
                session.add(
                    BackfillClip(
                        camera_id=self.cam.id,
                        external_id=ext_id,
                        kind=kind,
                        start=w_start,
                        end=w_end,
                    )
                )
                known.add(ext_id)
                covered.append((w_start, w_end))
                added += 1
            session.commit()
            pending = len(
                session.scalars(
                    select(BackfillClip.id).filter_by(
                        camera_id=self.cam.id, status="pending"
                    )
                ).all()
            )
        return ScanResult(added=added, skipped=skipped, pending=pending)

    def _event_windows(self, start, end, pad_s, min_score):
        adapter = build_adapter(self.cam, self.service.config)
        adapter.connect()
        try:
            events = adapter.list_recording_events(start, end, stream_id=self.cam.source)
        finally:
            adapter.close()
        pad = timedelta(seconds=pad_s)
        return [
            (ev.id, ev.type, ev.start - pad, ev.end + pad)
            for ev in events
            if ev.score >= min_score
        ]

    def _chunk_windows(self, start, end, chunk_minutes):
        step = timedelta(minutes=chunk_minutes)
        windows = []
        at = start
        while at < end:
            windows.append(
                (f"chunk:{at.isoformat()}", "chunk", at, min(at + step, end))
            )
            at += step
        return windows

    def process(
        self,
        *,
        limit: int | None = None,
        retry_failed: bool = False,
        progress=None,
    ) -> BackfillResult:
        """Download and ingest pending clips, oldest first. Resumable:
        every clip commits before the next starts, and an interrupt
        between clips leaves the rest pending for the next run."""
        with self.Session() as session:
            query = (
                select(BackfillClip)
                .filter(BackfillClip.camera_id == self.cam.id)
                .order_by(BackfillClip.start)
            )
            retried = 0
            if retry_failed:
                pending = list(
                    session.scalars(
                        query.filter(BackfillClip.status.in_(("pending", "failed")))
                    )
                )
                retried = sum(1 for c in pending if c.status == "failed")
            else:
                pending = list(session.scalars(query.filter_by(status="pending")))
            todo = pending if limit is None else pending[:limit]
            clip_ids = [c.id for c in todo]

        result = BackfillResult(
            processed=0, frames=0, failed=0, remaining=len(pending) - len(clip_ids),
            failed_total=0, retried=retried,
        )
        import contextlib

        phase = (
            progress.phase("Processing clips", total=len(clip_ids))
            if progress is not None
            else contextlib.nullcontext()
        )
        adapter = build_adapter(self.cam, self.service.config)
        adapter.connect()
        try:
            with phase, tempfile.TemporaryDirectory(prefix="siteloom-backfill-") as tmp:
                for clip_id in clip_ids:
                    before = result.frames
                    self._process_clip(adapter, clip_id, Path(tmp), result)
                    if progress is not None:
                        progress.advance(frames=result.frames - before)
                        progress.check_interrupt()
        finally:
            adapter.close()

        with self.Session() as session:
            result.failed_total = len(
                session.scalars(
                    select(BackfillClip.id).filter_by(
                        camera_id=self.cam.id, status="failed"
                    )
                ).all()
            )
            result.remaining = len(
                session.scalars(
                    select(BackfillClip.id).filter_by(
                        camera_id=self.cam.id, status="pending"
                    )
                ).all()
            )
        return result

    def _process_clip(self, adapter, clip_id: int, tmp: Path, result: BackfillResult):
        with self.Session() as session:
            clip = session.get(BackfillClip, clip_id)
            start = clip.start.replace(tzinfo=timezone.utc)
            end = clip.end.replace(tzinfo=timezone.utc)
            path = tmp / f"{clip.id}.mp4"
            try:
                adapter.download_clip(self.cam.source, start, end, path)
                source = FrameSource(
                    str(path), source_id=self.cam.id, is_file=True, base_time=start
                )
                frames = self.service.process_source(self.cam, source)
                self.service._process_audio(self.cam, str(path), base=start)
                clip.status = "done"
                clip.error = None
                clip.frames = frames
                result.processed += 1
                result.frames += frames
            except Exception as exc:
                # One rotten clip (failed export, corrupt segment) must
                # not sink the sweep — record it and move on, mirroring
                # the library indexer's failed-is-not-pending rule.
                log.exception("backfill clip %s failed", clip.external_id)
                clip.status = "failed"
                clip.error = f"{type(exc).__name__}: {exc}"
                result.failed += 1
            finally:
                clip.attempts += 1
                clip.processed_at = _naive_utc(datetime.now(timezone.utc))
                session.commit()
                path.unlink(missing_ok=True)
