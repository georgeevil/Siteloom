"""Drop-and-reindex a time window from the NVR's recordings.

The operator's "make the past match the current settings" tool: purge
the events/detections a window produced under old rules, then re-run the
UniFi backfill over the same window so the live pipeline (with today's
stitching, gating, and confidence settings) rebuilds it.

Purge is deliberately narrow: it deletes Events, their Detections,
EventIdentity links and PlateRead rows, their crop files, and the
BackfillClip bookkeeping for the window (so the re-scan re-registers the
NVR windows). It does
NOT touch Identity rows or their vectors — identities are the durable
layer that reindexed events re-match against; pruning them is a
separate, human decision.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from siteloom.config import CameraConfig, SiteConfig
from siteloom.store import BackfillClip, Detection, Event, EventIdentity, PlateRead

log = logging.getLogger(__name__)


@dataclass
class PurgeResult:
    events: int = 0
    detections: int = 0
    links: int = 0
    plate_reads: int = 0
    clips: int = 0
    crop_files: int = 0


@dataclass
class ReindexResult:
    purged: PurgeResult = field(default_factory=PurgeResult)
    clips_processed: int = 0
    clips_failed: int = 0
    frames: int = 0


def _naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def purge_window(
    Session,
    camera_ids: list[str],
    start: datetime,
    end: datetime,
    media_root: Path | None = None,
) -> PurgeResult:
    """Delete the given cameras' events overlapping [start, end].

    Batched commits (one per camera) so an interrupt loses at most one
    camera's progress; re-running is harmless — already-deleted rows are
    simply absent.
    """
    start, end = _naive_utc(start), _naive_utc(end)
    result = PurgeResult()
    for camera_id in camera_ids:
        with Session() as session:
            events = list(
                session.scalars(
                    select(Event).filter(
                        Event.camera_id == camera_id,
                        Event.external_id.is_(None),  # never externally-owned rows
                        Event.first_seen <= end,
                        Event.last_seen >= start,
                    )
                )
            )
            crop_paths: set[str] = set()
            for event in events:
                if event.best_crop_path:
                    crop_paths.add(event.best_crop_path)
                detections = session.scalars(
                    select(Detection).filter(Detection.event_id == event.id)
                ).all()
                for det in detections:
                    if det.crop_path:
                        crop_paths.add(det.crop_path)
                    session.delete(det)
                    result.detections += 1
                links = session.scalars(
                    select(EventIdentity).filter(EventIdentity.event_id == event.id)
                ).all()
                for link in links:
                    session.delete(link)
                    result.links += 1
                # Plate reads hang off the event like detections do
                # (CLD-85); leaving them behind would keep rows on
                # /plates whose event no longer exists.
                reads = session.scalars(
                    select(PlateRead).filter(PlateRead.event_id == event.id)
                ).all()
                for read in reads:
                    if read.crop_path:
                        crop_paths.add(read.crop_path)
                    session.delete(read)
                    result.plate_reads += 1
                session.delete(event)
                result.events += 1
            clips = session.scalars(
                select(BackfillClip).filter(
                    BackfillClip.camera_id == camera_id,
                    BackfillClip.start <= end,
                    BackfillClip.end >= start,
                )
            ).all()
            for clip in clips:
                session.delete(clip)
                result.clips += 1
            session.commit()
        if media_root is not None:
            for rel in crop_paths:
                path = (media_root / rel).resolve()
                # Confinement: never delete outside the media dir.
                if media_root.resolve() not in path.parents:
                    continue
                try:
                    path.unlink(missing_ok=True)
                    result.crop_files += 1
                except OSError:  # pragma: no cover — fs races
                    log.warning("could not remove crop %s", path)
    return result


def reindex_window(
    service,
    cameras: list[CameraConfig],
    start: datetime,
    end: datetime,
    progress=None,
) -> ReindexResult:
    """Purge [start, end] for the cameras, then re-backfill it from the NVR.

    `service` is a live IngestService; runs the exact live pipeline
    (PRD §6.6). Resumable at clip granularity like any backfill — the
    purge is idempotent and the clip table dedupes by NVR event id.
    """
    from siteloom.backfill import UnifiBackfill

    result = ReindexResult()
    media_root = Path(service.config.storage.media_dir)

    import contextlib

    phase = (
        progress.phase("Purging window", total=len(cameras))
        if progress is not None
        else contextlib.nullcontext()
    )
    with phase:
        result.purged = purge_window(
            service.Session, [c.id for c in cameras], start, end, media_root
        )
        if progress is not None:
            progress.advance(
                len(cameras),
                events=result.purged.events,
                detections=result.purged.detections,
            )
            progress.check_interrupt()

    for cam in cameras:
        backfill = UnifiBackfill(service, cam)
        scan_phase = (
            progress.phase(f"Scanning NVR: {cam.name or cam.id}")
            if progress is not None
            else contextlib.nullcontext()
        )
        with scan_phase:
            scan = backfill.scan(start, end)
        log.info(
            "reindex %s: %d clip(s) registered, %d pending",
            cam.id,
            scan.added,
            scan.pending,
        )
        processed = backfill.process(progress=progress)
        result.clips_processed += processed.processed
        result.clips_failed += processed.failed
        result.frames += processed.frames
    return result
