"""Ingestion service: adapter stream → dispatcher → event store.

This is application code in the PRD §7 sense — it talks only to the
JobDispatcher interface, so swapping LocalBackend for Celery/Ray later
does not touch this file.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import cv2

from siteloom.adapters import ADAPTERS
from siteloom.adapters.base import CameraAdapter, Frame
from siteloom.config import CameraConfig, SiteConfig
from siteloom.dispatch import Job, JobDispatcher, LocalBackend
from siteloom.modules.detection import DetectionModule
from siteloom.store import Camera, Detection, Event, get_session, init_db, make_engine

log = logging.getLogger(__name__)


def build_dispatcher(config: SiteConfig) -> JobDispatcher:
    if config.backend.kind == "local":
        dispatcher: JobDispatcher = LocalBackend()
    else:  # pragma: no cover — future backends
        raise ValueError(f"unknown backend {config.backend.kind!r}")
    dispatcher.register("detection", DetectionModule(config.detection))
    return dispatcher


def build_adapter(cam: CameraConfig, config: SiteConfig) -> CameraAdapter:
    cls = ADAPTERS[cam.adapter]
    if cam.adapter == "unifi":
        return cls(cam.source, unifi=config.unifi)
    return cls(cam.source)


class IngestService:
    def __init__(self, config: SiteConfig, dispatcher: JobDispatcher | None = None):
        self.config = config
        self.dispatcher = dispatcher or build_dispatcher(config)
        self.engine = make_engine(config.storage.db_url)
        init_db(self.engine)
        self.Session = get_session(self.engine)
        self.media_dir = Path(config.storage.media_dir)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self._sync_cameras()

    def _sync_cameras(self) -> None:
        with self.Session() as session:
            for cam in self.config.cameras:
                row = session.get(Camera, cam.id)
                if row is None:
                    row = Camera(id=cam.id, site_id=self.config.site_id)
                    session.add(row)
                row.name = cam.name or cam.id
                row.adapter = cam.adapter
            session.commit()

    def run_camera(self, cam: CameraConfig, max_frames: int | None = None) -> int:
        """Process one camera's stream; returns number of frames processed.

        For a FileAdapter this runs to end-of-file (the backfill shape);
        for live adapters it runs until the stream drops or the process
        is stopped.
        """
        adapter = build_adapter(cam, self.config)
        adapter.connect()

        if cam.adapter == "file":
            streams = adapter.list_streams()
        else:
            streams = [s for s in adapter.list_streams() if s.id == cam.source] or [
                type("S", (), {"id": cam.source})
            ]

        processed = 0
        try:
            for stream in streams:
                source = adapter.get_live_stream(stream.id)
                for frame in source.frames(cam.sample_fps):
                    self._process_frame(cam, frame)
                    processed += 1
                    if max_frames is not None and processed >= max_frames:
                        return processed
        finally:
            adapter.close()
        return processed

    def _process_frame(self, cam: CameraConfig, frame: Frame) -> None:
        if "detection" not in cam.modules:
            return
        ok, jpeg = cv2.imencode(".jpg", frame.image)
        if not ok:
            log.warning("frame encode failed for %s", cam.id)
            return
        job = Job(
            module="detection",
            payload={
                "image_jpeg": jpeg.tobytes(),
                "camera_id": cam.id,
                "timestamp": frame.timestamp.isoformat(),
                "zones": [z.model_dump() for z in cam.zones],
                "require_zone": cam.require_zone,
            },
        )
        result = self.dispatcher.submit_and_wait(job)
        if not result.ok:
            log.error("detection job failed on %s: %s", cam.id, result.error)
            return
        detections = result.result["detections"]
        if detections:
            self._store_detections(cam, frame.timestamp, detections)

    def _store_detections(
        self, cam: CameraConfig, timestamp: datetime, detections: list[dict]
    ) -> None:
        # SQLite DateTime columns are naive; store UTC without tzinfo.
        ts = timestamp.replace(tzinfo=None)
        with self.Session() as session:
            for det in detections:
                event = self._find_or_create_event(session, cam.id, det, ts)
                crop_path = self._save_crop(cam.id, det, ts)
                session.add(
                    Detection(
                        event_id=event.id,
                        timestamp=ts,
                        class_name=det["class_name"],
                        confidence=det["confidence"],
                        bbox=json.dumps(det["bbox"]),
                        zones=json.dumps(det["zones"]),
                        crop_path=crop_path,
                    )
                )
                event.last_seen = ts
                event.detection_count += 1
                if det["confidence"] > event.best_confidence and crop_path:
                    event.best_confidence = det["confidence"]
                    event.best_crop_path = crop_path
            session.commit()

    def _find_or_create_event(
        self, session, camera_id: str, det: dict, ts: datetime
    ) -> Event:
        event = None
        if det["track_id"] is not None:
            event = (
                session.query(Event)
                .filter_by(
                    camera_id=camera_id,
                    track_id=det["track_id"],
                    class_name=det["class_name"],
                )
                .order_by(Event.id.desc())
                .first()
            )
        if event is None:
            event = Event(
                camera_id=camera_id,
                track_id=det["track_id"],
                class_name=det["class_name"],
                first_seen=ts,
                last_seen=ts,
            )
            session.add(event)
            session.flush()  # assign event.id for the Detection FK
        return event

    def _save_crop(self, camera_id: str, det: dict, ts: datetime) -> str | None:
        crop_jpeg = det.get("crop_jpeg")
        if not crop_jpeg:
            return None
        day_dir = self.media_dir / camera_id / ts.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        name = f"{ts.strftime('%H%M%S_%f')}_{det['class_name']}_{det['track_id']}.jpg"
        path = day_dir / name
        path.write_bytes(crop_jpeg)
        return str(path)

    def run(self, max_frames: int | None = None) -> None:
        # Sequential over cameras for the PoC; per-camera threads or a
        # process pool arrive with the multi-machine backends.
        for cam in self.config.cameras:
            log.info("ingesting camera %s (%s)", cam.id, cam.adapter)
            count = self.run_camera(cam, max_frames=max_frames)
            log.info("camera %s: %d frames processed", cam.id, count)
