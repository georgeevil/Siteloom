"""Ingestion service: adapter stream → dispatcher → event store.

This is application code in the PRD §7 sense — it talks only to the
JobDispatcher interface, so swapping LocalBackend for Celery/Ray later
does not touch this file.
"""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from siteloom.adapters import ADAPTERS
from siteloom.adapters.base import CameraAdapter, Frame
from siteloom.config import CameraConfig, SiteConfig
from siteloom.dispatch import Job, JobDispatcher, LocalBackend
from siteloom.guests import GuestWindows
from siteloom.identity import IdentityResolver, VectorStore
from siteloom.modules.audio import AudioModule
from siteloom.modules.detection import DetectionModule
from siteloom.modules.identity import IdentityModule
from siteloom.store import (
    Camera,
    Detection,
    Event,
    EventIdentity,
    NoiseEvent,
    get_session,
    init_db,
    make_engine,
)

log = logging.getLogger(__name__)

# Live-stream reconnect pacing: exponential backoff between attempts,
# reset once a connection has stayed up long enough to count as stable.
LIVE_BACKOFF_S = 2.0
LIVE_BACKOFF_MAX_S = 60.0
LIVE_STABLE_S = 30.0


def build_dispatcher(config: SiteConfig) -> JobDispatcher:
    if config.backend.kind == "local":
        dispatcher: JobDispatcher = LocalBackend()
    else:  # pragma: no cover — future backends
        raise ValueError(f"unknown backend {config.backend.kind!r}")
    dispatcher.register("detection", DetectionModule(config.detection))
    if config.identity.enabled:
        dispatcher.register(
            "identity", IdentityModule(config.identity, device=config.detection.device)
        )
    if config.audio.enabled:
        dispatcher.register("audio", AudioModule(config.audio))
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

        self.resolver: IdentityResolver | None = None
        if config.identity.enabled:
            self.resolver = IdentityResolver(
                config.identity, VectorStore(config.identity.vector_db_path)
            )
        with self.Session() as session:
            self._guest_windows = GuestWindows(session, config.guests)

        # Optional outbound integrations: MQTT bus + webhooks. Both are
        # no-ops unless configured, and neither may break ingestion.
        from siteloom.integrations import MqttPublisher, WebhookNotifier

        self.publisher = MqttPublisher(config.integrations.mqtt)
        self.notifier = WebhookNotifier(config.integrations.webhooks)

        self._stop = threading.Event()

    def stop(self) -> None:
        """Ask a running ingest to finish in-flight frames and return."""
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

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
                if cam.adapter == "file":
                    self._process_audio(cam, stream.id)
        finally:
            adapter.close()
        return processed

    def _process_audio(self, cam: CameraConfig, media_path: str) -> None:
        """Loud-duration tracking over a file's audio stream (PRD §6.5).

        Live-stream audio arrives with the Celery/Ray backends; the file
        path covers backfill and NVR exports today.
        """
        if "audio" not in cam.modules or not self.config.audio.enabled:
            return
        from siteloom.adapters.file import IMAGE_EXTS

        if Path(media_path).suffix.lower() in IMAGE_EXTS:
            return
        base = datetime.fromtimestamp(Path(media_path).stat().st_mtime)
        result = self.dispatcher.submit_and_wait(
            Job(module="audio", payload={"media_path": media_path})
        )
        if not result.ok:
            log.error("audio job failed on %s: %s", cam.id, result.error)
            return
        episodes = result.result["episodes"]
        if not episodes:
            return
        with self.Session() as session:
            for ep in episodes:
                session.add(
                    NoiseEvent(
                        camera_id=cam.id,
                        start=base + timedelta(seconds=ep["start_s"]),
                        end=base + timedelta(seconds=ep["end_s"]),
                        peak_db=ep["peak_db"],
                        mean_db=ep["mean_db"],
                    )
                )
            session.commit()
        log.info("camera %s: %d noise episode(s) recorded", cam.id, len(episodes))

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
                self._identify(session, cam, event, det, ts, crop_path)
            session.commit()

    def _identify(
        self,
        session,
        cam: CameraConfig,
        event: Event,
        det: dict,
        ts: datetime,
        crop_path: str | None,
    ) -> None:
        """Second-pass identification on a detection crop (PRD §6.3/6.4).

        The crop goes through the dispatcher like any other job — the
        IdentityModule computes embeddings (edge work), the resolver
        matches/creates identities against the stores (central work).
        """
        if self.resolver is None or not det.get("crop_jpeg"):
            return
        if "identity" not in cam.modules:
            return  # per-camera module selection (NFR3)
        result = self.dispatcher.submit_and_wait(
            Job(
                module="identity",
                payload={
                    "crop_jpeg": det["crop_jpeg"],
                    "class_name": det["class_name"],
                },
            )
        )
        if not result.ok:
            log.error("identity job failed on %s: %s", cam.id, result.error)
            return
        registry = self.config.identity.identifiers
        for emb in result.result["embeddings"]:
            ident_cfg = registry.get(emb["identifier"])
            resolution = self.resolver.resolve(
                session,
                identifier_key=emb["identifier"],
                class_name=det["class_name"],
                vector=emb["vector"],
                plate=emb["plate"],
                timestamp=ts,
                crop_path=crop_path,
                threshold=ident_cfg.threshold if ident_cfg else None,
                max_vectors=ident_cfg.max_vectors_per_identity if ident_cfg else 20,
            )
            link = (
                session.query(EventIdentity)
                .filter_by(event_id=event.id, identity_id=resolution.identity.id)
                .first()
            )
            first_link = link is None
            if link is None:
                session.add(
                    EventIdentity(
                        event_id=event.id,
                        identity_id=resolution.identity.id,
                        identifier_key=emb["identifier"],
                        similarity=resolution.similarity,
                        matched_by=resolution.matched_by,
                        learned_plate=resolution.learned_plate,
                    )
                )
            else:
                link.hit_count += 1
                link.similarity = max(link.similarity, resolution.similarity)
                # Record the strongest evidence seen across frames: plate
                # outranks visual, and a link whose first frame *created*
                # the identity (no match, so None) picks up the first
                # re-match's mode.
                if resolution.matched_by == "plate":
                    link.matched_by = "plate"
                elif link.matched_by is None:
                    link.matched_by = resolution.matched_by
                link.learned_plate = link.learned_plate or resolution.learned_plate

            # Publish once per event+identity pairing, not per frame — a
            # 30-second visit is one match, not sixty notifications.
            if first_link:
                from siteloom.integrations.mqtt import identity_payload
                from siteloom.integrations.webhooks import classify_resolution

                payload = identity_payload(
                    event,
                    resolution.identity,
                    resolution.similarity,
                    resolution.is_new,
                )
                self.publisher.publish("identity", payload)
                self.notifier.fire(
                    classify_resolution(
                        resolution.identity,
                        resolution.is_new,
                        resolution.learned_plate,
                    ),
                    payload,
                )

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
                guest_window=self._guest_windows.contains(ts),
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
        """Ingest all configured cameras until stopped.

        File cameras run sequentially to completion (the backfill shape,
        PRD §6.6). Live cameras then run one worker thread each, so a
        slow or dropped stream never starves the others; each worker
        reconnects with backoff until stop() or a signal.
        """
        self._stop.clear()
        file_cams = [c for c in self.config.cameras if c.adapter == "file"]
        live_cams = [c for c in self.config.cameras if c.adapter != "file"]

        for cam in file_cams:
            if self._stop.is_set():
                return
            log.info("ingesting camera %s (%s)", cam.id, cam.adapter)
            count = self.run_camera(cam, max_frames=max_frames)
            log.info("camera %s: %d frames processed", cam.id, count)

        if not live_cams:
            return
        with self._stop_signals():
            workers = [
                threading.Thread(
                    target=self._run_live,
                    args=(cam, max_frames),
                    name=f"ingest-{cam.id}",
                    daemon=True,
                )
                for cam in live_cams
            ]
            for w in workers:
                w.start()
            # Join with a timeout so the main thread keeps handling signals.
            while any(w.is_alive() for w in workers):
                for w in workers:
                    w.join(timeout=0.5)

    def _run_live(self, cam: CameraConfig, max_frames: int | None) -> None:
        """One camera's live loop: connect, ingest, reconnect on drop.

        The adapter is rebuilt on every attempt — a fresh connect()
        re-resolves stream URLs, which Protect can rotate — and gaps are
        logged with their duration so a soak's blind spots are visible.
        """
        log.info("camera %s: live ingest started (%s)", cam.id, cam.adapter)
        processed = 0
        backoff = LIVE_BACKOFF_S

        def done() -> bool:
            return self._stop.is_set() or (
                max_frames is not None and processed >= max_frames
            )

        while not done():
            adapter = build_adapter(cam, self.config)
            connected_at = time.monotonic()
            got = 0
            try:
                adapter.connect()
                source = adapter.get_live_stream(cam.source)
                for frame in source.frames(cam.sample_fps):
                    self._process_frame(cam, frame)
                    processed += 1
                    got += 1
                    if done():
                        break
            except Exception:
                log.exception("camera %s: live stream error", cam.id)
            finally:
                adapter.close()
            if done():
                break
            uptime = time.monotonic() - connected_at
            if uptime >= LIVE_STABLE_S:
                backoff = LIVE_BACKOFF_S
            log.warning(
                "camera %s: stream dropped after %.0fs (%d frames); "
                "reconnecting in %.0fs",
                cam.id,
                uptime,
                got,
                backoff,
            )
            self._stop.wait(backoff)
            backoff = min(backoff * 2, LIVE_BACKOFF_MAX_S)
        log.info("camera %s: live ingest stopped (%d frames)", cam.id, processed)

    @contextmanager
    def _stop_signals(self):
        """Route SIGINT/SIGTERM/SIGHUP to a graceful stop (progress.py
        convention): first signal drains in-flight frames, second aborts."""
        from siteloom.progress import STOP_SIGNALS

        def handler(signum, frame):
            if self._stop.is_set():
                raise KeyboardInterrupt
            log.info(
                "stop signal received — finishing in-flight frames "
                "(send again to abort)"
            )
            self._stop.set()

        previous: dict[int, object] = {}
        try:
            for sig in STOP_SIGNALS:
                previous[sig] = signal.signal(sig, handler)
        except ValueError:  # not the main thread (tests, embedded use)
            pass
        try:
            yield
        finally:
            for sig, old in previous.items():
                signal.signal(sig, old)
