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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

from siteloom.adapters import ADAPTERS
from siteloom.adapters.base import CameraAdapter, Frame
from siteloom.config import CameraConfig, EventConfig, SiteConfig
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

# A detection only joins an existing event if that event was last seen
# this recently (in frame time). Track ids restart at 1 whenever a
# tracker is rebuilt — process restart, stream reconnect, the next
# backfill clip — so track id alone would staple today's visitor onto
# last week's event.
EVENT_LINK_GAP_S = 120.0


def _bbox_iou(
    a: tuple[float, float, float, float] | list[float],
    b: tuple[float, float, float, float] | list[float],
) -> float:
    """Intersection-over-union of two pixel-space xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


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
        # Effective per-camera event rules (site defaults + overrides).
        # Filled lazily so backfill's synthetic file cameras — which are
        # not in config.cameras — get rules too (PRD §6.6 parity).
        self._event_rules: dict[str, EventConfig] = {}
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
                processed += self.process_source(
                    cam,
                    source,
                    max_frames=None if max_frames is None else max_frames - processed,
                )
                if max_frames is not None and processed >= max_frames:
                    return processed
                if cam.adapter == "file":
                    self._process_audio(cam, stream.id)
        finally:
            adapter.close()
        return processed

    def process_source(
        self, cam: CameraConfig, source, max_frames: int | None = None
    ) -> int:
        """Run one FrameSource through the frame pipeline.

        This is the seam backfill shares with live ingest (PRD §6.6): a
        historical clip is just a FrameSource whose base_time is in the
        past, processed by exactly this code path.
        """
        processed = 0
        for frame in source.frames(cam.sample_fps):
            self._process_frame(cam, frame)
            processed += 1
            if max_frames is not None and processed >= max_frames:
                break
        return processed

    def _process_audio(
        self, cam: CameraConfig, media_path: str, base: datetime | None = None
    ) -> None:
        """Loud-duration tracking over a file's audio stream (PRD §6.5).

        Live-stream audio arrives with the Celery/Ray backends; the file
        path covers backfill and NVR exports today. `base` is the wall
        time of the file's first sample; without it the file's mtime is
        used (right for archives, wrong for downloaded clips).
        """
        if "audio" not in cam.modules or not self.config.audio.enabled:
            return
        from siteloom.adapters.file import IMAGE_EXTS

        if Path(media_path).suffix.lower() in IMAGE_EXTS:
            return
        if base is None:
            base = datetime.fromtimestamp(Path(media_path).stat().st_mtime)
        elif base.tzinfo is not None:
            base = base.astimezone(timezone.utc).replace(tzinfo=None)
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
        rules = self._rules_for(cam)
        with self.Session() as session:
            for det in detections:
                event = self._find_or_create_event(session, cam.id, det, ts, rules)
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
                # Flip significance before identifying so the frame that
                # crosses the gate still gets identity resolution.
                self._update_significance(event, rules)
                self._identify(session, cam, event, det, ts, crop_path, rules)
            session.commit()

    def _rules_for(self, cam: CameraConfig) -> EventConfig:
        rules = self._event_rules.get(cam.id)
        if rules is None:
            rules = self.config.events.for_camera(cam)
            self._event_rules[cam.id] = rules
        return rules

    @staticmethod
    def _update_significance(event: Event, rules: EventConfig) -> None:
        """Flip `significant` once the gate is met — monotonic, never unset."""
        if event.significant:
            return
        if event.detection_count < rules.min_detections:
            return
        if event.best_confidence < rules.min_confidence:
            return
        # Only gate on duration when configured: out-of-order frame
        # timestamps can leave last_seen before first_seen, and a negative
        # duration must not veto an otherwise-qualified event.
        if rules.min_duration_s > 0:
            duration = (event.last_seen - event.first_seen).total_seconds()
            if duration < rules.min_duration_s:
                return
        event.significant = True

    def _identify(
        self,
        session,
        cam: CameraConfig,
        event: Event,
        det: dict,
        ts: datetime,
        crop_path: str | None,
        rules: EventConfig,
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
        # Quality gates: an ephemeral fragment or a weak/tiny crop makes a
        # useless embedding — and every unresolved one mints a fresh
        # unknown identity, which is exactly the churn being gated out.
        if rules.identify_only_significant and not event.significant:
            return
        if det["confidence"] < rules.identify_min_confidence:
            return
        x1, y1, x2, y2 = det["bbox"]
        if min(x2 - x1, y2 - y1) < rules.identify_min_crop_px:
            return
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
        self, session, camera_id: str, det: dict, ts: datetime, rules: EventConfig
    ) -> Event:
        # Class groups absorb detector flapping (car↔truck mid-track):
        # any class in the group continues the event; per-frame truth is
        # on the Detection rows.
        group = rules.group_for(det["class_name"])
        event = None
        if det["track_id"] is not None:
            event = (
                session.query(Event)
                .filter(
                    Event.camera_id == camera_id,
                    Event.track_id == det["track_id"],
                    Event.class_name.in_(group),
                )
                .order_by(Event.id.desc())
                .first()
            )
            if (
                event is not None
                and abs((ts - event.last_seen).total_seconds()) > EVENT_LINK_GAP_S
            ):
                event = None
        if event is None:
            event = self._stitch_event(session, camera_id, det, ts, group, rules)
        if event is None:
            event = Event(
                camera_id=camera_id,
                track_id=det["track_id"],
                class_name=det["class_name"],
                first_seen=ts,
                last_seen=ts,
                guest_window=self._guest_windows.contains(ts),
                # Events start ephemeral and earn significance
                # (_update_significance); the model default stays True for
                # rows written by ungated writers and pre-column rows.
                significant=False,
            )
            session.add(event)
            session.flush()  # assign event.id for the Detection FK
        return event

    def _stitch_event(
        self,
        session,
        camera_id: str,
        det: dict,
        ts: datetime,
        group: list[str],
        rules: EventConfig,
    ) -> Event | None:
        """Reattach a trackless or fresh-track detection to a recent event.

        Sampled streams fragment one visit into many events: trackless
        detections have no track id at all, and tracker rebuilds hand out
        fresh ids mid-visit. If the same camera saw the same class group
        moments ago *in the same place* (IoU with that event's last
        detection), continue that event instead of starting another. The
        gap is symmetric (abs) so backfill clip order doesn't matter, and
        it is frame time, never wall clock — resume-equivalence depends
        on that.
        """
        if rules.stitch_gap_s <= 0:
            return None
        gap = timedelta(seconds=rules.stitch_gap_s)
        event = (
            session.query(Event)
            .filter(
                Event.camera_id == camera_id,
                Event.class_name.in_(group),
                Event.last_seen >= ts - gap,
                Event.last_seen <= ts + gap,
            )
            .order_by(Event.id.desc())
            .first()
        )
        if event is None:
            return None
        last_det = (
            session.query(Detection)
            .filter(Detection.event_id == event.id)
            .order_by(Detection.id.desc())
            .first()
        )
        if last_det is None:
            return None
        if _bbox_iou(json.loads(last_det.bbox), det["bbox"]) < rules.stitch_min_iou:
            return None
        if det["track_id"] is not None:
            # Adopt the new track so its later frames hit the fast path.
            event.track_id = det["track_id"]
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
