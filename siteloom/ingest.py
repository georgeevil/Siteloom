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
from siteloom.identity import IdentityResolver
from siteloom.modules.audio import AudioModule
from siteloom.modules.detection import DetectionModule
from siteloom.modules.identity import IdentityModule
from siteloom.store import (
    Camera,
    Detection,
    Event,
    EventIdentity,
    NoiseEvent,
    PlateRead,
    PlateWatch,
    get_session,
    init_db,
    make_engine,
)
from siteloom.store.claims import active_claim, fold_claim, link_claim

log = logging.getLogger(__name__)

# Live-stream reconnect pacing: exponential backoff between attempts,
# reset once a connection has stayed up long enough to count as stable.
LIVE_BACKOFF_S = 2.0
LIVE_BACKOFF_MAX_S = 60.0
LIVE_STABLE_S = 30.0

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
        # When each (event, identifier) last had an OCR attempt, in frame
        # time (CLD-130's cadence cap). Frame time, never wall clock, so
        # backfill and live ration identically. In-memory on purpose: a
        # restart forgetting it costs one extra OCR per visit, which is
        # cheaper than a table nobody else reads. Keyed by event id, so a
        # visit's cadence survives the CLD-40 merges (the id in hand at
        # decision time is the one the last frame used).
        self._plate_ocr_last: dict[tuple[int, str], datetime] = {}
        self._sync_cameras()

        self.resolver: IdentityResolver | None = None
        if config.identity.enabled:
            # Shared store, not a private client: embedded Qdrant is one
            # client per path per process, and the web process runs an
            # IngestService too (UI-triggered reindex) alongside the
            # recognition API and enrollment, which already share it.
            from siteloom.identity.vectors import get_shared_store

            self.resolver = IdentityResolver(
                config.identity, get_shared_store(config.identity.vector_db_path)
            )
        with self.Session() as session:
            self._guest_windows = GuestWindows(session, config.guests)

        # Optional outbound integrations: MQTT bus + webhooks. Both are
        # no-ops unless configured, and neither may break ingestion.
        from siteloom.integrations import MqttPublisher, WebhookNotifier

        self.publisher = MqttPublisher(config.integrations.mqtt)
        self.notifier = WebhookNotifier(config.integrations.webhooks)

        self._stop = threading.Event()
        # Optional heartbeat for a long soak (CLD-15). Set by run(); None
        # everywhere else so backfill and the tests stay reporter-free.
        self._progress = None

    def stop(self) -> None:
        """Ask a running ingest to finish in-flight frames and return."""
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    #: Aggregate heartbeat counters, seeded in this order so the progress
    #: bar's four-counter summary shows the totals rather than whichever
    #: per-camera key happened to be touched first.
    TICK_COUNTERS = ("frames", "detections", "matches", "reconnects")

    def _tick(self, cam_id: str, n: int = 0, **counters: int) -> None:
        """Advance the run's heartbeat, if one is attached (CLD-15).

        Every counter is recorded twice — once aggregated and once under
        `<camera>.<name>` — because the question a soak actually asks is
        "which camera went quiet at 3am", which a total cannot answer.

        This is also where the live workers notice a stop signal. The
        reporter owns the handlers (one owner, per progress.py), so the
        threads learn about Ctrl-C by polling it on their next tick
        instead of each installing a handler of its own.
        """
        p = self._progress
        if p is None:
            return
        wide = dict(counters)
        wide.update({f"{cam_id}.{key}": value for key, value in counters.items()})
        if n:
            p.advance(n, **wide)
        else:
            p.bump(**wide)
        if p.interrupt_requested:
            self._stop.set()

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
        # A frame with nothing in it is still a frame the soak got through;
        # counting only productive ones would make a quiet night look dead.
        self._tick(cam.id, 1, frames=1, detections=len(detections))

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
                detection = Detection(
                    event_id=event.id,
                    timestamp=ts,
                    class_name=det["class_name"],
                    confidence=det["confidence"],
                    bbox=json.dumps(det["bbox"]),
                    zones=json.dumps(det["zones"]),
                    crop_path=crop_path,
                )
                session.add(detection)
                event.last_seen = ts
                event.detection_count += 1
                event.confidence_sum += det["confidence"]
                if det["confidence"] > event.best_confidence and crop_path:
                    event.best_confidence = det["confidence"]
                    event.best_crop_path = crop_path
                # Flip significance before identifying so the frame that
                # crosses the gate still gets identity resolution.
                was_significant = event.significant
                self._update_significance(event, rules)
                if event.significant and not was_significant:
                    # The frames spent earning significance were stored but
                    # never identified; now that the event has proved real,
                    # they get the same pass (CLD-286). Before the current
                    # frame, so plate-OCR rationing stays chronological.
                    event = self._identify_backlog(
                        session, cam, event, detection, rules
                    )
                self._identify(
                    session, cam, event, det, ts, crop_path, rules, detection
                )
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
        detection: Detection | None = None,
    ) -> Event:
        """Second-pass identification on a detection crop (PRD §6.3/6.4).

        The crop goes through the dispatcher like any other job — the
        IdentityModule computes embeddings (edge work), the resolver
        matches/creates identities against the stores (central work).

        Returns the surviving event: the identity-aware merge (CLD-40)
        may fold `event` into a prior visit and delete its row, and a
        caller identifying several frames in a row (`_identify_backlog`)
        must follow the survivor rather than keep writing links against
        a deleted id.
        """
        if self.resolver is None or not det.get("crop_jpeg"):
            return event
        if "identity" not in cam.modules:
            return event  # per-camera module selection (NFR3)
        # Quality gates: an ephemeral fragment or a weak/tiny crop makes a
        # useless embedding — and every unresolved one mints a fresh
        # unknown identity, which is exactly the churn being gated out.
        if rules.identify_only_significant and not event.significant:
            return event
        if det["confidence"] < rules.identify_min_confidence:
            return event
        x1, y1, x2, y2 = det["bbox"]
        if min(x2 - x1, y2 - y1) < rules.identify_min_crop_px:
            return event
        identity_cfg = self.config.identity
        # Plate extras, resolved here because only this layer knows the
        # camera: the effective quality floors (CLD-128, one resolution
        # shared with any future replay) and the identifiers whose OCR is
        # rationed out this frame (CLD-130 — reading a parked car on
        # every sampled frame bought 1,437 rows and no information; the
        # embedding still runs, so re-ID is unchanged).
        plate_floors: dict[str, dict] = {}
        skip_plate_ocr: list[str] = []
        # Vehicle fingerprint (CLD-254): the floors ride in the payload
        # like plate floors do, and their absence is the module's off
        # switch — a directly-driven module (tests, replay) that sends
        # no key gets no read. One resolution, shared with the event
        # page's chip gate (`fingerprint_request`), so ingest and
        # display cannot disagree about what is fingerprinted.
        fingerprint_req = identity_cfg.fingerprint_request(det["class_name"])
        for key, ident in identity_cfg.identifiers.items():
            if not ident.plate_ocr:
                continue
            plate_floors[key] = identity_cfg.plate_floors_for(key, cam)._asdict()
            interval = ident.plate_ocr_interval_s
            last = self._plate_ocr_last.get((event.id, key))
            if (
                interval > 0
                and last is not None
                and (ts - last).total_seconds() < interval
            ):
                skip_plate_ocr.append(key)
        result = self.dispatcher.submit_and_wait(
            Job(
                module="identity",
                payload={
                    "crop_jpeg": det["crop_jpeg"],
                    "class_name": det["class_name"],
                    "plate_floors": plate_floors,
                    "skip_plate_ocr": skip_plate_ocr,
                    "fingerprint": fingerprint_req,
                },
            )
        )
        if not result.ok:
            log.error("identity job failed on %s: %s", cam.id, result.error)
            return event
        # The color read lands on the Detection row whether or not it
        # named a color, and before any resolution happens — it is a
        # measurement of the frame, not of the match (CLD-254). The
        # module computed it; writing rows stays this layer's job.
        color = result.result.get("fingerprint")
        if color is not None and detection is not None:
            detection.color_name = color["color"]
            detection.color_confidence = color["confidence"]
            detection.color_chroma = color["chroma_p95"]
            detection.color_saturation = color["saturation"]
            detection.color_crop_px = color["crop_px"]
            detection.color_reason = color["reason"]
            detection.color_min_px = color["min_px"]
            detection.color_chroma_floor = color["chroma_floor"]
        registry = identity_cfg.identifiers
        for emb in result.result["embeddings"]:
            ident_cfg = registry.get(emb["identifier"])
            if emb.get("plate_read") is not None:
                self._note_plate_ocr(event.id, emb["identifier"], ts)
            # The OCR attempt is recorded before anything is decided with
            # it, and whether or not it produced a plate (CLD-85). The
            # module computed it; writing rows is this layer's job, which
            # is what keeps the compute/state split intact.
            self._record_plate_read(
                session, cam, event, detection, ts, emb, crop_path
            )
            if emb["vector"] is None and emb.get("plate") is None:
                # Nothing to resolve — the entry exists only to carry a
                # failed read upstream. Resolving it would mint an
                # identity out of an embedding that does not exist.
                continue
            resolution = self.resolver.resolve(
                session,
                identifier_key=emb["identifier"],
                class_name=det["class_name"],
                vector=emb["vector"],
                plate=emb["plate"],
                timestamp=ts,
                crop_path=crop_path,
                # Per-camera threshold if this camera names one, else the
                # identifier's site-wide value (CLD-39). The resolver
                # already takes a threshold per call, so tuning one noisy
                # doorway is config, not a resolver change.
                threshold=identity_cfg.threshold_for(emb["identifier"], cam),
                max_vectors=ident_cfg.max_vectors_per_identity if ident_cfg else 20,
                camera_id=cam.id,
                # The identifier's own quality signal when the module
                # measured one (the face pipeline's YuNet score), else
                # the detector's box confidence. The box says "this is a
                # person", not "this face is legible" — feeding it to
                # `immediate_quality` is how a crisp walk-past minted an
                # identity per blurry face (CLD-139's mint half).
                quality=(
                    emb["quality"]
                    if emb.get("quality") is not None
                    else det["confidence"]
                ),
                # Names the visit this frame belongs to, which is what
                # bounds per-event learning and what makes a `wrong`
                # verdict stop further accretion mid-event (CLD-139).
                event_id=event.id,
            )
            if resolution.identity is None:
                # Quarantined (awaiting consistent sightings) or ambiguous
                # between known identities — no link either way (CLD-41).
                continue
            self._tick(cam.id, matches=1)
            # Identity-aware de-fragmentation (CLD-40): the same identity
            # moments apart on the same camera is one visit, even when the
            # subject moved too far between samples for the IoU stitch.
            # Merging before linking means the surviving event usually
            # already carries the link, so the pairing publishes once.
            event = self._merge_with_prior(session, event, resolution.identity.id, rules)
            # A watched plate is an alarm, not just a row. Checked after
            # the merge so the once-per-event dedupe counts reads on the
            # event that survived, not on a fragment about to be folded.
            if emb["plate"]:
                self._notify_watch_hit(session, event, emb["plate"])
            # An unlinked claim is one an operator repudiated (CLD-36);
            # a later frame matching the same identity must not revive it
            # by incrementing its hit count. It makes a fresh claim
            # instead, leaving the correction intact for review. A racing
            # writer that claimed the pair first is folded into, not
            # collided with (CLD-133).
            _link, first_link = link_claim(
                session,
                event_id=event.id,
                identity_id=resolution.identity.id,
                identifier_key=emb["identifier"],
                similarity=resolution.similarity,
                matched_by=resolution.matched_by,
                learned_plate=resolution.learned_plate,
            )

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
        return event

    def _identify_backlog(
        self,
        session,
        cam: CameraConfig,
        event: Event,
        current: Detection,
        rules: EventConfig,
    ) -> Event:
        """Identify the frames an event spent earning significance (CLD-286).

        The significance gate keeps identity work off ephemeral events, but
        it also meant a *departure* — whose biggest, most legible plate
        frames come first — spent exactly those frames on warm-up and then
        OCR'd only the shrinking tail. The evidence is already stored:
        Detection rows with crops on disk, at most `min_detections - 1` of
        them, so once the event proves real they get the same `_identify`
        pass the gate deferred. Runs inside the flipping frame's session,
        in timestamp order and before the current frame, so plate-OCR
        rationing (CLD-130) sees one chronological visit and resume
        equivalence holds (the flip and its backlog commit together).

        The per-frame quality gates still apply inside `_identify`; only
        the significance gate is behind us by construction. Returns the
        surviving event, because a backlog frame can trigger the
        identity-aware merge like any other.
        """
        if self.resolver is None or "identity" not in cam.modules:
            return event
        if not rules.identify_only_significant:
            return event  # nothing was gated, every frame already ran
        # The flipping frame's own row is identified by the caller; it
        # needs its id assigned before it can be excluded here.
        session.flush()
        rows = (
            session.query(Detection)
            .filter(Detection.event_id == event.id, Detection.id != current.id)
            .order_by(Detection.timestamp, Detection.id)
            .all()
        )
        for row in rows:
            if not row.crop_path:
                continue
            try:
                crop_jpeg = Path(row.crop_path).read_bytes()
            except OSError as exc:
                log.warning("backlog crop unreadable %s: %s", row.crop_path, exc)
                continue
            det = {
                "class_name": row.class_name,
                "confidence": row.confidence,
                "bbox": json.loads(row.bbox),
                "crop_jpeg": crop_jpeg,
            }
            event = self._identify(
                session, cam, event, det, row.timestamp, row.crop_path, rules, row
            )
        return event

    def _note_plate_ocr(self, event_id: int, identifier: str, ts: datetime) -> None:
        """Remember when this visit last had an OCR attempt (CLD-130).

        The map is advisory state, not a record — PlateRead rows are the
        record — so it is pruned by size rather than persisted: a 24 h
        soak must not grow it unboundedly, and dropping an old entry
        costs at most one extra OCR on a visit that outlived it.
        """
        self._plate_ocr_last[(event_id, identifier)] = ts
        if len(self._plate_ocr_last) > 1024:
            newest = sorted(
                self._plate_ocr_last.items(), key=lambda kv: kv[1], reverse=True
            )[:512]
            self._plate_ocr_last = dict(newest)

    def _notify_watch_hit(self, session, event: Event, plate: str) -> None:
        """Fire the watchlist alarm on the first accepted read of a
        watched plate on this event.

        Once per event+plate pairing — the same rule the identity publish
        follows — but keyed on the PlateRead rows already written (the
        accepted count reaching exactly 1), so the dedupe is restart-safe
        and survives event merges the way an in-memory flag would not.
        The query autoflushes this frame's pending row, so a first read
        counts itself. Both publishers degrade to a log line when their
        endpoint is down (NFR1); a missed alarm never blocks ingestion.
        """
        watch = session.query(PlateWatch).filter_by(plate=plate).first()
        if watch is None:
            return
        prior = (
            session.query(PlateRead)
            .filter(
                PlateRead.event_id == event.id,
                PlateRead.text == plate,
                PlateRead.accepted.is_(True),
            )
            .count()
        )
        if prior != 1:
            return
        from siteloom.integrations.mqtt import watchlist_payload

        payload = watchlist_payload(event, watch, plate)
        self.publisher.publish("watchlist", payload)
        self.notifier.fire("plate.watchlist", payload)

    def _record_plate_read(
        self,
        session,
        cam: CameraConfig,
        event: Event,
        detection: Detection | None,
        ts: datetime,
        emb: dict,
        crop_path: str | None,
    ) -> None:
        """Persist one OCR attempt, successful or not (CLD-85).

        Everything on the row was computed by IdentityModule and travelled
        here as scalars and JPEG bytes; nothing is recomputed and no
        second detector or OCR pass is bought. A failure is a row like any
        other — `reason` says which of the four ways it failed — because
        the reads that answer "how is plate OCR doing on motorcycles?"
        are precisely the short ones the old code dropped on the floor.
        """
        read = emb.get("plate_read")
        if not read:
            return
        if detection is not None and detection.id is None:
            # Assign the FK. Flushed only when there is a read to hang off
            # it, so the ordinary detection path is unchanged.
            session.flush()
        session.add(
            PlateRead(
                event_id=event.id,
                detection_id=detection.id if detection is not None else None,
                camera_id=cam.id,
                # Per-frame truth, not the event's class: an event's
                # class_name absorbs detector flapping (car↔truck), and
                # the screen's whole job is isolating motorcycles.
                class_name=(
                    detection.class_name if detection is not None else event.class_name
                ),
                identifier_key=emb["identifier"],
                at=ts,
                raw_text=read.get("raw_text"),
                text=read.get("normalized") or read.get("text"),
                accepted=read.get("text") is not None,
                reason=read.get("reason"),
                detector_confidence=read.get("detector_confidence"),
                ocr_confidence=read.get("ocr_confidence"),
                ocr_min_confidence=read.get("ocr_min_confidence"),
                plate_width=read.get("plate_width"),
                plate_height=read.get("plate_height"),
                sharpness=read.get("sharpness"),
                min_chars=int(read.get("min_chars") or 4),
                crop_path=self._save_plate_crop(
                    cam.id,
                    ts,
                    read.get("plate_jpeg"),
                    f"{detection.id if detection is not None else 'x'}"
                    f"-{emb['identifier']}",
                ),
                source_crop_path=crop_path,
            )
        )

    def _merge_with_prior(
        self, session, event: Event, identity_id: int, rules: EventConfig
    ) -> Event:
        """Fold `event` into a recent same-camera event of the same
        identity, returning whichever event survives (CLD-40).

        Both windows in the query compare frame time, symmetrically, so
        backfill clip order and restarts cannot change the outcome —
        `tests/test_resume_equivalence.py` holds this to account. The
        prior event absorbs the fragment: it is the row an operator may
        already have seen or reviewed.
        """
        if rules.merge_gap_s <= 0:
            return event
        gap = timedelta(seconds=rules.merge_gap_s)
        prior = (
            session.query(Event)
            .join(EventIdentity, EventIdentity.event_id == Event.id)
            .filter(
                Event.camera_id == event.camera_id,
                Event.id != event.id,
                EventIdentity.identity_id == identity_id,
                # Interval distance ≤ gap, whichever event came first.
                Event.last_seen >= event.first_seen - gap,
                Event.first_seen <= event.last_seen + gap,
            )
            .order_by(Event.id.desc())
            .first()
        )
        if prior is None:
            return event
        self._merge_events(session, prior, event, rules)
        return prior

    def _merge_events(
        self, session, target: Event, source: Event, rules: EventConfig
    ) -> None:
        """Move everything hanging off `source` onto `target`, combine the
        aggregates exactly, and delete the emptied source row."""
        for row in (
            session.query(Detection).filter(Detection.event_id == source.id).all()
        ):
            row.event_id = target.id
        # Plate reads follow their detections. Missing this would leave
        # rows pointing at an event about to be deleted — and a read whose
        # event no longer exists is unreviewable, which defeats the point
        # of keeping failures at all.
        for read in (
            session.query(PlateRead).filter(PlateRead.event_id == source.id).all()
        ):
            read.event_id = target.id
        for link in (
            session.query(EventIdentity)
            .filter(EventIdentity.event_id == source.id)
            .all()
        ):
            kept = None
            if link.is_active:
                # Only standing claims fold together. An unlinked row
                # (CLD-36) rides along to the survivor untouched — it is
                # a closed record of a repudiated claim, and merging it
                # into a live one would resurrect what an operator
                # detached.
                kept = active_claim(session, target.id, link.identity_id)
            if kept is None:
                link.event_id = target.id
                continue
            # Human review survives a merge (the Annotation philosophy) —
            # fold_claim keeps the stronger of the two verdicts.
            fold_claim(kept, link)
            session.delete(link)
        target.first_seen = min(target.first_seen, source.first_seen)
        target.last_seen = max(target.last_seen, source.last_seen)
        target.detection_count += source.detection_count
        target.confidence_sum += source.confidence_sum
        if source.best_confidence > target.best_confidence:
            target.best_confidence = source.best_confidence
            if source.best_crop_path:
                target.best_crop_path = source.best_crop_path
        target.guest_window = target.guest_window or source.guest_window
        target.missed_identity = target.missed_identity or source.missed_identity
        if target.missed_at is None:
            target.missed_at = source.missed_at
        if source.track_id is not None:
            # Adopt the fragment's track so its later frames fast-path here.
            target.track_id = source.track_id
        target.significant = target.significant or source.significant
        self._update_significance(target, rules)
        # Children are re-pointed above; flush before the delete so the
        # ORM never tries to null their FKs on the source's behalf.
        session.flush()
        session.delete(source)
        session.flush()

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
                and abs((ts - event.last_seen).total_seconds()) > rules.track_link_gap_s
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

        The top stitch_candidates events in the window are all tried and
        the best overlap wins: with two subjects in frame the newest
        event is usually the other subject, and a single-candidate probe
        minted one fresh event per frame (CLD-40).
        """
        if rules.stitch_gap_s <= 0:
            return None
        gap = timedelta(seconds=rules.stitch_gap_s)
        candidates = (
            session.query(Event)
            .filter(
                Event.camera_id == camera_id,
                Event.class_name.in_(group),
                Event.last_seen >= ts - gap,
                Event.last_seen <= ts + gap,
            )
            .order_by(Event.id.desc())
            .limit(max(1, rules.stitch_candidates))
            .all()
        )
        best: Event | None = None
        best_iou = 0.0
        for candidate in candidates:
            last_det = (
                session.query(Detection)
                .filter(Detection.event_id == candidate.id)
                .order_by(Detection.id.desc())
                .first()
            )
            if last_det is None:
                continue
            iou = _bbox_iou(json.loads(last_det.bbox), det["bbox"])
            # Strict > keeps the newest candidate on equal overlap, which
            # keeps the choice deterministic across restarts.
            if iou >= rules.stitch_min_iou and iou > best_iou:
                best, best_iou = candidate, iou
        if best is None:
            return None
        if det["track_id"] is not None:
            # Adopt the new track so its later frames hit the fast path.
            best.track_id = det["track_id"]
        return best

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

    def _save_plate_crop(
        self, camera_id: str, ts: datetime, plate_jpeg: bytes | None, tag: str
    ) -> str | None:
        """Write the plate sub-crop — a third image, its own directory.

        `crop_jpeg` is doing two jobs already (display thumbnail and
        embedder input) and changing it is a re-enroll event, so the
        evidence image for an OCR read is written beside it and never
        over it. `_save_crop` above is untouched by this.
        """
        if not plate_jpeg:
            return None
        day_dir = self.media_dir / camera_id / ts.strftime("%Y-%m-%d") / "plates"
        day_dir.mkdir(parents=True, exist_ok=True)
        # `tag` disambiguates two vehicles read in the same frame — the
        # timestamp alone collides and the second would overwrite the
        # first, silently attaching one read's evidence to another's row.
        path = day_dir / f"{ts.strftime('%H%M%S_%f')}_{tag}.jpg"
        path.write_bytes(plate_jpeg)
        return str(path)

    def run(self, max_frames: int | None = None, progress=None) -> None:
        """Ingest all configured cameras until stopped.

        File cameras run sequentially to completion (the backfill shape,
        PRD §6.6). Live cameras then run one worker thread each, so a
        slow or dropped stream never starves the others; each worker
        reconnects with backoff until stop() or a signal.

        `progress` is an optional live `ProgressReporter` (CLD-15). It
        makes a day-long soak visible to `siteloom jobs` and `/jobs`, and
        it takes over signal handling while attached — two owners racing
        for SIGINT is how a Ctrl-C gets swallowed.

        On "resume" for this command (CLD-12's open question): there is
        nothing to skip. Live ingest has no unit of done-ness — the frames
        it missed while stopped are simply gone — so the resume command is
        the same invocation, and restarting it means "reconnect", not
        "continue". Bounded catch-up over that gap is `backfill-unifi`.
        """
        self._stop.clear()
        self._progress = progress
        if progress is not None:
            # Seed the aggregates so the bar's first four counters are the
            # totals, not whichever per-camera key was touched first.
            progress.bump(**{key: 0 for key in self.TICK_COUNTERS})
        try:
            self._run_cameras(max_frames)
        finally:
            self._progress = None

    @contextmanager
    def _phase(self, name: str):
        """The reporter's phase if one is attached, otherwise nothing."""
        if self._progress is None:
            yield
        else:
            with self._progress.phase(name):
                yield

    def _run_cameras(self, max_frames: int | None) -> None:
        file_cams = [c for c in self.config.cameras if c.adapter == "file"]
        live_cams = [c for c in self.config.cameras if c.adapter != "file"]

        if file_cams:
            with self._phase(f"Ingesting {len(file_cams)} file camera(s)"):
                for cam in file_cams:
                    if self._stop.is_set():
                        return
                    log.info("ingesting camera %s (%s)", cam.id, cam.adapter)
                    count = self.run_camera(cam, max_frames=max_frames)
                    log.info("camera %s: %d frames processed", cam.id, count)

        if not live_cams:
            return
        # The reporter installs STOP_SIGNALS itself; installing ours on top
        # would leave whichever won holding a stop the other never sees.
        with self._stop_signals(active=self._progress is None), self._phase(
            f"Live ingest ({len(live_cams)} camera(s))"
        ):
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
            # Join with a timeout so the main thread keeps handling signals
            # — and, when a reporter owns them, so a stop reaches the
            # workers even while every camera is asleep in its backoff.
            while any(w.is_alive() for w in workers):
                if self._progress is not None and self._progress.interrupt_requested:
                    self._stop.set()
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
            self._tick(cam.id, reconnects=1)
            self._stop.wait(backoff)
            backoff = min(backoff * 2, LIVE_BACKOFF_MAX_S)
        log.info("camera %s: live ingest stopped (%d frames)", cam.id, processed)

    @contextmanager
    def _stop_signals(self, active: bool = True):
        """Route SIGINT/SIGTERM/SIGHUP to a graceful stop (progress.py
        convention): first signal drains in-flight frames, second aborts.

        `active=False` is a no-op, for when a ProgressReporter already owns
        these signals — it needs the interrupt to record the run.
        """
        from siteloom.progress import STOP_SIGNALS

        if not active:
            yield
            return

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
