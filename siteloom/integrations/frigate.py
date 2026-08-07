"""Frigate event consumer: recognition on top of an existing Frigate NVR.

The wiring this implements (the Double Take pattern, with Siteloom's
identity layer standing in for Double Take + CompreFace):

    UniFi/RTSP cameras → Frigate (motion + object detection)
    Frigate → MQTT broker            topic: frigate/events
    Siteloom subscribes to that topic. On a person/vehicle event it
    fetches the triggering snapshot from Frigate's own HTTP API — never
    the camera directly — runs the identity pipeline over it, stores the
    event + identity, republishes the result on siteloom/identity, and
    fires configured webhooks.

Frigate emits three message types per tracked object: "new" when first
confirmed, "update" as tracking/score improves, "end" when it leaves.
We process "new" and rate-limited "update"s (later frames often carry a
better face view than the first), deduped onto one Siteloom Event via
Event.external_id = the Frigate event id.

Identity state remains exactly the shared store: a face enrolled from
the photo backfill matches on a Frigate camera and vice versa.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy import select

from siteloom.config import SiteConfig
from siteloom.dispatch import Job, JobDispatcher
from siteloom.integrations.mqtt import MqttPublisher, identity_payload
from siteloom.integrations.webhooks import WebhookNotifier, classify_resolution
from siteloom.store import Camera, Event, EventIdentity

log = logging.getLogger(__name__)

# Frigate label -> Siteloom detection class (identical today, mapped
# explicitly so a divergence is a table entry, not a code path).
LABEL_MAP = {
    "person": "person",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "dog": "dog",
    "cat": "cat",
}


@dataclass
class ConsumerStats:
    received: int = 0
    processed: int = 0
    skipped: int = 0
    identities: int = 0
    errors: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped += 1
        self.by_reason[reason] = self.by_reason.get(reason, 0) + 1


# MQTT payloads are untrusted input (CLD-48): the event id is
# interpolated into a Frigate API URL, the camera id becomes a media_dir
# subpath and a Camera row, and the label feeds detection-class routing.
# `.` is in the class because Frigate ids look like "1712345678.123-abc",
# so the regex alone does not stop `..` — dotted traversal is rejected
# explicitly in _safe_id.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _safe_id(value: Any) -> str | None:
    """`value` if it is a string safe for URL/path/row use, else None."""
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        return None
    if value.startswith(".") or ".." in value:
        return None
    return value


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def fetch_snapshot(api_url: str, frigate_event_id: str) -> bytes | None:
    """The event's best snapshot from Frigate's API (never the camera)."""
    url = f"{api_url.rstrip('/')}/api/events/{frigate_event_id}/snapshot.jpg"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.read()
    except Exception as exc:
        log.warning("snapshot fetch failed for %s: %s", frigate_event_id, exc)
        return None


class FrigateConsumer:
    def __init__(
        self,
        config: SiteConfig,
        session_factory,
        dispatcher: JobDispatcher,
        resolver,
        publisher: MqttPublisher | None = None,
        notifier: WebhookNotifier | None = None,
        snapshot_fetcher: Callable[[str, str], bytes | None] = fetch_snapshot,
    ):
        self.config = config
        self.cfg = config.integrations.frigate
        self.Session = session_factory
        self.dispatcher = dispatcher
        self.resolver = resolver
        self.publisher = publisher
        self.notifier = notifier
        self.fetch = snapshot_fetcher
        self.stats = ConsumerStats()
        # frigate event id -> monotonic time of last processed snapshot,
        # to rate-limit "update" storms per tracked object.
        self._last_processed: dict[str, float] = {}

    # -- message handling ----------------------------------------------

    def handle_message(self, raw: bytes | str) -> bool:
        """One MQTT message off frigate/events. Returns True if a
        snapshot was processed."""
        self.stats.received += 1
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            self.stats.skip("bad-json")
            return False

        kind = message.get("type")
        after = message.get("after") or message.get("before") or {}
        if not isinstance(after, dict):
            self.stats.skip("bad-input")
            return False
        frigate_id = after.get("id")
        if not frigate_id:
            self.stats.skip("no-id")
            return False
        if _safe_id(frigate_id) is None:
            self.stats.skip("bad-input")
            return False
        if kind == "end":
            self._finalize(after)
            self._last_processed.pop(frigate_id, None)
            return False
        if kind not in ("new", "update"):
            self.stats.skip("type")
            return False

        label = after.get("label", "")
        if _safe_id(label) is None:
            self.stats.skip("bad-input")
            return False
        if self.cfg.labels and label not in self.cfg.labels:
            self.stats.skip("label")
            return False
        camera = after.get("camera", "")
        if _safe_id(camera) is None:
            self.stats.skip("bad-input")
            return False
        if self.cfg.cameras and camera not in self.cfg.cameras:
            self.stats.skip("camera")
            return False
        try:
            score = float(after.get("top_score") or after.get("score") or 0.0)
        except (TypeError, ValueError):
            self.stats.skip("bad-input")
            return False
        # Frigate scores are confidences in [0, 1]; anything else (NaN,
        # inf, negative, huge) is a hostile or corrupt payload.
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            self.stats.skip("bad-input")
            return False
        if score < self.cfg.min_score:
            self.stats.skip("score")
            return False
        if not after.get("has_snapshot", True):
            self.stats.skip("no-snapshot")
            return False
        now = time.monotonic()
        last = self._last_processed.get(frigate_id)
        if kind == "update" and last is not None:
            if now - last < self.cfg.update_interval_s:
                self.stats.skip("rate-limited")
                return False

        snapshot = self.fetch(self.cfg.api_url, frigate_id)
        if snapshot is None:
            self.stats.errors += 1
            return False
        self._last_processed[frigate_id] = now
        self._process_snapshot(frigate_id, camera, label, score, snapshot)
        self.stats.processed += 1
        return True

    # -- pipeline ---------------------------------------------------------

    def _process_snapshot(
        self, frigate_id: str, camera: str, label: str, score: float, jpeg: bytes
    ) -> None:
        class_name = LABEL_MAP.get(label, label)
        ts = _now()
        with self.Session() as session:
            event = self._event_for(session, frigate_id, camera, class_name, ts)
            event.last_seen = ts
            event.detection_count += 1
            event.confidence_sum += score
            if score > event.best_confidence:
                event.best_confidence = score
            crop_path = self._save_snapshot(frigate_id, event, jpeg, ts)
            if crop_path and not event.best_crop_path:
                event.best_crop_path = crop_path
            session.commit()

            if self.resolver is not None:
                self._identify(session, event, class_name, jpeg, crop_path)

            if self.publisher is not None:
                from siteloom.integrations.mqtt import event_payload

                self.publisher.publish("events", event_payload(event, camera))

    def _event_for(
        self, session, frigate_id: str, camera: str, class_name: str, ts: datetime
    ) -> Event:
        if session.get(Camera, camera) is None:
            session.add(
                Camera(
                    id=camera,
                    site_id=self.config.site_id,
                    name=camera,
                    adapter="frigate",
                )
            )
            session.flush()
        event = session.scalar(select(Event).filter_by(external_id=frigate_id))
        if event is None:
            event = Event(
                camera_id=camera,
                external_id=frigate_id,
                class_name=class_name,
                first_seen=ts,
                last_seen=ts,
                guest_window=False,
                # Frigate already motion/score-gates its events; the
                # significance gate is ingest's fragmentation fix and
                # must not hide externally curated events.
                significant=True,
            )
            session.add(event)
            session.flush()
        return event

    def _identify(self, session, event, class_name, jpeg, crop_path) -> None:
        result = self.dispatcher.submit_and_wait(
            Job(
                module="identity",
                payload={"crop_jpeg": jpeg, "class_name": class_name},
            )
        )
        if not result.ok:
            log.error("identity failed on frigate event %s: %s", event.external_id, result.error)
            self.stats.errors += 1
            return
        registry = self.config.identity.identifiers
        for emb in result.result["embeddings"]:
            ident_cfg = registry.get(emb["identifier"])
            resolution = self.resolver.resolve(
                session,
                identifier_key=emb["identifier"],
                class_name=class_name,
                vector=emb["vector"],
                plate=emb["plate"],
                timestamp=event.last_seen,
                crop_path=crop_path,
                threshold=ident_cfg.threshold if ident_cfg else None,
                max_vectors=ident_cfg.max_vectors_per_identity if ident_cfg else 20,
                camera_id=event.camera_id,
                # Frigate serves its best snapshot for the event; its own
                # score is the closest thing to detection quality here.
                quality=event.best_confidence or None,
            )
            if resolution.identity is None:
                # Quarantined or ambiguous (CLD-41) — no link, no publish.
                continue
            link = session.scalar(
                select(EventIdentity).filter_by(
                    event_id=event.id, identity_id=resolution.identity.id
                )
            )
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
            session.commit()
            self.stats.identities += 1

            payload = identity_payload(
                event, resolution.identity, resolution.similarity, resolution.is_new
            )
            if self.publisher is not None:
                self.publisher.publish("identity", payload)
            if self.notifier is not None:
                self.notifier.fire(
                    classify_resolution(
                        resolution.identity,
                        resolution.is_new,
                        resolution.learned_plate,
                    ),
                    payload,
                )

    def _finalize(self, after: dict[str, Any]) -> None:
        """On "end", stamp the event's true duration from Frigate's data."""
        frigate_id = after.get("id")
        end_time = after.get("end_time")
        if not frigate_id or not end_time:
            return
        try:
            end_ts = float(end_time)
        except (TypeError, ValueError):
            self.stats.skip("bad-input")
            return
        if not math.isfinite(end_ts):
            self.stats.skip("bad-input")
            return
        with self.Session() as session:
            event = session.scalar(select(Event).filter_by(external_id=frigate_id))
            if event is not None:
                event.last_seen = datetime.fromtimestamp(
                    end_ts, tz=timezone.utc
                ).replace(tzinfo=None)
                session.commit()

    def _save_snapshot(self, frigate_id, event, jpeg: bytes, ts) -> str | None:
        from pathlib import Path

        media_dir = Path(self.config.storage.media_dir).resolve()
        path = (
            media_dir
            / event.camera_id
            / ts.strftime("%Y-%m-%d")
            / f"frigate_{frigate_id}_{event.detection_count}.jpg"
        ).resolve()
        # Defense in depth behind _safe_id: whatever the ids contained,
        # the *resolved* path must stay inside media_dir (is_relative_to
        # on resolved paths, not a string-prefix check — "media-x" must
        # not pass for "media").
        if not path.is_relative_to(media_dir):
            log.warning(
                "refusing snapshot write outside media_dir for event %s", frigate_id
            )
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg)
        return str(path)

    # -- run loop -----------------------------------------------------------

    def run(self) -> None:  # pragma: no cover — needs a live broker
        """Blocking MQTT subscribe loop."""
        import paho.mqtt.client as mqtt

        mq = self.config.integrations.mqtt
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"{mq.client_id}-frigate",
        )
        if mq.username:
            client.username_pw_set(mq.username, mq.password)

        def on_connect(client, userdata, flags, reason_code, properties):
            log.info("connected to mqtt %s:%s (%s)", mq.host, mq.port, reason_code)
            client.subscribe(self.cfg.mqtt_topic)
            log.info("subscribed to %s", self.cfg.mqtt_topic)

        def on_message(client, userdata, message):
            try:
                self.handle_message(message.payload)
            except Exception:
                self.stats.errors += 1
                log.exception("error handling frigate event")

        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(mq.host, mq.port, keepalive=60)
        log.info(
            "consuming frigate events (labels=%s, min_score=%.2f, api=%s)",
            self.cfg.labels or "all",
            self.cfg.min_score,
            self.cfg.api_url,
        )
        client.loop_forever(retry_first_connection=True)
