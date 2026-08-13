"""MQTT publishing: Siteloom events onto the home-automation bus.

Anything that speaks MQTT (Home Assistant, Node-RED, a Double Take-style
consumer pointed at us instead of Frigate) can react to Siteloom
detections and identity matches without polling our API.

Topics, under `integrations.mqtt.base_topic` (default "siteloom"):
    siteloom/events    — one message per stored event update
    siteloom/identity  — one message per identity resolution
    siteloom/noise     — one message per noise episode
    siteloom/watchlist — one message per watched-plate sighting (per event)

Payloads are flat JSON with a stable shape; new keys may be added but
existing ones are not repurposed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from siteloom.config import MqttConfig

log = logging.getLogger(__name__)


class MqttPublisher:
    """Thin wrapper over paho with lazy connect and quiet failure.

    Publishing telemetry must never take down ingestion — a broker being
    offline degrades to a log line, not an exception (NFR1: nothing may
    assume the uplink is always reachable).
    """

    def __init__(self, cfg: MqttConfig):
        self.cfg = cfg
        self._client = None
        self._failed_once = False

    def _connect(self):
        if self._client is not None:
            return self._client
        import paho.mqtt.client as mqtt

        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.cfg.client_id,
        )
        if self.cfg.username:
            client.username_pw_set(self.cfg.username, self.cfg.password)
        client.connect(self.cfg.host, self.cfg.port, keepalive=60)
        client.loop_start()  # background thread services the socket
        self._client = client
        return client

    def publish(self, subtopic: str, payload: dict[str, Any]) -> bool:
        if not self.cfg.enabled:
            return False
        try:
            client = self._connect()
            topic = f"{self.cfg.base_topic}/{subtopic}"
            client.publish(topic, json.dumps(payload), qos=0)
            self._failed_once = False
            return True
        except Exception as exc:
            if not self._failed_once:  # log the outage once, not per event
                log.warning("mqtt publish failed (%s); continuing without", exc)
                self._failed_once = True
            self._client = None
            return False

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._client = None


def event_payload(event, camera_name: str = "") -> dict[str, Any]:
    return {
        "source": "siteloom",
        "type": "event",
        "event_id": event.id,
        "external_id": event.external_id,
        "camera": camera_name or event.camera_id,
        "label": event.class_name,
        "track_id": event.track_id,
        "first_seen": event.first_seen.isoformat(),
        "last_seen": event.last_seen.isoformat(),
        "detections": event.detection_count,
        "top_confidence": event.best_confidence,
        "guest_window": event.guest_window,
    }


def watchlist_payload(event, watch, plate: str) -> dict[str, Any]:
    """One watched-plate sighting. Carries the watch's label and note so
    an automation can route on *why* the plate is watched without a
    second lookup against our API."""
    return {
        "source": "siteloom",
        "type": "watchlist",
        "event_id": event.id,
        "camera": event.camera_id,
        "label": event.class_name,
        "plate": plate,
        "watch_label": watch.label,
        "watch_note": watch.note,
        "timestamp": event.last_seen.isoformat(),
    }


def identity_payload(event, identity, similarity: float, is_new: bool) -> dict[str, Any]:
    return {
        "source": "siteloom",
        "type": "identity",
        "event_id": event.id,
        "camera": event.camera_id,
        "label": event.class_name,
        "identifier": identity.identifier_key,
        "identity_id": identity.id,
        "name": identity.label,  # null while in the unknown bucket
        "display_name": identity.display_name,
        "known": identity.label is not None,
        "new_identity": is_new,
        "similarity": round(similarity, 4),
        "plate": identity.plate,
        "timestamp": event.last_seen.isoformat(),
    }
