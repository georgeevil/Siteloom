"""Webhook notifications on recognition results.

The Double Take role in the reference stack: when a face/vehicle is
matched (or a new unknown appears), POST JSON to configured URLs so
external systems — Home Assistant automations, a phone-notification
relay, a booking system — can react.

Delivery is fire-and-forget on a worker thread: a slow or dead endpoint
must not stall the ingest loop, and a lost notification is preferable to
blocked recognition.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
from typing import Any

from siteloom.config import WebhookConfig

log = logging.getLogger(__name__)

TIMEOUT_S = 10


class WebhookNotifier:
    def __init__(self, hooks: list[WebhookConfig]):
        self.hooks = hooks

    def fire(self, event_type: str, payload: dict[str, Any]) -> int:
        """Dispatch to every hook subscribed to event_type. Returns the
        number of hooks queued (not delivered — delivery is async)."""
        matching = [h for h in self.hooks if event_type in h.events]
        body = dict(payload, event_type=event_type)
        for hook in matching:
            threading.Thread(
                target=self._post, args=(hook, body), daemon=True
            ).start()
        return len(matching)

    def _post(self, hook: WebhookConfig, body: dict[str, Any]) -> None:
        try:
            request = urllib.request.Request(
                hook.url,
                data=json.dumps(body).encode(),
                headers={
                    "Content-Type": "application/json",
                    **(
                        {"Authorization": f"Bearer {hook.token}"}
                        if hook.token
                        else {}
                    ),
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp:
                if resp.status >= 400:
                    log.warning("webhook %s returned %s", hook.url, resp.status)
        except Exception as exc:
            log.warning("webhook %s failed: %s", hook.url, exc)


def classify_resolution(identity, is_new: bool, learned_plate: bool) -> str:
    """Map an identity resolution onto the webhook event taxonomy."""
    if learned_plate:
        return "identity.new_plate"
    if identity.label is not None:
        return "identity.match"
    return "identity.unknown"
