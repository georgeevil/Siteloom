"""The site-timezone admin actions (CLD-100).

The setting itself is `SiteConfig.timezone` + `timezone_source`
(siteloom/config.py); the rendering it drives is `siteloom/localtime.py`.
This module is the write path: the small "Site time" panel on /classes
posts here, and each endpoint is one rung of the resolution chain made
into an operator action:

1. **Set** — an admin types (or picks) an IANA name. Validated before
   anything is applied, the /classes/detection contract (CLD-61): a typo
   is refused with a 400, never stored, never half-applied.
2. **Detect from NVR** — connect to the UniFi NVR once, read the zone its
   own settings carry, store it, disconnect. An *action*, not a
   per-request read: the serve process holds no NVR connection, and the
   NVR's answer only changes when someone reconfigures the NVR.
3. **Seed from browser** — the admin's browser proposes
   `Intl.DateTimeFormat().resolvedOptions().timeZone` (no permission
   prompt, no geolocation — the Intl API is deliberately the only
   mechanism) and the admin confirms. Applies only while the setting is
   unset: a seed is a starting point, not an override.

Rung 4 (default UTC, labelled) is the absence of a row here: an empty
`timezone` renders as UTC and the panel says so.

All three paths write back through the same config persistence
/classes/detection uses, so the setting survives a restart, and all three
are admin by prefix (`/classes/timezone` in auth.ADMIN_PREFIXES) — the
one middleware gates and audits them like every other mutation.

Detect and seed answer with a redirect carrying a short outcome code
(`/classes?tz=<code>`) the panel renders as a notice; codes are fixed
strings, never echoes of anything a client sent.
"""

from __future__ import annotations

import logging

from fastapi import Form, HTTPException
from fastapi.responses import RedirectResponse

from siteloom import localtime

log = logging.getLogger(__name__)


def read_nvr_timezone(config) -> str:
    """One connect-read-disconnect against the configured NVR.

    Module-level and injectable-by-monkeypatch so the route is testable
    with no NVR (the test-suite rule); raises on any failure — the route
    turns that into the polite notice.
    """
    from siteloom.adapters.unifi import UniFiProtectAdapter

    adapter = UniFiProtectAdapter(unifi=config.unifi)
    adapter.connect()
    try:
        return adapter.nvr_timezone()
    finally:
        adapter.close()


def register(app, templates, Session, config) -> None:
    def _persist() -> None:
        # The same write-back /classes/detection uses: live config object
        # plus YAML, silently skipped for a config built in memory.
        from siteloom.web.library_routes import _persist_config

        _persist_config(config)

    def _store(name: str, source: str) -> None:
        """Apply an already-validated zone. Empty clears back to UTC."""
        config.timezone = name
        config.timezone_source = source if name else ""
        _persist()

    @app.post("/classes/timezone")
    def set_timezone(timezone: str = Form("")):
        """Rung 1: the admin says what zone this site lives in.

        Parse before apply (CLD-61): the name is validated in full before
        the live config or the YAML is touched, so a typo changes nothing
        anywhere and says so.
        """
        try:
            name = localtime.validate_timezone(timezone)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        _store(name, "admin")
        return RedirectResponse(
            "/classes?tz=" + ("set" if name else "cleared"), status_code=303
        )

    @app.post("/classes/timezone/detect")
    def detect_timezone():
        """Rung 2: ask the NVR what wall clock its cameras live on."""
        if not config.unifi.host:
            return RedirectResponse("/classes?tz=no-unifi", status_code=303)
        try:
            name = localtime.validate_timezone(read_nvr_timezone(config))
            if not name:
                raise ValueError("NVR reported an empty timezone")
        except Exception as exc:
            # Unreachable NVR, bad credentials, a zone name this host's
            # tzdata does not know — all the same polite answer, with the
            # detail in the log where the operator debugging it will look.
            log.warning("NVR timezone detect failed: %s", exc)
            return RedirectResponse("/classes?tz=nvr-failed", status_code=303)
        _store(name, "nvr")
        return RedirectResponse("/classes?tz=detected", status_code=303)

    @app.post("/classes/timezone/seed")
    def seed_timezone(timezone: str = Form("")):
        """Rung 3: adopt the zone the admin's browser proposed — once.

        Only while unset. A site whose zone is already known (any rung)
        keeps it: the browser of whoever happens to open the panel next
        must never quietly move every rendered timestamp.
        """
        if config.timezone:
            return RedirectResponse("/classes?tz=already-set", status_code=303)
        try:
            name = localtime.validate_timezone(timezone)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        if not name:
            raise HTTPException(400, "the browser sent no timezone to seed")
        _store(name, "browser")
        return RedirectResponse("/classes?tz=seeded", status_code=303)
