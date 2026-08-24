"""sd_notify, in stdlib, for `Type=notify` units.

Opt-in (`siteloom service install --notify`), not the default, and worth
saying why. What `Type=notify` buys is honest: `systemctl start` blocks
until the server is actually accepting connections instead of returning
the moment the process exists. What it costs is a readiness signal that
has to be told the truth by hand, where `Type=exec` — the default here —
already catches the failure that actually happens, which is a bad exec:
a moved venv, a config that will not load.

`WatchdogSec` is deliberately not offered. The only place to ping from is
a background thread, and a thread keeps pinging happily while uvicorn's
event loop is wedged — a watchdog that lies is worse than none, and
`/readyz` is a better liveness probe because it goes through the loop.

Every function here is a silent no-op when `$NOTIFY_SOCKET` is unset,
which is every run that is not under systemd, including every test. This
follows the rule the MQTT publisher does: telemetry that cannot reach its
sink degrades to nothing, never to an exception (NFR1).
"""

from __future__ import annotations

import logging
import os
import socket

log = logging.getLogger(__name__)

# SOCK_CLOEXEC is a Linux-only constant, and reading it off `socket`
# elsewhere is an AttributeError — which the OSError guard in `_send` does
# not catch, in the one module whose whole promise is that it never raises.
# Python has set FD_CLOEXEC on new file descriptors by default since PEP
# 446, so falling back to 0 gives up nothing on the platforms that lack it.
_CLOEXEC = getattr(socket, "SOCK_CLOEXEC", 0)


def _send(message: bytes) -> bool:
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return False
    # A leading '@' means the Linux abstract namespace, which is spelled
    # as a leading NUL byte in the sockaddr.
    if address[0] == "@":
        address = "\0" + address[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | _CLOEXEC) as sock:
            sock.connect(address)
            sock.sendall(message)
        return True
    except OSError as exc:
        log.debug("sd_notify %r failed: %s", message, exc)
        return False


def ready() -> bool:
    """Tell systemd the service is up. Returns whether anything listened."""
    return _send(b"READY=1")


def stopping() -> bool:
    return _send(b"STOPPING=1")


def status(text: str) -> bool:
    return _send(b"STATUS=" + text.encode())
