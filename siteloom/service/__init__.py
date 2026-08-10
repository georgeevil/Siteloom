"""Running Siteloom under the operating system's own service manager.

The split here is the same one `health.py` makes: `ServiceSpec` is a set
of facts about a service and knows nothing about plist XML or systemd
INI, so both renderers, `service print-unit` and the tests share one
definition. `manager.py` is the only module that touches the filesystem
or a subprocess.

Siteloom does not daemonize itself. There is no `--daemon` and no PID
file: the process runs in the foreground, logs where it is told, and
stops on SIGTERM, and the supervisor owns backgrounding, restart and
boot ordering. A PID file would also be a second, worse liveness source
next to `OperationRun.pid` and `/healthz`.
"""

from siteloom.service.manager import (
    STATUS_NOT_INSTALLED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    Backend,
    InstalledUnit,
    ServiceError,
    backend_for_platform,
)
from siteloom.service.spec import UNITS, ServiceSpec, spec_from_config

__all__ = [
    "STATUS_NOT_INSTALLED",
    "STATUS_RUNNING",
    "STATUS_STOPPED",
    "UNITS",
    "Backend",
    "InstalledUnit",
    "ServiceError",
    "ServiceSpec",
    "backend_for_platform",
    "spec_from_config",
]
