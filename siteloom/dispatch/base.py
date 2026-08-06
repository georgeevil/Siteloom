"""Job execution backend interface (PRD §7).

Processing modules are plain callables with `process(job) -> result`; they
never touch a specific execution backend. A `JobDispatcher` wraps whichever
backend is configured. The interface is modeled on Ray's core API
(`remote()` returning a future, `get()` to resolve) so a future RayBackend
is a thin wrapper and a CeleryBackend maps onto AsyncResult naturally.

Job payloads must stay serializable — references (paths, bytes, ids),
never live Python objects — so the same payload works across backends.
"""

from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Job:
    """A unit of work routed to a processing module.

    `module` names the registered ProcessingModule ("detection", "face",
    ...). `payload` holds only serializable data — e.g. JPEG bytes plus
    source camera id and timestamp, never a numpy array or open handle.
    """

    module: str
    payload: dict[str, Any]
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class JobResult:
    job_id: str
    module: str
    ok: bool
    result: Any = None
    error: str | None = None


class JobHandle(abc.ABC):
    """Future-like handle returned by submit() (Ray ObjectRef analog)."""

    @abc.abstractmethod
    def wait(self, timeout: float | None = None) -> JobResult: ...


class JobDispatcher(abc.ABC):
    """Uniform front for any execution backend.

    Application code (ingestion service, backfill job) only ever sees this
    interface, so moving from one laptop to a distributed fleet is a config
    change, not a code change.
    """

    @abc.abstractmethod
    def register(self, name: str, module: Any) -> None:
        """Register a ProcessingModule under a routing name."""

    @abc.abstractmethod
    def submit(self, job: Job) -> JobHandle: ...

    def submit_and_wait(self, job: Job, timeout: float | None = None) -> JobResult:
        return self.submit(job).wait(timeout)

    def submit_batch(self, jobs: list[Job]) -> list[JobHandle]:
        return [self.submit(j) for j in jobs]

    def shutdown(self) -> None:  # noqa: B027 — optional hook
        pass
