"""Processing module protocol (PRD §7).

A module is a plain class with `process(job) -> result`. It never imports
or calls an execution backend — backends wrap modules, not the reverse.
Results must be serializable (dicts/lists/bytes) for the same reason job
payloads must be: they may cross a process or network boundary later.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from siteloom.dispatch.base import Job


@runtime_checkable
class ProcessingModule(Protocol):
    def process(self, job: Job) -> Any: ...
