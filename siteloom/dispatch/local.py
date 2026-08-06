"""LocalBackend: synchronous, in-process execution (PRD §7).

No queue — the module runs directly in submit(). Modules are instantiated
once and kept warm (mirrors Ray's actor-per-model-type pattern), so model
weights load a single time per process.
"""

from __future__ import annotations

from typing import Any

from siteloom.dispatch.base import Job, JobDispatcher, JobHandle, JobResult


class _ResolvedHandle(JobHandle):
    def __init__(self, result: JobResult):
        self._result = result

    def wait(self, timeout: float | None = None) -> JobResult:
        return self._result


class LocalBackend(JobDispatcher):
    def __init__(self) -> None:
        self._modules: dict[str, Any] = {}

    def register(self, name: str, module: Any) -> None:
        self._modules[name] = module

    def submit(self, job: Job) -> JobHandle:
        module = self._modules.get(job.module)
        if module is None:
            return _ResolvedHandle(
                JobResult(
                    job_id=job.job_id,
                    module=job.module,
                    ok=False,
                    error=f"no module registered under {job.module!r}",
                )
            )
        try:
            result = module.process(job)
            return _ResolvedHandle(
                JobResult(job_id=job.job_id, module=job.module, ok=True, result=result)
            )
        except Exception as exc:  # jobs must not kill the ingest loop
            return _ResolvedHandle(
                JobResult(
                    job_id=job.job_id,
                    module=job.module,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
