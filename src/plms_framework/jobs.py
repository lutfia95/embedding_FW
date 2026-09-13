from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from .artifacts import ArtifactStore
from .errors import FrameworkError, ValidationError


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    progress: float = 0.0
    stage: str = "queued"
    submitted_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    result: dict | None = None
    error: dict | None = None
    artifacts: list[dict] = field(default_factory=list)

    def public(self, include_result: bool = False) -> dict:
        value = asdict(self)
        if not include_result:
            value.pop("result", None)
        return value


class JobManager:
    def __init__(self, artifact_store: ArtifactStore, max_workers: int = 1) -> None:
        self.artifact_store = artifact_store
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="plms-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(
        self, kind: str, function: Callable[[], dict], *, render: Callable | None = None
    ) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job, function, render)
        return job

    def _run(self, job: Job, function: Callable[[], dict], render: Callable | None) -> None:
        job.status, job.stage, job.started_at = "running", "compute", datetime.now(UTC).isoformat()
        job.progress = 0.05
        try:
            result = function()
            job.progress, job.stage = 0.8, "artifacts"
            artifacts = [self.artifact_store.save_json(job.id, "result.json", result)]
            if render is not None:
                artifacts.extend(render(result, self.artifact_store, job.id))
            manifest = {
                "job_id": job.id,
                "kind": job.kind,
                "result_provenance": result.get("provenance", {}),
                "artifacts": artifacts,
            }
            artifacts.append(self.artifact_store.save_json(job.id, "manifest.json", manifest))
            job.result, job.artifacts = result, artifacts
            job.status, job.stage, job.progress = "succeeded", "complete", 1.0
        except FrameworkError as exc:
            job.error = exc.as_dict()
            job.status, job.stage = "failed", "failed"
        # A worker boundary must translate unexpected backend exceptions into job state.
        except Exception as exc:  # noqa: BLE001
            job.error = {"code": "unexpected_job_failure", "message": str(exc), "context": {}}
            job.status, job.stage = "failed", "failed"
        finally:
            job.finished_at = datetime.now(UTC).isoformat()

    def get(self, job_id: str) -> Job:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise ValidationError(f"Unknown job: {job_id}") from exc
