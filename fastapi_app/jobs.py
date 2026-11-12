"""Small file-based job queue shared by the API and the local operations worker."""

import json
import time
import uuid
from pathlib import Path

from filelock import FileLock, Timeout


class BusyError(Exception):
    """Another operation is queued or running."""


class WorkerUnavailable(Exception):
    """The local worker has not been started."""


def write(path: Path, content: dict) -> None:
    """Replace JSON atomically so the UI cannot read a half-written result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def lock(directory: Path, name: str) -> FileLock:
    directory.mkdir(parents=True, exist_ok=True)
    return FileLock(directory / f"{name}.lock", timeout=0)


def worker_available(directory: Path) -> bool:
    # The OS releases this lock if the worker exits, including after a crash.
    try:
        with lock(directory, "worker"):
            return False
    except Timeout:
        return True


def list_jobs(directory: Path) -> list[dict]:
    return sorted(
        [read(path) for path in (directory / "jobs").glob("*.json")],
        key=lambda job: job["created_at"],
        reverse=True,
    )


def get_job(directory: Path, job_id: str) -> dict:
    return read(directory / "jobs" / f"{uuid.UUID(job_id).hex}.json")


def save_job(directory: Path, job: dict) -> None:
    write(directory / "jobs" / f"{job['id']}.json", job)


def log_path(directory: Path, job_id: str) -> Path:
    return directory / "jobs" / f"{uuid.UUID(job_id).hex}.log"


def submit(directory: Path, kind: str, arguments: dict, queued: bool) -> dict:
    """API jobs require a worker; direct CLI jobs execute in their own process."""
    if queued and not worker_available(directory):
        raise WorkerUnavailable("Start the local operations worker before submitting a job.")
    try:
        with lock(directory, "queue"):
            if any(job["state"] in ("queued", "running") for job in list_jobs(directory)):
                raise BusyError("Another operation is already queued or running.")
            job = {
                "id": uuid.uuid4().hex,
                "kind": kind,
                "arguments": arguments,
                "state": "queued" if queued else "running",
                "created_at": time.time(),
                "started_at": None if queued else time.time(),
                "finished_at": None,
                "steps": [],
                "source": None,
                "result": None,
                "error": None,
            }
            save_job(directory, job)
            return job
    except Timeout as exc:
        raise BusyError("The job queue is being updated. Try again.") from exc


def claim(directory: Path) -> dict | None:
    """Caller holds the execution lock; claim the single queued operation."""
    with lock(directory, "queue"):
        for job in list_jobs(directory):
            if job["state"] == "queued":
                job.update(state="running", started_at=time.time())
                save_job(directory, job)
                return job
    return None


def recover_interrupted(directory: Path) -> None:
    """Caller holds the execution lock, so any recorded running job was interrupted."""
    with lock(directory, "queue"):
        for job in list_jobs(directory):
            if job["state"] == "running":
                job.update(
                    state="failed",
                    error="The operations process stopped before this job completed.",
                    finished_at=time.time(),
                )
                save_job(directory, job)
