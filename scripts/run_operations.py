"""Run predefined operations locally, either directly or through the UI job queue."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from filelock import Timeout

from fastapi_app import jobs
from fraud_detection.common import read_config, source_metadata
from scripts import release

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("init", "reproduce", "retrain", "check", "ci", "release", "deploy", "rollback")


def candidate_info() -> dict:
    config = read_config(ROOT / "params.yaml")
    info = jobs.read(ROOT / config["outputs"]["artifacts"] / "model_info.json")
    report = jobs.read(ROOT / config["outputs"]["reports"] / "results.json")
    comparison = jobs.read(ROOT / config["outputs"]["candidates"] / "comparison.json")
    if (
        info["comparison_run_id"] != comparison["parent_run_id"]
        or report["comparison_run_id"] != comparison["parent_run_id"]
        or report["registered_model_version"] != int(info["model_version"])
        or report["selected_model"] != info["selected_model"]
        or comparison["candidates"][info["selected_model"]]["run_id"] != info["run_id"]
    ):
        raise ValueError("Candidate, evaluation report and MLflow run identities disagree.")
    return info


def execute(directory: Path, job: dict) -> bool:
    """Caller holds the execution lock. A failed step stops all following steps."""
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    env["PIPELINE_JOB_ID"] = job["id"]
    env["PYTHONUNBUFFERED"] = "1"

    with jobs.log_path(directory, job["id"]).open("w", buffering=1) as log:

        def step(name, operation, *arguments):
            record = {
                "name": name,
                "state": "running",
                "started_at": time.time(),
                "seconds": None,
                "execution": None,
                "error": None,
            }
            job["steps"].append(record)
            jobs.save_job(directory, job)
            started = time.perf_counter()
            print(f"\n--- {name} ---", file=log, flush=True)
            try:
                result = operation(*arguments)
            except Exception as exc:
                record.update(state="failed", error=str(exc), seconds=time.perf_counter() - started)
                jobs.save_job(directory, job)
                raise
            record.update(state="success", seconds=time.perf_counter() - started)
            if isinstance(result, dict) and "execution" in result:
                record["execution"] = result["execution"]
            jobs.save_job(directory, job)
            return result

        def command(arguments):
            release.command(arguments, log, env)

        def run_stage(name, output, force):
            arguments = [sys.executable, "-m", "dvc", "repro", "--single-item", name]
            if force:
                arguments.append("--force")
            command(arguments)
            recorded = jobs.read(output)
            return {
                "execution": "executed"
                if recorded.get("pipeline_job_id") == job["id"]
                else "reused"
            }

        def pipeline(force):
            config = read_config(ROOT / "params.yaml")
            outputs = {
                "prepare": ROOT / config["data"]["prepared"] / "preparation.json",
                "train": ROOT / config["outputs"]["candidates"] / "comparison.json",
                "evaluate": ROOT / config["outputs"]["reports"] / "results.json",
            }
            for name, output in outputs.items():
                step(name, run_stage, name, output, force)

        def require_current_pipeline():
            status = subprocess.check_output(
                [sys.executable, "-m", "dvc", "status", "--json"],
                cwd=ROOT,
                env=env,
                text=True,
            )
            if json.loads(status):
                raise ValueError(
                    "DVC outputs are stale. Run CI and review its changes before committing."
                )

        def checks():
            paths = ["fraud_detection", "fastapi_app", "scripts", "tests"]
            commands = [
                (
                    "Style checks",
                    ["ruff", "check", "--line-length", "100", "--select", "E,F,I", *paths],
                ),
                ("Formatting", ["ruff", "format", "--check", "--line-length", "100", *paths]),
                ("Tests", ["pytest", "tests", "-q", f"--junitxml=reports/tests/{job['id']}.xml"]),
            ]
            for name, arguments in commands:
                step(name, command, [sys.executable, "-m", *arguments])

        try:
            job["source"] = source_metadata((ROOT / "params.yaml").resolve())
            job["source"]["workspace_dirty"] = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=ROOT,
                    text=True,
                ).strip()
            )
            job["dag"] = subprocess.check_output(
                [sys.executable, "-m", "dvc", "dag", "--dot"],
                cwd=ROOT,
                env=env,
                text=True,
                stderr=subprocess.STDOUT,
            )
            jobs.save_job(directory, job)
            kind = job["kind"]
            if kind == "init":
                job["result"] = step("Record running baseline", release.initialise, directory)
            elif kind in ("reproduce", "retrain", "check", "ci", "release"):
                if kind in ("check", "ci", "release"):
                    checks()
                if kind == "check":
                    step("Check DVC outputs", require_current_pipeline)
                else:
                    pipeline(force=kind == "retrain")
                info = candidate_info()
                if kind in ("check", "ci"):
                    job["result"] = step(
                        "Validate candidate", release.check_model, int(info["model_version"])
                    )
                elif kind == "release":
                    image = f"aiops-project:job-{job['id'][:12]}"
                    step("Build image", command, ["docker", "build", "-t", image, "."])
                    job["result"] = step(
                        "Deploy", release.deploy, directory, int(info["model_version"]), image, log
                    )
                else:
                    job["result"] = info
            elif kind == "deploy":
                image = job["arguments"]["image"]
                if image is None:
                    image = release.read_env()["APP_IMAGE"]
                job["result"] = step(
                    "Deploy", release.deploy, directory, job["arguments"]["version"], image, log
                )
            elif kind == "rollback":
                job["result"] = step("Rollback", release.rollback, directory, log)
            else:
                raise ValueError(f"Unknown operation: {kind}")
        except Exception as exc:
            traceback.print_exc(file=log)
            job.update(state="failed", error=str(exc), finished_at=time.time())
            jobs.save_job(directory, job)
            return False
    job.update(state="success", finished_at=time.time())
    jobs.save_job(directory, job)
    return True


def worker(directory: Path) -> None:
    stop = False

    def stop_after_current_job(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stop_after_current_job)
    signal.signal(signal.SIGINT, stop_after_current_job)
    with jobs.lock(directory, "worker"):
        print("Operations worker ready; stop with Ctrl+C after the current job.", flush=True)
        while not stop:
            try:
                with jobs.lock(directory, "execution"):
                    jobs.recover_interrupted(directory)
                    job = jobs.claim(directory)
                    if job is not None:
                        execute(directory, job)
            except Timeout:
                pass  # A direct CLI operation is running; leave it in control.
            time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("worker", *KINDS))
    parser.add_argument("--version", type=int)
    parser.add_argument("--image", help="Explicit application image for an operator deployment.")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error("Run this command from the project root.")
    if args.action == "deploy" and (args.version is None or args.version < 1):
        parser.error("deploy requires --version with a positive registry version.")
    if args.action != "deploy" and (args.version is not None or args.image is not None):
        parser.error("--version and --image apply only to deploy.")
    directory = ROOT / "operations"
    try:
        if args.action == "worker":
            worker(directory)
            return
        with jobs.lock(directory, "execution").acquire(timeout=5):
            jobs.recover_interrupted(directory)
            job = jobs.submit(
                directory, args.action, {"version": args.version, "image": args.image}, queued=False
            )
            success = execute(directory, job)
        print(
            json.dumps(
                {key: job[key] for key in ("id", "kind", "state", "error", "result")}, indent=2
            )
        )
        print(f"Log: {jobs.log_path(directory, job['id'])}")
        raise SystemExit(0 if success else 1)
    except (Timeout, jobs.BusyError) as exc:
        parser.exit(1, f"Another operations process is active: {exc}\n")


if __name__ == "__main__":
    main()
