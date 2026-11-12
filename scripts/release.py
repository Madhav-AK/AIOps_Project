"""Validate a registered model, deploy with Compose, and restore a known release."""

import json
import shlex
import subprocess
import tempfile
import time
from importlib.metadata import version as installed_version
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import requests
from dotenv import dotenv_values, set_key
from mlflow import MlflowClient
from packaging.requirements import Requirement

from fastapi_app import jobs
from fraud_detection.common import read_config
from fraud_detection.features import INPUT_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ("model-server", "backend", "frontend")


def command(arguments: list[str], log, env=None, check=True) -> None:
    """Execute a fixed argument list and retain its actual output and exit code."""
    print("$ " + shlex.join(arguments), file=log, flush=True)
    subprocess.run(
        arguments,
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=check,
    )


def image_id(image: str) -> str:
    return subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def check_model(model_version: int) -> dict:
    """Check packaging and the inference contract, without a model-quality gate."""
    config = read_config(ROOT / "params.yaml")
    uri = config["tracking"]["uri"]
    model_uri = f"models:/{config['tracking']['model_name']}/{model_version}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    registered = client.get_model_version(config["tracking"]["model_name"], str(model_version))
    if registered.tags.get("selection") != "validation_f1":
        raise ValueError("Only validation-selected model versions can be deployed.")
    parent_id = client.get_run(registered.run_id).data.tags["mlflow.parentRunId"]
    with tempfile.TemporaryDirectory(prefix="fraud-release-") as temporary:
        model_path = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                tracking_uri=uri,
                dst_path=temporary,
            )
        )
        for line in (model_path / "requirements.txt").read_text().splitlines():
            requirement = Requirement(line)
            if installed_version(requirement.name) not in requirement.specifier:
                raise ValueError(f"Model requires {line}; the active environment differs.")
        model = mlflow.pyfunc.load_model(model_path)
        if model.metadata.signature.inputs.input_names() != list(INPUT_COLUMNS):
            raise ValueError("Model input fields do not match the application contract.")
        threshold = model.metadata.metadata["decision_threshold"]
        if not 0 < threshold < 1:
            raise ValueError("Model decision threshold must be between zero and one.")
        examples = [
            json.loads((ROOT / "data/examples" / name).read_text())
            for name in ("not_flagged.json", "flag_for_review.json")
        ]
        scores = np.asarray(model.predict(pd.DataFrame(examples)))
        if (
            scores.shape != (2, 2)
            or scores.dtype.kind not in "fi"
            or not np.isfinite(scores).all()
            or (scores < 0).any()
            or (scores > 1).any()
            or not np.allclose(scores.sum(axis=1), 1)
        ):
            raise ValueError("Model must return two valid class scores for each example.")
        comparison = json.loads(
            Path(
                client.download_artifacts(
                    parent_id,
                    "comparison.json",
                    temporary,
                )
            ).read_text()
        )
        source = comparison["source"]
        if not source["git_commit"] or not source["source_files_sha256"]:
            raise ValueError("The model has no recorded source lineage.")
    return {
        "model_uri": model_uri,
        "model_version": str(model_version),
        "decision_threshold": float(threshold),
        "comparison_run_id": parent_id,
        "source_git_commit": source["git_commit"],
        "source_worktree_dirty": source["source_worktree_dirty"],
        "cases": [
            {
                "input": example,
                "score": float(score),
                "label": "Flag for review" if score >= threshold else "Not flagged",
            }
            for example, score in zip(examples, scores[:, 1])
        ],
    }


def read_env() -> dict:
    values = dotenv_values(ROOT / ".env")
    if not all(values.get(key) for key in ("MODEL_URI", "DECISION_THRESHOLD", "APP_IMAGE")):
        raise ValueError(".env must explicitly set MODEL_URI, DECISION_THRESHOLD and APP_IMAGE.")
    return values


def updated_env(directory: Path, values: dict) -> str:
    """Prepare settings without changing the active Compose configuration."""
    pending = directory / "deployment.env.pending"
    pending.write_text((ROOT / ".env").read_text())
    for key, value in values.items():
        set_key(pending, key, str(value), quote_mode="never")
    content = pending.read_text()
    pending.unlink()
    return content


def restore_env(directory: Path, snapshot: dict) -> None:
    pending = directory / "deployment.env.pending"
    pending.write_text(snapshot["env_text"])
    pending.replace(ROOT / ".env")


def compose_command(snapshot: dict, *arguments: str) -> list[str]:
    """Use the recorded deployment files, including its HTTPS override if enabled."""
    files = snapshot["operations"]["compose_files"]
    if not files or any(not (ROOT / name).is_file() for name in files):
        raise ValueError("Every configured Compose file must exist.")
    return ["docker", "compose", *[part for name in files for part in ("-f", name)], *arguments]


def verify_running(snapshot: dict) -> None:
    config = snapshot["operations"]
    url = config["api_url"].rstrip("/")
    verify = True if config["ca_bundle"] is None else str(ROOT / config["ca_bundle"])
    with requests.Session() as client:
        client.verify = verify
        for route in ("/ready", "/model"):
            response = client.get(url + route, timeout=8)
            response.raise_for_status()
            payload = response.json()
            version = payload["model_version"] if route == "/ready" else payload["version"]
            if version != snapshot["model_version"]:
                raise ValueError("The API reports a different deployed model version.")
        for case in snapshot["validation"]["cases"]:
            response = client.post(url + "/predict", json=case["input"], timeout=8)
            response.raise_for_status()
            result = response.json()
            if (
                result["model_uri"] != snapshot["model_uri"]
                or result["label"] != case["label"]
                or abs(result["score"] - case["score"]) > 1e-6
                or result["decision_threshold"] != snapshot["decision_threshold"]
            ):
                raise ValueError("Deployed inference differs from the validated model.")
    for service in SERVICES:
        container = subprocess.check_output(
            compose_command(snapshot, "ps", "-q", service),
            cwd=ROOT,
            text=True,
        ).strip()
        actual = subprocess.check_output(
            ["docker", "inspect", "--format", "{{.Image}}", container],
            text=True,
        ).strip()
        if actual != snapshot["image_id"]:
            raise ValueError(f"The {service} container is running a different image.")


def public(snapshot: dict) -> dict:
    return {
        key: snapshot[key]
        for key in (
            "model_uri",
            "model_version",
            "decision_threshold",
            "image",
            "image_id",
            "deployed_at",
        )
    }


def initialise(directory: Path) -> dict:
    """Explicitly record the actual running baseline once, before the first release."""
    path = directory / "deployment.json"
    if path.exists():
        raise ValueError("Deployment history has already been initialized.")
    values = read_env()
    name, version = values["MODEL_URI"].removeprefix("models:/").rsplit("/", 1)
    config = read_config(ROOT / "params.yaml")
    if name != config["tracking"]["model_name"] or not version.isdecimal():
        raise ValueError("The baseline must use an immutable numeric model version.")
    validation = check_model(int(version))
    if float(values["DECISION_THRESHOLD"]) != validation["decision_threshold"]:
        raise ValueError("The configured threshold differs from the model package.")
    snapshot = {
        **{key: validation[key] for key in ("model_uri", "model_version", "decision_threshold")},
        "image": values["APP_IMAGE"],
        "image_id": image_id(values["APP_IMAGE"]),
        "env_text": (ROOT / ".env").read_text(),
        "validation": validation,
        "operations": config["operations"],
        "deployed_at": time.time(),
    }
    verify_running(snapshot)
    jobs.write(path, {"current": snapshot, "previous": None, "events": []})
    return public(snapshot)


def switch(directory: Path, state: dict, target: dict, action: str, log) -> dict:
    """A failed switch restores and verifies the exact previous model and image."""
    previous = state["current"]
    if image_id(previous["image"]) != previous["image_id"]:
        raise ValueError("The running image tag changed; preserve its recorded image for rollback.")
    start = ("up", "-d", "--no-build", "--wait", "--wait-timeout", "90", *SERVICES)
    target_command = compose_command(target, *start)
    recovery_command = compose_command(previous, *start)
    diagnostic_command = compose_command(target, "logs", "--no-color", "--tail", "30")
    event = {
        "action": action,
        "state": "running",
        "started_at": time.time(),
        "from": public(previous),
        "to": public(target),
        "error": None,
    }
    state["events"].append(event)
    jobs.write(directory / "deployment.json", state)
    try:
        restore_env(directory, target)
        command(target_command, log)
        verify_running(target)
    except Exception as exc:
        print(f"Deployment failed: {exc}. Restoring the previous release.", file=log, flush=True)
        # Retain container errors before recovery removes/recreates failed containers.
        command(diagnostic_command, log, check=False)
        try:
            restore_env(directory, previous)
            command(recovery_command, log)
            verify_running(previous)
        except Exception as recovery_error:
            event.update(
                state="recovery_failed", error=str(recovery_error), finished_at=time.time()
            )
            jobs.write(directory / "deployment.json", state)
            raise RuntimeError(
                "Deployment and automatic recovery failed; inspect the job log."
            ) from recovery_error
        event.update(state="restored_previous", error=str(exc), finished_at=time.time())
        jobs.write(directory / "deployment.json", state)
        raise RuntimeError(
            "Deployment failed; the previous release was restored and verified."
        ) from exc
    target["deployed_at"] = time.time()
    state.update(current=target, previous=previous)
    event.update(state="success", finished_at=time.time())
    jobs.write(directory / "deployment.json", state)
    return public(target)


def deploy(directory: Path, model_version: int, image: str, log) -> dict:
    state = jobs.read(directory / "deployment.json")
    validation = check_model(model_version)
    identity = image_id(image)
    content = updated_env(
        directory,
        {
            "MODEL_URI": validation["model_uri"],
            "DECISION_THRESHOLD": validation["decision_threshold"],
            "APP_IMAGE": image,
        },
    )
    target = {
        **{key: validation[key] for key in ("model_uri", "model_version", "decision_threshold")},
        "image": image,
        "image_id": identity,
        "env_text": content,
        "validation": validation,
        "operations": read_config(ROOT / "params.yaml")["operations"],
        "deployed_at": time.time(),
    }
    return switch(directory, state, target, "deploy", log)


def rollback(directory: Path, log) -> dict:
    state = jobs.read(directory / "deployment.json")
    if state["previous"] is None:
        raise ValueError("There is no previous successful release to restore.")
    target = state["previous"]
    if image_id(target["image"]) != target["image_id"]:
        raise ValueError("The previous image tag no longer identifies its recorded image.")
    return switch(directory, state, target, "rollback", log)
