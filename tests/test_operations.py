"""Queue exclusion, fixed API actions, pipeline failures and deployment recovery."""

import io
import json
import subprocess

import pytest
import yaml
from fastapi.testclient import TestClient

from fastapi_app import jobs
from fastapi_app.app import app
from scripts import release, run_operations


@pytest.fixture
def operations_api(monkeypatch, tmp_path, monitoring_environment):
    monkeypatch.setenv("MODEL_URI", "models:/transaction-fraud/9")
    monkeypatch.setenv("DECISION_THRESHOLD", "0.5")
    monkeypatch.setenv("MODEL_SERVER_URL", "http://unused.test")
    monkeypatch.setenv("MLFLOW_PUBLIC_URL", "http://127.0.0.1:5000")
    monkeypatch.setenv("APP_IMAGE", "test-image")
    monkeypatch.setenv("OPERATIONS_DIR", str(tmp_path))
    with TestClient(app) as client:
        yield client, tmp_path


def test_api_refuses_jobs_without_a_worker(operations_api):
    client, directory = operations_api
    assert client.get("/operations").json() == {"worker_running": False, "busy": False}
    assert client.post("/pipeline/runs", json={"action": "reproduce"}).status_code == 503
    assert jobs.list_jobs(directory) == []


def test_queue_accepts_one_job_and_exposes_its_real_failure(operations_api):
    client, directory = operations_api
    with jobs.lock(directory, "worker"):
        response = client.post("/pipeline/runs", json={"action": "retrain"})
        assert response.status_code == 202
        job = response.json()
        assert job["state"] == "queued"
        assert client.post("/pipeline/runs", json={"action": "ci"}).status_code == 409
        with jobs.lock(directory, "execution"):
            running = jobs.claim(directory)
            assert running["id"] == job["id"]
            running.update(state="failed", error="Invalid labelled data.")
            jobs.save_job(directory, running)
            jobs.log_path(directory, job["id"]).write_text("Validation failed before training.\n")
        detail = client.get(f"/pipeline/runs/{job['id']}").json()
        assert detail["state"] == "failed"
        assert detail["error"] == "Invalid labelled data."
        assert "before training" in detail["log_tail"]
        assert client.get("/operations").json()["busy"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "shell"},
        {"action": "reproduce", "command": "echo unsafe"},
        {},
    ],
)
def test_api_accepts_only_predefined_pipeline_actions(operations_api, payload):
    client, directory = operations_api
    assert client.post("/pipeline/runs", json=payload).status_code == 422
    assert jobs.list_jobs(directory) == []


@pytest.mark.parametrize("version", [-1, 0, True, "9"])
def test_deployment_requires_a_positive_integer_version(operations_api, version):
    client, directory = operations_api
    assert client.post("/deployments", json={"version": version}).status_code == 422
    assert jobs.list_jobs(directory) == []


def test_api_submits_version_without_accepting_shell_or_image_input(operations_api):
    client, directory = operations_api
    with jobs.lock(directory, "worker"):
        assert (
            client.post("/deployments", json={"version": 10, "image": "other"}).status_code == 422
        )
        response = client.post("/deployments", json={"version": 10})
        assert response.status_code == 202
        assert response.json()["arguments"] == {"version": 10, "image": None}
        assert response.json()["kind"] == "deploy"


def test_unknown_job_is_not_replaced_with_a_successful_record(operations_api):
    client, _ = operations_api
    assert client.get("/pipeline/runs/" + "a" * 32).status_code == 404
    assert client.get("/pipeline/runs/not-a-job-id").status_code == 422


def test_worker_exit_and_interruption_are_visible(tmp_path):
    with jobs.lock(tmp_path, "worker"):
        assert jobs.worker_available(tmp_path)
    assert not jobs.worker_available(tmp_path)
    job = jobs.submit(tmp_path, "reproduce", {}, queued=False)
    with jobs.lock(tmp_path, "execution"):
        jobs.recover_interrupted(tmp_path)
    record = jobs.get_job(tmp_path, job["id"])
    assert record["state"] == "failed"
    assert "stopped" in record["error"]


def test_pipeline_command_failure_stops_downstream_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(run_operations, "ROOT", tmp_path)
    monkeypatch.setattr(run_operations, "source_metadata", lambda path: {"git_commit": "test"})
    monkeypatch.setattr(run_operations.subprocess, "check_output", lambda *a, **kw: "digraph {}")
    (tmp_path / "params.yaml").write_text(
        yaml.safe_dump(
            {
                "data": {"prepared": "prepared"},
                "outputs": {"candidates": "candidates", "reports": "reports"},
            }
        )
    )
    commands = []

    def fail(arguments, log, env):
        commands.append(arguments)
        assert env["PIPELINE_JOB_ID"] == job["id"]
        raise subprocess.CalledProcessError(1, arguments)

    monkeypatch.setattr(release, "command", fail)
    directory = tmp_path / "operations"
    job = jobs.submit(directory, "retrain", {}, queued=False)
    with jobs.lock(directory, "execution"):
        assert run_operations.execute(directory, job) is False
    saved = jobs.get_job(directory, job["id"])
    assert saved["state"] == "failed"
    assert [(step["name"], step["state"]) for step in saved["steps"]] == [("prepare", "failed")]
    assert len(commands) == 1


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "ROOT", tmp_path)
    directory = tmp_path / "operations"
    directory.mkdir()
    env = "MODEL_URI=models:/transaction-fraud/9\nDECISION_THRESHOLD=0.5\nAPP_IMAGE=old-image\n"
    (tmp_path / ".env").write_text(env)
    operations = {
        "api_url": "http://127.0.0.1:8000",
        "ca_bundle": None,
        "compose_files": ["compose.yaml"],
    }
    (tmp_path / "params.yaml").write_text(yaml.safe_dump({"operations": operations}))
    (tmp_path / "compose.yaml").write_text("services: {}\n")
    old = {
        "model_uri": "models:/transaction-fraud/9",
        "model_version": "9",
        "decision_threshold": 0.5,
        "image": "old-image",
        "image_id": "old-id",
        "env_text": env,
        "validation": {"cases": []},
        "operations": operations,
        "deployed_at": 1,
    }
    jobs.write(directory / "deployment.json", {"current": old, "previous": None, "events": []})
    validation = {
        "model_uri": "models:/transaction-fraud/10",
        "model_version": "10",
        "decision_threshold": 0.5,
        "cases": [],
    }
    monkeypatch.setattr(release, "check_model", lambda version: validation)
    monkeypatch.setattr(
        release, "image_id", lambda image: {"old-image": "old-id", "new-image": "new-id"}[image]
    )
    return directory, tmp_path


def test_rejected_candidate_never_changes_the_running_configuration(deployment, monkeypatch):
    directory, root = deployment
    before = (root / ".env").read_text()

    def reject(version):
        raise ValueError("Invalid model signature.")

    monkeypatch.setattr(release, "check_model", reject)
    with pytest.raises(ValueError, match="signature"):
        release.deploy(directory, 10, "new-image", io.StringIO())
    assert (root / ".env").read_text() == before
    assert jobs.read(directory / "deployment.json")["events"] == []


def test_failed_restart_restores_previous_settings_and_verifies_recovery(deployment, monkeypatch):
    directory, root = deployment
    seen = []
    verified = []
    tls = {
        "api_url": "https://127.0.0.1:8000",
        "ca_bundle": "certs/cert.pem",
        "compose_files": ["compose.yaml", "compose.tls.yaml"],
    }
    (root / "params.yaml").write_text(yaml.safe_dump({"operations": tls}))
    (root / "compose.tls.yaml").write_text("services: {}\n")
    commands = []

    def restart(arguments, log, **options):
        if options.get("check") is False:
            return  # Diagnostic log collection must not block recovery.
        commands.append(arguments)
        seen.append(release.read_env()["MODEL_URI"])
        if len(seen) == 1:
            raise RuntimeError("Backend did not start.")

    monkeypatch.setattr(release, "command", restart)
    monkeypatch.setattr(
        release, "verify_running", lambda snapshot: verified.append(snapshot["model_version"])
    )
    with pytest.raises(RuntimeError, match="restored and verified"):
        release.deploy(directory, 10, "new-image", io.StringIO())
    assert seen == ["models:/transaction-fraud/10", "models:/transaction-fraud/9"]
    assert "compose.tls.yaml" in commands[0]
    assert "compose.tls.yaml" not in commands[1]
    assert verified == ["9"]
    assert "APP_IMAGE=old-image" in (root / ".env").read_text()
    state = jobs.read(directory / "deployment.json")
    assert state["current"]["model_version"] == "9"
    assert state["events"][-1]["state"] == "restored_previous"


def test_successful_release_and_manual_rollback_preserve_both_identities(deployment, monkeypatch):
    directory, _ = deployment
    verified = []
    monkeypatch.setattr(release, "command", lambda arguments, log: None)
    monkeypatch.setattr(
        release, "verify_running", lambda snapshot: verified.append(snapshot["model_version"])
    )
    release.deploy(directory, 10, "new-image", io.StringIO())
    state = jobs.read(directory / "deployment.json")
    assert state["current"]["image_id"] == "new-id"
    assert state["previous"]["image_id"] == "old-id"
    release.rollback(directory, io.StringIO())
    assert verified == ["10", "9"]
    assert release.read_env()["MODEL_URI"] == "models:/transaction-fraud/9"
    assert [event["state"] for event in jobs.read(directory / "deployment.json")["events"]] == [
        "success",
        "success",
    ]


def test_api_keeps_saved_environment_contents_private(operations_api):
    client, directory = operations_api
    snapshot = {
        "model_uri": "models:/transaction-fraud/9",
        "model_version": "9",
        "decision_threshold": 0.5,
        "image": "image",
        "image_id": "id",
        "deployed_at": 1,
        "env_text": "PRIVATE_SETTING=secret",
    }
    jobs.write(directory / "deployment.json", {"current": snapshot, "previous": None, "events": []})
    response = client.get("/deployments")
    assert response.status_code == 200
    assert "PRIVATE_SETTING" not in json.dumps(response.json())


@pytest.mark.parametrize("stale", [False, True])
def test_push_checks_never_reproduce_build_or_deploy(tmp_path, monkeypatch, stale):
    monkeypatch.setattr(run_operations, "ROOT", tmp_path)
    monkeypatch.setattr(run_operations, "source_metadata", lambda path: {"git_commit": "test"})

    def output(arguments, **kwargs):
        if "status" in arguments and "dvc" in arguments:
            return json.dumps({"train": [{"changed deps": "train.py"}]} if stale else {})
        return "digraph {}"

    monkeypatch.setattr(run_operations.subprocess, "check_output", output)
    commands = []
    monkeypatch.setattr(release, "command", lambda arguments, *args: commands.append(arguments))
    monkeypatch.setattr(run_operations, "candidate_info", lambda: {"model_version": "16"})
    validated = []
    monkeypatch.setattr(release, "check_model", lambda v: validated.append(v) or {"version": v})
    directory = tmp_path / "operations"
    job = jobs.submit(directory, "check", {}, queued=False)
    assert run_operations.execute(directory, job) is (not stale)
    assert len(commands) == 3
    assert [arguments[2] for arguments in commands] == ["ruff", "ruff", "pytest"]
    assert validated == ([] if stale else [16])
    if stale:
        assert "Run CI" in job["error"]


@pytest.mark.parametrize("score, label", [(0.1, "Not flagged"), (0.9, "Flag for review")])
def test_release_accepts_constant_predictions_without_a_quality_gate(
    tmp_path, monkeypatch, score, label
):
    from types import SimpleNamespace

    import numpy as np

    from fraud_detection.features import INPUT_COLUMNS

    monkeypatch.setattr(release, "ROOT", tmp_path)
    (tmp_path / "params.yaml").write_text(
        yaml.safe_dump(
            {
                "tracking": {"uri": "http://tracking", "model_name": "transaction-fraud"},
            }
        )
    )
    examples = tmp_path / "data/examples"
    examples.mkdir(parents=True)
    for name in ("not_flagged.json", "flag_for_review.json"):
        (examples / name).write_text(json.dumps(dict(zip(INPUT_COLUMNS, ["PAYMENT", 1, 2, 3]))))
    package = tmp_path / "model"
    package.mkdir()
    (package / "requirements.txt").write_text("")
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "source": {
                    "git_commit": "test",
                    "source_files_sha256": {"train.py": "hash"},
                    "source_worktree_dirty": False,
                }
            }
        )
    )
    client = SimpleNamespace(
        get_model_version=lambda *args: SimpleNamespace(
            tags={"selection": "validation_f1"}, run_id="r"
        ),
        get_run=lambda *args: SimpleNamespace(
            data=SimpleNamespace(tags={"mlflow.parentRunId": "p"})
        ),
        download_artifacts=lambda *args: str(comparison),
    )
    monkeypatch.setattr(release, "MlflowClient", lambda **kwargs: client)
    monkeypatch.setattr(release.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        release.mlflow.artifacts, "download_artifacts", lambda **kwargs: str(package)
    )
    model = SimpleNamespace(
        metadata=SimpleNamespace(
            signature=SimpleNamespace(
                inputs=SimpleNamespace(input_names=lambda: list(INPUT_COLUMNS))
            ),
            metadata={"decision_threshold": 0.5},
        ),
        predict=lambda frame: np.array([[1 - score, score]] * len(frame)),
    )
    monkeypatch.setattr(release.mlflow.pyfunc, "load_model", lambda path: model)
    result = release.check_model(16)
    assert [case["label"] for case in result["cases"]] == [label, label]
