"""Checks for data identity, MLflow packaging, and validation-only selection."""

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pytest
import yaml

from fraud_detection.common import file_hash, source_metadata, write_json
from fraud_detection.data import sample_and_split
from fraud_detection.evaluate import run as evaluate_run
from fraud_detection.evaluate import select_candidate
from fraud_detection.features import INPUT_COLUMNS, TARGET
from fraud_detection.prepare import prepare_batch
from fraud_detection.train import run as train_run


def test_ray_batch_logic_preserves_source_identity(transactions):
    frame = transactions.iloc[20:30].assign(unused="discard")
    transformed = prepare_batch(frame)
    assert transformed.index.tolist() == frame.index.tolist()
    assert set(transformed.columns) == {*INPUT_COLUMNS, TARGET}
    assert str(transformed[TARGET].dtype) == "int8"
    assert str(transformed["type"].dtype) == "category"


def test_ray_batch_logic_rejects_invalid_labels(transactions):
    transactions.loc[0, TARGET] = 7
    with pytest.raises(ValueError, match="binary labels"):
        prepare_batch(transactions)


def test_selection_uses_validation_and_has_deterministic_ties():
    candidates = {
        "always_non_fraud": {"validation": {"f1": 1.0}},
        "xgboost_unweighted": {"validation": {"f1": 0.6}, "test": {"f1": 1.0}},
        "xgboost_weighted": {"validation": {"f1": 0.7}, "test": {"f1": 0.0}},
    }
    assert select_candidate(candidates) == "xgboost_weighted"
    candidates["xgboost_unweighted"]["validation"]["f1"] = 0.7
    assert select_candidate(candidates) == "xgboost_unweighted"


def test_container_lineage_can_be_supplied_without_git(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOURCE_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("SOURCE_WORKTREE_DIRTY", "false")
    result = source_metadata(Path("params.yaml"))
    assert result["git_commit"] == "a" * 40
    assert result["source_worktree_dirty"] is False


def test_tracking_registration_and_fresh_process_model_load(transactions, tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_ENABLE_TELEMETRY", "false")
    monkeypatch.setenv("SOURCE_GIT_COMMIT", "unit-test-source")
    monkeypatch.setenv("SOURCE_WORKTREE_DIRTY", "true")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    raw = tmp_path / "raw.csv"
    transactions.to_csv(raw, index=False)
    data_config = {
        "sample_size": 100,
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
        "minimum_fraud_per_split": 2,
        "prepared": str(prepared),
        "path": str(raw),
    }
    splits = sample_and_split(transactions, data_config, 42)
    for name, frame in splits.items():
        frame.to_parquet(prepared / f"{name}.parquet")
    write_json(prepared / "preparation.json", {"dataset_sha256": file_hash(raw)})
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    config = {
        "seed": 42,
        "data": data_config,
        "model": {"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
        "decision_threshold": 0.5,
        "tracking": {"uri": uri, "experiment": "unit-test", "model_name": "test-fraud"},
        "outputs": {
            "candidates": str(tmp_path / "candidates"),
            "artifacts": str(tmp_path / "model"),
            "reports": str(tmp_path / "reports"),
        },
    }
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.safe_dump(config))
    mlflow.set_tracking_uri(uri)
    experiment_id = mlflow.create_experiment(
        "unit-test", artifact_location=(tmp_path / "mlartifacts").as_uri()
    )
    project_run = mlflow.MlflowClient().create_run(experiment_id)
    monkeypatch.setenv("MLFLOW_RUN_ID", project_run.info.run_id)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", experiment_id)
    comparison = train_run(config_path)
    assert len(comparison["candidates"]) == 3
    report = evaluate_run(config_path)
    assert report["test"]["rows"] == len(splits["test"])
    info = json.loads((tmp_path / "model/model_info.json").read_text())
    local = joblib.load(tmp_path / "model/model.joblib")
    packaged = mlflow.pyfunc.load_model(info["model_uri"])
    inputs = transactions.loc[:, list(INPUT_COLUMNS)]
    np.testing.assert_allclose(local.predict_proba(inputs), packaged.predict(inputs))
    signature = mlflow.models.get_model_info(info["model_uri"]).signature
    assert set(signature.inputs.input_names()) == set(INPUT_COLUMNS)
    assert packaged.metadata.metadata["decision_threshold"] == 0.5
    # New interpreter, outside the source tree: only the packaged code can satisfy imports.
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["MLFLOW_TRACKING_URI"] = uri
    code = (
        "import json, mlflow, pandas as pd; "
        f"model=mlflow.pyfunc.load_model({info['model_uri']!r}); "
        f"result=model.predict(pd.DataFrame({inputs.head(1).to_dict('records')!r})); "
        "print(json.dumps(result.tolist()))"
    )
    child = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    scores = json.loads(child.stdout)
    assert len(scores[0]) == 2
    assert sum(scores[0]) == pytest.approx(1.0)


@pytest.mark.parametrize("missing", ["MLFLOW_RUN_ID", "MLFLOW_TRACKING_URI"])
def test_training_rejects_missing_project_context(tmp_path, monkeypatch, missing):
    monkeypatch.setenv("MLFLOW_RUN_ID", "explicit-test-run")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "explicit-test-uri")
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="requires an MLflow Project run"):
        train_run(tmp_path / "params.yaml")


@pytest.mark.parametrize("missing", ["SOURCE_GIT_COMMIT", "SOURCE_WORKTREE_DIRTY"])
def test_lineage_rejects_partial_container_configuration(tmp_path, monkeypatch, missing):
    monkeypatch.setenv("SOURCE_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("SOURCE_WORKTREE_DIRTY", "false")
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="Supply both"):
        source_metadata(tmp_path / "params.yaml")


def test_lineage_rejects_unknown_source_state(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("SOURCE_WORKTREE_DIRTY", "unknown")
    with pytest.raises(ValueError, match="must be true or false"):
        source_metadata(tmp_path / "params.yaml")
