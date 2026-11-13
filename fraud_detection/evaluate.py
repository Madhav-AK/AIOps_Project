"""Select on validation, evaluate held-out data, and register the packaged candidate."""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from fraud_detection.common import read_config, write_json
from fraud_detection.features import INPUT_COLUMNS, TARGET

LOGGER = logging.getLogger(__name__)


def score_model(model, frame: pd.DataFrame, threshold: float) -> dict:
    """Use continuous scores for average precision and fixed-threshold labels."""
    labels = frame[TARGET]
    scores = model.predict_proba(frame.loc[:, list(INPUT_COLUMNS)])[:, 1]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Model scores must be finite and in [0, 1].")
    predictions = (scores >= threshold).astype(int)
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "flagged_count": int(predictions.sum()),
        "rows": len(frame),
    }


def select_candidate(candidates: dict) -> str:
    """Exclude the dummy; favour unweighted on ties; never inspect test results."""
    return max(
        ["xgboost_unweighted", "xgboost_weighted"],
        key=lambda name: candidates[name]["validation"]["f1"],
    )


def run(config_path: Path) -> dict:
    config = read_config(config_path)
    candidates_path = Path(config["outputs"]["candidates"])
    comparison = json.loads((candidates_path / "comparison.json").read_text())
    selected_name = select_candidate(comparison["candidates"])
    candidate = comparison["candidates"][selected_name]
    model = joblib.load(candidates_path / f"{selected_name}.joblib")
    prepared = Path(config["data"]["prepared"])
    test = pd.read_parquet(prepared / "test.parquet")
    metrics = score_model(model, test, config["decision_threshold"])
    feature_importance = sorted(
        [
            {"feature": str(k), "importance": float(v)}
            for k, v in zip(
                model.named_steps["encode"].get_feature_names_out(),
                model.named_steps["classifier"].feature_importances_,
            )
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )
    mlflow.set_tracking_uri(comparison["tracking_uri"])
    with mlflow.start_run(run_id=comparison["parent_run_id"]):
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
        )
        mlflow.set_tags(
            {
                "selected_model": selected_name,
                "selection_rule": "highest validation F1 at fixed threshold",
            }
        )
        mlflow.log_dict(metrics, "test.json")
        mlflow.log_dict(feature_importance, "feature_importance.json")
        registered = mlflow.register_model(candidate["model_uri"], config["tracking"]["model_name"])
        client = MlflowClient()
        client.set_registered_model_alias(registered.name, "candidate", registered.version)
        client.set_model_version_tag(
            registered.name,
            registered.version,
            "source_git_commit",
            comparison["source"]["git_commit"],
        )
        client.set_model_version_tag(
            registered.name, registered.version, "selection", "validation_f1"
        )
        info = {
            "selected_model": selected_name,
            "decision_threshold": config["decision_threshold"],
            "input_columns": list(INPUT_COLUMNS),
            "run_id": candidate["run_id"],
            "comparison_run_id": comparison["parent_run_id"],
            "registered_model": registered.name,
            "model_version": registered.version,
            "model_uri": f"models:/{registered.name}/{registered.version}",
        }
        mlflow.log_dict(info, "selected_model.json")
    output = Path(config["outputs"]["artifacts"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates_path / f"{selected_name}.joblib", output / "model.joblib")
    write_json(output / "model_info.json", info)
    # Keep a small Git/DVC metrics snapshot. Full per-run evidence lives in MLflow.
    report = {
        "training_rows": comparison["source"]["training_rows"],
        "labelled_batch": comparison["source"]["labelled_batch"],
        "validation": {
            name: values["validation"] for name, values in comparison["candidates"].items()
        },
        "test": metrics,
        "selected_model": selected_name,
        "decision_threshold": config["decision_threshold"],
        "comparison_run_id": comparison["parent_run_id"],
        "registered_model_version": int(registered.version),
    }
    if "PIPELINE_JOB_ID" in os.environ:
        report["pipeline_job_id"] = os.environ["PIPELINE_JOB_ID"]
    write_json(Path(config["outputs"]["reports"]) / "results.json", report)
    LOGGER.info(
        "Selected %s; registry version %s; test metrics %s",
        selected_name,
        registered.version,
        metrics,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.config)


if __name__ == "__main__":
    main()
