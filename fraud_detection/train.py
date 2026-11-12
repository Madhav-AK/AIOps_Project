"""Train the simple comparison on prepared data through an MLflow Project."""

import argparse
import json
import logging
import os
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier

from fraud_detection.common import (
    file_hash,
    read_config,
    snapshot_source,
    source_metadata,
    write_json,
)
from fraud_detection.data import training_positive_weight
from fraud_detection.evaluate import score_model
from fraud_detection.features import INPUT_COLUMNS, TARGET, build_model

LOGGER = logging.getLogger(__name__)


def run(config_path: Path) -> dict:
    if not all(os.environ.get(key) for key in ("MLFLOW_RUN_ID", "MLFLOW_TRACKING_URI")):
        raise ValueError(
            "Training requires an MLflow Project run; start the server and use dvc repro."
        )
    config = read_config(config_path)
    threshold = config["decision_threshold"]
    if not 0 < threshold < 1:
        raise ValueError("Decision threshold must be between 0 and 1.")
    prepared = Path(config["data"]["prepared"])
    train = pd.read_parquet(prepared / "train.parquet")
    validation = pd.read_parquet(prepared / "validation.parquet")
    preparation = json.loads((prepared / "preparation.json").read_text())
    output = Path(config["outputs"]["candidates"])
    output.mkdir(parents=True, exist_ok=True)
    inputs, labels = train.loc[:, list(INPUT_COLUMNS)], train[TARGET]
    weight = training_positive_weight(labels)
    metadata = source_metadata(config_path)
    metadata["dataset_sha256"] = preparation["dataset_sha256"]
    if "TRAINING_IMAGE_ID" in os.environ:
        metadata["training_image_id"] = os.environ["TRAINING_IMAGE_ID"]
    uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(uri)
    example = inputs.head(3).astype({"type": "str"})
    candidates = {}
    # Resume the required Project run. MLflow consumes MLFLOW_RUN_ID before child runs start.
    with mlflow.start_run(run_name="fraud-comparison") as parent:
        mlflow.set_tags(
            {
                "source_git_commit": metadata["git_commit"],
                "source_worktree_dirty": str(metadata["source_worktree_dirty"]),
                "dataset_sha256": preparation["dataset_sha256"],
            }
        )
        mlflow.log_params(
            {
                "seed": config["seed"],
                "decision_threshold": threshold,
                "training_rows": len(train),
                "validation_rows": len(validation),
                "sample_size": config["data"]["sample_size"],
                **{f"model.{k}": v for k, v in config["model"].items()},
            }
        )
        mlflow.log_dict(config, "config.json")
        with tempfile.TemporaryDirectory(prefix="fraud-source-") as temporary:
            snapshot = Path(temporary) / "source_snapshot.zip"
            snapshot_source(snapshot)
            metadata["source_snapshot_sha256"] = file_hash(snapshot)
            mlflow.log_artifact(str(snapshot))
        mlflow.log_dict(metadata, "lineage.json")
        mlflow.log_dict(preparation, "data_preparation.json")
        if "PIPELINE_JOB_ID" in os.environ:
            mlflow.set_tag("pipeline_job_id", os.environ["PIPELINE_JOB_ID"])
        for name, positive_weight in [
            ("always_non_fraud", None),
            ("xgboost_unweighted", 1.0),
            ("xgboost_weighted", weight),
        ]:
            with mlflow.start_run(run_name=name, nested=True) as child:
                model = (
                    DummyClassifier(strategy="constant", constant=0)
                    if positive_weight is None
                    else build_model(config["model"], config["seed"], positive_weight)
                )
                LOGGER.info("Training %s", name)
                started = time.perf_counter()
                model.fit(inputs, labels)
                duration = time.perf_counter() - started
                metrics = score_model(model, validation, threshold)
                path = output / f"{name}.joblib"
                joblib.dump(model, path)
                mlflow.log_params(
                    {"candidate": name, "seed": config["seed"], "decision_threshold": threshold}
                )
                if positive_weight is not None:
                    mlflow.log_params({**config["model"], "scale_pos_weight": positive_weight})
                mlflow.log_metrics(
                    {
                        f"validation_{k}": v
                        for k, v in metrics.items()
                        if isinstance(v, (int, float))
                    }
                )
                mlflow.log_metric("training_seconds", duration)
                mlflow.set_tags(
                    {
                        "source_git_commit": metadata["git_commit"],
                        "dataset_sha256": preparation["dataset_sha256"],
                    }
                )
                mlflow.log_dict(metrics, "validation.json")
                signature = infer_signature(example, model.predict_proba(example))
                logged = mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    pyfunc_predict_fn="predict_proba",
                    metadata={"decision_threshold": threshold},
                    code_paths=["fraud_detection"],
                    signature=signature,
                    input_example=example,
                    pip_requirements=[
                        f"{name}=={version(name)}"
                        for name in (
                            "mlflow",
                            "scikit-learn",
                            "xgboost-cpu",
                            "numpy",
                            "pandas",
                            "scipy",
                            "joblib",
                            "cloudpickle",
                            # MLflow 3.6 serving requires the compatible HTTP stack.
                            "fastapi",
                            "starlette",
                        )
                    ],
                )
                candidates[name] = {
                    "run_id": child.info.run_id,
                    "model_uri": logged.model_uri,
                    "validation": metrics,
                    "training_seconds": duration,
                    "scale_pos_weight": positive_weight,
                }
        manifest = {
            "parent_run_id": parent.info.run_id,
            "tracking_uri": uri,
            "candidates": candidates,
            "source": metadata,
        }
        if "PIPELINE_JOB_ID" in os.environ:
            manifest["pipeline_job_id"] = os.environ["PIPELINE_JOB_ID"]
        mlflow.log_dict(manifest, "comparison.json")
    write_json(output / "comparison.json", manifest)
    LOGGER.info("MLflow comparison run: %s", manifest["parent_run_id"])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.config)


if __name__ == "__main__":
    main()
