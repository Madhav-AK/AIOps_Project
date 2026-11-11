"""Run the basic XGBoost comparison; operational tracking is added in commit 2."""

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from fraud_detection.data import describe, load_dataset, sample_and_split, training_positive_weight
from fraud_detection.features import INPUT_COLUMNS, TARGET, build_model

LOGGER = logging.getLogger(__name__)


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def evaluate(model, frame: pd.DataFrame, threshold: float) -> dict:
    """Use continuous scores for average precision and a fixed threshold for labels."""
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


def write_json(path: Path, content: dict | list) -> None:
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def run(config_path: Path) -> dict:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text())
    threshold = config["decision_threshold"]
    if not 0 < threshold < 1:
        raise ValueError("Decision threshold must be between 0 and 1.")
    reports = Path(config["outputs"]["reports"])
    artifacts = Path(config["outputs"]["artifacts"])
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    source = Path(config["data"]["path"])
    LOGGER.info("Loading and validating %s", source)
    data_started = time.perf_counter()
    raw = load_dataset(source)
    summary = {"source": describe(raw)}
    splits = sample_and_split(raw, config["data"], config["seed"])
    summary["splits"] = {name: describe(frame) for name, frame in splits.items()}
    summary["sample_rows"] = sum(len(frame) for frame in splits.values())
    summary["load_validate_split_seconds"] = time.perf_counter() - data_started
    LOGGER.info("Read %s rows; sampled %s", len(raw), summary["sample_rows"])
    del raw
    # Index membership makes split disjointness and future reproduction inspectable.
    np.savez(artifacts / "split_indices.npz", **{k: v.index.to_numpy() for k, v in splits.items()})
    train = splits["train"]
    inputs, labels = train.loc[:, list(INPUT_COLUMNS)], train[TARGET]
    weight = training_positive_weight(labels)

    dummy = DummyClassifier(strategy="constant", constant=0).fit(inputs, labels)
    validation = {"always_non_fraud": evaluate(dummy, splits["validation"], threshold)}
    candidates = {}
    training_times = {}
    for name, positive_weight in [("xgboost_unweighted", 1.0), ("xgboost_weighted", weight)]:
        model = build_model(config["model"], config["seed"], positive_weight)
        LOGGER.info("Training %s (scale_pos_weight=%.3f)", name, positive_weight)
        fit_started = time.perf_counter()
        model.fit(inputs, labels)
        training_times[name] = time.perf_counter() - fit_started
        validation[name] = evaluate(model, splits["validation"], threshold)
        candidates[name] = model
        LOGGER.info("%s validation: %s", name, validation[name])

    # The fixed selection rule sees validation only. The test split is evaluated afterwards.
    selected_name = max(candidates, key=lambda name: validation[name]["f1"])
    selected = candidates[selected_name]
    test_metrics = evaluate(selected, splits["test"], threshold)
    joblib.dump(selected, artifacts / "model.joblib")
    write_json(
        artifacts / "model_info.json",
        {
            "selected_model": selected_name,
            "decision_threshold": threshold,
            "input_columns": list(INPUT_COLUMNS),
        },
    )
    names = selected.named_steps["encode"].get_feature_names_out()
    importances = selected.named_steps["classifier"].feature_importances_
    feature_importances = sorted(
        [{"feature": str(k), "importance": float(v)} for k, v in zip(names, importances)],
        key=lambda item: item["importance"],
        reverse=True,
    )
    metrics = {
        "validation": validation,
        "selection_rule": "Highest validation F1 at fixed threshold; ties favour unweighted.",
        "selected_model": selected_name,
        "decision_threshold": threshold,
        "scale_pos_weight": {"xgboost_unweighted": 1.0, "xgboost_weighted": weight},
        "training_seconds": training_times,
        "test": test_metrics,
    }
    source_files = [
        *Path("fraud_detection").glob("*.py"),
        config_path,
        Path("requirements.txt"),
    ]
    metadata = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_run": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_worktree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "source_files_sha256": {p.as_posix(): file_hash(p) for p in source_files if p.exists()},
        "dataset_sha256": file_hash(source),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ["numpy", "pandas", "scikit-learn", "xgboost-cpu", "joblib"]
        },
        "total_seconds": time.perf_counter() - started,
    }
    write_json(
        reports / "results.json",
        {
            "data_summary": summary,
            "metrics": metrics,
            "feature_importance": feature_importances,
            "config": config,
            "run_metadata": metadata,
        },
    )
    LOGGER.info("Selected %s; held-out test: %s", selected_name, test_metrics)
    LOGGER.info("Reports: %s; fitted pipeline: %s", reports, artifacts / "model.joblib")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(args.config)
    except (ValueError, OSError, KeyError) as exc:
        LOGGER.error("Training failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
