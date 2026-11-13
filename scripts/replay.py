"""Replay held-out inputs through the real API; labels are evaluated only by this client."""

import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import f1_score, precision_score, recall_score

from fastapi_app.jobs import write
from fraud_detection.common import file_hash, read_config
from fraud_detection.features import INPUT_COLUMNS, TARGET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--amount-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    if (
        args.rows < 1
        or args.concurrency < 1
        or not np.isfinite(args.amount_multiplier)
        or args.amount_multiplier <= 0
    ):
        parser.error("Rows, concurrency and amount multiplier must be positive and finite.")
    config = read_config(Path("params.yaml"))
    source = Path(config["data"]["prepared"]) / "test.parquet"
    frame = pd.read_parquet(source).sample(n=args.rows, random_state=config["seed"])
    inputs = frame.loc[:, list(INPUT_COLUMNS)].copy()
    inputs["amount"] *= args.amount_multiplier
    url = config["operations"]["api_url"].rstrip("/")
    verify = (
        True if config["operations"]["ca_bundle"] is None else config["operations"]["ca_bundle"]
    )
    metadata = requests.get(url + "/model", timeout=8, verify=verify)
    metadata.raise_for_status()
    model = metadata.json()

    def predict(record):
        started = time.perf_counter()
        response = requests.post(
            url + "/predict", json=record, timeout=10, verify=verify, headers={"X-Client": "replay"}
        )
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        result = response.json()
        if result["model_uri"] != model["model_uri"]:
            raise ValueError(
                "Deployed model changed during replay; no mixed-version report is produced."
            )
        return result["score"], elapsed

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(predict, inputs.to_dict("records")))
    elapsed = time.perf_counter() - started
    scores, latencies = np.array(results).T
    report = {
        "model_uri": model["model_uri"],
        "decision_threshold": model["decision_threshold"],
        "rows": len(frame),
        "concurrency": args.concurrency,
        "seed": config["seed"],
        "throughput_scope": (
            "Warm API; client wall time includes per-request connections; excludes data loading."
        ),
        "amount_multiplier": args.amount_multiplier,
        "source_sha256": file_hash(source),
        "source_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "script_sha256": file_hash(Path(__file__)),
        "total_seconds": elapsed,
        "requests_per_second": len(frame) / elapsed,
        "p50_seconds": float(np.quantile(latencies, 0.5)),
        "p95_seconds": float(np.quantile(latencies, 0.95)),
        "finished_at": time.time(),
        "labelled": args.amount_multiplier == 1.0,
    }
    if report["labelled"]:
        predictions = scores >= model["decision_threshold"]
        labels = frame[TARGET]
        report.update(
            precision=float(precision_score(labels, predictions, zero_division=0)),
            recall=float(recall_score(labels, predictions, zero_division=0)),
            f1=float(f1_score(labels, predictions, zero_division=0)),
            positive_labels=int(labels.sum()),
            predicted_positive=int(predictions.sum()),
        )
    name = "labelled_replay" if report["labelled"] else "shifted_replay"
    write(Path(f"reports/monitoring/{name}.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
