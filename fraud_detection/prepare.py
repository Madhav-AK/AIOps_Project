"""Validate/transform CSV batches on Ray workers, then persist fixed data splits."""

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import ray

from fraud_detection.common import file_hash, read_config, write_json
from fraud_detection.data import describe, sample_and_split
from fraud_detection.features import (
    INPUT_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET,
    TRANSACTION_TYPES,
    validate_inputs,
)

LOGGER = logging.getLogger(__name__)


def prepare_batch(frame: pd.DataFrame) -> pd.DataFrame:
    """Project columns, normalize storage types, and reject invalid training rows."""
    result = frame.loc[:, [*INPUT_COLUMNS, TARGET]].copy()
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    validate_inputs(result.loc[:, list(INPUT_COLUMNS)])
    if result[TARGET].isna().any() or not result[TARGET].isin([0, 1]).all():
        raise ValueError("isFraud must contain only the binary labels 0 and 1.")
    result[TARGET] = result[TARGET].astype("int8")
    result["type"] = result["type"].astype(pd.CategoricalDtype(TRANSACTION_TYPES))
    return result


def run(config_path: Path) -> dict:
    config = read_config(config_path)
    data_config = config["data"]
    workers, batch_rows = data_config["workers"], data_config["batch_rows"]
    if workers < 1 or batch_rows < 1:
        raise ValueError("Ray workers and batch_rows must be positive.")
    started = time.perf_counter()
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(
        num_cpus=workers,
        num_gpus=0,
        include_dashboard=False,
        log_to_driver=False,
        logging_level=logging.WARNING,
    )
    startup_seconds = time.perf_counter() - started
    process_batch = ray.remote(prepare_batch)
    completed, pending = [], []
    try:
        for batch in pd.read_csv(
            data_config["path"], usecols=[*INPUT_COLUMNS, TARGET], chunksize=batch_rows
        ):
            pending.append(process_batch.remote(batch))
            # Bound in-flight data and preserve CSV row order, independent of task completion order.
            if len(pending) == workers:
                completed.extend(ray.get(pending))
                pending.clear()
        if pending:
            completed.extend(ray.get(pending))
    finally:
        ray.shutdown()
    if not completed:
        raise ValueError("The input dataset is empty.")
    frame = pd.concat(completed).sort_index()
    splits = sample_and_split(frame, data_config, config["seed"])
    output = Path(data_config["prepared"])
    output.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(output / f"{name}.parquet", index=True)
    elapsed = time.perf_counter() - started
    summary = {
        "dataset_sha256": file_hash(Path(data_config["path"])),
        "source": describe(frame),
        "splits": {name: describe(part) for name, part in splits.items()},
        "seed": config["seed"],
        "workers": workers,
        "batches": len(completed),
        "startup_seconds": startup_seconds,
        "total_seconds": elapsed,
        "rows_per_second": len(frame) / elapsed,
        "throughput_scope": "CSV -> Ray -> split -> Parquet; includes startup",
    }
    write_json(output / "preparation.json", summary)
    LOGGER.info(
        "Prepared %s rows in %.2fs (%.0f rows/s), %s Ray batches",
        len(frame),
        elapsed,
        summary["rows_per_second"],
        len(completed),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.config)


if __name__ == "__main__":
    main()
