"""Simulate arriving labels using PaySim source rows excluded from all original splits."""

import argparse
from pathlib import Path

import pandas as pd

from fraud_detection.common import read_config
from fraud_detection.features import INPUT_COLUMNS, TARGET, validate_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20000)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("rows must be positive.")
    config = read_config(Path("params.yaml"))
    prepared = Path(config["data"]["prepared"])
    used = set()
    for name in ("train", "validation", "test"):
        used.update(pd.read_parquet(prepared / f"{name}.parquet", columns=[]).index)
    selected = []
    for chunk in pd.read_csv(
        config["data"]["path"], usecols=[*INPUT_COLUMNS, TARGET], chunksize=250000
    ):
        selected.append(chunk.loc[~chunk.index.isin(used)])
        if sum(len(part) for part in selected) >= args.rows:
            break
    batch = pd.concat(selected).head(args.rows)
    if len(batch) != args.rows:
        raise ValueError("Not enough unused labelled source rows.")
    validate_inputs(batch.loc[:, list(INPUT_COLUMNS)])
    if not batch[TARGET].isin([0, 1]).all():
        raise ValueError("Labels must be binary.")
    destination = Path("data/incoming/labelled_batch.parquet")
    if destination.exists():
        raise FileExistsError(
            "The batch already exists; version any deliberate replacement with DVC."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    batch.to_parquet(destination, index=True)
    print(
        f"Saved {len(batch)} unused labelled rows "
        f"({int(batch[TARGET].sum())} fraud) to {destination}"
    )


if __name__ == "__main__":
    main()
