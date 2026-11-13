"""CSV validation and fixed-seed sampling/splitting for the initial baseline."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.features import INPUT_COLUMNS, NUMERIC_COLUMNS, TARGET, validate_inputs


def load_dataset(path: Path) -> pd.DataFrame:
    """Load only permitted inputs and target; account IDs/outcome balances stay out."""
    frame = pd.read_csv(
        path,
        usecols=[*INPUT_COLUMNS, TARGET],
        dtype={"type": "category", **dict.fromkeys(NUMERIC_COLUMNS, "float64")},
    )
    validate_inputs(frame.loc[:, list(INPUT_COLUMNS)])
    if frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("isFraud must contain only the binary labels 0 and 1.")
    frame[TARGET] = frame[TARGET].astype("int8")
    return frame


def describe(frame: pd.DataFrame) -> dict:
    """A compact EDA/validation summary, including the natural class imbalance."""
    return {
        "rows": len(frame),
        "class_counts": {str(int(k)): int(v) for k, v in frame[TARGET].value_counts().items()},
        "fraud_fraction": float(frame[TARGET].mean()),
        "transaction_types": {str(k): int(v) for k, v in frame["type"].value_counts().items()},
        "missing_values": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "numeric_ranges": {
            column: {"minimum": float(frame[column].min()), "maximum": float(frame[column].max())}
            for column in NUMERIC_COLUMNS
        },
    }


def sample_and_split(frame: pd.DataFrame, config: dict, seed: int) -> dict[str, pd.DataFrame]:
    """Uniform sample, then 70/15/15 stratification; original row indices survive."""
    size = config["sample_size"]
    validation_fraction = config["validation_fraction"]
    test_fraction = config["test_fraction"]
    if size <= 0 or not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError(
            "Sample size must be positive and split fractions must be between 0 and 1."
        )
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must leave a nonempty training split.")
    if not frame.index.is_unique:
        raise ValueError("Source row indices must be unique for auditable split membership.")
    sample = frame.sample(n=min(size, len(frame)), random_state=seed)
    train, remaining = train_test_split(
        sample,
        test_size=validation_fraction + test_fraction,
        stratify=sample[TARGET],
        random_state=seed,
    )
    validation, test = train_test_split(
        remaining,
        test_size=test_fraction / (validation_fraction + test_fraction),
        stratify=remaining[TARGET],
        random_state=seed,
    )
    splits = {"train": train, "validation": validation, "test": test}
    minimum = config["minimum_fraud_per_split"]
    for name, part in splits.items():
        if set(part[TARGET].unique()) != {0, 1} or part[TARGET].sum() < minimum:
            raise ValueError(f"{name} needs both classes and at least {minimum} fraudulent rows.")
    return splits


def training_positive_weight(labels: pd.Series) -> float:
    """Compute imbalance weight using training labels alone."""
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        raise ValueError("Training requires both classes.")
    return negative / positive


def validate_labelled_batch(batch: pd.DataFrame, used_indices: pd.Index) -> None:
    """Keep maintenance labels out of every existing train/validation/test row."""
    validate_inputs(batch.loc[:, list(INPUT_COLUMNS)])
    if not batch[TARGET].isin([0, 1]).all():
        raise ValueError("Maintenance labels must be binary and present.")
    if not batch.index.is_unique or not batch.index.intersection(used_indices).empty:
        raise ValueError("Maintenance source rows must be unique and outside all original splits.")
