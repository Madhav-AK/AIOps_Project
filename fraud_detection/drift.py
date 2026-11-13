"""A training-only amount reference and a small population stability index (PSI)."""

import numpy as np


def amount_reference(amounts) -> dict:
    values = np.asarray(amounts, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Reference amounts must be finite, nonnegative and nonempty.")
    cuts = np.unique(np.quantile(values, [0.2, 0.4, 0.6, 0.8]))
    counts, _ = np.histogram(values, bins=[-np.inf, *cuts, np.inf])
    return {
        "feature": "amount",
        "cutpoints": cuts.tolist(),
        "proportions": (counts / len(values)).tolist(),
        "training_rows": len(values),
    }


def population_stability_index(amounts, reference: dict) -> float:
    values = np.asarray(amounts, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Observed amounts must be finite, nonnegative and nonempty.")
    expected = np.asarray(reference["proportions"], dtype=float)
    cuts = np.asarray(reference["cutpoints"], dtype=float)
    if (
        len(expected) != len(cuts) + 1
        or not np.isfinite(cuts).all()
        or (np.diff(cuts) <= 0).any()
        or (expected < 0).any()
        or not np.isfinite(expected).all()
        or not np.isclose(expected.sum(), 1)
    ):
        raise ValueError("Invalid amount reference bins/proportions.")
    observed, _ = np.histogram(values, bins=[-np.inf, *cuts, np.inf])
    # A documented floor keeps empty bins finite; normalize again after flooring.
    expected = np.maximum(expected, 1e-6)
    actual = np.maximum(observed / len(values), 1e-6)
    expected /= expected.sum()
    actual /= actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))
