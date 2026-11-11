"""Data leakage, input-contract, metric, and serialization checks for commit 1."""

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import load_dataset, sample_and_split, training_positive_weight
from fraud_detection.evaluate import score_model as evaluate
from fraud_detection.features import (
    INPUT_COLUMNS,
    TARGET,
    AddBalanceGap,
    build_model,
    validate_inputs,
)


def split_config():
    return {
        "sample_size": 80,
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
        "minimum_fraud_per_split": 2,
    }


def test_sampling_is_reproducible_disjoint_and_stratified(transactions):
    first = sample_and_split(transactions, split_config(), 42)
    second = sample_and_split(transactions, split_config(), 42)
    membership = [set(part.index) for part in first.values()]
    assert sum(map(len, membership)) == len(set.union(*membership)) == 80
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
        assert set(first[name][TARGET]) == {0, 1}
        assert first[name][TARGET].mean() == pytest.approx(0.2, abs=0.05)


def test_sampling_does_not_take_only_first_rows(transactions):
    splits = sample_and_split(transactions, split_config(), 42)
    selected = set.union(*(set(part.index) for part in splits.values()))
    assert selected != set(range(80))


def test_insufficient_fraud_fails_before_training(transactions):
    config = {**split_config(), "minimum_fraud_per_split": 100}
    with pytest.raises(ValueError, match="fraudulent rows"):
        sample_and_split(transactions, config, 42)


def test_invalid_split_fractions_are_rejected(transactions):
    config = {**split_config(), "test_fraction": 0.9}
    with pytest.raises(ValueError, match="nonempty training"):
        sample_and_split(transactions, config, 42)


def test_weight_uses_the_supplied_training_labels():
    assert training_positive_weight(pd.Series([0, 0, 0, 1])) == 3.0
    with pytest.raises(ValueError, match="both classes"):
        training_positive_weight(pd.Series([0, 0]))


@pytest.mark.parametrize("invalid", [-1.0, np.inf, np.nan])
def test_invalid_numeric_inputs_rejected(transactions, invalid):
    frame = transactions.loc[:, list(INPUT_COLUMNS)].copy()
    frame.loc[0, "amount"] = invalid
    with pytest.raises(ValueError, match="finite, nonnegative"):
        validate_inputs(frame)


def test_unknown_transaction_type_rejected(transactions):
    frame = transactions.loc[:, list(INPUT_COLUMNS)].copy()
    frame.loc[0, "type"] = "UNKNOWN"
    with pytest.raises(ValueError, match="transaction type"):
        validate_inputs(frame)


@pytest.mark.parametrize("value", ["not-a-number", True])
def test_strings_and_booleans_are_not_numeric_inputs(transactions, value):
    frame = transactions.loc[:, list(INPUT_COLUMNS)].copy()
    frame["amount"] = value
    with pytest.raises(ValueError, match="must be numeric"):
        validate_inputs(frame)


def test_missing_column_rejected(transactions):
    frame = transactions.loc[:, list(INPUT_COLUMNS)].drop(columns="amount")
    with pytest.raises(ValueError, match="missing=.*amount"):
        validate_inputs(frame)


def test_target_cannot_be_passed_to_inference(transactions):
    with pytest.raises(ValueError, match="extra=.*isFraud"):
        validate_inputs(transactions)


def test_derived_feature_does_not_mutate_input(transactions):
    inputs = transactions.loc[:, list(INPUT_COLUMNS)]
    original = inputs.copy()
    transformed = AddBalanceGap().fit_transform(inputs)
    assert transformed.iloc[0]["amount_minus_sender_balance"] == -80.0
    pd.testing.assert_frame_equal(inputs, original)


def test_csv_rejects_nonbinary_target(transactions, tmp_path):
    transactions.loc[0, TARGET] = 2
    path = tmp_path / "invalid.csv"
    transactions.to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary labels"):
        load_dataset(path)


def test_csv_excludes_identifiers_and_post_transaction_columns(transactions, tmp_path):
    frame = transactions.assign(nameOrig="C123", newbalanceOrig=999.0, isFlaggedFraud=1)
    path = tmp_path / "transactions.csv"
    frame.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert set(loaded.columns) == {*INPUT_COLUMNS, TARGET}


def test_model_serialization_keeps_preprocessing_and_predictions(transactions, tmp_path):
    inputs = transactions.loc[:, list(INPUT_COLUMNS)]
    model = build_model({"n_estimators": 5, "max_depth": 2, "n_jobs": 1}, 42)
    model.fit(inputs, transactions[TARGET])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    # Column ordering in a JSON request must not change the named-column transformation.
    reordered = inputs.loc[:, list(reversed(INPUT_COLUMNS))]
    np.testing.assert_allclose(model.predict_proba(inputs), restored.predict_proba(reordered))


def test_average_precision_uses_scores_not_thresholded_labels(transactions):
    class ScoredModel:
        def predict_proba(self, inputs):
            scores = np.array([0.1, 0.2, 0.8, 0.7])
            return np.column_stack([1 - scores, scores])

    frame = transactions.iloc[:4].copy()
    frame[TARGET] = [0, 1, 1, 0]
    result = evaluate(ScoredModel(), frame, 0.5)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["f1"] == 0.5
    assert result["average_precision"] == pytest.approx(5 / 6)
