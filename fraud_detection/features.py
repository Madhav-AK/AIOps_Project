"""The inference input contract and shared train/serve transformations."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

TRANSACTION_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
NUMERIC_COLUMNS = ("amount", "oldbalanceOrg", "oldbalanceDest")
INPUT_COLUMNS = ("type", *NUMERIC_COLUMNS)
DERIVED_COLUMN = "amount_minus_sender_balance"
TARGET = "isFraud"


def validate_inputs(frame: pd.DataFrame) -> None:
    """Reject invalid inputs identically during fitting and later inference."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Expected a nonempty transaction DataFrame.")
    missing = set(INPUT_COLUMNS) - set(frame.columns)
    extra = set(frame.columns) - set(INPUT_COLUMNS)
    if missing or extra:
        raise ValueError(f"Invalid input columns: missing={sorted(missing)}, extra={sorted(extra)}")
    if not frame["type"].isin(TRANSACTION_TYPES).all():
        raise ValueError("Unknown or missing transaction type.")
    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(
            frame[column]
        ):
            raise ValueError(f"{column} must be numeric.")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{column} must contain finite, nonnegative values.")


class AddBalanceGap(TransformerMixin, BaseEstimator):
    """Add one interpretable feature without learning from labels or other rows."""

    def fit(self, X: pd.DataFrame, y=None):
        validate_inputs(X)
        self.n_features_in_ = len(INPUT_COLUMNS)
        self.feature_names_in_ = np.array(INPUT_COLUMNS, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        validate_inputs(X)
        result = X.loc[:, list(INPUT_COLUMNS)].copy()
        result[DERIVED_COLUMN] = result["amount"] - result["oldbalanceOrg"]
        return result

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array([*INPUT_COLUMNS, DERIVED_COLUMN], dtype=object)


def build_model(model_params: dict, seed: int, positive_weight: float = 1.0) -> Pipeline:
    """Package preprocessing and CPU XGBoost as one serializable estimator."""
    encoder = ColumnTransformer(
        [
            (
                "type",
                OneHotEncoder(categories=[list(TRANSACTION_TYPES)], sparse_output=False),
                ["type"],
            ),
            ("numeric", "passthrough", [*NUMERIC_COLUMNS, DERIVED_COLUMN]),
        ]
    )
    classifier = XGBClassifier(
        **model_params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        random_state=seed,
        scale_pos_weight=positive_weight,
    )
    return Pipeline(
        [("features", AddBalanceGap()), ("encode", encoder), ("classifier", classifier)]
    )
