"""Small synthetic transactions shared by pipeline tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def transactions():
    labels = np.array([0] * 80 + [1] * 20)
    return pd.DataFrame(
        {
            "type": np.where(labels == 1, "TRANSFER", "PAYMENT"),
            "amount": np.where(labels == 1, 200.0, 20.0),
            "oldbalanceOrg": np.full(100, 100.0),
            "oldbalanceDest": np.full(100, 50.0),
            "isFraud": labels,
        }
    )
