"""Drift, metric integrity and maintenance-data leakage checks."""

import math
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from fastapi_app import monitoring
from fastapi_app.app import app
from fraud_detection.data import validate_labelled_batch
from fraud_detection.drift import amount_reference, population_stability_index


def test_psi_is_zero_for_the_reference_and_detects_a_large_shift():
    inputs = np.arange(1, 1001, dtype=float)
    reference = amount_reference(inputs)
    assert population_stability_index(inputs, reference) == pytest.approx(0)
    assert population_stability_index(inputs * 1000, reference) > 0.2
    assert math.isfinite(population_stability_index([0] * 100, reference))


@pytest.mark.parametrize("values", [[], [-1], [np.nan], [np.inf]])
def test_invalid_amounts_do_not_create_a_drift_measurement(values):
    with pytest.raises(ValueError, match="finite, nonnegative and nonempty"):
        amount_reference(values)
    with pytest.raises(ValueError, match="finite, nonnegative and nonempty"):
        population_stability_index(values, amount_reference([1, 2, 3]))


def test_malformed_reference_is_rejected():
    with pytest.raises(ValueError, match="reference bins"):
        population_stability_index([1], {"cutpoints": [2, 1], "proportions": [0.2, 0.3, 0.5]})


def test_labelled_maintenance_rejects_overlap_and_invalid_labels(transactions):
    batch = transactions.iloc[:5].copy()
    validate_labelled_batch(batch, transactions.index[10:])
    with pytest.raises(ValueError, match="outside all original splits"):
        validate_labelled_batch(batch, transactions.index)
    batch.loc[0, "isFraud"] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_labelled_batch(batch, transactions.index[10:])


@pytest.fixture
def client(monkeypatch, tmp_path, monitoring_environment):
    settings = {
        "MODEL_URI": "models:/transaction-fraud/25",
        "DECISION_THRESHOLD": "0.5",
        "MODEL_SERVER_URL": "http://model.test",
        "MLFLOW_PUBLIC_URL": "http://tracking.test",
        "APP_IMAGE": "test-image",
        "OPERATIONS_DIR": str(tmp_path / "operations"),
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    with TestClient(app) as client:
        app.state.client.close()
        app.state.client = httpx.Client(
            base_url="http://model.test",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"predictions": [[0.8, 0.2]]})
            ),
        )
        yield client


def test_window_minimum_bounding_invalid_input_and_cached_reference(client, monkeypatch):
    def unexpected_reload(*args):
        pytest.fail("Reference must stay tied to the deployed model until restart.")

    monkeypatch.setattr(monitoring, "load_reference", unexpected_reload)
    valid = {"type": "PAYMENT", "amount": 50.0, "oldbalanceOrg": 100.0, "oldbalanceDest": 0.0}
    assert client.post("/predict", json={**valid, "amount": -1}).status_code == 422
    assert client.get("/monitoring").json()["window_count"] == 0
    for amount in range(1, 100):
        assert client.post("/predict", json={**valid, "amount": float(amount)}).status_code == 200
    assert client.get("/monitoring").json()["amount_psi"] is None
    client.post("/predict", json={**valid, "amount": 100.0})
    assert client.get("/monitoring").json()["amount_psi"] == pytest.approx(0)
    for _ in range(210):
        client.post("/predict", json={**valid, "amount": 100000.0})
    result = client.get("/monitoring").json()
    assert result["window_count"] == result["window_size"] == 200
    assert result["amount_psi"] > result["drift_threshold"]
    assert result["reference_run_id"] == "test-comparison"


def test_metrics_count_real_statuses_and_bound_route_client_labels(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        monitoring.requests, "get", lambda *args, **kwargs: SimpleNamespace(ok=True)
    )
    monkeypatch.setattr(monitoring, "ROOT", tmp_path)
    client.get("/unexpected/private-identifier", headers={"X-Client": "private-identifier"})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "private-identifier" not in response.text
    assert 'route="unmatched"' in response.text
    labels = {"method": "GET", "route": "unmatched", "status": "404", "client": "other"}
    assert app.state.monitoring.registry.get_sample_value("aiops_http_requests_total", labels) == 1
    assert math.isnan(
        app.state.monitoring.registry.get_sample_value("aiops_preparation_rows_per_second")
    )
    assert app.state.monitoring.registry.get_sample_value("aiops_worker_up") == 0
