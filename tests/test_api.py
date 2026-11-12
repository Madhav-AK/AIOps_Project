"""Application contract and upstream failure checks, without loading a model in the API."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from fastapi_app.app import app

VALID = {"type": "TRANSFER", "amount": 100.0, "oldbalanceOrg": 100.0, "oldbalanceDest": 0.0}


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("MODEL_URI", "models:/transaction-fraud/8")
    monkeypatch.setenv("DECISION_THRESHOLD", "0.5")
    monkeypatch.setenv("MODEL_SERVER_URL", "http://model.test:8080")
    monkeypatch.setenv("MLFLOW_PUBLIC_URL", "http://127.0.0.1:5000")
    with TestClient(app) as client:
        yield client


def upstream(handler):
    app.state.client.close()
    app.state.client = httpx.Client(
        base_url="http://model.test:8080", transport=httpx.MockTransport(handler)
    )


def test_prediction_forwards_named_float_fields_and_returns_pinned_identity(api_client):
    def handler(request):
        assert request.url.path == "/invocations"
        assert json.loads(request.content) == {"dataframe_records": [VALID]}
        return httpx.Response(200, json={"predictions": [[0.1, 0.9]]})

    upstream(handler)
    response = api_client.post("/predict", json=VALID)
    assert response.status_code == 200
    result = response.json()
    assert result["label"] == "Flag for review"
    assert result["score"] == 0.9
    assert result["model_uri"] == "models:/transaction-fraud/8"
    assert response.headers["X-Request-ID"] == result["request_id"]


@pytest.mark.parametrize(
    "patch",
    [
        {"amount": -1},
        {"amount": True},
        {"amount": "100"},
        {"type": "UNKNOWN"},
        {"isFraud": 1},
        {"amount": None},
    ],
)
def test_invalid_transaction_never_reaches_model(api_client, patch):
    def handler(request):
        pytest.fail("Invalid application inputs reached the model server.")

    upstream(handler)
    response = api_client.post("/predict", json={**VALID, **patch})
    assert response.status_code == 422


def test_missing_field_is_rejected(api_client):
    response = api_client.post("/predict", json={"type": "PAYMENT"})
    assert response.status_code == 422


@pytest.mark.parametrize("value", ["NaN", "Infinity"])
def test_nonfinite_json_is_rejected_with_serializable_error(api_client, value):
    body = json.dumps(VALID).replace("100.0", value, 1)
    response = api_client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"]


@pytest.mark.parametrize(
    "reply",
    [
        {},
        {"predictions": []},
        {"predictions": [[0.2, 1.2]]},
        {"predictions": [[0.2, 0.3]]},
        {"predictions": [[0.2, float("nan")]]},
        {"predictions": [[True, 0.0]]},
    ],
)
def test_invalid_model_output_cannot_become_a_prediction(api_client, reply):
    upstream(
        lambda request: httpx.Response(
            200, text=json.dumps(reply), headers={"Content-Type": "application/json"}
        )
    )
    response = api_client.post("/predict", json=VALID)
    assert response.status_code == 502
    assert "score" not in response.json()


@pytest.mark.parametrize("error,expected", [(httpx.ConnectError, 503), (httpx.ReadTimeout, 504)])
def test_upstream_failure_is_explicit(api_client, error, expected):
    def handler(request):
        raise error("Service failure", request=request)

    upstream(handler)
    assert api_client.post("/predict", json=VALID).status_code == expected


def test_readiness_and_liveness_are_separate(api_client):
    upstream(lambda request: httpx.Response(503))
    assert api_client.get("/ready").status_code == 503
    assert api_client.get("/health").status_code == 200


def test_examples_obey_the_api_contract(api_client):
    response = api_client.get("/examples")
    assert response.status_code == 200
    assert set(response.json()) == {"payment", "transfer"}
    assert all(set(record) == set(VALID) for record in response.json().values())


def test_unavailable_pipeline_evidence_is_not_replaced_with_fake_values(
    api_client, monkeypatch, tmp_path
):
    monkeypatch.setattr("fastapi_app.app.ROOT", tmp_path)
    response = api_client.get("/pipeline")
    assert response.status_code == 503


def test_model_version_alias_is_rejected_at_startup(monkeypatch):
    monkeypatch.setenv("MODEL_URI", "models:/transaction-fraud@candidate")
    with pytest.raises(ValueError, match="must pin"):
        with TestClient(app):
            pass
