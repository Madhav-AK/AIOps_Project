"""Exercise the actual Streamlit form using mocked REST responses."""

import json as jsonlib
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests
from streamlit.testing.v1 import AppTest


@pytest.fixture
def frontend(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://backend.test:8000")
    calls = []
    behaviour = {"predict_status": 200, "pipeline_status": 200}
    examples = {
        "payment": jsonlib.loads(Path("data/examples/not_flagged.json").read_text()),
        "transfer": jsonlib.loads(Path("data/examples/flag_for_review.json").read_text()),
    }

    def request(method, url, json=None, timeout=None, headers=None):
        path = urlparse(url).path
        if path == "/model":
            payload = {
                "name": "transaction-fraud",
                "version": "8",
                "decision_threshold": 0.5,
                "transaction_types": ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
                "mlflow_url": "http://127.0.0.1:5000",
                "image": "test-image",
            }
        elif path == "/monitoring":
            payload = {
                "model_uri": "models:/transaction-fraud/8",
                "window_count": 0,
                "reference_rows": 100,
                "window_size": 200,
                "minimum_window": 100,
                "amount_psi": None,
                "drift_threshold": 0.2,
                "labelled_replay": None,
                "grafana_url": "http://127.0.0.1:3000/d/fraud-operations",
                "prometheus_url": "http://127.0.0.1:9090/alerts",
            }
        elif path == "/examples":
            payload = examples
        elif path == "/ready":
            payload = {"status": "ready"}
        elif path == "/predict":
            calls.append(json)
            payload = {
                "label": "Flag for review",
                "score": 0.9,
                "decision_threshold": 0.5,
                "model_version": "8",
                "request_id": "test-request",
            }
        elif path == "/pipeline":
            payload = {
                "rows": 1000,
                "preparation_seconds": 2.0,
                "rows_per_second": 500.0,
                "workers": 2,
                "batches": 4,
                "splits": {"train": 700, "validation": 150, "test": 150},
                "test_metrics": {"precision": 0.5, "recall": 0.5, "f1": 0.5},
                "evaluated_model_version": 8,
                "comparison_run_id": "test-run",
            }
        elif path == "/operations":
            payload = {"worker_running": False, "busy": False}
        elif path == "/deployments":
            payload = {"current": {"model_version": "8"}, "previous": None, "events": []}
        elif path == "/pipeline/runs":
            payload = []
        else:
            raise AssertionError(f"Unexpected REST request: {url}")
        response = requests.Response()
        response.status_code = {
            "/predict": behaviour["predict_status"],
            "/pipeline": behaviour["pipeline_status"],
        }.get(path, 200)
        response._content = jsonlib.dumps(payload).encode()
        response.url = url
        return response

    monkeypatch.setattr(requests, "request", request)
    return (
        AppTest.from_file(Path("fastapi_app/frontend.py").resolve(), default_timeout=10),
        calls,
        behaviour,
    )


def test_blank_form_does_not_submit_guessed_values(frontend):
    app, calls, _ = frontend
    app.run()
    app.button[2].click().run()
    assert not app.exception
    assert calls == []
    assert "enter all three numeric values" in app.error[0].value


def test_example_populates_form_and_one_submit_produces_result(frontend):
    app, calls, _ = frontend
    app.run()
    app.button[1].click().run()
    assert app.selectbox[0].value == "TRANSFER"
    app.button[2].click().run()
    assert not app.exception
    assert len(calls) == 1
    assert app.metric[0].value == "0.900"
    assert app.warning[0].value == "Flag for review"


def test_failed_prediction_clears_previous_result(frontend):
    app, calls, behaviour = frontend
    app.run().button[1].click().run().button[2].click().run()
    assert app.metric
    behaviour["predict_status"] = 503
    app.button[2].click().run()
    assert not app.exception
    assert len(calls) == 2
    assert not app.metric
    assert "could not be completed" in app.error[0].value


def test_operations_screen_displays_recorded_evidence(frontend):
    app, calls, _ = frontend
    app.run().radio[0].set_value("Operations").run()
    assert not app.exception
    assert app.title[0].value == "Pipeline overview"
    assert len(app.metric) == 6
    assert calls == []


def test_missing_backend_configuration_is_visible(monkeypatch):
    monkeypatch.delenv("BACKEND_URL", raising=False)
    app = AppTest.from_file(Path("fastapi_app/frontend.py").resolve()).run()
    assert not app.exception
    assert "not configured" in app.error[0].value


def test_pipeline_outputs_being_rebuilt_do_not_hide_operations_controls(frontend):
    app, _, behaviour = frontend
    behaviour["pipeline_status"] = 503
    app.run().radio[0].set_value("Operations").run()
    assert not app.exception
    assert any(button.label == "Run pipeline" for button in app.button)
    assert any("outputs are currently unavailable" in message.value for message in app.info)
