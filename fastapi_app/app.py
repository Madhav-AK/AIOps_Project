"""Validate application requests and forward inference to MLflow over HTTP."""

import json
import logging
import math
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, get_args

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from fastapi_app import jobs

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
TransactionType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    type: TransactionType
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)


class Prediction(BaseModel):
    label: Literal["Flag for review", "Not flagged"]
    score: float
    decision_threshold: float
    model_version: str
    model_uri: str
    request_id: str


class PipelineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["reproduce", "retrain", "ci", "release"]


class DeploymentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: int = Field(gt=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Read required settings at startup; close the model-server client on shutdown."""
    model_uri = os.environ["MODEL_URI"]
    match = re.fullmatch(r"models:/([\w.-]+)/([1-9]\d*)", model_uri)
    if match is None:
        raise ValueError("MODEL_URI must pin models:/name/version; aliases are not accepted.")
    threshold = float(os.environ["DECISION_THRESHOLD"])
    if not 0 < threshold < 1:
        raise ValueError("DECISION_THRESHOLD must be between 0 and 1.")
    app.state.model = {
        "model_uri": model_uri,
        "name": match.group(1),
        "version": match.group(2),
        "decision_threshold": threshold,
        "input_fields": list(Transaction.model_fields),
        "transaction_types": list(get_args(TransactionType)),
        "mlflow_url": os.environ["MLFLOW_PUBLIC_URL"],
        "image": os.environ["APP_IMAGE"],
    }
    app.state.operations = Path(os.environ["OPERATIONS_DIR"])
    app.state.client = httpx.Client(base_url=os.environ["MODEL_SERVER_URL"], timeout=5.0)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        yield
    finally:
        app.state.client.close()


app = FastAPI(title="Transaction fraud screening", version="1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    # Report fields/messages without echoing raw input (which may include non-JSON NaN/Infinity).
    issues = [{key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": issues})


@app.middleware("http")
async def log_request(request: Request, call_next):
    """Attach one request ID and record its route, status and duration."""
    request.state.request_id = uuid.uuid4().hex
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    LOGGER.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request.state.request_id,
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
    )
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict:
    try:
        response = request.app.state.client.get("/ping")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        LOGGER.warning("Model readiness check failed: %s", type(exc).__name__)
        raise HTTPException(503, "Prediction service is unavailable.") from exc
    return {"status": "ready", "model_version": request.app.state.model["version"]}


@app.get("/model")
def model_details(request: Request) -> dict:
    return request.app.state.model


@app.get("/examples")
def examples() -> dict:
    try:
        return {
            name: Transaction.model_validate(
                json.loads((ROOT / "data/examples" / filename).read_text())
            ).model_dump()
            for name, filename in [
                ("payment", "not_flagged.json"),
                ("transfer", "flag_for_review.json"),
            ]
        }
    except (OSError, ValueError) as exc:
        LOGGER.exception("Example transactions could not be read")
        raise HTTPException(503, "Example transactions are unavailable.") from exc


@app.get("/pipeline")
def pipeline_overview() -> dict:
    """Expose recorded preparation and evaluation evidence."""
    try:
        prepared = json.loads((ROOT / "data/prepared/preparation.json").read_text())
        results = json.loads((ROOT / "reports/model/results.json").read_text())
        return {
            "rows": prepared["source"]["rows"],
            "fraud_rows": prepared["source"]["class_counts"]["1"],
            "preparation_seconds": prepared["total_seconds"],
            "rows_per_second": prepared["rows_per_second"],
            "workers": prepared["workers"],
            "batches": prepared["batches"],
            "splits": {name: part["rows"] for name, part in prepared["splits"].items()},
            "comparison_run_id": results["comparison_run_id"],
            "evaluated_model_version": results["registered_model_version"],
            "selected_model": results["selected_model"],
            "test_metrics": results["test"],
        }
    except (OSError, ValueError, KeyError) as exc:
        LOGGER.exception("Recorded pipeline results could not be read")
        raise HTTPException(503, "Recorded pipeline results are unavailable.") from exc


def submit_operation(request: Request, kind: str, arguments: dict) -> dict:
    try:
        return jobs.submit(request.app.state.operations, kind, arguments, queued=True)
    except jobs.WorkerUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except jobs.BusyError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/operations")
def operations_status(request: Request) -> dict:
    directory = request.app.state.operations
    active = [job for job in jobs.list_jobs(directory) if job["state"] in ("queued", "running")]
    return {"worker_running": jobs.worker_available(directory), "busy": bool(active)}


@app.post("/pipeline/runs", status_code=202)
def start_pipeline(payload: PipelineRequest, request: Request) -> dict:
    return submit_operation(request, payload.action, {})


@app.get("/pipeline/runs")
def pipeline_runs(request: Request) -> list[dict]:
    return jobs.list_jobs(request.app.state.operations)[:20]


@app.get("/pipeline/runs/{job_id}")
def pipeline_run(job_id: uuid.UUID, request: Request) -> dict:
    directory = request.app.state.operations
    try:
        job = jobs.get_job(directory, job_id.hex)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Unknown operations job.") from exc
    path = jobs.log_path(directory, job_id.hex)
    # A queued job legitimately has no log until the worker starts it.
    if path.exists():
        with path.open("rb") as log:
            log.seek(max(0, path.stat().st_size - 16000))
            job["log_tail"] = log.read().decode("utf-8", errors="replace")
    else:
        job["log_tail"] = "The job has not produced a log yet."
    return job


@app.get("/deployments")
def deployments(request: Request) -> dict:
    try:
        state = jobs.read(request.app.state.operations / "deployment.json")
    except FileNotFoundError as exc:
        raise HTTPException(503, "Record the running deployment baseline first.") from exc
    # Keep saved environment file contents private to the local release process.
    fields = (
        "model_uri",
        "model_version",
        "decision_threshold",
        "image",
        "image_id",
        "deployed_at",
    )
    return {
        "current": {key: state["current"][key] for key in fields},
        "previous": None
        if state["previous"] is None
        else {key: state["previous"][key] for key in fields},
        "events": state["events"],
    }


@app.post("/deployments", status_code=202)
def start_deployment(payload: DeploymentRequest, request: Request) -> dict:
    return submit_operation(request, "deploy", {"version": payload.version, "image": None})


@app.post("/deployments/rollback", status_code=202)
def start_rollback(request: Request) -> dict:
    return submit_operation(request, "rollback", {})


def fraud_score(payload: dict) -> float:
    """Enforce the native MLflow predict_proba contract for one binary transaction."""
    (probabilities,) = payload["predictions"]
    non_fraud, fraud = probabilities
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in (non_fraud, fraud)
    ):
        raise ValueError("Invalid class scores.")
    if not math.isclose(non_fraud + fraud, 1.0, abs_tol=1e-5):
        raise ValueError("Class scores must sum to one.")
    return float(fraud)


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction, request: Request) -> Prediction:
    try:
        response = request.app.state.client.post(
            "/invocations", json={"dataframe_records": [transaction.model_dump()]}
        )
        response.raise_for_status()
        score = fraud_score(response.json())
    except httpx.TimeoutException as exc:
        LOGGER.warning("Model request timed out; request_id=%s", request.state.request_id)
        raise HTTPException(504, "Prediction timed out. Please try again.") from exc
    except httpx.RequestError as exc:
        LOGGER.warning("Model unavailable; request_id=%s", request.state.request_id)
        raise HTTPException(503, "Prediction service is unavailable.") from exc
    except (httpx.HTTPStatusError, ValueError, KeyError, TypeError) as exc:
        LOGGER.exception("Invalid model response; request_id=%s", request.state.request_id)
        raise HTTPException(502, "Prediction service returned an invalid response.") from exc
    metadata = request.app.state.model
    return Prediction(
        label="Flag for review" if score >= metadata["decision_threshold"] else "Not flagged",
        score=score,
        decision_threshold=metadata["decision_threshold"],
        model_version=metadata["version"],
        model_uri=metadata["model_uri"],
        request_id=request.state.request_id,
    )
