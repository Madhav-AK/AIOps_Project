"""Prometheus counters, a bounded input window, and observed component/job health."""

import json
import math
import os
import tempfile
import threading
from collections import Counter as Counts
from collections import deque
from pathlib import Path

import psutil
import requests
from mlflow import MlflowClient
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info, ProcessCollector

from fastapi_app import jobs
from fraud_detection.common import read_config
from fraud_detection.drift import population_stability_index

ROOT = Path(__file__).resolve().parents[1]


def load_reference(model_uri: str, tracking_uri: str) -> dict:
    """Fetch the reference from this deployed model's training run, never the latest candidate."""
    name, version = model_uri.removeprefix("models:/").rsplit("/", 1)
    client = MlflowClient(tracking_uri=tracking_uri)
    registered = client.get_model_version(name, version)
    parent = client.get_run(registered.run_id).data.tags["mlflow.parentRunId"]
    with tempfile.TemporaryDirectory(prefix="fraud-reference-") as directory:
        path = client.download_artifacts(parent, "amount_reference.json", directory)
        reference = json.loads(Path(path).read_text())
    if reference["comparison_run_id"] != parent:
        raise ValueError("Monitoring reference belongs to a different training run.")
    population_stability_index([0.0], reference)  # Validate the saved schema/bins at startup.
    return reference


class Monitoring:
    """One instance per backend process; no multiprocess/shared-counter framework."""

    def __init__(self, model_uri: str, directory: Path):
        self.config = read_config(ROOT / "params.yaml")["monitoring"]
        self.reference = load_reference(model_uri, os.environ["MLFLOW_TRACKING_URI"])
        self.model_uri = model_uri
        self.directory = directory
        self.amounts = deque(maxlen=self.config["window_size"])
        if not 1 <= self.config["minimum_window"] <= self.amounts.maxlen:
            raise ValueError("Monitoring minimum must fit in its bounded window.")
        self.lock = threading.Lock()
        self.registry = CollectorRegistry()
        ProcessCollector(registry=self.registry)
        self.model_info = Info(
            "aiops_model", "Active model and training reference", registry=self.registry
        )
        self.model_info.info(
            {"model_uri": model_uri, "reference_run_id": self.reference["comparison_run_id"]}
        )
        self.reference_rows = Gauge(
            "aiops_reference_rows",
            "Training inputs in the deployed reference",
            registry=self.registry,
        )
        self.reference_rows.set(self.reference["training_rows"])
        self.requests = Counter(
            "aiops_http_requests",
            "Backend responses",
            ["method", "route", "status", "client"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "aiops_http_request_seconds",
            "Backend request duration",
            ["route"],
            registry=self.registry,
        )
        self.model_latency = Histogram(
            "aiops_model_inference_seconds",
            "MLflow round-trip time, including errors",
            registry=self.registry,
        )
        self.predictions = Counter(
            "aiops_predictions", "Successful predictions", ["label"], registry=self.registry
        )
        for label in ("Flag for review", "Not flagged"):
            self.predictions.labels(label)
        self.scores = Histogram(
            "aiops_fraud_score",
            "Observed scores, not calibrated probabilities",
            buckets=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1],
            registry=self.registry,
        )
        self.psi = Gauge(
            "aiops_amount_psi",
            "Amount PSI; NaN until enough valid predictions",
            registry=self.registry,
        )
        self.psi.set(math.nan)
        self.window_count = Gauge(
            "aiops_amount_window_count", "Inputs in current bounded window", registry=self.registry
        )
        self.threshold = Gauge(
            "aiops_drift_threshold", "Demonstration drift threshold", registry=self.registry
        )
        self.threshold.set(self.config["drift_threshold"])
        self.minimum = Gauge(
            "aiops_drift_minimum",
            "Minimum input count before computing PSI",
            registry=self.registry,
        )
        self.minimum.set(self.config["minimum_window"])
        self.services = Gauge(
            "aiops_service_up",
            "Observed HTTP component health",
            ["component"],
            registry=self.registry,
        )
        self.worker = Gauge(
            "aiops_worker_up", "Operations worker holds its OS lock", registry=self.registry
        )
        self.recorded_jobs = Gauge(
            "aiops_recorded_jobs",
            "Retained operations records",
            ["kind", "state"],
            registry=self.registry,
        )
        self.stage_seconds = Gauge(
            "aiops_pipeline_stage_seconds",
            "Latest pipeline stage wall time",
            ["stage", "execution"],
            registry=self.registry,
        )
        self.preparation_rate = Gauge(
            "aiops_preparation_rows_per_second",
            "Last actual Ray preparation throughput",
            registry=self.registry,
        )
        self.memory = Gauge(
            "aiops_machine_available_memory_bytes",
            "Available memory reported by the host kernel",
            registry=self.registry,
        )
        self.disk = Gauge(
            "aiops_disk_free_bytes",
            "Available space on the application filesystem",
            registry=self.registry,
        )
        self.replay_metrics = Gauge(
            "aiops_labelled_replay",
            "Last labelled replay for the active model",
            ["metric"],
            registry=self.registry,
        )
        self.replay_timestamp = Gauge(
            "aiops_labelled_replay_timestamp_seconds",
            "Timestamp of last labelled replay for this model",
            registry=self.registry,
        )

    def observe_request(self, method, route, status, client, seconds):
        if route == "/metrics":
            return
        client = client if client in ("frontend", "replay") else "other"
        self.requests.labels(method, route, str(status), client).inc()
        self.latency.labels(route).observe(seconds)

    def observe_prediction(self, amount, score, label):
        self.predictions.labels(label).inc()
        self.scores.observe(score)
        with self.lock:
            self.amounts.append(amount)
            self.window_count.set(len(self.amounts))
            if len(self.amounts) >= self.config["minimum_window"]:
                self.psi.set(population_stability_index(self.amounts, self.reference))

    def summary(self) -> dict:
        with self.lock:
            count = len(self.amounts)
            psi = (
                population_stability_index(self.amounts, self.reference)
                if count >= self.config["minimum_window"]
                else None
            )
        replay_path = ROOT / "reports/monitoring/labelled_replay.json"
        replay = json.loads(replay_path.read_text()) if replay_path.exists() else None
        return {
            "model_uri": self.model_uri,
            "reference_run_id": self.reference["comparison_run_id"],
            "reference_rows": self.reference["training_rows"],
            "window_count": count,
            "window_size": self.amounts.maxlen,
            "minimum_window": self.config["minimum_window"],
            "amount_psi": psi,
            "drift_threshold": self.config["drift_threshold"],
            "labelled_replay": replay,
            "grafana_url": self.config["grafana_url"],
            "prometheus_url": self.config["prometheus_url"],
        }

    def refresh(self):
        targets = {**self.config["health_urls"], "frontend": os.environ["FRONTEND_HEALTH_URL"]}
        for name, url in targets.items():
            try:
                response = requests.get(url, timeout=0.5)
                healthy = response.ok
            except requests.RequestException:
                healthy = False  # An observed failed probe, not an invented successful value.
            self.services.labels(name).set(int(healthy))
        self.worker.set(int(jobs.worker_available(self.directory)))
        records = jobs.list_jobs(self.directory)
        self.recorded_jobs.clear()
        for (kind, state), count in Counts((j["kind"], j["state"]) for j in records).items():
            self.recorded_jobs.labels(kind, state).set(count)
        self.stage_seconds.clear()
        for record in records:
            steps = [
                s
                for s in record["steps"]
                if s["name"] in ("prepare", "train", "evaluate") and s["seconds"] is not None
            ]
            if steps:
                for step in steps:
                    self.stage_seconds.labels(step["name"], step["execution"] or "failed").set(
                        step["seconds"]
                    )
                break
        try:
            preparation = json.loads((ROOT / "data/prepared/preparation.json").read_text())
        except (OSError, ValueError):
            self.preparation_rate.set(math.nan)  # DVC can temporarily replace the directory.
        else:
            self.preparation_rate.set(preparation["rows_per_second"])
        self.memory.set(psutil.virtual_memory().available)
        self.disk.set(psutil.disk_usage(ROOT).free)
        self.replay_metrics.clear()
        replay = self.summary()["labelled_replay"]
        self.replay_timestamp.set(math.nan)
        if replay is not None and replay["model_uri"] == self.model_uri:
            for name in (
                "precision",
                "recall",
                "f1",
                "requests_per_second",
                "p50_seconds",
                "p95_seconds",
            ):
                self.replay_metrics.labels(name).set(replay[name])
            self.replay_timestamp.set(replay["finished_at"])
