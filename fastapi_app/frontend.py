"""Streamlit UI: transaction screening and a separate pipeline overview."""

import logging
import math
import os
from datetime import datetime, timezone

import requests
import streamlit as st

st.set_page_config(page_title="Transaction review")
LOGGER = logging.getLogger(__name__)

if "BACKEND_URL" not in os.environ:
    LOGGER.error("BACKEND_URL is required")
    st.error("The application is not configured. Contact the operator.")
    st.stop()
BACKEND_URL = os.environ["BACKEND_URL"].rstrip("/")


def api(method: str, path: str, payload=None):
    """Call the configured backend; failed requests remain errors."""
    response = requests.request(
        method, f"{BACKEND_URL}{path}", json=payload, timeout=8, headers={"X-Client": "frontend"}
    )
    response.raise_for_status()
    return response.json()


page = st.sidebar.radio("Workspace", ["Check transaction", "Operations"])
st.sidebar.caption("Demonstration using synthetic PaySim transactions.")

try:
    model = api("GET", "/model")
except (requests.RequestException, ValueError):
    LOGGER.exception("Backend metadata request failed")
    st.error("The application service is unavailable. Please try again shortly.")
    st.stop()

if page == "Check transaction":
    st.title("Transaction review")
    st.write("Enter a transaction to check whether it should be flagged for review.")
    try:
        examples = api("GET", "/examples")
        api("GET", "/ready")
    except (requests.RequestException, ValueError):
        LOGGER.exception("Prediction service readiness check failed")
        st.error("The prediction service is unavailable. Please try again shortly.")
        st.stop()

    def load_example(name):
        for field, value in examples[name].items():
            st.session_state[field] = value

    st.write("Start with an example, or enter your own details below.")
    first, second = st.columns(2)
    first.button("Payment example", on_click=load_example, args=("payment",))
    second.button("Transfer example", on_click=load_example, args=("transfer",))

    with st.form("transaction"):
        values = {
            "type": st.selectbox(
                "Transaction type", model["transaction_types"], index=None, key="type"
            ),
            "amount": st.number_input(
                "Transaction amount", min_value=0.0, value=None, key="amount"
            ),
            "oldbalanceOrg": st.number_input(
                "Sender balance before", min_value=0.0, value=None, key="oldbalanceOrg"
            ),
            "oldbalanceDest": st.number_input(
                "Recipient balance before", min_value=0.0, value=None, key="oldbalanceDest"
            ),
        }
        st.caption("Use balances from before the transaction. Zero balances are allowed.")
        submitted = st.form_submit_button("Check transaction")

    # Render only this submission's result, so an error cannot display an older success.
    if submitted:
        if any(value is None for value in values.values()) or any(
            not math.isfinite(value) for field, value in values.items() if field != "type"
        ):
            st.error("Choose a type and enter all three numeric values.")
        else:
            try:
                result = api("POST", "/predict", values)
            except (requests.RequestException, ValueError):
                LOGGER.exception("Prediction request failed")
                st.error("A prediction could not be completed. Please try again shortly.")
            else:
                if result["label"] == "Flag for review":
                    st.warning(result["label"])
                else:
                    st.success(result["label"])
                st.metric("Model score", f"{result['score']:.3f}")
                st.caption(
                    f"Checked: {values['type']}, amount {values['amount']:,.2f}. "
                    f"Review threshold {result['decision_threshold']:.2f}. "
                    f"Model version {result['model_version']}."
                )
                st.caption(f"Request reference: {result['request_id']}")
    st.caption(
        "A higher score indicates greater suspicion. It is not a calibrated fraud probability."
    )

else:

    def submit_operation(path, payload):
        try:
            job = api("POST", path, payload)
        except (requests.RequestException, ValueError):
            LOGGER.exception("Operation could not be submitted")
            st.error(
                "The operation could not be submitted. Check worker availability and active jobs."
            )
        else:
            st.session_state["selected_job"] = job["id"]
            st.rerun()

    @st.fragment(run_every="2s")
    def operations_page():
        st.title("Pipeline overview")
        try:
            deployed = api("GET", "/model")
            status = api("GET", "/operations")
            deployment = api("GET", "/deployments")
        except (requests.RequestException, ValueError):
            LOGGER.exception("Operations overview request failed")
            st.error(
                "Operations information is unavailable. Check the backend and deployment setup."
            )
            return
        try:
            pipeline = api("GET", "/pipeline")
        except (requests.RequestException, ValueError):
            pipeline = None  # DVC replaces these outputs while preparing/evaluating.
        st.write(
            f"Deployed model: {deployed['name']}, version {deployed['version']}, "
            f"threshold {deployed['decision_threshold']:.2f}"
        )
        st.caption(f"Application image: {deployed['image']}")
        st.link_button("Open MLflow", deployed["mlflow_url"])
        st.caption("Status and logs refresh every two seconds.")

        with st.expander("Monitoring", expanded=True):
            try:
                observed = api("GET", "/monitoring")
            except (requests.RequestException, ValueError):
                st.warning("Monitoring information is unavailable.")
            else:
                st.link_button("Open Grafana", observed["grafana_url"])
                st.link_button("Open alerts", observed["prometheus_url"])
                st.caption(f"Drift reference: {observed['reference_rows']:,} rows from training.")
                psi = observed["amount_psi"]
                st.write(f"Amount window: {observed['window_count']}/{observed['window_size']}")
                if psi is None:
                    st.info(
                        f"Collect {observed['minimum_window']} valid predictions to compute drift."
                    )
                elif psi > observed["drift_threshold"]:
                    st.warning(f"Amount drift detected: PSI {psi:.3f}.")
                else:
                    st.write(f"Amount PSI: {psi:.3f}; threshold {observed['drift_threshold']}.")
                st.caption(
                    "Input drift does not prove model quality has fallen. "
                    "Replay labels are evaluated separately."
                )
                replay = observed["labelled_replay"]
                if replay is not None:
                    st.write(
                        f"Last labelled replay: {replay['rows']} rows for {replay['model_uri']}; "
                        f"F1 {replay['f1']:.3f}."
                    )

        st.subheader("Pipeline controls")
        if not status["worker_running"]:
            st.warning("The local operations worker is stopped. Start it to run operations.")
            st.code("python -m scripts.run_operations worker", language="bash")
        if status["busy"]:
            st.info("An operation is queued or running. Wait for it to finish.")
        disabled = not status["worker_running"] or status["busy"]
        actions = {
            "Reproduce pipeline": "reproduce",
            "Retrain model": "retrain",
            "CI checks": "ci",
            "Check and deploy": "release",
        }
        action = st.selectbox("Pipeline action", list(actions))
        st.caption(
            "CI checks reproduce and validate the pipeline. Check and deploy also "
            "builds and deploys the image."
        )
        if st.button("Run pipeline", disabled=disabled):
            submit_operation("/pipeline/runs", {"action": actions[action]})

        st.subheader("Deployment")
        if pipeline is not None:
            version = st.number_input(
                "Model version",
                min_value=1,
                value=pipeline["evaluated_model_version"],
                step=1,
            )
            if st.button("Deploy model", disabled=disabled):
                submit_operation("/deployments", {"version": version})
        else:
            st.info("Candidate results are unavailable while pipeline outputs are regenerated.")
        previous = deployment["previous"]
        if previous is not None:
            st.caption(
                f"Previous release: model version {previous['model_version']}, "
                f"image {previous['image']}."
            )
        if st.button("Restore previous release", disabled=disabled or previous is None):
            submit_operation("/deployments/rollback", {})
        with st.expander("Deployment history"):
            st.json(deployment["events"])

        st.subheader("Run history")
        try:
            runs = api("GET", "/pipeline/runs")
            if runs:
                st.dataframe(
                    [
                        {
                            "Job": run["id"][:8],
                            "Action": run["kind"],
                            "State": run["state"],
                            "Created (UTC)": datetime.fromtimestamp(
                                run["created_at"], timezone.utc
                            ).strftime("%H:%M:%S"),
                        }
                        for run in runs
                    ]
                )
                selected = st.selectbox(
                    "Job details",
                    [run["id"] for run in runs],
                    key="selected_job",
                    format_func=lambda value: value[:8],
                )
                detail = api("GET", f"/pipeline/runs/{selected}")
                st.write(f"**{detail['kind']} — {detail['state']}**")
                if detail["error"] is not None:
                    st.error(detail["error"])
                if "dag" in detail:
                    st.graphviz_chart(detail["dag"])
                st.dataframe(detail["steps"])
                if detail["source"] is not None:
                    st.caption(f"Source commit {detail['source']['git_commit']}.")
                    if "workspace_dirty" in detail["source"]:
                        st.caption(
                            f"Uncommitted workspace changes: {detail['source']['workspace_dirty']}."
                        )
                    else:
                        st.caption(
                            "Uncommitted model-source changes: "
                            f"{detail['source']['source_worktree_dirty']}."
                        )
                with st.expander("Job log"):
                    st.code(detail["log_tail"], language="text")
            else:
                st.info("No operations have been recorded yet.")
        except (requests.RequestException, ValueError):
            LOGGER.exception("Operations history request failed")
            st.error("Run history is unavailable.")

        if pipeline is None:
            st.info("Preparation and evaluation outputs are currently unavailable.")
            return
        st.subheader("Latest recorded preparation")
        rows, duration, throughput = st.columns(3)
        rows.metric("Rows processed", f"{pipeline['rows']:,}")
        duration.metric("Preparation time", f"{pipeline['preparation_seconds']:.2f} s")
        throughput.metric("Throughput", f"{pipeline['rows_per_second']:,.0f} rows/s")
        st.caption(
            f"{pipeline['workers']} Ray workers, {pipeline['batches']} batches. "
            "Timing includes startup, CSV reading, validation, splitting and Parquet writes."
        )
        st.write("Prepared splits:", pipeline["splits"])
        st.subheader("Latest recorded evaluation")
        metrics = pipeline["test_metrics"]
        precision, recall, f1 = st.columns(3)
        precision.metric("Precision", f"{metrics['precision']:.3f}")
        recall.metric("Recall", f"{metrics['recall']:.3f}")
        f1.metric("F1", f"{metrics['f1']:.3f}")
        st.caption(
            f"Evaluated model version {pipeline['evaluated_model_version']}. "
            f"Comparison run {pipeline['comparison_run_id']}"
        )

    operations_page()
