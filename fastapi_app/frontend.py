"""Streamlit UI: transaction screening and a separate pipeline overview."""

import logging
import math
import os

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
    response = requests.request(method, f"{BACKEND_URL}{path}", json=payload, timeout=8)
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
    st.title("Pipeline overview")
    st.write(
        f"Deployed model: {model['name']}, version {model['version']}, "
        f"threshold {model['decision_threshold']:.2f}"
    )
    st.link_button("Open MLflow", model["mlflow_url"])
    st.graphviz_chart(
        'digraph { rankdir=LR; "Raw data" -> "Ray prepare / split" '
        '-> "Train + MLflow" -> "Evaluate / register"; }'
    )
    try:
        pipeline = api("GET", "/pipeline")
    except (requests.RequestException, ValueError):
        LOGGER.exception("Recorded pipeline details request failed")
        st.error("Recorded pipeline details are unavailable.")
        st.stop()

    st.subheader("Recorded preparation")
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
