AI OPs Project Submission

- By Madhav AK
- DA24B012

This README file goes through all major submission points.

- Main report: [REPORT.pdf](REPORT.pdf).
- The five documentation PDFs and test report: [docs](docs/).
- AI usage details: [AI_DISCLOSURE.md](AI_DISCLOSURE.md).
- Video Link: https://drive.google.com/file/d/1cPaMSbcB42UD9COzOZRbsjQFW30cGylR/view?usp=sharing

- `docs` contains the five required documentation PDFs and the test report.
- `fraud_detection` contains data preparation, training and evaluation.
- `fastapi_app` contains the FastAPI backend and Streamlit frontend.
- `scripts` contains the download, pipeline, deployment and replay commands.
- `data` contains the dataset, examples and DVC files.
- `monitoring` contains the Prometheus settings, alerts and Grafana dashboard.
- `tests` contains the automated tests.
- `reports` contains model results, replay results and test output.
- `artifacts` contains the generated model files.
- `mlflow_storage` stores the MLflow experiments and registered models.
- `operations` stores job records and deployment history.

### Setup

Requires Python 3.12.3 and Docker Compose on Linux. Setup and training below are
for a new checkout. On this machine, go to Running the application.

```bash
cd ~/AIOps_Project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
dvc repro prepare
python -m scripts.create_labelled_batch
mkdir -p mlflow_storage operations
```

### MLflow and training

Start MLflow in a separate activated terminal:

```bash
mlflow server --host 127.0.0.1 --port 5000 --workers 1 \
  --backend-store-uri sqlite:///mlflow_storage/tracking.db \
  --serve-artifacts --artifacts-destination mlflow_storage/artifacts
```

Then in another activated terminal:

```bash
dvc repro --force
cat artifacts/model/model_info.json
docker tag aiops-project:training aiops-project:initial
docker tag aiops-project:training aiops-project:tracking
```

Create `.env` using the actual model URI and threshold from `model_info.json`:

```text
MODEL_URI=<model_uri>
DECISION_THRESHOLD=<decision_threshold>
APP_IMAGE=aiops-project:initial
```

Stop the foreground MLflow server with Ctrl+C and wait for it to finish. Then:

```bash
docker compose up -d --wait
python -m scripts.run_operations init
```

### Running the application

```bash
source .venv/bin/activate
docker compose up -d
```

Keep one worker running in a separate activated terminal:

```bash
python -m scripts.run_operations worker
```

- Application: http://127.0.0.1:8501
- API documentation: http://127.0.0.1:8000/docs
- MLflow: http://127.0.0.1:5000
- Prometheus alerts: http://127.0.0.1:9090/alerts
- Grafana: http://127.0.0.1:3000/d/fraud-operations

### Pipeline

```bash
dvc dag
dvc status
python -m scripts.run_operations ci
python -m scripts.run_operations release
python -m scripts.run_operations rollback
```

### Replay

Normal traffic, followed by changed amounts for the drift check:

```bash
python -m scripts.replay --rows 2000 --concurrency 4
python -m scripts.replay --rows 2000 --concurrency 4 --amount-multiplier 1000
```

### Tests

```bash
python -m pytest tests -q --junitxml=reports/tests/final.xml
```
