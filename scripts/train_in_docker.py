"""Run the MLflow training Project in the same Docker environment as the app."""

import os
import subprocess
from pathlib import Path

from fraud_detection.common import read_config


def main() -> None:
    root = Path.cwd().resolve()
    config = read_config(root / "params.yaml")
    image = "aiops-project:training"
    subprocess.run(["docker", "build", "-t", image, "."], check=True)
    identity = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True
    ).strip()
    # On this Linux machine, host networking keeps the configured tracking URL reachable.
    arguments = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--volume",
        f"{root}:/app",
        "--workdir",
        "/app",
        "--env",
        f"MLFLOW_TRACKING_URI={config['tracking']['uri']}",
        "--env",
        f"TRAINING_IMAGE_ID={identity}",
    ]
    if "PIPELINE_JOB_ID" in os.environ:
        arguments += ["--env", f"PIPELINE_JOB_ID={os.environ['PIPELINE_JOB_ID']}"]
    arguments += [
        image,
        "mlflow",
        "run",
        ".",
        "--env-manager",
        "local",
        "--entry-point",
        "train",
        "--experiment-name",
        config["tracking"]["experiment"],
        "-P",
        "config=params.yaml",
    ]
    subprocess.run(arguments, check=True)


if __name__ == "__main__":
    main()
