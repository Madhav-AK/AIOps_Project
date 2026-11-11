"""Small shared helpers for the three pipeline stages."""

import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import yaml


def read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def write_json(path: Path, content: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def source_metadata(config_path: Path) -> dict:
    """Record exact source identity; containers can receive it explicitly."""
    paths = [
        *sorted(Path("fraud_detection").glob("*.py")),
        *sorted(Path("scripts").glob("*.py")),
        config_path,
        Path("requirements.txt"),
        Path("MLproject"),
        Path("python_env.yaml"),
        Path("dvc.yaml"),
    ]
    source_keys = ("SOURCE_GIT_COMMIT", "SOURCE_WORKTREE_DIRTY")
    if any(key in os.environ for key in source_keys):
        if not all(os.environ.get(key) for key in source_keys):
            raise ValueError("Supply both SOURCE_GIT_COMMIT and SOURCE_WORKTREE_DIRTY explicitly.")
        commit = os.environ["SOURCE_GIT_COMMIT"]
        dirty_value = os.environ["SOURCE_WORKTREE_DIRTY"]
        if dirty_value not in ("true", "false"):
            raise ValueError("SOURCE_WORKTREE_DIRTY must be true or false.")
        dirty = dirty_value == "true"
    else:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain", "--", *map(str, paths)], text=True
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                "Outside a Git checkout, supply SOURCE_GIT_COMMIT and SOURCE_WORKTREE_DIRTY."
            ) from exc
    return {
        "git_commit": commit,
        "source_worktree_dirty": dirty,
        "source_files_sha256": {p.as_posix(): file_hash(p) for p in paths if p.is_file()},
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in [
                "numpy",
                "pandas",
                "scikit-learn",
                "xgboost-cpu",
                "joblib",
                "ray",
                "mlflow",
                "dvc",
            ]
        },
    }
