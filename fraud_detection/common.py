"""Small shared helpers for the three pipeline stages."""

import hashlib
import json
import platform
import subprocess
import zipfile
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
    """Record source identity from the Git checkout mounted into training."""
    config_path = config_path.resolve()
    if config_path.is_relative_to(Path.cwd()):
        config_path = config_path.relative_to(Path.cwd())
    paths = [
        *sorted(Path("fraud_detection").glob("*.py")),
        Path("scripts/download_data.py"),
        Path("scripts/train_in_docker.py"),
        Path("Dockerfile"),
        config_path,
        Path("requirements.txt"),
        Path("MLproject"),
        Path("python_env.yaml"),
        Path("dvc.yaml"),
    ]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(
            subprocess.check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--",
                    *[str(p) for p in paths if not p.is_absolute()],
                ],
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Run training from the project Git checkout.") from exc
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


def snapshot_source(path: Path) -> None:
    """Save draft changes on top of the recorded Git commit; ignored data stays in DVC."""
    patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"])
    untracked = (
        subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"])
        .decode()
        .split("\0")
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("working_tree.patch", patch)
        for name in untracked:
            if name and Path(name).is_file():
                archive.write(name, "untracked/" + name)
