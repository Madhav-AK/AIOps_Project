"""Download the original PaySim Kaggle archive without account credentials."""

import argparse
import csv
import hashlib
import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

LOGGER = logging.getLogger(__name__)
DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1?datasetVersionNumber=2"
)
SOURCE_PAGE = "https://www.kaggle.com/datasets/ealaxi/paysim1"
EXPECTED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}


def sha256_file(path: Path) -> str:
    """Hash in chunks so the whole CSV need not fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(output: Path, manifest_path: Path) -> None:
    """Stage and check the download before installing the CSV and its provenance."""
    if output.exists():
        if not manifest_path.exists():
            raise ValueError(f"Existing {output} has no source manifest; inspect it first.")
        manifest = json.loads(manifest_path.read_text())
        if sha256_file(output) != manifest["sha256"]:
            raise ValueError("Existing CSV differs from its recorded checksum.")
        LOGGER.info("CSV already exists and checksum matches: %s", output)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    # Temporary files stay on the destination filesystem for an atomic final rename.
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        archive_path = Path(temporary) / "paysim.zip"
        request = Request(DOWNLOAD_URL, headers={"User-Agent": "AIOps-Course-Project"})
        LOGGER.info("Downloading Kaggle dataset version 2")
        downloaded = 0
        with urlopen(request, timeout=90) as response, archive_path.open("wb") as target:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                target.write(chunk)
                downloaded += len(chunk)
                if downloaded % (20 * 1024 * 1024) == 0:
                    LOGGER.info("Downloaded %.0f MiB", downloaded / 1024**2)

        staged_csv = Path(temporary) / "paysim.csv"
        with ZipFile(archive_path) as archive:
            csv_members = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(csv_members) != 1:
                raise ValueError("Expected exactly one CSV in the Kaggle archive.")
            # Copy one stream rather than extracting untrusted archive paths.
            with archive.open(csv_members[0]) as source, staged_csv.open("wb") as target:
                shutil.copyfileobj(source, target)

        with staged_csv.open(newline="") as source:
            columns = next(csv.reader(source))
        if set(columns) != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected dataset schema: {columns}")

        manifest = {
            "dataset": "PaySim synthetic financial transactions",
            "source_page": SOURCE_PAGE,
            "download_url": DOWNLOAD_URL,
            "dataset_version": 2,
            "archive_member": csv_members[0],
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "path": output.as_posix(),
            "bytes": staged_csv.stat().st_size,
            "sha256": sha256_file(staged_csv),
            "columns": columns,
        }
        staged_csv.replace(output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info("Dataset ready: %s (%s)", output, manifest["sha256"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/paysim.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/source.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download(args.output, args.manifest)


if __name__ == "__main__":
    main()
