"""Load the saved pipeline in a fresh process and score one transaction JSON file."""

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/model"))
    args = parser.parse_args()
    try:
        record = json.loads(args.input.read_text())
        if not isinstance(record, dict):
            raise ValueError(
                "Input must be one JSON object containing the four transaction fields."
            )
        info = json.loads((args.model_dir / "model_info.json").read_text())
        model = joblib.load(args.model_dir / "model.joblib")
        score = float(model.predict_proba(pd.DataFrame([record]))[0, 1])
        flagged = score >= info["decision_threshold"]
        print(
            json.dumps(
                {
                    "label": "Flag for review" if flagged else "Not flagged",
                    "score": score,
                    "decision_threshold": info["decision_threshold"],
                    "model": info["selected_model"],
                },
                indent=2,
            )
        )
    except (OSError, ValueError, KeyError) as exc:
        logging.error("Prediction failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
