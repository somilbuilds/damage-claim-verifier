"""
Sample-set evaluation runner.

Runs the same orchestrator pipeline on dataset/sample_claims.csv, compares the
predicted fields with the sample labels, prints per-field accuracy, and saves
code/evaluation/eval_results.json.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import loader  # noqa: E402
import orchestrator  # noqa: E402

EVAL_FIELDS = [
    "evidence_standard_met",
    "claim_status",
    "issue_type",
    "object_part",
    "severity",
    "valid_image",
]


def load_dotenv(dotenv_path: Path = REPO_ROOT / ".env") -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _accuracy(expected_rows: list[dict], predicted_rows: list[dict]) -> dict:
    results = {}
    total = min(len(expected_rows), len(predicted_rows))
    for field in EVAL_FIELDS:
        correct = 0
        for expected, predicted in zip(expected_rows, predicted_rows):
            if str(expected.get(field, "")).strip().lower() == str(predicted.get(field, "")).strip().lower():
                correct += 1
        results[field] = {
            "correct": correct,
            "total": total,
            "accuracy": (correct / total) if total else 0.0,
        }
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    sample_path = DATASET_DIR / "sample_claims.csv"
    predictions_path = CODE_DIR / "evaluation" / "sample_predictions.csv"
    results_path = CODE_DIR / "evaluation" / "eval_results.json"

    claims = loader.load_claims(sample_path)
    user_history = loader.load_user_history(DATASET_DIR / "user_history.csv")
    evidence_requirements = loader.load_evidence_requirements(DATASET_DIR / "evidence_requirements.csv")

    orchestrator.run_pipeline(
        claims=claims,
        user_history_by_id=user_history,
        evidence_requirements=evidence_requirements,
        output_csv_path=predictions_path,
        dataset_dir=DATASET_DIR,
    )

    expected_rows = _read_csv(sample_path)
    predicted_rows = _read_csv(predictions_path)
    field_results = _accuracy(expected_rows, predicted_rows)

    for field, result in field_results.items():
        print(f"{field}: {result['correct']}/{result['total']} = {result['accuracy']:.3f}")

    payload = {
        "sample_claims": len(expected_rows),
        "predicted_rows": len(predicted_rows),
        "fields": field_results,
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
