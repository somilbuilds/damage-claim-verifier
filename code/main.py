"""
CLI entry point for producing output.csv from dataset/claims.csv.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import loader
import orchestrator

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"


def load_dotenv(dotenv_path: Path = REPO_ROOT / ".env") -> None:
    """Load simple KEY=VALUE pairs without overriding existing environment."""
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    claims = loader.load_claims(DATASET_DIR / "claims.csv")
    user_history = loader.load_user_history(DATASET_DIR / "user_history.csv")
    evidence_requirements = loader.load_evidence_requirements(DATASET_DIR / "evidence_requirements.csv")

    orchestrator.run_pipeline(
        claims=claims,
        user_history_by_id=user_history,
        evidence_requirements=evidence_requirements,
        output_csv_path=REPO_ROOT / "output.csv",
        dataset_dir=DATASET_DIR,
    )


if __name__ == "__main__":
    main()
