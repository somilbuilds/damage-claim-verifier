"""
orchestrator.py - Sequential claim review pipeline coordinator.

Runs the required order:
Agent 2 claim extraction -> Agent 3 risk -> Agent 1 vision -> Agent 4 decision.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import agent1_vision
import agent2_claim
import agent3_risk
import agent4_decision
import loader
import parser

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
VALID_CLAIM_OBJECTS = {"car", "laptop", "package"}

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


def _claim_id(index: int) -> str:
    return f"claim_{index + 1:06d}"


def _input_fields(row: dict) -> dict:
    return {
        "user_id": row.get("user_id", ""),
        "image_paths": row.get("image_paths", ""),
        "user_claim": row.get("user_claim", ""),
        "claim_object": row.get("claim_object", ""),
    }


def _write_row(writer: csv.DictWriter, row: dict, output_file) -> None:
    writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})
    output_file.flush()


def _safe_output_row(input_row: dict) -> dict:
    outputs = parser.ensure_manual_review_flag(parser.build_safe_default_row())
    final = _input_fields(input_row)
    final.update(outputs)
    return final


def _merge_agent_failure_flags(row: dict, agent2: dict | None, agent3: dict | None, agent1: dict | None) -> dict:
    if agent2 and agent2.get("prompt_injection_flag"):
        row = parser.ensure_manual_review_flag(row)
        if row["risk_flags"] == "manual_review_required":
            row["risk_flags"] = "text_instruction_present;manual_review_required"
        elif "text_instruction_present" not in row["risk_flags"].split(";"):
            row["risk_flags"] = "text_instruction_present;" + row["risk_flags"]

    if agent3:
        risk_flags = str(agent3.get("risk_flags", "")).strip()
        if risk_flags and risk_flags != "none":
            existing = [] if row.get("risk_flags") in {"", "none"} else row["risk_flags"].split(";")
            for flag in risk_flags.split(";"):
                flag = flag.strip()
                if flag and flag not in existing:
                    existing.append(flag)
            row["risk_flags"] = ";".join(existing) if existing else "none"

    if agent1 is None:
        row = parser.ensure_manual_review_flag(row)

    return row


def _process_valid_batch(
    claims_batch: list[dict],
    user_history_by_id: dict[str, dict],
    evidence_requirements: dict[str, list[dict]],
    dataset_dir: str | Path | None,
) -> list[dict]:
    agent2_inputs = [
        {
            "claim_id": claim["_claim_id"],
            "claim_object": claim.get("claim_object", ""),
            "user_claim": claim.get("user_claim", ""),
        }
        for claim in claims_batch
    ]
    agent2_results = agent2_claim.run_claim_agent(agent2_inputs)
    agent2_by_id = {item["claim_id"]: item for item in agent2_results}

    agent3_inputs = [
        {
            "claim_id": claim["_claim_id"],
            "user_id": claim.get("user_id", ""),
        }
        for claim in claims_batch
    ]
    agent3_results = agent3_risk.run_risk_agent(agent3_inputs, user_history_by_id)
    agent3_by_id = {item["claim_id"]: item for item in agent3_results}

    agent1_by_id: dict[str, dict | None] = {}
    valid_image_ids_by_id: dict[str, list[str]] = {}
    for claim in claims_batch:
        claim_id = claim["_claim_id"]
        images = loader.load_images(claim.get("image_paths", ""), dataset_dir=dataset_dir)
        valid_image_ids_by_id[claim_id] = [image["image_id"] for image in images]
        agent2_context = agent2_by_id.get(claim_id)
        if not agent2_context:
            logger.warning("[%s] Agent 2 output missing; skipping Agent 1.", claim_id)
            agent1_by_id[claim_id] = None
            continue
        agent1_by_id[claim_id] = agent1_vision.run_vision_agent(
            images=images,
            claim_context={
                "sanitized_summary": agent2_context.get("sanitized_summary", ""),
                "object_part": agent2_context.get("object_part", "unknown"),
                "issue_type": agent2_context.get("issue_type", "unknown"),
                "claim_object": claim.get("claim_object", ""),
            },
            claim_id=claim_id,
        )

    agent4_inputs = []
    skipped_agent4_outputs = {}
    for claim in claims_batch:
        claim_id = claim["_claim_id"]
        claim_object = claim.get("claim_object", "")
        if agent1_by_id.get(claim_id) is None:
            logger.warning("[%s] Agent 1 failed; using safe defaults without calling Agent 4 for this claim.", claim_id)
            skipped_agent4_outputs[claim_id] = parser.ensure_manual_review_flag(
                parser.validate_output_row(
                    None,
                    claim_object=claim_object,
                    claim_id=claim_id,
                    valid_image_ids=valid_image_ids_by_id.get(claim_id, []),
                )
            )
            continue
        agent4_inputs.append(
            {
                "claim_id": claim_id,
                "claim_object": claim_object,
                "agent1": agent1_by_id.get(claim_id),
                "agent2": agent2_by_id.get(claim_id),
                "agent3": agent3_by_id.get(claim_id),
                "requirements": loader.get_requirements_for_object(evidence_requirements, claim_object),
                "valid_image_ids": valid_image_ids_by_id.get(claim_id, []),
            }
        )

    agent4_by_id = dict(skipped_agent4_outputs)
    if agent4_inputs:
        agent4_results = agent4_decision.run_decision_agent(agent4_inputs)
        if agent4_results is None:
            logger.warning("Agent 4 failed for batch starting %s; using safe defaults.", agent4_inputs[0]["claim_id"])
            agent4_results = [
                parser.ensure_manual_review_flag(
                    parser.validate_output_row(
                        None,
                        claim_object=record.get("claim_object", ""),
                        claim_id=record["claim_id"],
                        valid_image_ids=record.get("valid_image_ids", []),
                    )
                )
                for record in agent4_inputs
            ]
        for record, result in zip(agent4_inputs, agent4_results, strict=True):
            agent4_by_id[record["claim_id"]] = result

    final_rows = []
    for claim in claims_batch:
        claim_id = claim["_claim_id"]
        output_fields = agent4_by_id[claim_id]
        output_fields = _merge_agent_failure_flags(
            dict(output_fields),
            agent2_by_id.get(claim_id),
            agent3_by_id.get(claim_id),
            agent1_by_id.get(claim_id),
        )
        final = _input_fields(claim)
        final.update(output_fields)
        final_rows.append(final)
    return final_rows


def run_pipeline(
    claims: list[dict],
    user_history_by_id: dict[str, dict],
    evidence_requirements: dict[str, list[dict]],
    output_csv_path: str | Path,
    dataset_dir: str | Path | None = None,
) -> None:
    """Run the full pipeline and write completed rows incrementally."""
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        output_file.flush()

        for start in range(0, len(claims), BATCH_SIZE):
            raw_batch = claims[start : start + BATCH_SIZE]
            valid_claims = []
            immediate_rows: dict[int, dict] = {}

            for offset, claim in enumerate(raw_batch):
                absolute_index = start + offset
                claim["_claim_id"] = _claim_id(absolute_index)
                claim_object = str(claim.get("claim_object", "")).strip()
                if claim_object not in VALID_CLAIM_OBJECTS:
                    logger.warning(
                        "[%s] Invalid claim_object=%r; skipping all agents and using safe defaults.",
                        claim["_claim_id"],
                        claim.get("claim_object", ""),
                    )
                    immediate_rows[offset] = _safe_output_row(claim)
                else:
                    valid_claims.append(claim)

            processed_by_claim_id = {}
            if valid_claims:
                processed_rows = _process_valid_batch(
                    valid_claims,
                    user_history_by_id,
                    evidence_requirements,
                    dataset_dir=dataset_dir,
                )
                processed_by_claim_id = {
                    row.get("_claim_id", valid_claims[index]["_claim_id"]): row
                    for index, row in enumerate(processed_rows)
                }
                for claim, row in zip(valid_claims, processed_rows, strict=True):
                    processed_by_claim_id[claim["_claim_id"]] = row

            for offset, claim in enumerate(raw_batch):
                row = immediate_rows.get(offset)
                if row is None:
                    row = processed_by_claim_id[claim["_claim_id"]]
                _write_row(writer, row, output_file)


def process_dataset(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    user_history_by_id: dict[str, dict] | None = None,
    evidence_requirements: dict[str, list[dict]] | None = None,
    dataset_dir: str | Path | None = None,
) -> None:
    """Convenience wrapper that loads CSV inputs and runs the pipeline."""
    claims = loader.load_claims(input_csv_path)
    if user_history_by_id is None:
        user_history_by_id = loader.load_user_history()
    if evidence_requirements is None:
        evidence_requirements = loader.load_evidence_requirements()
    run_pipeline(
        claims=claims,
        user_history_by_id=user_history_by_id,
        evidence_requirements=evidence_requirements,
        output_csv_path=output_csv_path,
        dataset_dir=dataset_dir,
    )
