"""
agent3_risk.py - User history risk agent.

Runs second in the pipeline. It evaluates user history for risk context in
batches of up to 10, while skipping API calls entirely for claims whose user_id
has no history row.
"""

from __future__ import annotations

import logging

from groq import Groq

from parser import ALLOWED_RISK_FLAGS, parse_json_response
from ratelimiter import call_with_rate_limit

logger = logging.getLogger(__name__)

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_BATCH_SIZE = 10


def _no_history_result(claim: dict) -> dict:
    return {
        "claim_id": claim.get("claim_id", "unknown"),
        "user_id": claim.get("user_id", ""),
        "risk_flags": "",
        "risk_reason": "no history available",
    }


def _default_history_result(claim: dict, reason: str) -> dict:
    return {
        "claim_id": claim.get("claim_id", "unknown"),
        "user_id": claim.get("user_id", ""),
        "risk_flags": "manual_review_required",
        "risk_reason": reason,
    }


def _build_prompt(history_items: list[dict]) -> str:
    records_text = []
    for index, item in enumerate(history_items, start=1):
        history = item["history"]
        records_text.append(
            f"""--- HISTORY {index} ---
claim_id: {item.get("claim_id", "unknown")}
user_id: {item.get("user_id", "")}
past_claim_count: {history.get("past_claim_count", "0")}
accept_claim: {history.get("accept_claim", "0")}
manual_review_claim: {history.get("manual_review_claim", "0")}
rejected_claim: {history.get("rejected_claim", "0")}
last_90_days_claim_count: {history.get("last_90_days_claim_count", "0")}
history_flags: {history.get("history_flags", "none")}
history_summary: {history.get("history_summary", "")}
"""
        )

    return f"""You are Agent 3, the user history risk agent.

Assess whether each user's history adds relevant claim risk. User history is
context only; it must never decide visual support or contradiction.

Allowed risk_flags values for this agent:
- none
- user_history_risk
- manual_review_required

Use "none" when there is no history risk. Use
"user_history_risk;manual_review_required" only when the history suggests
recent frequency, repeated rejection, exaggeration, authenticity problems, or
other review-worthy risk.

Return raw JSON only, with this exact shape:
{{
  "results": [
    {{
      "claim_id": "<same claim_id>",
      "user_id": "<same user_id>",
      "risk_flags": "<none or semicolon-separated allowed flags>",
      "risk_reason": "<short reason>"
    }}
  ]
}}

History records:
{''.join(records_text)}
"""


def _call_groq(client: Groq, model_name: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _clean_flags(raw_flags: object) -> str:
    text = str(raw_flags or "none").strip().lower()
    if not text or text in {"none", "null", "n/a", "na"}:
        return "none"
    allowed = {"user_history_risk", "manual_review_required"}
    tokens = [token.strip() for token in text.split(";") if token.strip()]
    cleaned = [token for token in tokens if token in allowed and token in ALLOWED_RISK_FLAGS]
    return ";".join(dict.fromkeys(cleaned)) if cleaned else "none"


def _clean_result(raw: dict, item: dict) -> dict:
    reason = str(raw.get("risk_reason", "")).strip()
    if not reason:
        reason = "No notable user history risk."
    return {
        "claim_id": item.get("claim_id", raw.get("claim_id", "unknown")),
        "user_id": item.get("user_id", raw.get("user_id", "")),
        "risk_flags": _clean_flags(raw.get("risk_flags", "none")),
        "risk_reason": reason,
    }


def run_risk_agent(
    claims_batch: list[dict],
    user_history_by_id: dict[str, dict],
    model_name: str = DEFAULT_GROQ_MODEL,
) -> list[dict]:
    """Evaluate history risk for a batch of up to 10 claims."""
    if not claims_batch:
        return []
    if len(claims_batch) > MAX_BATCH_SIZE:
        raise ValueError("agent3_risk batches must contain at most 10 claims")

    results_by_claim_id = {}
    history_items = []
    for claim in claims_batch:
        user_id = claim.get("user_id", "")
        history = user_history_by_id.get(user_id)
        if history is None:
            result = _no_history_result(claim)
            results_by_claim_id[result["claim_id"]] = result
            continue
        history_items.append(
            {
                "claim_id": claim.get("claim_id", "unknown"),
                "user_id": user_id,
                "history": history,
            }
        )

    if history_items:
        first_claim_id = history_items[0].get("claim_id", "unknown")
        prompt = _build_prompt(history_items)
        try:
            client = Groq()
        except Exception as exc:
            logger.warning("Agent 3 Groq client initialization failed: %s", exc)
            for item in history_items:
                result = _default_history_result(item, "Unable to evaluate user history automatically.")
                results_by_claim_id[result["claim_id"]] = result
        else:
            raw_text = call_with_rate_limit(
                "groq",
                _call_groq,
                client,
                model_name,
                prompt,
                claim_id=f"agent3_{first_claim_id}",
            )
            parsed = parse_json_response(raw_text, claim_id=f"agent3_{first_claim_id}")
            if parsed is None:
                for item in history_items:
                    result = _default_history_result(item, "Unable to evaluate user history automatically.")
                    results_by_claim_id[result["claim_id"]] = result
            else:
                raw_by_id = {
                    str(item.get("claim_id", "")): item
                    for item in parsed.get("results", [])
                    if isinstance(item, dict)
                }
                for item in history_items:
                    claim_id = str(item.get("claim_id", "unknown"))
                    result = _clean_result(raw_by_id.get(claim_id, {}), item)
                    results_by_claim_id[result["claim_id"]] = result

    return [
        results_by_claim_id.get(claim.get("claim_id", "unknown"), _default_history_result(claim, "Missing risk result."))
        for claim in claims_batch
    ]
