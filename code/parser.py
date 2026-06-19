"""
parser.py — JSON response parsing and field validation for all agents.

Strips markdown code fences, parses JSON, validates every field against
the allowed-values lists from the problem statement. Invalid or missing
fields get replaced with the safest fallback value.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed values per field (from problem_statement.md)
# ---------------------------------------------------------------------------

ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ALLOWED_ISSUE_TYPE = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
}

ALLOWED_OBJECT_PART = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender",
        "quarter_panel", "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid",
        "corner", "port", "base", "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label",
        "contents", "item", "unknown",
    },
}

ALLOWED_RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}

ALLOWED_SEVERITY = {"none", "low", "medium", "high", "unknown"}

ALLOWED_BOOL = {"true", "false"}

ALLOWED_CLAIM_OBJECTS = {"car", "laptop", "package"}

# ---------------------------------------------------------------------------
# Safe defaults — used when an agent returns None, bad JSON, or invalid fields
# ---------------------------------------------------------------------------

SAFE_DEFAULTS = {
    "evidence_standard_met": "false",
    "evidence_standard_met_reason": "Unable to verify evidence automatically.",
    "risk_flags": "manual_review_required",
    "issue_type": "unknown",
    "object_part": "unknown",
    "claim_status": "not_enough_information",
    "claim_status_justification": "Unable to process claim automatically.",
    "supporting_image_ids": "none",
    "valid_image": "false",
    "severity": "unknown",
}


def build_safe_default_row() -> dict:
    """Return a fresh copy of all safe-default output fields."""
    return dict(SAFE_DEFAULTS)


def validate_claim_object(claim_object: str | None, claim_id: str = "unknown") -> bool:
    """Return True if claim_object is valid (car/laptop/package).

    Logs a warning and returns False if missing or invalid.
    The orchestrator should build a safe-default row when this returns False.
    """
    if not claim_object or claim_object.strip().lower() not in ALLOWED_CLAIM_OBJECTS:
        logger.warning(
            "[%s] Invalid claim_object=%r. Expected one of %s. "
            "Row will use safe defaults.",
            claim_id, claim_object, ALLOWED_CLAIM_OBJECTS,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Markdown fence stripping
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)


def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers if present, return inner content."""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_response(
    raw_text: str | None,
    claim_id: str = "unknown",
) -> dict | None:
    """Parse an agent's raw text response into a dict.

    Returns None if the text is empty, None, or unparseable JSON.
    Logs a warning on failure with the claim_id for traceability.
    """
    if not raw_text:
        logger.warning(
            "[%s] JSON parse failure: empty or None response.", claim_id
        )
        return None

    cleaned = strip_markdown_fences(raw_text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(
            "[%s] JSON parse failure: %s. Raw (first 300 chars): %.300s",
            claim_id, e, cleaned,
        )
        return None

    if not isinstance(parsed, dict):
        logger.warning(
            "[%s] JSON parse failure: expected dict, got %s.",
            claim_id, type(parsed).__name__,
        )
        return None

    return parsed


# ---------------------------------------------------------------------------
# Field validators
# ---------------------------------------------------------------------------

def _validate_enum(
    data: dict,
    field: str,
    allowed: set[str],
    fallback: str,
    claim_id: str,
) -> str:
    """Validate a single enum field. Return valid value or fallback."""
    raw = data.get(field, "").strip().lower()
    if raw in allowed:
        return raw
    if raw:
        logger.warning(
            "[%s] Invalid %s=%r, falling back to %r.", claim_id, field, raw, fallback
        )
    else:
        logger.warning(
            "[%s] Missing %s, falling back to %r.", claim_id, field, fallback
        )
    return fallback


def _validate_bool_field(
    data: dict,
    field: str,
    fallback: str,
    claim_id: str,
) -> str:
    """Validate a boolean string field (true/false)."""
    raw = str(data.get(field, "")).strip().lower()
    if raw in ALLOWED_BOOL:
        return raw
    logger.warning(
        "[%s] Invalid %s=%r, falling back to %r.", claim_id, field, raw, fallback
    )
    return fallback


def _validate_semicolon_list(
    data: dict,
    field: str,
    allowed: set[str] | None,
    claim_id: str,
) -> str:
    """Validate a semicolon-separated field. Drop invalid tokens.

    Returns "none" if result is empty after filtering.
    If allowed is None, accept any non-empty token (used for image IDs).
    """
    raw = str(data.get(field, "")).strip().lower()
    if not raw or raw in ("none", "null", "n/a", "na", ""):
        return "none"

    tokens = [t.strip() for t in raw.split(";") if t.strip()]

    if allowed is not None:
        valid = []
        for t in tokens:
            if t in allowed:
                valid.append(t)
            elif t not in ("none",):
                logger.warning(
                    "[%s] Invalid token %r in %s, dropping.", claim_id, t, field
                )
        tokens = valid

    if not tokens:
        return "none"

    return ";".join(tokens)


def _validate_free_text(
    data: dict,
    field: str,
    fallback: str,
    claim_id: str,
) -> str:
    """Validate a free-text field (justification / reason). Must be non-empty."""
    raw = str(data.get(field, "")).strip()
    if raw and raw.lower() not in ("none", "null", "n/a", ""):
        return raw
    logger.warning(
        "[%s] Missing %s, falling back to default.", claim_id, field
    )
    return fallback


# ---------------------------------------------------------------------------
# Full row validation
# ---------------------------------------------------------------------------

def validate_output_row(
    data: dict | None,
    claim_object: str,
    claim_id: str = "unknown",
    valid_image_ids: list[str] | None = None,
) -> dict:
    """Validate and clean a complete output row from agent 4.

    Args:
        data: Parsed JSON dict from the decision agent (or None on failure).
        claim_object: "car", "laptop", or "package" — needed for object_part validation.
        claim_id: Row identifier for log messages.
        valid_image_ids: Actual image IDs available for this claim row.
            supporting_image_ids are cross-checked against this list.

    Returns:
        A dict with all 10 output-only fields validated and safe.
        (The 4 input fields — user_id, image_paths, user_claim, claim_object —
         are added by the orchestrator, not here.)
    """
    if data is None:
        logger.warning(
            "[%s] Full row fallback to safe defaults (no data).", claim_id
        )
        return build_safe_default_row()

    # Get the allowed object_part set for this claim_object.
    part_allowed = ALLOWED_OBJECT_PART.get(claim_object, set())
    # Always allow "unknown" as fallback.
    part_allowed = part_allowed | {"unknown"}

    row = {}

    row["evidence_standard_met"] = _validate_bool_field(
        data, "evidence_standard_met", "false", claim_id
    )
    row["evidence_standard_met_reason"] = _validate_free_text(
        data, "evidence_standard_met_reason",
        SAFE_DEFAULTS["evidence_standard_met_reason"], claim_id
    )
    row["risk_flags"] = _validate_semicolon_list(
        data, "risk_flags", ALLOWED_RISK_FLAGS, claim_id
    )
    row["issue_type"] = _validate_enum(
        data, "issue_type", ALLOWED_ISSUE_TYPE, "unknown", claim_id
    )
    row["object_part"] = _validate_enum(
        data, "object_part", part_allowed, "unknown", claim_id
    )
    row["claim_status"] = _validate_enum(
        data, "claim_status", ALLOWED_CLAIM_STATUS,
        "not_enough_information", claim_id
    )
    row["claim_status_justification"] = _validate_free_text(
        data, "claim_status_justification",
        SAFE_DEFAULTS["claim_status_justification"], claim_id
    )
    # Cross-check supporting_image_ids against the actual images for this claim.
    image_id_allowed = set(valid_image_ids) | {"none"} if valid_image_ids else None
    row["supporting_image_ids"] = _validate_semicolon_list(
        data, "supporting_image_ids", image_id_allowed, claim_id
    )
    row["valid_image"] = _validate_bool_field(
        data, "valid_image", "false", claim_id
    )
    row["severity"] = _validate_enum(
        data, "severity", ALLOWED_SEVERITY, "unknown", claim_id
    )

    return row


def ensure_manual_review_flag(row: dict) -> dict:
    """Append manual_review_required to risk_flags if not already present.

    Used when a claim falls back to safe defaults due to API or parse failure.
    """
    flags = row.get("risk_flags", "none")
    if "manual_review_required" not in flags:
        if flags == "none":
            flags = "manual_review_required"
        else:
            flags = flags + ";manual_review_required"
        row["risk_flags"] = flags
    return row
