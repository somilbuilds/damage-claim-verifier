# Damage Claim Evidence Pipeline

This solution uses a fixed four-agent sequence:

1. `agent2_claim.py` extracts the claimed `object_part`, `issue_type`, prompt-injection flag, and `sanitized_summary` from `user_claim` conversations using Groq `llama-3.3-70b-versatile` in batches of 10.
2. `agent3_risk.py` evaluates user history with Groq `llama-3.3-70b-versatile` in batches of 10. Claims with no `user_history.csv` row skip the API call and return `risk_reason="no history available"`.
3. `agent1_vision.py` inspects loaded images one claim at a time using Gemini `gemini-3.1-flash-lite`, with Groq `llama-4-scout-17b-16e-instruct` as the vision fallback. It receives only Agent 2's sanitized summary, object part, and issue type.
4. `agent4_decision.py` combines the three agent outputs with evidence requirements, calls Groq `llama-3.3-70b-versatile` in batches of 10, and validates final rows with `parser.validate_output_row`.

`orchestrator.py` validates `claim_object` before any agent call. Invalid objects skip all agents, receive parser safe defaults plus `manual_review_required`, and are still written with the original input value. Completed rows are written incrementally to the output CSV.

## Running Predictions

From the repository root:

```bash
python code/main.py
```

This loads `.env`, reads `dataset/claims.csv`, `dataset/user_history.csv`, and `dataset/evidence_requirements.csv`, then writes `output.csv` at the repo root.

## Running Evaluation

From the repository root:

```bash
python code/evaluation/main.py
```

This runs the same pipeline on `dataset/sample_claims.csv`, writes `code/evaluation/sample_predictions.csv`, prints per-field accuracy for the labeled fields, and saves `code/evaluation/eval_results.json`.

## Environment Variables

Create a local `.env` file or export these variables:

```text
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

Secrets must come from environment variables only.
