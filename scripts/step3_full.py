"""Step 3 full run - 50 conflict + 10 control dilemmas x 5 resamples x 5 models.

Scaled-up version of step3_pilot.py, incorporating what the pilot taught us:
- gpt-4o-mini dropped (config.py MODELS still lists it; excluded here to save
  cost - see PROJECT.md discussion, it was always marked optional).
- Same temperature=1.0, same reasoning-field capture for CoT models.
- Same per-call error handling (log and continue) plus a rate-limit retry,
  since a run this size is far more likely to hit a transient error somewhere.
- Incremental JSONL writes so a crash only costs the unfinished tail.
- A running cost tracker with a hard safety cap well above the ~$9.54
  estimate, in case per-call costs come in higher than the pilot suggested.

Run from the project root with the venv active:
    python scripts/step3_full.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai  # noqa: E402

from config import client, MODELS  # noqa: E402

EXCLUDED_MODELS = {"gpt-4o-mini"}  # qwen's provider outage was resolved and
# backfilled separately (scripts/step3_qwen_backfill.py) - all 5 models are
# in scope again.
RUN_MODELS = {k: v for k, v in MODELS.items() if k not in EXCLUDED_MODELS}

TEMPERATURE = 1.0
N_RESAMPLES = 5
COST_CAP_USD = 15.00  # ~1.6x the $9.54 estimate, as a safety net
MAX_RETRIES = 5

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def load_dilemmas() -> list[dict]:
    with (RESULTS_DIR / "conflict_candidates_50.json").open(encoding="utf-8") as f:
        conflict = json.load(f)
    with (RESULTS_DIR / "control_candidates_10.json").open(encoding="utf-8") as f:
        control = json.load(f)

    dilemmas = []
    for row in conflict:
        dilemmas.append({
            "id": f"dd_{row['dilemma_idx']}",
            "type": "conflict",
            "topic_group": row["topic_group"],
            "text": row["text"],
            "source": f"daily_dilemmas:{row['dilemma_idx']}",
        })
    for row in control:
        dilemmas.append({
            "id": f"control_{row['dilemma_idx']}",
            "type": "control",
            "topic_group": row["topic_group"],
            "text": row["text"],
            "source": row.get("source", f"daily_dilemmas:{row['dilemma_idx']}"),
        })
    return dilemmas


def call_model(dilemma: dict, model_key: str, model_cfg: dict, sample_index: int) -> dict:
    record = {
        "dilemma_id": dilemma["id"],
        "dilemma_type": dilemma["type"],
        "topic_group": dilemma["topic_group"],
        "dilemma_source": dilemma["source"],
        "dilemma_text": dilemma["text"],
        "model_key": model_key,
        "slug": model_cfg["slug"],
        "sample_index": sample_index,
        "temperature": TEMPERATURE,
        "extra_body": model_cfg["extra_body"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_cfg["slug"],
                messages=[{"role": "user", "content": dilemma["text"]}],
                temperature=TEMPERATURE,
                extra_body=model_cfg["extra_body"],
            )

            # OpenRouter sometimes returns HTTP 200 with a soft error payload
            # (choices=None, error={code, message}) instead of raising - e.g.
            # a transient 522 gateway timeout from the upstream provider.
            # Treat that the same as a retryable transport error.
            soft_error = getattr(response, "error", None)
            if soft_error or not response.choices:
                raise RuntimeError(f"Provider soft error: {soft_error}")

            record["response"] = response.choices[0].message.content
            record["reasoning"] = getattr(response.choices[0].message, "reasoning", None)
            record["usage"] = response.usage.model_dump() if response.usage else None
            record["error"] = None
            return record
        except openai.RateLimitError:
            wait = 10 * (attempt + 1)
            print(f"    rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
        except Exception as exc:  # noqa: BLE001 - retry transient errors, then give up
            wait = 5 * (attempt + 1)
            print(f"    error ({exc}), retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
            record["error"] = str(exc)

    record["response"] = None
    record["reasoning"] = None
    record["usage"] = None
    record["error"] = record.get("error") or "Exceeded max retries, no error captured"
    return record


def main() -> None:
    dilemmas = load_dilemmas()
    total_calls = len(dilemmas) * N_RESAMPLES * len(RUN_MODELS)
    print(f"Plan: {len(dilemmas)} dilemmas x {N_RESAMPLES} resamples x "
          f"{len(RUN_MODELS)} models = {total_calls} calls")
    print(f"Models: {list(RUN_MODELS.keys())}\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"step3_full_{timestamp}.jsonl"

    running_cost = 0.0
    completed = 0
    errors = 0

    with out_path.open("w", encoding="utf-8") as f:
        for dilemma in dilemmas:
            for model_key, model_cfg in RUN_MODELS.items():
                for sample_index in range(N_RESAMPLES):
                    record = call_model(dilemma, model_key, model_cfg, sample_index)
                    f.write(json.dumps(record) + "\n")
                    f.flush()

                    completed += 1
                    if record["error"]:
                        errors += 1
                        print(f"[{completed}/{total_calls}] ERROR "
                              f"{dilemma['id']}/{model_key}/{sample_index}: {record['error']}")
                    else:
                        cost = (record["usage"] or {}).get("cost") or 0.0
                        running_cost += cost
                        if completed % 25 == 0 or completed == total_calls:
                            print(f"[{completed}/{total_calls}] ... "
                                  f"running cost ${running_cost:.4f}")

                    if running_cost > COST_CAP_USD:
                        print(f"\nSTOPPING: running cost ${running_cost:.2f} "
                              f"exceeded cap ${COST_CAP_USD:.2f}")
                        print(f"Partial results saved to {out_path}")
                        return

    print(f"\nDone. {completed - errors}/{completed} calls succeeded.")
    print(f"Total cost: ${running_cost:.4f}")
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
