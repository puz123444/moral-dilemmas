"""One-off: generate + judge a single replacement control dilemma, then splice
it into the existing combined generation and judged files in place of the
dilemma it's replacing.

Usage:
    python scripts/swap_control_dilemma.py <old_dilemma_id> <new_dilemma_id> <new_dilemma_text>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODELS  # noqa: E402
from step3_full import call_model, N_RESAMPLES, EXCLUDED_MODELS  # noqa: E402
from step4_judge import judge_batch, COT_MODELS  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
COMBINED_GEN_PATH = RESULTS_DIR / "step3_full_combined.jsonl"
COMBINED_JUDGED_PATH = RESULTS_DIR / "judged_pilot_batched_step3_full_combined.jsonl"

RUN_MODELS = {k: v for k, v in MODELS.items() if k not in EXCLUDED_MODELS}


def main() -> None:
    old_id, new_id_raw, new_text = sys.argv[1], sys.argv[2], sys.argv[3]
    new_dilemma_id = f"control_{new_id_raw}"

    dilemma = {
        "id": new_dilemma_id,
        "type": "control",
        "topic_group": "family",
        "text": new_text,
        "source": f"daily_dilemmas:{new_id_raw}",
    }

    print(f"Generating {new_dilemma_id} across {list(RUN_MODELS.keys())}...")
    gen_records = []
    for model_key, model_cfg in RUN_MODELS.items():
        for sample_index in range(N_RESAMPLES):
            record = call_model(dilemma, model_key, model_cfg, sample_index)
            gen_records.append(record)
            status = "ERROR" if record["error"] else "ok"
            print(f"  {model_key}/{sample_index}: {status}")

    gen_cost = sum((r["usage"] or {}).get("cost") or 0.0 for r in gen_records if not r["error"])
    print(f"Generation done. Cost: ${gen_cost:.4f}\n")

    print("Judging (batched)...")
    judged_rows = []
    judge_cost_total = 0.0
    by_model = {}
    for r in gen_records:
        by_model.setdefault(r["model_key"], []).append(r)

    for model_key, recs in by_model.items():
        recs = sorted(recs, key=lambda r: r["sample_index"])
        texts = [r["response"] for r in recs]
        judgments, cost = judge_batch(dilemma["text"], texts, "content")
        judge_cost_total += cost
        for rec, j in zip(recs, judgments):
            judged_rows.append({
                "dilemma_id": new_dilemma_id, "dilemma_type": "control",
                "dilemma_source": dilemma["source"], "model_key": model_key,
                "sample_index": rec["sample_index"], "field": "content",
                "label": j["label"], "discloses_conflict": j["discloses_conflict"],
                "cost": cost / len(recs),
            })

        if model_key in COT_MODELS:
            reasoning_recs = [r for r in recs if r.get("reasoning")]
            if reasoning_recs:
                texts = [r["reasoning"] for r in reasoning_recs]
                judgments, cost = judge_batch(dilemma["text"], texts, "reasoning")
                judge_cost_total += cost
                for rec, j in zip(reasoning_recs, judgments):
                    judged_rows.append({
                        "dilemma_id": new_dilemma_id, "dilemma_type": "control",
                        "dilemma_source": dilemma["source"], "model_key": model_key,
                        "sample_index": rec["sample_index"], "field": "reasoning",
                        "label": j["label"], "discloses_conflict": j["discloses_conflict"],
                        "cost": cost / len(reasoning_recs),
                    })

    print(f"Judging done. Cost: ${judge_cost_total:.4f}")
    print(f"Total cost: ${gen_cost + judge_cost_total:.4f}\n")

    # --- Splice into the combined generation file ---
    with COMBINED_GEN_PATH.open(encoding="utf-8") as f:
        gen_all = [json.loads(l) for l in f if l.strip()]
    old_full_id = f"control_{old_id}"
    gen_all = [r for r in gen_all if r["dilemma_id"] != old_full_id]
    gen_all.extend(gen_records)
    with COMBINED_GEN_PATH.open("w", encoding="utf-8") as f:
        for r in gen_all:
            f.write(json.dumps(r) + "\n")
    print(f"Spliced into {COMBINED_GEN_PATH} ({len(gen_all)} total records)")

    # --- Splice into the combined judged file ---
    with COMBINED_JUDGED_PATH.open(encoding="utf-8") as f:
        judged_all = [json.loads(l) for l in f if l.strip()]
    judged_all = [r for r in judged_all if r["dilemma_id"] != old_full_id]
    judged_all.extend(judged_rows)
    with COMBINED_JUDGED_PATH.open("w", encoding="utf-8") as f:
        for r in judged_all:
            f.write(json.dumps(r) + "\n")
    print(f"Spliced into {COMBINED_JUDGED_PATH} ({len(judged_all)} total records)")


if __name__ == "__main__":
    main()
