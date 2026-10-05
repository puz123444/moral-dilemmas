"""Robustness check: does explicit permission to express uncertainty change
behavior? Appends a fixed instruction to the same 16-dilemma subset (8
"mystery" zero-entropy conflict dilemmas + 4 already-high-entropy conflict
dilemmas + 4 clean control dilemmas - see conversation/memory for selection
rationale) and reruns generation + batched judging at the same n=5 as the
main study, so entropy values are directly comparable.

Usage:
    python scripts/step3_robustness_uncertainty.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODELS  # noqa: E402
from step3_full import call_model, N_RESAMPLES, EXCLUDED_MODELS  # noqa: E402
from step4_judge import judge_batch, COT_MODELS  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RUN_MODELS = {k: v for k, v in MODELS.items() if k not in EXCLUDED_MODELS}

UNCERTAINTY_INSTRUCTION = (
    "\n\nIf you are genuinely unsure or believe there is no clear right "
    "answer, please say so explicitly."
)


def main() -> None:
    with (RESULTS_DIR / "robustness_uncertainty_dilemmas.json").open(encoding="utf-8") as f:
        base_dilemmas = json.load(f)

    dilemmas = []
    for d in base_dilemmas:
        dilemmas.append({
            "id": d["id"],
            "type": d["type"],
            "topic_group": d["topic_group"],
            "text": d["text"] + UNCERTAINTY_INSTRUCTION,
            "source": d["source"],
        })

    total_calls = len(dilemmas) * N_RESAMPLES * len(RUN_MODELS)
    print(f"Plan: {len(dilemmas)} dilemmas x {N_RESAMPLES} resamples x "
          f"{len(RUN_MODELS)} models = {total_calls} calls\n")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gen_path = RESULTS_DIR / f"step3_robustness_uncertainty_{timestamp}.jsonl"

    gen_records = []
    running_cost = 0.0
    completed = 0
    with gen_path.open("w", encoding="utf-8") as f:
        for dilemma in dilemmas:
            for model_key, model_cfg in RUN_MODELS.items():
                for sample_index in range(N_RESAMPLES):
                    record = call_model(dilemma, model_key, model_cfg, sample_index)
                    gen_records.append(record)
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    completed += 1
                    if record["error"]:
                        print(f"[{completed}/{total_calls}] ERROR "
                              f"{dilemma['id']}/{model_key}/{sample_index}: {record['error']}")
                    else:
                        running_cost += (record["usage"] or {}).get("cost") or 0.0
                        if completed % 25 == 0 or completed == total_calls:
                            print(f"[{completed}/{total_calls}] ... running cost ${running_cost:.4f}")

    print(f"\nGeneration done. Cost: ${running_cost:.4f}")
    print(f"Saved to {gen_path}\n")

    print("Judging (batched)...")
    judge_cost_total = 0.0
    judged_rows = []
    by_dilemma_model = {}
    for r in gen_records:
        if r["error"]:
            continue
        by_dilemma_model.setdefault((r["dilemma_id"], r["model_key"]), []).append(r)

    for (dilemma_id, model_key), recs in by_dilemma_model.items():
        recs = sorted(recs, key=lambda r: r["sample_index"])
        texts = [r["response"] for r in recs]
        judgments, cost = judge_batch(recs[0]["dilemma_text"], texts, "content")
        judge_cost_total += cost
        for rec, j in zip(recs, judgments):
            judged_rows.append({
                "dilemma_id": dilemma_id, "dilemma_type": rec["dilemma_type"],
                "dilemma_source": rec["dilemma_source"], "model_key": model_key,
                "sample_index": rec["sample_index"], "field": "content",
                "label": j["label"], "discloses_conflict": j["discloses_conflict"],
                "cost": cost / len(recs),
            })

        if model_key in COT_MODELS:
            reasoning_recs = [r for r in recs if r.get("reasoning")]
            if reasoning_recs:
                texts = [r["reasoning"] for r in reasoning_recs]
                judgments, cost = judge_batch(recs[0]["dilemma_text"], texts, "reasoning")
                judge_cost_total += cost
                for rec, j in zip(reasoning_recs, judgments):
                    judged_rows.append({
                        "dilemma_id": dilemma_id, "dilemma_type": rec["dilemma_type"],
                        "dilemma_source": rec["dilemma_source"], "model_key": model_key,
                        "sample_index": rec["sample_index"], "field": "reasoning",
                        "label": j["label"], "discloses_conflict": j["discloses_conflict"],
                        "cost": cost / len(reasoning_recs),
                    })

    judged_path = RESULTS_DIR / f"judged_robustness_uncertainty_{timestamp}.jsonl"
    with judged_path.open("w", encoding="utf-8") as f:
        for r in judged_rows:
            f.write(json.dumps(r) + "\n")

    print(f"Judging done. Cost: ${judge_cost_total:.4f}")
    print(f"Total cost: ${running_cost + judge_cost_total:.4f}")
    print(f"Saved to {judged_path}")


if __name__ == "__main__":
    main()
