"""Resume judging for the uncertainty-permission robustness run after the
judge crashed on a soft-error (fixed in step4_judge.py). Reads the already-
saved generation file so the $1.00 already spent on generation isn't wasted.

Writes incrementally (a crash only loses the unfinished tail) and falls back
to per-item judging if a batched call persistently fails for one group,
rather than losing that group's data entirely.

Usage:
    python scripts/resume_judge_robustness.py <gen_file.jsonl>
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from step4_judge import judge_batch, judge_one, COT_MODELS  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def judge_group(dilemma_text: str, recs: list[dict], field: str) -> tuple[list[dict], float]:
    texts = [r["response"] if field == "content" else r["reasoning"] for r in recs]
    try:
        judgments, cost = judge_batch(dilemma_text, texts, field)
        return judgments, cost
    except Exception as exc:  # noqa: BLE001
        print(f"    batched judge failed ({exc}), falling back to per-item judging...")
        judgments = []
        total_cost = 0.0
        for text in texts:
            j = judge_one(dilemma_text, text, field)
            judgments.append({"label": j["label"], "discloses_conflict": j["discloses_conflict"]})
            total_cost += j["cost"]
        return judgments, total_cost


def main() -> None:
    gen_path = Path(sys.argv[1])
    with gen_path.open(encoding="utf-8") as f:
        gen_records = [json.loads(l) for l in f if l.strip()]

    print(f"Loaded {len(gen_records)} generation records from {gen_path.name}")

    by_dilemma_model = {}
    for r in gen_records:
        if r["error"]:
            continue
        by_dilemma_model.setdefault((r["dilemma_id"], r["model_key"]), []).append(r)

    print(f"Judging {len(by_dilemma_model)} (dilemma, model) groups...\n")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    judged_path = RESULTS_DIR / f"judged_robustness_uncertainty_{timestamp}.jsonl"

    judge_cost_total = 0.0
    skipped = []
    with judged_path.open("w", encoding="utf-8") as f:
        for i, ((dilemma_id, model_key), recs) in enumerate(by_dilemma_model.items()):
            recs = sorted(recs, key=lambda r: r["sample_index"])
            dilemma_text = recs[0]["dilemma_text"]

            try:
                judgments, cost = judge_group(dilemma_text, recs, "content")
                judge_cost_total += cost
                for rec, j in zip(recs, judgments):
                    row = {
                        "dilemma_id": dilemma_id, "dilemma_type": rec["dilemma_type"],
                        "dilemma_source": rec["dilemma_source"], "model_key": model_key,
                        "sample_index": rec["sample_index"], "field": "content",
                        "label": j["label"], "discloses_conflict": j["discloses_conflict"],
                        "cost": cost / len(recs),
                    }
                    f.write(json.dumps(row) + "\n")
                f.flush()
            except Exception as exc:  # noqa: BLE001
                print(f"  SKIPPING content for {dilemma_id}/{model_key}: {exc}")
                skipped.append((dilemma_id, model_key, "content"))

            if model_key in COT_MODELS:
                reasoning_recs = [r for r in recs if r.get("reasoning")]
                if reasoning_recs:
                    try:
                        judgments, cost = judge_group(dilemma_text, reasoning_recs, "reasoning")
                        judge_cost_total += cost
                        for rec, j in zip(reasoning_recs, judgments):
                            row = {
                                "dilemma_id": dilemma_id, "dilemma_type": rec["dilemma_type"],
                                "dilemma_source": rec["dilemma_source"], "model_key": model_key,
                                "sample_index": rec["sample_index"], "field": "reasoning",
                                "label": j["label"], "discloses_conflict": j["discloses_conflict"],
                                "cost": cost / len(reasoning_recs),
                            }
                            f.write(json.dumps(row) + "\n")
                        f.flush()
                    except Exception as exc:  # noqa: BLE001
                        print(f"  SKIPPING reasoning for {dilemma_id}/{model_key}: {exc}")
                        skipped.append((dilemma_id, model_key, "reasoning"))

            print(f"[{i + 1}/{len(by_dilemma_model)}] {dilemma_id}/{model_key} -> "
                  f"running judge cost ${judge_cost_total:.4f}")

    print(f"\nJudging done. Cost: ${judge_cost_total:.4f}")
    if skipped:
        print(f"Skipped {len(skipped)} groups after fallback also failed: {skipped}")
    print(f"Saved to {judged_path}")


if __name__ == "__main__":
    main()
