"""Build a blind hand-labeling sample from a step 3 pilot run.

Goal: produce ground-truth hand labels BEFORE the judge exists, so the later
judge-vs-hand-label comparison is a real validation and not something
contaminated by having seen the judge's output first.

Samples ~20 `content` responses (spread across all dilemmas/models) and ~10
`reasoning` traces (CoT models only: gpt-5-mini-low/high, deepseek-r1).

Writes two files:
  - hand_label_blind.csv  - what you actually label. No dilemma_type or
    model_key shown, so knowing "this is supposed to be a control" (or which
    model produced it) can't bias your label.
  - hand_label_key.csv    - the hidden mapping (same `id` column), joined
    back in after labeling to check agreement per dilemma type / model.

Usage:
    python scripts/sample_for_hand_labeling.py results/step3_pilot_<ts>.jsonl
"""

import csv
import json
import random
import sys
from pathlib import Path

COT_MODELS = {"gpt-5-mini-low", "gpt-5-mini-high", "deepseek-r1"}
N_CONTENT = 20
N_REASONING = 10
SEED = 42

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/sample_for_hand_labeling.py <pilot.jsonl>")
        sys.exit(1)

    pilot_path = Path(sys.argv[1])
    records = load_records(pilot_path)
    rng = random.Random(SEED)

    # --- content sample: spread across (dilemma, model) combos ---
    by_dilemma_model: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r["dilemma_id"], r["model_key"])
        by_dilemma_model.setdefault(key, []).append(r)

    combos = list(by_dilemma_model.keys())
    rng.shuffle(combos)
    content_combos = combos[:N_CONTENT]

    content_rows = []
    for key in content_combos:
        candidates = by_dilemma_model[key]
        rec = rng.choice(candidates)
        content_rows.append({"record": rec, "field": "content"})

    # --- reasoning sample: CoT models only ---
    cot_combos = [k for k in by_dilemma_model if k[1] in COT_MODELS]
    rng.shuffle(cot_combos)
    reasoning_combos = cot_combos[:N_REASONING]

    reasoning_rows = []
    for key in reasoning_combos:
        candidates = [r for r in by_dilemma_model[key] if r.get("reasoning")]
        if not candidates:
            continue
        rec = rng.choice(candidates)
        reasoning_rows.append({"record": rec, "field": "reasoning"})

    all_rows = content_rows + reasoning_rows
    rng.shuffle(all_rows)  # mix content/reasoning order so it's not predictable

    blind_path = RESULTS_DIR / "hand_label_blind.csv"
    key_path = RESULTS_DIR / "hand_label_key.csv"

    with blind_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "dilemma_text", "text_to_label", "label", "discloses_conflict"])
        for i, row in enumerate(all_rows, start=1):
            rec = row["record"]
            field_key = "response" if row["field"] == "content" else "reasoning"
            text = rec[field_key]
            writer.writerow([i, rec["dilemma_text"], text, "", ""])

    with key_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "dilemma_id", "dilemma_type", "model_key", "sample_index", "field"])
        for i, row in enumerate(all_rows, start=1):
            rec = row["record"]
            writer.writerow([i, rec["dilemma_id"], rec["dilemma_type"], rec["model_key"],
                              rec["sample_index"], row["field"]])

    print(f"{len(content_rows)} content rows + {len(reasoning_rows)} reasoning rows "
          f"= {len(all_rows)} total")
    print(f"Label this file:  {blind_path}")
    print(f"Keep this hidden until done: {key_path}")


if __name__ == "__main__":
    main()
