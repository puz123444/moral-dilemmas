"""Backfill qwen-2.5-7b for the full run.

qwen's only OpenRouter provider (Phala) was returning 525 errors when the
main step3_full.py run started, so it was temporarily excluded (see that
script's EXCLUDED_MODELS comment). Provider recovered later - this script
runs just qwen over the same 60 dilemmas x 5 resamples, producing a JSONL in
the same record format so it can be concatenated with the main run's output.

Run from the project root with the venv active:
    python scripts/step3_qwen_backfill.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODELS  # noqa: E402
from step3_full import RESULTS_DIR, N_RESAMPLES, call_model, load_dilemmas  # noqa: E402
import json  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

QWEN_ONLY = {"qwen-2.5-7b": MODELS["qwen-2.5-7b"]}


def main() -> None:
    dilemmas = load_dilemmas()
    total_calls = len(dilemmas) * N_RESAMPLES * len(QWEN_ONLY)
    print(f"Backfilling qwen-2.5-7b: {len(dilemmas)} dilemmas x {N_RESAMPLES} "
          f"resamples = {total_calls} calls\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"step3_qwen_backfill_{timestamp}.jsonl"

    running_cost = 0.0
    completed = 0
    errors = 0

    with out_path.open("w", encoding="utf-8") as f:
        for dilemma in dilemmas:
            for model_key, model_cfg in QWEN_ONLY.items():
                for sample_index in range(N_RESAMPLES):
                    record = call_model(dilemma, model_key, model_cfg, sample_index)
                    f.write(json.dumps(record) + "\n")
                    f.flush()

                    completed += 1
                    if record["error"]:
                        errors += 1
                        print(f"[{completed}/{total_calls}] ERROR "
                              f"{dilemma['id']}/{sample_index}: {record['error']}")
                    else:
                        cost = (record["usage"] or {}).get("cost") or 0.0
                        running_cost += cost
                        if completed % 25 == 0 or completed == total_calls:
                            print(f"[{completed}/{total_calls}] ... running cost ${running_cost:.4f}")

    print(f"\nDone. {completed - errors}/{completed} calls succeeded.")
    print(f"Total cost: ${running_cost:.4f}")
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
