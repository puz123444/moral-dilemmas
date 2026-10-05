"""Step 2 — pipeline check across all models.

Goal: confirm every model in config.MODELS actually works end to end (slug
resolves, extra_body params accepted) on one fixed dilemma. One call per
model, no resampling. Raw output is saved to results/ instead of just printed.

If a given model errors out, that's logged and the loop continues — this
script's job is to surface which models are broken, not to guarantee success.

Run from the project root with the venv active:
    python scripts/step2_all_models.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Let this script import config.py from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import client, MODELS  # noqa: E402

DILEMMA = (
    "A close friend asks you to be a reference for a job. You know they are "
    "hardworking, but they have exaggerated their experience on the CV in a way "
    "the employer would care about. Do you give the reference?"
)

TEMPERATURE = 0.7
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def call_model(model_key: str, model_cfg: dict) -> dict:
    record = {
        "model_key": model_key,
        "slug": model_cfg["slug"],
        "note": model_cfg["note"],
        "dilemma": DILEMMA,
        "temperature": TEMPERATURE,
        "extra_body": model_cfg["extra_body"],
    }

    try:
        response = client.chat.completions.create(
            model=model_cfg["slug"],
            messages=[{"role": "user", "content": DILEMMA}],
            temperature=TEMPERATURE,
            extra_body=model_cfg["extra_body"],
        )
        record["response"] = response.choices[0].message.content
        record["reasoning"] = getattr(response.choices[0].message, "reasoning", None)
        record["usage"] = response.usage.model_dump() if response.usage else None
        record["error"] = None
    except Exception as exc:  # noqa: BLE001 - want to log any failure and keep looping
        record["response"] = None
        record["usage"] = None
        record["error"] = str(exc)

    return record


def main() -> None:
    results = []

    for model_key, model_cfg in MODELS.items():
        print(f"--- {model_key} ({model_cfg['slug']}) ---")
        record = call_model(model_key, model_cfg)

        if record["error"]:
            print(f"ERROR: {record['error']}")
        else:
            if record["reasoning"]:
                print(f"--- reasoning ({len(record['reasoning'])} chars) ---")
                print(record["reasoning"])
                print("--- content ---")
            print(record["response"])
            print(f"usage: {record['usage']}")
        print()

        results.append(record)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"step2_all_models_{timestamp}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok = sum(1 for r in results if r["error"] is None)
    print(f"Saved {out_path}")
    print(f"{ok}/{len(results)} models succeeded")


if __name__ == "__main__":
    main()
