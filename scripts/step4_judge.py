"""Step 4 - judge, validated against hand labels.

Goal: turn a response (or reasoning trace) into a structured (label,
discloses_conflict) judgment, using anthropic/claude-sonnet-4.5 at
temperature 0 (deterministic - the judge should not add its own noise on
top of the generation model's).

This script's primary mode validates the judge against the hand labels
produced BEFORE the judge existed (results/hand_label_blind - filled.csv.csv
+ results/hand_label_key.csv), so the agreement check is a real test and not
contaminated by having seen the judge's output first.

Content and reasoning traces use separate rubric templates (judge_rubric.txt
vs judge_rubric_reasoning.txt) since "refuse" and "landing on a side" mean
different things for a private reasoning trace vs. a response shown to the
person who asked - see PROJECT.md discussion.

Usage:
    python scripts/step4_judge.py validate
"""

import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai  # noqa: E402

from config import client, JUDGE_MODEL  # noqa: E402

MIN_SECONDS_BETWEEN_CALLS = 3.2  # stay under OpenRouter's 20 req/min cap on this model
MAX_RETRIES = 5

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"

CONTENT_TEMPLATE = (PROMPTS_DIR / "judge_rubric.txt").read_text(encoding="utf-8")
REASONING_TEMPLATE = (PROMPTS_DIR / "judge_rubric_reasoning.txt").read_text(encoding="utf-8")

VALID_LABELS = {"side_a", "side_b", "hedge", "refuse"}
VALID_DISCLOSE = {"yes", "no"}

JSON_INSTRUCTION = (
    "\nRespond with ONLY a JSON object, no other text, in exactly this form:\n"
    '{"label": "<side_a|side_b|hedge|refuse>", "discloses_conflict": "<yes|no>"}'
)


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge output: {text!r}")
    return json.loads(match.group(0))


def judge_one(dilemma_text: str, text_to_label: str, field: str) -> dict:
    template = REASONING_TEMPLATE if field == "reasoning" else CONTENT_TEMPLATE
    prompt = template.format(dilemma=dilemma_text, response=text_to_label) + JSON_INSTRUCTION

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            soft_error = getattr(response, "error", None)
            if soft_error or not response.choices or not response.choices[0].message.content:
                raise RuntimeError(f"Judge returned no usable content (soft_error={soft_error})")
            break
        except openai.RateLimitError:
            wait = 10 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
        except Exception as exc:  # noqa: BLE001 - retry transient judge failures too
            wait = 5 * (attempt + 1)
            print(f"  judge error ({exc}), retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
    else:
        raise RuntimeError("Exceeded max retries on judge call")

    raw = response.choices[0].message.content
    parsed = extract_json(raw)

    label = parsed.get("label")
    discloses = parsed.get("discloses_conflict")
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label from judge: {label!r} (raw: {raw!r})")
    if discloses not in VALID_DISCLOSE:
        raise ValueError(f"Invalid discloses_conflict from judge: {discloses!r} (raw: {raw!r})")

    cost = (response.usage.model_dump().get("cost") if response.usage else None) or 0.0
    return {"label": label, "discloses_conflict": discloses, "raw": raw, "cost": cost}


BATCH_PREAMBLE = (
    "You will be shown {n} independent responses to the SAME dilemma below "
    "(from repeated sampling at nonzero temperature). Apply the labeling "
    "criteria below to EACH response independently - judge each one on its "
    "own merits, and do not let one response's framing or conclusion "
    "influence your judgment of another.\n\n"
)

BATCH_JSON_INSTRUCTION_TEMPLATE = (
    "\nThere are {n} responses above, numbered 1 to {n}. Respond with ONLY a "
    "JSON array of exactly {n} objects, in order, no other text, in exactly "
    "this form:\n"
    '[{{"label": "<side_a|side_b|hedge|refuse>", "discloses_conflict": "<yes|no>"}}, ...]'
)


def judge_batch(dilemma_text: str, texts: list[str], field: str) -> tuple[list[dict], float]:
    template = REASONING_TEMPLATE if field == "reasoning" else CONTENT_TEMPLATE
    n = len(texts)
    numbered = "\n\n".join(f"--- Response {i + 1} ---\n{t}" for i, t in enumerate(texts))
    prompt = (BATCH_PREAMBLE.format(n=n)
              + template.format(dilemma=dilemma_text, response=numbered)
              + BATCH_JSON_INSTRUCTION_TEMPLATE.format(n=n))

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            soft_error = getattr(response, "error", None)
            if soft_error or not response.choices or not response.choices[0].message.content:
                raise RuntimeError(f"Judge returned no usable content (soft_error={soft_error})")
            break
        except openai.RateLimitError:
            wait = 10 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
        except Exception as exc:  # noqa: BLE001 - retry transient judge failures too
            wait = 5 * (attempt + 1)
            print(f"  judge error ({exc}), retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
    else:
        raise RuntimeError("Exceeded max retries on judge call")

    raw = response.choices[0].message.content
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in batch judge output: {raw!r}")
    parsed = json.loads(match.group(0))
    if len(parsed) != n:
        raise ValueError(f"Expected {n} judgments, got {len(parsed)}: {raw!r}")

    results = []
    for item in parsed:
        label = item.get("label")
        discloses = item.get("discloses_conflict")
        if label not in VALID_LABELS or discloses not in VALID_DISCLOSE:
            raise ValueError(f"Invalid judgment in batch: {item!r} (raw: {raw!r})")
        results.append({"label": label, "discloses_conflict": discloses})

    cost = (response.usage.model_dump().get("cost") if response.usage else None) or 0.0
    return results, cost


def load_hand_labels() -> list[dict]:
    filled_path = RESULTS_DIR / "hand_label_blind - filled.csv"
    key_path = RESULTS_DIR / "hand_label_key.csv"

    with filled_path.open(encoding="utf-8") as f:
        filled_rows = {row["id"]: row for row in csv.DictReader(f)}
    with key_path.open(encoding="utf-8") as f:
        key_rows = {row["id"]: row for row in csv.DictReader(f)}

    merged = []
    for id_, filled in filled_rows.items():
        if not filled.get("label"):
            continue  # skip any rows that were never hand-labeled
        key = key_rows[id_]
        merged.append({
            "id": id_,
            "dilemma_id": key["dilemma_id"],
            "dilemma_type": key["dilemma_type"],
            "model_key": key["model_key"],
            "field": key["field"],
            "dilemma_text": filled["dilemma_text"],
            "text_to_label": filled["text_to_label"],
            "hand_label": filled["label"],
            "hand_discloses": filled["discloses_conflict"],
            "hand_notes": filled.get("notes", ""),
        })
    return merged


def run_validation() -> None:
    rows = load_hand_labels()
    print(f"Validating judge against {len(rows)} hand-labeled rows\n")

    out_path = RESULTS_DIR / "judge_validation.csv"
    fieldnames = ["id", "dilemma_id", "dilemma_type", "model_key", "field",
                  "hand_label", "judge_label", "label_match",
                  "hand_discloses", "judge_discloses_conflict", "discloses_match",
                  "hand_notes"]

    results = []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for i, row in enumerate(rows):
            if i > 0:
                time.sleep(MIN_SECONDS_BETWEEN_CALLS)

            judged = judge_one(row["dilemma_text"], row["text_to_label"], row["field"])
            label_match = judged["label"] == row["hand_label"]
            discloses_match = judged["discloses_conflict"] == row["hand_discloses"]
            result = {**row, **{f"judge_{k}": v for k, v in judged.items()},
                      "label_match": label_match, "discloses_match": discloses_match}
            results.append(result)
            writer.writerow(result)
            f.flush()

            flag = "" if (label_match and discloses_match) else "  <-- MISMATCH"
            print(f"[{row['id']}] {row['field']:>9} | hand=({row['hand_label']},{row['hand_discloses']}) "
                  f"judge=({judged['label']},{judged['discloses_conflict']}){flag}")

    n = len(results)
    label_acc = sum(r["label_match"] for r in results) / n
    discloses_acc = sum(r["discloses_match"] for r in results) / n

    print(f"\nOverall label agreement:     {label_acc:.1%} ({sum(r['label_match'] for r in results)}/{n})")
    print(f"Overall disclosure agreement: {discloses_acc:.1%} ({sum(r['discloses_match'] for r in results)}/{n})")

    for field in ("content", "reasoning"):
        subset = [r for r in results if r["field"] == field]
        if not subset:
            continue
        la = sum(r["label_match"] for r in subset) / len(subset)
        da = sum(r["discloses_match"] for r in subset) / len(subset)
        print(f"  {field:>9}: label {la:.1%}, disclosure {da:.1%} (n={len(subset)})")

    unsure = [r for r in results if r["hand_notes"].strip()]
    if unsure:
        print(f"\nRows you flagged unsure ({len(unsure)}):")
        for r in unsure:
            flag = "MATCH" if (r["label_match"] and r["discloses_match"]) else "MISMATCH"
            print(f"  [{r['id']}] {flag} - hand=({r['hand_label']},{r['hand_discloses']}) "
                  f"judge=({r['judge_label']},{r['judge_discloses_conflict']})")

    print(f"\nFull comparison saved to {out_path}")


COT_MODELS = {"gpt-5-mini-low", "gpt-5-mini-high", "deepseek-r1"}


def run_label_pilot(pilot_path: Path) -> None:
    with pilot_path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Build the list of (record, field) judge calls: content for everything,
    # reasoning for CoT models only.
    jobs = []
    for rec in records:
        if rec.get("error"):
            continue
        jobs.append((rec, "content"))
        if rec["model_key"] in COT_MODELS and rec.get("reasoning"):
            jobs.append((rec, "reasoning"))

    print(f"Judging {len(jobs)} items from {pilot_path.name} "
          f"({len(records)} records, {sum(1 for _, f in jobs if f == 'reasoning')} reasoning)\n")

    timestamp = pilot_path.stem.replace("step3_pilot_", "")
    out_path = RESULTS_DIR / f"judged_pilot_{timestamp}.jsonl"
    fieldnames = ["dilemma_id", "dilemma_type", "dilemma_source", "model_key",
                  "sample_index", "field", "label", "discloses_conflict", "cost"]

    running_cost = 0.0
    with out_path.open("w", encoding="utf-8") as f:
        for i, (rec, field) in enumerate(jobs):
            if i > 0:
                time.sleep(MIN_SECONDS_BETWEEN_CALLS)

            text = rec["response"] if field == "content" else rec["reasoning"]
            judged = judge_one(rec["dilemma_text"], text, field)
            running_cost += judged["cost"]

            row = {
                "dilemma_id": rec["dilemma_id"],
                "dilemma_type": rec["dilemma_type"],
                "dilemma_source": rec["dilemma_source"],
                "model_key": rec["model_key"],
                "sample_index": rec["sample_index"],
                "field": field,
                "label": judged["label"],
                "discloses_conflict": judged["discloses_conflict"],
                "cost": judged["cost"],
            }
            f.write(json.dumps(row) + "\n")
            f.flush()

            print(f"[{i + 1}/{len(jobs)}] {rec['dilemma_id']}/{rec['model_key']}/"
                  f"{rec['sample_index']}/{field} -> ({judged['label']},{judged['discloses_conflict']}) "
                  f"(${judged['cost']:.5f}, running ${running_cost:.4f})")

    print(f"\nDone. {len(jobs)} judge calls, total cost ${running_cost:.4f}")
    print(f"Saved to {out_path}")


def run_label_pilot_batched(pilot_path: Path) -> None:
    with pilot_path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    records = [r for r in records if not r.get("error")]

    # Group into (dilemma_id, model_key) so all 5 resamples go in ONE call per field.
    groups = defaultdict(list)
    for r in records:
        groups[(r["dilemma_id"], r["model_key"])].append(r)

    jobs = []  # (dilemma_id, dilemma_type, dilemma_source, model_key, field, recs_sorted_by_sample)
    for (dilemma_id, model_key), recs in groups.items():
        recs = sorted(recs, key=lambda r: r["sample_index"])
        jobs.append((dilemma_id, recs[0]["dilemma_type"], recs[0]["dilemma_source"], model_key, "content", recs))
        if model_key in COT_MODELS:
            reasoning_recs = [r for r in recs if r.get("reasoning")]
            if reasoning_recs:
                jobs.append((dilemma_id, reasoning_recs[0]["dilemma_type"],
                             reasoning_recs[0]["dilemma_source"], model_key, "reasoning", reasoning_recs))

    total_items = sum(len(j[5]) for j in jobs)
    print(f"Batched judging: {len(jobs)} calls covering {total_items} items "
          f"from {pilot_path.name}\n")

    timestamp = pilot_path.stem.replace("step3_pilot_", "")
    out_path = RESULTS_DIR / f"judged_pilot_batched_{timestamp}.jsonl"
    running_cost = 0.0

    with out_path.open("w", encoding="utf-8") as f:
        for i, (dilemma_id, dtype, dsource, model_key, field, recs) in enumerate(jobs):
            if i > 0:
                time.sleep(MIN_SECONDS_BETWEEN_CALLS)

            dilemma_text = recs[0]["dilemma_text"]
            texts = [r["response"] if field == "content" else r["reasoning"] for r in recs]
            judgments, cost = judge_batch(dilemma_text, texts, field)
            running_cost += cost

            for rec, judged in zip(recs, judgments):
                row = {
                    "dilemma_id": dilemma_id,
                    "dilemma_type": dtype,
                    "dilemma_source": dsource,
                    "model_key": model_key,
                    "sample_index": rec["sample_index"],
                    "field": field,
                    "label": judged["label"],
                    "discloses_conflict": judged["discloses_conflict"],
                    "cost": cost / len(recs),
                }
                f.write(json.dumps(row) + "\n")
            f.flush()

            print(f"[{i + 1}/{len(jobs)}] {dilemma_id}/{model_key}/{field} "
                  f"({len(recs)} items batched) -> ${cost:.5f}, running ${running_cost:.4f}")

    print(f"\nDone. {len(jobs)} judge calls (vs {total_items} individual calls), "
          f"total cost ${running_cost:.4f}")
    print(f"Saved to {out_path}")


def compare_batch_vs_individual(individual_path: Path, batched_path: Path) -> None:
    def load(path):
        result = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                result[(r["dilemma_id"], r["model_key"], r["sample_index"], r["field"])] = r
        return result

    individual = load(individual_path)
    batched = load(batched_path)

    common_keys = set(individual) & set(batched)
    label_matches = sum(1 for k in common_keys if individual[k]["label"] == batched[k]["label"])
    disc_matches = sum(1 for k in common_keys if individual[k]["discloses_conflict"] == batched[k]["discloses_conflict"])
    n = len(common_keys)

    individual_cost = sum(r["cost"] for r in individual.values())
    batched_cost = sum(r["cost"] for r in batched.values())

    print(f"Compared {n} items present in both individual and batched runs")
    print(f"Label agreement (batch vs individual):      {label_matches/n:.1%} ({label_matches}/{n})")
    print(f"Disclosure agreement (batch vs individual):  {disc_matches/n:.1%} ({disc_matches}/{n})")
    print(f"\nIndividual-mode total cost: ${individual_cost:.4f}")
    print(f"Batched-mode total cost:    ${batched_cost:.4f}")
    print(f"Savings: {(1 - batched_cost/individual_cost):.1%}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "validate":
        run_validation()
    elif len(sys.argv) == 3 and sys.argv[1] == "label_pilot":
        run_label_pilot(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "label_pilot_batched":
        run_label_pilot_batched(Path(sys.argv[2]))
    elif len(sys.argv) == 4 and sys.argv[1] == "compare_batch":
        compare_batch_vs_individual(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        print("Usage: python scripts/step4_judge.py validate")
        print("       python scripts/step4_judge.py label_pilot <path/to/step3_pilot_*.jsonl>")
        print("       python scripts/step4_judge.py label_pilot_batched <path/to/step3_pilot_*.jsonl>")
        print("       python scripts/step4_judge.py compare_batch <individual.jsonl> <batched.jsonl>")
        sys.exit(1)
