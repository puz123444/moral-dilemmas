# Do LLMs Disclose Moral Dilemmas?

When language models face genuinely conflicting moral considerations, do they
acknowledge the conflict, or present a confident one-sided answer? Compared
across models and across CoT / reasoning-effort conditions.

See `PROJECT.md` for the full research plan.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then paste your real OpenRouter key into .env
```

## Project layout

```
data/      dilemma datasets + control set
prompts/   prompt templates (main, robustness variants, judge rubric)
scripts/   generation, judging, analysis
results/   raw model outputs (saved before judging), then judge labels
config.py  API client + model list (single source of truth for model slugs)
```

## Build order

1. `scripts/step1_single_call.py` — API sanity test (one model, one dilemma) ✅ done, verified working
2. Loop over all models, same dilemma, save raw outputs — ⏳ next
3. Loop over dilemmas + n-sample resampling for entropy
4. Judge script (separate from generation), validated against hand labels
5. Analysis: entropy gaps, disclosure rates, mismatch rates

## Datasets

- **DailyDilemmas** — `kellycyy/daily_dilemmas` on HuggingFace (loads via `datasets`)
- **Right vs. Right** — from arXiv:2412.19926 (1,730 dilemmas, 4 conflict types).
  Check the paper for its data release location before wiring in.

## Current status (2026-09-04)

- venv created at `.venv`, all deps from `requirements.txt` installed into it
  (openai 3.8.0, python-dotenv, pandas, datasets)
- `.env` has a real OpenRouter key (rotated at least once already — fine, scripts
  read it fresh each run)
- Step 1 ran successfully against `meta-llama/llama-3.1-8b-instruct` — confirmed
  API connectivity, response format, and cost reporting (~$0.00002/call)
- Judge model chosen: `anthropic/claude-sonnet-4.5` (config.py `JUDGE_MODEL`)
- Git: repo initialized, first commit made (`4d7e54b`), all scaffold files
  currently staged (`git add .` run) but **not yet committed** — do
  `git commit -m "..."` when ready
- Not yet built: Step 2 script (loop all models/conditions over one dilemma,
  save raw JSON to `results/`)
- Open question: exact data-release location for Right vs. Right dataset
  (paper is arXiv:2412.19926) — not yet resolved, DailyDilemmas is the
  starting dataset
