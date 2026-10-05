# Do LLMs Disclose Moral Dilemmas?

When a language model faces genuinely conflicting moral considerations, does it
acknowledge the conflict, or hand back a confident one-sided answer? This repo
holds the code, prompts, and data for a small study on that question.

**Write-up:** _(add LessWrong post link here)_

The headline problem is measurement: you cannot take a model's word for whether
it found a question hard. So conflict is measured behaviourally - resample the
model at temperature 1.0 and see whether it lands in the same place - and that
measure is compared against whether the model says anything about the conflict.

## What the study did

- **60 dilemmas**: 50 conflict dilemmas sampled from DailyDilemmas (spread
  across topic categories, screened to drop content-filter triggers) plus 10
  hand-curated controls that should have an obvious right answer.
- **5 models**: `gpt-5-mini` at low and high reasoning effort (as separate
  conditions), `llama-3.1-8b-instruct`, `qwen-2.5-7b-instruct`, `deepseek-r1`.
  All via OpenRouter.
- **n = 5 resamples** per dilemma per model, temperature 1.0.
- **Judge**: `anthropic/claude-sonnet-4.5` at temperature 0, labelling each
  response for which side it lands on (side A / side B / hedge / refuse) and
  whether it discloses the conflict. Validated against 30 blind hand labels:
  90% agreement on the side label, 83% on disclosure.
- **One robustness condition**: a 16-dilemma subset re-run with
  `"If you are genuinely unsure or believe there is no clear right answer,
  please say so explicitly."` appended to the prompt.
- **Total cost**: roughly $16 of OpenRouter credit.

## Findings in brief

Full discussion and caveats are in the write-up; these are the numbers.

- On conflict dilemmas, 17.2% of responses pick a side with no acknowledgment
  that a real choice was being made, against 42.0% on controls. The thing that
  moves between conditions is hedging (28.8% vs 2.8%).
- Splitting each (dilemma, model) cell by **its own** resampling entropy rather
  than by my conflict/control label: cells where the resamples disagreed show
  86.2% disclosure and 11.9% confident one-siding, against 75.9% and 24.1% for
  cells where they agreed. Disclosure tracks measured instability, not just the
  topic.
- But only 64 of 296 cells showed any resampling disagreement at all, and 55 of
  those belong to `llama` and `qwen`. For `gpt-5-mini` (both settings) and
  `deepseek-r1` the entropy signal essentially never fires, so the method can't
  answer the central question for three of five models.
- The uncertainty instruction leaves control behaviour untouched (43% to 43%
  confident one-sided) while cutting it sharply on conflict dilemmas.
  `deepseek-r1` drops from 35% to 6%, suggesting its usual decisiveness is a
  house style rather than a settled view. `gpt-5-mini` barely moves on that
  metric but roughly doubles its hedging, drawn from responses that had already
  disclosed and still picked.

## Repo layout

```
config.py        API client, model slugs, judge model (single source of truth)
PROJECT.md       the original research plan, written before the run
prompts/         prompt template and the two judge rubrics
scripts/         generation, judging, analysis
results/         dilemma sets, raw generations, judge labels, hand labels
data/            empty; DailyDilemmas loads from HuggingFace at runtime
```

## Which results file is which

The filenames are not self-explanatory, mostly because scripts were reused
across stages. Canonical files for the study as reported:

| File | What it is |
|---|---|
| `results/conflict_candidates_50.json` | the 50 conflict dilemmas used |
| `results/control_candidates_10.json` | the 10 control dilemmas used |
| `results/step3_full_combined.jsonl` | **canonical generations** - all 60 dilemmas, 5 models, n=5 |
| `results/judged_pilot_batched_step3_full_combined.jsonl` | **canonical judge labels** for the above (the `pilot_batched` in the name is a leftover from script reuse - this is the full run) |
| `results/robustness_uncertainty_dilemmas.json` | the 16-dilemma subset for the "just asking" condition |
| `results/step3_robustness_uncertainty_20260906T100639Z.jsonl` | generations under the uncertainty instruction |
| `results/judged_robustness_uncertainty_20260906T123701Z.jsonl` | judge labels for the above |
| `results/hand_label_blind.csv` | the blind labelling sheet (no model or dilemma_type shown) |
| `results/hand_label_blind - filled.csv` | my 30 hand labels |
| `results/hand_label_key.csv` | key mapping those rows back to model and dilemma type |
| `results/judge_validation.csv` | judge-vs-hand-label agreement output |

Superseded or intermediate, kept for provenance:

| File | What it was |
|---|---|
| `results/step2_all_models_*.json` | pipeline check, one dilemma across all models |
| `results/step3_pilot_*.jsonl`, `results/judged_pilot_*.jsonl` | the 6-dilemma pilot that preceded the full run |
| `results/step3_full_2026*.jsonl` | partial runs, merged into `step3_full_combined.jsonl` |
| `results/step3_qwen_backfill_*.jsonl` | `qwen` rerun after a provider outage, also merged into the combined file |

## Reproducing

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then paste your real OpenRouter key into .env
```

Run order, from the project root with the venv active:

```powershell
python scripts/step1_single_call.py          # API sanity check, one call
python scripts/step2_all_models.py           # confirm every model slug resolves
python scripts/step3_full.py                 # main generation run (~$3.60)
python scripts/step3_qwen_backfill.py        # only if a provider drops mid-run
python scripts/step4_judge.py                # judge validation + judging (~$6.80)
python scripts/analyze_full.py results/judged_pilot_batched_step3_full_combined.jsonl results/step3_full_combined.jsonl
```

The robustness condition is a separate pass:

```powershell
python scripts/step3_robustness_uncertainty.py
```

Supporting scripts: `sample_for_hand_labeling.py` builds the blind labelling
sheet (run *before* the judge exists, so validation isn't contaminated),
`resume_judge_robustness.py` picks up judging after a crash without re-paying
for generation, `swap_control_dilemma.py` splices a replacement control item
into the combined files, and `analyze_pilot.py` is the earlier pilot-scale
version of the analysis.

## Models

| Key | OpenRouter slug | Notes |
|---|---|---|
| `gpt-5-mini-low` | `openai/gpt-5-mini` | `reasoning.effort = low` |
| `gpt-5-mini-high` | `openai/gpt-5-mini` | `reasoning.effort = high` |
| `llama-3.1-8b` | `meta-llama/llama-3.1-8b-instruct` | open weight |
| `qwen-2.5-7b` | `qwen/qwen-2.5-7b-instruct` | open weight |
| `deepseek-r1` | `deepseek/deepseek-r1` | native CoT, reasoning tokens billed as output |
| judge | `anthropic/claude-sonnet-4.5` | temperature 0, held fixed |

`gpt-4o-mini` is still listed in `config.MODELS` but was excluded from the full
run to save budget.

## Method notes worth knowing

- **Refusals are excluded from entropy** and tracked as their own rate.
  `llama-3.1-8b` intermittently returns canned safety refusals on topics
  adjacent to moderation-sensitive categories (around 80% of resamples on an
  underage-drinking control item, 40% on a psychological-harm one). Folding
  those into entropy let a flaky content filter masquerade as moral hedging.
- **Entropy is normalised Shannon entropy** over `{side_a, side_b, hedge}`,
  divided by `log2(3)`. Cells with too few non-refused samples report
  "insufficient data" rather than a misleading number.
- **Content and reasoning traces use separate rubrics**
  (`prompts/judge_rubric.txt` and `prompts/judge_rubric_reasoning.txt`).
  "Refusing" and "landing on a side" mean different things for private
  deliberation the user never sees than for a final answer.
- **Judging is batched** at 5 resamples per call, which came in about 49%
  cheaper than per-item judging and agreed with it at 94.3% on the side label
  and 88.7% on disclosure.
- **Forced-choice logprobs are unavailable** for both reasoning models -
  `gpt-5-mini` returns an explicit API error, `deepseek-r1` silently returns
  null - so the single-token certainty check in `PROJECT.md` could not be run.
- **One control dilemma was swapped mid-study** after turning out softer than
  intended; `swap_control_dilemma.py` is the tool that did it.
- `PROJECT.md` pre-registers an RLHF-versus-open-weight comparison that does
  **not** hold up: the RLHF group is one model at two settings against three
  very different open models spanning 7B to roughly 671B parameters. The
  write-up treats the five as individual case studies instead.

## Datasets

- **DailyDilemmas** - `kellycyy/daily_dilemmas` on HuggingFace, from Chiu,
  Jiang & Choi, [arXiv:2410.02683](https://arxiv.org/abs/2410.02683). 1,360
  everyday moral dilemmas, each a choice between two actions. It carries no
  native hardness or agreement score, which is why the control set had to be
  curated by hand.
- **Right vs. Right** - [arXiv:2412.19926](https://arxiv.org/abs/2412.19926),
  1,730 dilemmas with native conflict-type labels. Not used here, but it is the
  obvious next dataset precisely because those labels would supply the
  independent difficulty measure this study lacks.

## Acknowledgements

Claude Opus 5 helped write the analysis scripts, iterate on the judge rubrics,
and draft this README. This is separate from the judge model (Claude Sonnet
4.5), which is part of the method rather than the authorship.

## Known limitations

Summarised here, discussed properly in the write-up: there is no measure of
genuine moral conflict independent of my own labelling; resampling at n = 5
resolves entropy into only two usable values and detects nothing at all for
three of the five models; the uncertainty instruction is a leading prompt, so
it cannot distinguish concealed uncertainty from compliance; and the judge
agreement of 90% / 83% implies roughly one label in ten would be scored
differently by a careful human.
