# Project: Do LLMs Disclose Moral Dilemmas?

## Research question
When language models are put in situations with genuinely conflicting moral
considerations, do they acknowledge the conflict (in their response and/or
chain-of-thought), or do they present a confident, one-sided answer? Compare
across models and across CoT vs. non-CoT / reasoning-effort conditions.

## Why this matters
Prior work shows LLMs form internal representations of demographic/user
attributes and that CoT reasoning is not always faithful to the actual
computation driving an answer. This project asks a related but distinct
question: independent of *why* a model settles on an answer, does it accurately
represent and communicate its own uncertainty when facing a genuine value
conflict — or does it default to confident-sounding resolution regardless of
whether the situation warrants it?

## Ground-truth conflict detection (independent of what the model says)
- **Primary — entropy via resampling:** sample the model n times (temp > 0),
  judge each response into a categorical label (side A / side B / hedge / refuse),
  compute entropy over that distribution. Uniform across all models.
- **Sanity check:** where logprobs are available, also compute single-call
  entropy over a forced answer token and check agreement with resampling.
- **Control set:** clear-cut, low-tension dilemmas pulled from the datasets'
  own hardness/agreement scores. Entropy should be clearly lower here.
- Normalize entropy by log2(number of labels).
- Do NOT use "CoT mentions a conflict" as ground-truth (circular — it is also an
  outcome metric).

## Datasets
- DailyDilemmas (`kellycyy/daily_dilemmas`)
- Right vs. Right (~1,730 dilemmas, 4 conflict types: truth/loyalty,
  individual/community, short-term/long-term, justice/mercy) — arXiv:2412.19926
- Prefer dataset-native control items over hand-written ones.

## Conditions / robustness checks
- Reworded/paraphrased versions
- User states a side upfront (anchoring)
- Explicit "if you're not sure, say so" instruction
- Record response length per condition
- Order-effects check: does A-before-B predict which side wins?

## Primary metrics
1. Disclosure rate on genuine conflicts (baseline-adjusted vs. control)
2. Resolution-type breakdown: discloses+hedges / discloses+picks / no-disclose+picks / refuses
3. CoT-response mismatch rate (both directions) — visible-CoT models only
4. Entropy gap: conflict vs. control, per model
5. Broken out by dataset-provided hardness where available
6. GPT-5-mini: low vs. high reasoning_effort

## Models
| Model | Role | Notes |
|---|---|---|
| GPT-5-mini | RLHF, native reasoning | reasoning_effort low + high as separate conditions |
| GPT-4o-mini | RLHF, no native CoT | optional, prompted-CoT only |
| Llama-3.1-8B-Instruct | Open weight | logprobs available |
| Qwen2.5-7B-Instruct | Open weight | logprobs available |
| DeepSeek-R1 | Open, native CoT | reasoning tokens billed as output |

Access: all via one OpenRouter API key (OpenAI-SDK-compatible).
Judge: `anthropic/claude-sonnet-4.5`, held fixed.

Known constraint: reasoning models likely don't expose temperature/logprobs on
reasoning endpoints — confirm in pilot. Where unavailable, resampling entropy only.

## Pre-registered headline comparison
Entropy gap and disclosure-rate gap (conflict vs. control), broken out by
training approach (RLHF vs. open-weight-lighter-safety) and by CoT condition
(reasoning-effort low vs. high; CoT vs. non-CoT models). Everything else is
secondary/exploratory.

## Implementation plan
1. Environment setup (done)
2. Project structure (done)
3. Step 1 — single call sanity test (done)
4. Step 2 — loop over all models, same dilemma, save raw outputs
5. Step 3 — loop over dilemmas + resampling for entropy
6. Step 4 — judge script (separate), validate against 30-50 hand labels
7. Step 5 — analysis in pandas
8. Pilot end-to-end on 5-10 dilemmas before the full run

## Budget
~$4-5 USD for a ~100-dilemma run across all model/conditions with resampling and
robustness variants. Within a EUR 20 budget.
