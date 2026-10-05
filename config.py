"""Central configuration: API client + model list.

Import this from every script so model slugs and client setup live in one place.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY or "REPLACE_ME" in OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing or still the placeholder. "
        "Edit the .env file and paste your real OpenRouter key."
    )

# OpenRouter speaks the OpenAI API protocol; we just point the client at its URL.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- Models under test -------------------------------------------------------
# Slugs must match exactly what OpenRouter lists at https://openrouter.ai/models
# Verify each one in the pilot run before scaling up.
MODELS = {
    "gpt-5-mini-low": {
        "slug": "openai/gpt-5-mini",
        "extra_body": {"reasoning": {"effort": "low"}},
        "note": "RLHF, native reasoning, low effort condition",
    },
    "gpt-5-mini-high": {
        "slug": "openai/gpt-5-mini",
        "extra_body": {"reasoning": {"effort": "high"}},
        "note": "RLHF, native reasoning, high effort condition",
    },
    "gpt-4o-mini": {
        "slug": "openai/gpt-4o-mini",
        "extra_body": {},
        "note": "RLHF, no native CoT (prompted-CoT only). Optional.",
    },
    "llama-3.1-8b": {
        "slug": "meta-llama/llama-3.1-8b-instruct",
        "extra_body": {},
        "note": "Open weight, lighter safety tuning, logprobs available",
    },
    "qwen-2.5-7b": {
        "slug": "qwen/qwen-2.5-7b-instruct",
        "extra_body": {},
        "note": "Open weight, lighter safety tuning, logprobs available",
    },
    "deepseek-r1": {
        "slug": "deepseek/deepseek-r1",
        "extra_body": {},
        "note": "Open, native CoT. Reasoning tokens billed as output.",
    },
}

# --- Judge (held fixed across the whole project) ----------------------------
JUDGE_MODEL = "anthropic/claude-sonnet-4.5"

# Model used for the first single-call sanity test
SANITY_MODEL = "meta-llama/llama-3.1-8b-instruct"
