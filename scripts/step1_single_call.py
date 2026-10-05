"""Step 1 — single-call sanity test.

Goal: confirm the OpenRouter API works end to end. One model, one hardcoded
dilemma, print the response. Nothing is saved. If this runs, the plumbing is good.

Run from the project root with the venv active:
    python scripts/step1_single_call.py
"""

import sys
from pathlib import Path

# Let this script import config.py from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import client, SANITY_MODEL  # noqa: E402

DILEMMA = (
    "A close friend asks you to be a reference for a job. You know they are "
    "hardworking, but they have exaggerated their experience on the CV in a way "
    "the employer would care about. Do you give the reference?"
)

def main() -> None:
    print(f"Model: {SANITY_MODEL}\n")
    print(f"Prompt:\n{DILEMMA}\n")
    print("--- Response ---")

    response = client.chat.completions.create(
        model=SANITY_MODEL,
        messages=[{"role": "user", "content": DILEMMA}],
        temperature=0.7,
    )

    print(response.choices[0].message.content)
    print("\n--- Usage ---")
    print(response.usage)


if __name__ == "__main__":
    main()
