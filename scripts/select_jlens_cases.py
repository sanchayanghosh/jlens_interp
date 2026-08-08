#!/usr/bin/env python3
"""Select the six matched cases and materialize the 24-prompt experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.data import build_experiment_payload, read_jsonl  # noqa: E402

DEFAULT_INPUT = (
    REPO_ROOT
    / "output"
    / "sycophancy_natural_train__aman313-2dd5-sycophancy.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "selected_cases.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_experiment_payload(read_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Selected {payload['design']['n_cases']} cases / {payload['design']['n_prompts']} prompts")
    for case in payload["cases"]:
        print(
            f"  {case['correct_letter']}->{case['original_wrong_letter']} "
            f"{case['historical_label']:<11} source={case['source_id']} "
            f"{case['question']}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
