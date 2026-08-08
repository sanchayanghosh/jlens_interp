#!/usr/bin/env python3
"""Materialize either the pilot or expanded matched JLENS experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.data import (  # noqa: E402
    build_expanded_experiment_payload,
    build_experiment_payload,
    read_jsonl,
)

DEFAULT_INPUT = (
    REPO_ROOT
    / "output"
    / "sycophancy_natural_train__aman313-2dd5-sycophancy.jsonl"
)
DEFAULT_OUTPUTS = {
    "pilot": REPO_ROOT / "experiments" / "selected_cases.json",
    "expanded": REPO_ROOT / "experiments" / "expanded_cases.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--selection", choices=sorted(DEFAULT_OUTPUTS), default="pilot")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    groups = read_jsonl(args.input)
    payload = (
        build_expanded_experiment_payload(groups)
        if args.selection == "expanded"
        else build_experiment_payload(groups)
    )
    output = args.output or DEFAULT_OUTPUTS[args.selection]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Selected {payload['design']['n_cases']} cases / "
        f"{payload['design']['n_prompts']} prompts"
    )
    for case in payload["cases"][:12]:
        print(
            f"  {case['correct_letter']}->{case['original_wrong_letter']} "
            f"{case['historical_label']:<11} source={case['source_id']} "
            f"{case['question']}"
        )
    if len(payload["cases"]) > 12:
        print(f"  ... {len(payload['cases']) - 12} additional cases")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
