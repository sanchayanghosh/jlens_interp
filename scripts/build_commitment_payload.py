#!/usr/bin/env python3
"""Build the matched single-shot commitment experiment payload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.commitment_data import build_commitment_payload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=REPO_ROOT / "experiments" / "expanded_cases.json",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "expanded_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments" / "commitment_cases.json",
    )
    args = parser.parse_args()
    payload = build_commitment_payload(
        json.loads(args.cases.read_text()),
        json.loads(args.summary.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["design"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
