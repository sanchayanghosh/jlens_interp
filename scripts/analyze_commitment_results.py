#!/usr/bin/env python3
"""Analyze and render the single-shot commitment experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.commitment_analysis import (  # noqa: E402
    analyze_commitment_results,
    render_commitment_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "commitment_raw.json",
    )
    parser.add_argument(
        "--findings",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "commitment_findings.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "commitment_results.md",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    findings = analyze_commitment_results(raw, args.bootstrap_samples)
    args.findings.write_text(json.dumps(findings, indent=2) + "\n")
    args.markdown.write_text(render_commitment_markdown(findings))
    print(json.dumps(findings["preregistered"], indent=2))
    print(f"Wrote {args.findings}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
