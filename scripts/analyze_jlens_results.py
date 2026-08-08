#!/usr/bin/env python3
"""Analyze a raw Modal JLENS export and write JSON plus Markdown summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analysis import analyze_results, render_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=REPO_ROOT / "experiments" / "results" / "raw_results.json",
        type=Path,
    )
    parser.add_argument(
        "--summary-json",
        default=REPO_ROOT / "experiments" / "results" / "summary.json",
        type=Path,
    )
    parser.add_argument(
        "--summary-markdown",
        default=REPO_ROOT / "experiments" / "results" / "preliminary_results.md",
        type=Path,
    )
    args = parser.parse_args()

    results = json.loads(args.input.read_text())
    summary = analyze_results(results)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    args.summary_markdown.write_text(render_markdown(summary))
    print(json.dumps({
        "behavior": summary["regenerated_behavior_counts"],
        "historical_wrong1_match": summary["historical_wrong1_match"],
        "summary_json": str(args.summary_json),
        "summary_markdown": str(args.summary_markdown),
    }, indent=2))


if __name__ == "__main__":
    main()
