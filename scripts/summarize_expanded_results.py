#!/usr/bin/env python3
"""Create report-ready findings from the expanded Modal summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.expanded_report import (  # noqa: E402
    build_expanded_findings,
    render_expanded_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "expanded_summary.json",
    )
    parser.add_argument(
        "--findings",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "expanded_findings.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "expanded_results.md",
    )
    args = parser.parse_args()
    summary = json.loads(args.input.read_text())
    findings = build_expanded_findings(summary)
    args.findings.write_text(json.dumps(findings, indent=2) + "\n")
    args.markdown.write_text(render_expanded_markdown(findings))
    print(json.dumps(findings["behavior"], indent=2))
    print(f"Wrote {args.findings}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
