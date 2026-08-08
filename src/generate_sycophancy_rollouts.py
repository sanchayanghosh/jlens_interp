#!/usr/bin/env python3
"""Generate and deterministically label sycophancy rollouts with Together.ai."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from sycophancy import PromptSpec, Variant, build_prompt_group, label_group, shuffled_split
from together_api import generate_response

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "input_data" / "open_trivia_shuffled.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-items", type=int, required=True, help="Questions to process; 3 calls each")
    parser.add_argument(
        "--variant",
        choices=["instruction_prompted", "natural", "incentivised"],
        default="instruction_prompted",
    )
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", required=True, help="Together.ai model identifier")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Append, skipping source IDs already present")
    parser.add_argument("--mock", action="store_true", help="Do not call Together.ai")
    args = parser.parse_args()
    if args.num_items < 1:
        parser.error("--num-items must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def default_output_path(variant: str, split: str, model: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")
    return PROJECT_ROOT / "output" / f"sycophancy_{variant}_{split}__{safe_model}.jsonl"


def load_completed_source_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                completed.add(str(json.loads(line)["source_id"]))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid resume file {path} at line {line_number}") from exc
    return completed


def create_groups(
    data: pd.DataFrame,
    num_items: int,
    seed: int,
    completed: set[str],
    variant: Variant,
) -> list[list[PromptSpec]]:
    rng = random.Random(seed)
    groups: list[list[PromptSpec]] = []
    valid_position = 0
    for source_index, row in data.iterrows():
        group = build_prompt_group(str(source_index), valid_position, row, rng, variant)
        if group is None:
            continue
        valid_position += 1
        if group[0].source_id in completed:
            continue
        groups.append(group)
        if len(groups) >= num_items:
            break
    return groups


def run_group(
    specs: list[PromptSpec],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    mock: bool,
) -> dict[str, Any]:
    if mock:
        responses = ["Mocked Response" for _ in specs]
    else:
        responses = [
            generate_response(spec.prompt, model, api_key, temperature, max_tokens)
            for spec in specs
        ]
    return {
        "source_id": specs[0].source_id,
        "dataset_position": specs[0].dataset_position,
        "correct_letter": specs[0].correct_letter,
        "model": model,
        "rollouts": label_group(specs, responses),
    }


def summarize(path: Path) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    groups = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            group = json.loads(line)
            groups += 1
            labels.update(rollout["label"] for rollout in group["rollouts"])
    evaluable = sum(count for label, count in labels.items() if label != "skip")
    sycophantic = labels["sycophantic"]
    return {
        "question_groups": groups,
        "total_rollouts": sum(labels.values()),
        "label_counts": dict(sorted(labels.items())),
        "evaluable_rollouts": evaluable,
        "sycophantic_fraction_of_evaluable": sycophantic / evaluable if evaluable else None,
    }


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("TOGETHER_API_KEY", "")
    if not args.mock and not api_key:
        raise SystemExit("TOGETHER_API_KEY is missing. Add it to .env or the environment.")

    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output or default_output_path(args.variant, args.split, args.model)
    ).expanduser().resolve()
    if output_path.exists() and not args.resume:
        raise SystemExit(f"Output exists: {output_path}. Use --resume or choose another --output.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_data = pd.read_csv(input_path)
    split_data = shuffled_split(raw_data, args.split)
    completed = load_completed_source_ids(output_path) if args.resume else set()
    groups = create_groups(split_data, args.num_items, args.seed, completed, args.variant)
    if not groups:
        raise SystemExit("No new valid question groups selected.")

    print(f"Model: {args.model}")
    print(f"Variant: {args.variant}")
    print(f"Question groups this run: {len(groups)}")
    print(f"Completions this run: {len(groups) * 3}")
    print(f"Output: {output_path}")

    mode = "a" if args.resume else "w"
    failures: list[tuple[str, str]] = []
    with output_path.open(mode) as output_handle, ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        future_to_source = {
            executor.submit(
                run_group,
                specs,
                args.model,
                api_key,
                args.temperature,
                args.max_tokens,
                args.mock,
            ): specs[0].source_id
            for specs in groups
        }
        for future in tqdm(as_completed(future_to_source), total=len(future_to_source)):
            source_id = future_to_source[future]
            try:
                record = future.result()
            except Exception as exc:
                failures.append((source_id, str(exc)))
                continue
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()

    summary = summarize(output_path)
    summary.update(
        {
            "input": str(input_path),
            "split": args.split,
            "variant": args.variant,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "failed_groups_this_run": len(failures),
            "failures": [{"source_id": source_id, "error": error} for source_id, error in failures],
        }
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")
    if failures:
        raise SystemExit(f"{len(failures)} group(s) failed; rerun with --resume.")


if __name__ == "__main__":
    main()
