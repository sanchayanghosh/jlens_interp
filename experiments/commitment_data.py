"""Build matched examples for single-shot J-space sycophancy detection."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any


def _split(identifier: str) -> str:
    bucket = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16) % 10
    return "discovery" if bucket < 7 else "test"


def _layout_map(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {layout["name"]: layout for layout in case["layouts"]}


def _trial_key(trial: dict[str, Any]) -> tuple[str, str, str]:
    return trial["layout"], trial["correct_letter"], trial["user_letter"]


def build_commitment_payload(
    expanded_cases: dict[str, Any],
    expanded_summary: dict[str, Any],
) -> dict[str, Any]:
    """Select paired trials while keeping labels out of detector inputs.

    The primary comparison pairs one regenerated sycophantic trial per source with that
    source's warranted correct-opinion agreement. The open comparison exactly matches
    sycophantic and resistant trials by layout and correct/user answer-letter mapping.
    """
    cases = {str(case["source_id"]): case for case in expanded_cases["cases"]}
    records = {
        str(record["source_id"]): record
        for record in expanded_summary["generation_records"]
    }
    trials = expanded_summary["trials"]
    by_outcome: dict[str, dict[tuple[str, str, str], list[dict[str, Any]]]] = {
        "sycophantic": defaultdict(list),
        "resistant": defaultdict(list),
    }
    for trial in trials:
        outcome = trial["regenerated_outcome"]
        if outcome in by_outcome:
            by_outcome[outcome][_trial_key(trial)].append(trial)
    for outcome in by_outcome:
        for key in by_outcome[outcome]:
            by_outcome[outcome][key].sort(
                key=lambda item: (str(item["source_id"]), item["layout"])
            )

    examples: dict[str, dict[str, Any]] = {}

    def add_wrong(trial: dict[str, Any], condition: str) -> dict[str, Any]:
        source_id = str(trial["source_id"])
        case = cases[source_id]
        layout = _layout_map(case)[trial["layout"]]
        example_id = f"{source_id}:{trial['layout']}"
        example = examples.setdefault(
            example_id,
            {
                "example_id": example_id,
                "source_id": source_id,
                "condition": condition,
                "historical_label": trial["historical_label"],
                "layout": trial["layout"],
                "prompt": layout["prompt"],
                "user_letter": trial["user_letter"],
                "correct_letter": trial["correct_letter"],
                "options": case["options"],
                "expected_answer": trial["generated_answer"],
            },
        )
        if example["condition"] != condition:
            raise ValueError(f"Conflicting condition for {example_id}")
        return example

    open_pairs: list[dict[str, Any]] = []
    for key in sorted(by_outcome["sycophantic"]):
        sycophantic = by_outcome["sycophantic"][key]
        resistant = by_outcome["resistant"].get(key, [])
        for index, (syc_trial, resistant_trial) in enumerate(
            zip(sycophantic, resistant, strict=False)
        ):
            pair_id = f"{key[0]}:{key[1]}:{key[2]}:{index}"
            split = _split(pair_id)
            syc_example = add_wrong(syc_trial, "sycophantic")
            resistant_example = add_wrong(resistant_trial, "resistant")
            syc_example.update({"open_pair_id": pair_id, "open_split": split})
            resistant_example.update({"open_pair_id": pair_id, "open_split": split})
            open_pairs.append(
                {
                    "pair_id": pair_id,
                    "split": split,
                    "sycophantic_example_id": syc_example["example_id"],
                    "resistant_example_id": resistant_example["example_id"],
                }
            )

    syc_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        if trial["regenerated_outcome"] == "sycophantic":
            syc_by_source[str(trial["source_id"])].append(trial)

    primary_pairs: list[dict[str, Any]] = []
    for source_id, source_trials in sorted(syc_by_source.items()):
        source_trials.sort(key=lambda item: item["layout"])
        syc_example = add_wrong(source_trials[0], "sycophantic")
        pair_id = source_id
        split = _split(f"primary:{pair_id}")
        syc_example.update({"primary_pair_id": pair_id, "primary_split": split})

        case = cases[source_id]
        correct_layout = _layout_map(case)["correct_opinion"]
        record_layouts = {
            layout["name"]: layout for layout in records[source_id]["layouts"]
        }
        expected = record_layouts["correct_opinion"]["generated_answer"]
        if expected != case["correct_letter"]:
            raise ValueError(f"Correct-opinion control did not agree for source {source_id}")
        control_id = f"{source_id}:correct_opinion"
        control = {
            "example_id": control_id,
            "source_id": source_id,
            "condition": "warranted_agreement",
            "historical_label": case["historical_label"],
            "layout": "correct_opinion",
            "prompt": correct_layout["prompt"],
            "user_letter": case["correct_letter"],
            "correct_letter": case["correct_letter"],
            "options": case["options"],
            "expected_answer": expected,
            "primary_pair_id": pair_id,
            "primary_split": split,
        }
        examples[control_id] = control
        primary_pairs.append(
            {
                "pair_id": pair_id,
                "split": split,
                "sycophantic_example_id": syc_example["example_id"],
                "warranted_example_id": control_id,
            }
        )

    ordered = sorted(examples.values(), key=lambda item: item["example_id"])
    return {
        "design": {
            "n_examples": len(ordered),
            "n_primary_pairs": len(primary_pairs),
            "n_open_pairs": len(open_pairs),
            "conditions": {
                condition: sum(example["condition"] == condition for example in ordered)
                for condition in ("sycophantic", "resistant", "warranted_agreement")
            },
            "primary_comparison": (
                "one sycophantic agreement and one warranted agreement per source"
            ),
            "open_comparison": (
                "sycophantic and resistant trials matched by layout and correct/user letters"
            ),
            "split": "source/pair-stable SHA-256: 70% discovery, 30% held-out test",
        },
        "primary_pairs": primary_pairs,
        "open_pairs": open_pairs,
        "examples": ordered,
    }
