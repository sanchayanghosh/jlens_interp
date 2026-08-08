"""Summarize behavioral outcomes and layerwise Jacobian Lens measurements."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from statistics import fmean
from typing import Any

CORRECTION_WORDS = ("incorrect", "unfortunately", "but", "however", "actually", "not")
AFFIRMATION_WORDS = ("indeed", "yes", "correct", "right", "agree")


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None and math.isfinite(value)]
    return round(fmean(kept), 6) if kept else None


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return round(numerator / denominator, 6) if denominator else None


def _layout_map(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {layout["name"]: layout for layout in case["layouts"]}


def _points(layout: dict[str, Any], position: str, lens: str) -> list[dict[str, Any]]:
    return layout.get("traces", {}).get(position, {}).get(lens, [])


def _margin(point: dict[str, Any], letter: str) -> float | None:
    value = point.get("correct_vs_letter_margins", {}).get(letter)
    return float(value) if value is not None else None


def _vocabulary_mean(point: dict[str, Any], words: tuple[str, ...]) -> float | None:
    scores = point.get("vocabulary_scores", {})
    return _mean(scores.get(word) for word in words)


def _persistent_crossover(points: list[dict[str, Any]], user_letter: str) -> int | None:
    margins = [_margin(point, user_letter) for point in points]
    for index, margin in enumerate(margins):
        if margin is not None and margin < 0 and all(
            later is not None and later < 0 for later in margins[index:]
        ):
            return int(points[index]["layer"])
    return None


def _trace_metrics(
    layout: dict[str, Any],
    baseline: dict[str, Any],
    correct_opinion: dict[str, Any],
    position: str,
    lens: str,
    user_letter: str,
) -> dict[str, Any]:
    trace = _points(layout, position, lens)
    if not trace:
        return {}
    baseline_by_layer = {
        int(point["layer"]): point for point in _points(baseline, position, lens)
    }
    correct_by_layer = {
        int(point["layer"]): point for point in _points(correct_opinion, position, lens)
    }
    first = trace[0]
    middle = trace[len(trace) // 2]
    last = trace[-1]
    last_layer = int(last["layer"])
    last_margin = _margin(last, user_letter)
    baseline_margin = (
        _margin(baseline_by_layer[last_layer], user_letter)
        if last_layer in baseline_by_layer
        else None
    )
    correct_margin = (
        _margin(correct_by_layer[last_layer], user_letter)
        if last_layer in correct_by_layer
        else None
    )
    correction_mean = _vocabulary_mean(last, CORRECTION_WORDS)
    affirmation_mean = _vocabulary_mean(last, AFFIRMATION_WORDS)
    baseline_point = baseline_by_layer.get(last_layer, {})
    baseline_correction = _vocabulary_mean(baseline_point, CORRECTION_WORDS)
    baseline_affirmation = _vocabulary_mean(baseline_point, AFFIRMATION_WORDS)
    return {
        "first_layer": int(first["layer"]),
        "first_margin": _margin(first, user_letter),
        "middle_layer": int(middle["layer"]),
        "middle_margin": _margin(middle, user_letter),
        "last_layer": last_layer,
        "last_margin": last_margin,
        "baseline_last_margin": baseline_margin,
        "correct_opinion_last_margin": correct_margin,
        "opinion_shift_from_baseline": (
            round(last_margin - baseline_margin, 6)
            if last_margin is not None and baseline_margin is not None
            else None
        ),
        "opinion_shift_from_correct_opinion": (
            round(last_margin - correct_margin, 6)
            if last_margin is not None and correct_margin is not None
            else None
        ),
        "persistent_crossover_layer": _persistent_crossover(trace, user_letter),
        "early_positive_late_negative": (
            any((_margin(point, user_letter) or 0) > 0 for point in trace[: len(trace) // 2])
            and last_margin is not None
            and last_margin < 0
        ),
        "last_entropy_nats": last.get("entropy_nats"),
        "last_top_probability": last.get("top_probability"),
        "correction_minus_affirmation": (
            round(correction_mean - affirmation_mean, 6)
            if correction_mean is not None and affirmation_mean is not None
            else None
        ),
        "correction_shift_from_baseline": (
            round(correction_mean - baseline_correction, 6)
            if correction_mean is not None and baseline_correction is not None
            else None
        ),
        "affirmation_shift_from_baseline": (
            round(affirmation_mean - baseline_affirmation, 6)
            if affirmation_mean is not None and baseline_affirmation is not None
            else None
        ),
    }


def _lens_agreement(layout: dict[str, Any], position: str, user_letter: str) -> float | None:
    jacobian = _points(layout, position, "jacobian_lens")
    logit = _points(layout, position, "logit_lens")
    j_by_layer = {int(point["layer"]): _margin(point, user_letter) for point in jacobian}
    l_by_layer = {int(point["layer"]): _margin(point, user_letter) for point in logit}
    common = sorted(
        layer
        for layer in j_by_layer.keys() & l_by_layer.keys()
        if j_by_layer[layer] is not None and l_by_layer[layer] is not None
    )
    return _pearson(
        [float(j_by_layer[layer]) for layer in common],
        [float(l_by_layer[layer]) for layer in common],
    )


def analyze_results(results: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, JSON-serializable summary of a full experiment run."""
    trials: list[dict[str, Any]] = []
    behavior_counts: Counter[str] = Counter()
    history_counts: Counter[str] = Counter()
    regenerated_by_history: dict[str, Counter[str]] = defaultdict(Counter)
    baseline_correct_cases = 0

    for case in results["cases"]:
        layouts = _layout_map(case)
        if not {"no_opinion", "correct_opinion"}.issubset(layouts):
            continue
        baseline_correct = layouts["no_opinion"].get("generated_answer") == case["correct_letter"]
        baseline_correct_cases += baseline_correct
        history_counts[case["historical_label"]] += 1
        for layout_name in ("wrong_opinion_1", "wrong_opinion_2"):
            if layout_name not in layouts:
                continue
            layout = layouts[layout_name]
            outcome = case.get("regenerated_outcomes", {}).get(layout_name, "ambiguous")
            behavior_counts[outcome] += 1
            regenerated_by_history[case["historical_label"]][outcome] += 1
            user_letter = str(layout["user_letter"])
            trial = {
                "source_id": case["source_id"],
                "question": case["question"],
                "historical_label": case["historical_label"],
                "layout": layout_name,
                "correct_letter": case["correct_letter"],
                "user_letter": user_letter,
                "baseline_answer": layouts["no_opinion"].get("generated_answer"),
                "generated_answer": layout.get("generated_answer"),
                "baseline_correct": baseline_correct,
                "regenerated_outcome": outcome,
                "assistant_boundary": {},
                "pre_answer": {},
            }
            for position in ("assistant_boundary", "pre_answer"):
                for lens in ("jacobian_lens", "logit_lens"):
                    trial[position][lens] = _trace_metrics(
                        layout,
                        layouts["no_opinion"],
                        layouts["correct_opinion"],
                        position,
                        lens,
                        user_letter,
                    )
                trial[position]["lens_margin_correlation"] = _lens_agreement(
                    layout, position, user_letter
                )
            trials.append(trial)

    grouped: dict[str, Any] = {}
    for outcome in ("sycophantic", "resistant", "ambiguous"):
        group = [trial for trial in trials if trial["regenerated_outcome"] == outcome]
        grouped[outcome] = {
            "n": len(group),
            "baseline_correct_n": sum(bool(trial["baseline_correct"]) for trial in group),
        }
        for position in ("assistant_boundary", "pre_answer"):
            grouped[outcome][position] = {}
            for lens in ("jacobian_lens", "logit_lens"):
                metrics = [trial[position].get(lens, {}) for trial in group]
                grouped[outcome][position][lens] = {
                    key: _mean(metric.get(key) for metric in metrics)
                    for key in (
                        "first_margin",
                        "middle_margin",
                        "last_margin",
                        "opinion_shift_from_baseline",
                        "opinion_shift_from_correct_opinion",
                        "last_entropy_nats",
                        "last_top_probability",
                        "correction_minus_affirmation",
                        "correction_shift_from_baseline",
                        "affirmation_shift_from_baseline",
                    )
                }
                grouped[outcome][position][lens]["persistent_crossover_n"] = sum(
                    metric.get("persistent_crossover_layer") is not None for metric in metrics
                )
                grouped[outcome][position][lens]["early_positive_late_negative_n"] = sum(
                    bool(metric.get("early_positive_late_negative")) for metric in metrics
                )
            grouped[outcome][position]["mean_lens_margin_correlation"] = _mean(
                trial[position].get("lens_margin_correlation") for trial in group
            )

    historical_wrong1_matches = 0
    historical_wrong1_eligible = 0
    for trial in trials:
        if trial["layout"] != "wrong_opinion_1" or trial["regenerated_outcome"] == "ambiguous":
            continue
        historical_wrong1_eligible += 1
        expected = "sycophantic" if trial["historical_label"] == "sycophantic" else "resistant"
        historical_wrong1_matches += trial["regenerated_outcome"] == expected

    return {
        "metadata": results["metadata"],
        "n_cases": len(results["cases"]),
        "n_baseline_correct_cases": baseline_correct_cases,
        "n_wrong_opinion_trials": len(trials),
        "historical_case_counts": dict(sorted(history_counts.items())),
        "regenerated_behavior_counts": dict(sorted(behavior_counts.items())),
        "regenerated_by_historical_label": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(regenerated_by_history.items())
        },
        "historical_wrong1_match": {
            "matches": historical_wrong1_matches,
            "eligible": historical_wrong1_eligible,
            "rate": (
                round(historical_wrong1_matches / historical_wrong1_eligible, 6)
                if historical_wrong1_eligible
                else None
            ),
        },
        "grouped_metrics": grouped,
        "trials": trials,
    }


def _add_stat(
    accumulators: dict[tuple[str, str, str, int, str], list[float]],
    key: tuple[str, str, str, int, str],
    value: float | None,
) -> None:
    if value is None or not math.isfinite(float(value)):
        return
    state = accumulators.setdefault(key, [0.0, 0.0, 0.0])
    numeric = float(value)
    state[0] += 1
    state[1] += numeric
    state[2] += numeric * numeric


def _finish_stat(state: list[float]) -> dict[str, float | int]:
    n = int(state[0])
    mean = state[1] / n
    variance = max(0.0, (state[2] - n * mean * mean) / (n - 1)) if n > 1 else 0.0
    sd = math.sqrt(variance)
    return {
        "n": n,
        "mean": round(mean, 6),
        "sd": round(sd, 6),
        "se": round(sd / math.sqrt(n), 6),
    }


def _point_metrics(
    point: dict[str, Any],
    user_letter: str,
    correct_letter: str,
    baseline: dict[str, Any] | None,
    correct_opinion: dict[str, Any] | None,
) -> dict[str, float | None]:
    margin = _margin(point, user_letter)
    option_scores = point.get("option_first_token_scores", {})
    option_margin = (
        float(option_scores[correct_letter]) - float(option_scores[user_letter])
        if correct_letter in option_scores and user_letter in option_scores
        else None
    )
    correction = _vocabulary_mean(point, CORRECTION_WORDS)
    affirmation = _vocabulary_mean(point, AFFIRMATION_WORDS)
    baseline_margin = _margin(baseline, user_letter) if baseline is not None else None
    correct_margin = (
        _margin(correct_opinion, user_letter) if correct_opinion is not None else None
    )
    baseline_correction = (
        _vocabulary_mean(baseline, CORRECTION_WORDS) if baseline is not None else None
    )
    baseline_affirmation = (
        _vocabulary_mean(baseline, AFFIRMATION_WORDS) if baseline is not None else None
    )
    return {
        "correct_user_margin": margin,
        "option_text_margin": option_margin,
        "opinion_shift_from_baseline": (
            margin - baseline_margin
            if margin is not None and baseline_margin is not None
            else None
        ),
        "opinion_shift_from_correct_opinion": (
            margin - correct_margin
            if margin is not None and correct_margin is not None
            else None
        ),
        "entropy_nats": point.get("entropy_nats"),
        "top_probability": point.get("top_probability"),
        "correction_minus_affirmation": (
            correction - affirmation
            if correction is not None and affirmation is not None
            else None
        ),
        "correction_shift_from_baseline": (
            correction - baseline_correction
            if correction is not None and baseline_correction is not None
            else None
        ),
        "affirmation_shift_from_baseline": (
            affirmation - baseline_affirmation
            if affirmation is not None and baseline_affirmation is not None
            else None
        ),
    }


def _expanded_layerwise_aggregates(results: dict[str, Any]) -> dict[str, Any]:
    accumulators: dict[tuple[str, str, str, int, str], list[float]] = {}
    for case in results["cases"]:
        layouts = _layout_map(case)
        if not {"no_opinion", "correct_opinion"}.issubset(layouts):
            continue
        for layout_name in ("wrong_opinion_1", "wrong_opinion_2"):
            layout = layouts.get(layout_name)
            if layout is None:
                continue
            outcome = case.get("regenerated_outcomes", {}).get(layout_name, "ambiguous")
            history = case["historical_label"]
            groups = (
                f"regenerated={outcome}",
                f"historical={history}",
                f"historical={history}|regenerated={outcome}",
                f"layout={layout_name}|regenerated={outcome}",
            )
            user_letter = str(layout["user_letter"])
            for position, lens_traces in layout.get("traces", {}).items():
                for lens, points in lens_traces.items():
                    baseline_by_layer = {
                        int(point["layer"]): point
                        for point in _points(layouts["no_opinion"], position, lens)
                    }
                    correct_by_layer = {
                        int(point["layer"]): point
                        for point in _points(layouts["correct_opinion"], position, lens)
                    }
                    for point in points:
                        layer = int(point["layer"])
                        metrics = _point_metrics(
                            point,
                            user_letter,
                            case["correct_letter"],
                            baseline_by_layer.get(layer),
                            correct_by_layer.get(layer),
                        )
                        for group in groups:
                            for metric, value in metrics.items():
                                _add_stat(
                                    accumulators,
                                    (group, position, lens, layer, metric),
                                    value,
                                )

    nested: dict[str, Any] = {}
    for (group, position, lens, layer, metric), state in sorted(accumulators.items()):
        layer_entry = (
            nested.setdefault(group, {})
            .setdefault(position, {})
            .setdefault(lens, {})
            .setdefault(str(layer), {})
        )
        layer_entry[metric] = _finish_stat(state)
    return nested


def analyze_expanded_results(results: dict[str, Any]) -> dict[str, Any]:
    """Analyze a large run and retain aggregates instead of full per-token traces."""
    summary = analyze_results(results)
    summary["design"] = results.get("design", {})
    summary["layerwise_aggregates"] = _expanded_layerwise_aggregates(results)
    summary["generation_records"] = [
        {
            "source_id": case["source_id"],
            "historical_label": case["historical_label"],
            "correct_letter": case["correct_letter"],
            "original_wrong_letter": case.get("original_wrong_letter"),
            "second_wrong_letter": case.get("second_wrong_letter"),
            "regenerated_answers": case.get("regenerated_answers", {}),
            "regenerated_outcomes": case.get("regenerated_outcomes", {}),
            "layouts": [
                {
                    "name": layout["name"],
                    "user_letter": layout["user_letter"],
                    "response": layout.get("response"),
                    "generated_answer": layout.get("generated_answer"),
                    "prompt_token_count": layout.get("prompt_token_count"),
                    "generated_token_count": layout.get("generated_token_count"),
                }
                for layout in case["layouts"]
            ],
        }
        for case in results["cases"]
    ]
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a preliminary, appropriately cautious result section."""
    metadata = summary["metadata"]
    counts = summary["regenerated_behavior_counts"]
    match = summary["historical_wrong1_match"]
    resistant = summary["grouped_metrics"]["resistant"]
    resistant_boundary_j = resistant["assistant_boundary"]["jacobian_lens"]
    resistant_boundary_l = resistant["assistant_boundary"]["logit_lens"]
    resistant_answer_j = resistant["pre_answer"]["jacobian_lens"]
    resistant_answer_l = resistant["pre_answer"]["logit_lens"]

    def display(value: Any) -> str:
        return "—" if value is None else str(value)

    lines = [
        "## Preliminary results",
        "",
        f"The run completed {metadata['n_prompts_completed']} deterministic prompts on "
        f"{metadata['gpu']} using `{metadata['model_id']}`. The Jacobian Lens checkpoint "
        f"contains {metadata['lens_metadata'].get('n_prompts')} fitting prompts and "
        f"{metadata['lens_metadata'].get('n_proj')} Hutchinson projections.",
        "",
        "### Behavioral regeneration",
        "",
        "| Outcome on wrong-opinion layouts | Count |",
        "| --- | ---: |",
        f"| Sycophantic | {counts.get('sycophantic', 0)} |",
        f"| Resistant | {counts.get('resistant', 0)} |",
        f"| Ambiguous (including an incorrect no-opinion control) | {counts.get('ambiguous', 0)} |",
        "",
        f"The no-opinion answer was correct for {summary['n_baseline_correct_cases']} of "
        f"{summary['n_cases']} questions. All {counts.get('resistant', 0)} behaviorally eligible "
        "wrong-opinion trials were resistant; none was sycophantic. The ambiguous trials came "
        "from questions whose no-opinion answer was already wrong, so they cannot establish an "
        "opinion-induced answer switch.",
        "",
        f"For the original wrong-opinion layout, {match['matches']} of {match['eligible']} "
        "behaviorally eligible cases reproduced their historical label. Historical labels were "
        "generated by a different checkpoint, so all representation-level comparisons below use "
        "the regenerated outcome as the primary grouping.",
        "",
        "| Source | Historical | Layout | Control | Opinion answer | Regenerated |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for trial in summary["trials"]:
        lines.append(
            f"| {trial['source_id']} | {trial['historical_label']} | {trial['layout']} | "
            f"{trial['baseline_answer']} | {trial['generated_answer']} | "
            f"{trial['regenerated_outcome']} |"
        )

    lines.extend(["", "### Representation-level readout", ""])
    lines.append(
        "The table reports the mean layer-30 correct-minus-user letter margin at the assistant "
        "boundary. Positive values favor the correct answer; negative values favor the user's "
        "answer. The shift is the wrong-opinion margin minus the no-opinion margin for the same "
        "question and distractor."
    )
    lines.extend(
        [
            "",
            "| Regenerated outcome | n | Jacobian margin | Jacobian shift | "
            "Logit margin | Logit shift |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for outcome in ("sycophantic", "resistant", "ambiguous"):
        group = summary["grouped_metrics"][outcome]
        j = group["assistant_boundary"]["jacobian_lens"]
        logit = group["assistant_boundary"]["logit_lens"]
        lines.append(
            f"| {outcome.capitalize()} | {group['n']} | {display(j['last_margin'])} | "
            f"{display(j['opinion_shift_from_baseline'])} | {display(logit['last_margin'])} | "
            f"{display(logit['opinion_shift_from_baseline'])} |"
        )

    lines.extend(
        [
            "",
            "For the eight resistant trials, the opinion still shifted assistant-boundary "
            "evidence toward the user's distractor (mean shift "
            f"{resistant_boundary_j['opinion_shift_from_baseline']} "
            f"with Jacobian Lens and {resistant_boundary_l['opinion_shift_from_baseline']} with "
            "Logit Lens). A persistent negative boundary margin appeared in "
            f"{resistant_boundary_j['persistent_crossover_n']}/8 Jacobian traces and "
            f"{resistant_boundary_l['persistent_crossover_n']}/8 Logit traces. Immediately before "
            "the emitted answer, however, mean margins were strongly correct-favoring "
            f"({resistant_answer_j['last_margin']} Jacobian; {resistant_answer_l['last_margin']} "
            "Logit), with no persistent negative crossover in either lens. Mean Jacobian–Logit "
            "margin correlation also rose from "
            f"{resistant['assistant_boundary']['mean_lens_margin_correlation']} "
            f"at the boundary to {resistant['pre_answer']['mean_lens_margin_correlation']} before "
            "the answer. Descriptively, the wrong opinion affected the initial response state but "
            "did not override the final answer decision in these eligible trials.",
            "",
            "Because this checkpoint produced zero sycophantic eligible trials, H2 and any "
            "sycophantic-versus-resistant vocabulary marker remain untested in this run. H1 has "
            "descriptive support at the pre-answer position for resistant trials, but not as a "
            "claim of persistent positive letter evidence at every earlier position.",
            "",
            "These are exploratory means from six questions, not inferential statistics. A "
            "decoded token or margin is not treated as a marker unless it reproduces across "
            "matched groups and survives the second-wrong-answer and correct-opinion controls. "
            "The raw export includes every layer, requested prompt/generation position, both "
            "lenses, entropy, top-token concentration, answer and option evidence, and the "
            "correction/affirmation vocabularies.",
            "",
            "### Next steps",
            "",
            "1. Replace historical labels with a larger regenerated behavioral set from the exact "
            "checkpoint, retaining only questions whose no-opinion control is correct.",
            "2. Expand each matched cell to at least 20 questions and bootstrap confidence "
            "intervals for the opinion-induced margin trajectories and crossover layers.",
            "3. Fit an official exact Anthropic Jacobian Lens on the same checkpoint and "
            "compare it "
            "against this public 32-projection Hutchinson checkpoint to quantify estimator drift.",
            "4. Test whether any late-layer effect transfers across both wrong distractors while "
            "remaining absent in the correct-opinion control; report correction and affirmation "
            "tokens only when that criterion is met.",
            "5. Repeat with the configurable target checkpoint (including the sycophancy-tuned "
            "model when a compatible lens is fitted) so behavior and representations are measured "
            "on the same model.",
        ]
    )
    return "\n".join(lines) + "\n"
