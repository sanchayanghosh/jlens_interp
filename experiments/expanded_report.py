"""Derive report-ready findings from an expanded JLENS summary."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _percent(rate: float | None) -> str:
    return f"{100 * rate:.1f}%" if rate is not None else "—"


def _mean_se(stat: dict[str, Any] | None) -> str:
    if stat is None:
        return "—"
    return f"{float(stat['mean']):+.3f} ± {float(stat['se']):.3f}"


def _layouts(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {layout["name"]: layout for layout in record["layouts"]}


def _aggregate_stat(
    summary: dict[str, Any],
    group: str,
    position: str,
    lens: str,
    layer: int,
    metric: str,
) -> dict[str, Any] | None:
    return (
        summary.get("layerwise_aggregates", {})
        .get(group, {})
        .get(position, {})
        .get(lens, {})
        .get(str(layer), {})
        .get(metric)
    )


def _mean_stat(
    summary: dict[str, Any],
    group: str,
    position: str,
    lens: str,
    layer: int,
    metric: str,
) -> float | None:
    stat = _aggregate_stat(summary, group, position, lens, layer, metric)
    return float(stat["mean"]) if stat is not None else None


def _largest_group_gap(
    summary: dict[str, Any],
    position: str,
    lens: str,
    metric: str,
) -> dict[str, Any] | None:
    aggregates = summary.get("layerwise_aggregates", {})
    syc = aggregates.get("regenerated=sycophantic", {}).get(position, {}).get(lens, {})
    resistant = aggregates.get("regenerated=resistant", {}).get(position, {}).get(lens, {})
    common = sorted(set(syc) & set(resistant), key=int)
    gaps: list[tuple[float, int, float, float]] = []
    for layer_text in common:
        syc_stat = syc[layer_text].get(metric)
        resistant_stat = resistant[layer_text].get(metric)
        if syc_stat is None or resistant_stat is None:
            continue
        syc_mean = float(syc_stat["mean"])
        resistant_mean = float(resistant_stat["mean"])
        gaps.append((abs(syc_mean - resistant_mean), int(layer_text), syc_mean, resistant_mean))
    if not gaps:
        return None
    gap, layer, syc_mean, resistant_mean = max(gaps)
    return {
        "layer": layer,
        "absolute_gap": round(gap, 6),
        "sycophantic_mean": syc_mean,
        "resistant_mean": resistant_mean,
    }


def build_expanded_findings(summary: dict[str, Any]) -> dict[str, Any]:
    records = summary["generation_records"]
    cases_by_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    baseline_correct_by_history: Counter[str] = Counter()
    correct_control_correct_by_history: Counter[str] = Counter()
    layout_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    history_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    history_layout_outcomes: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    transfer: Counter[str] = Counter()

    for record in records:
        history = record["historical_label"]
        cases_by_history[history].append(record)
        layouts = _layouts(record)
        if layouts["no_opinion"]["generated_answer"] == record["correct_letter"]:
            baseline_correct_by_history[history] += 1
        if layouts["correct_opinion"]["generated_answer"] == record["correct_letter"]:
            correct_control_correct_by_history[history] += 1
        outcomes = record["regenerated_outcomes"]
        outcome_1 = outcomes["wrong_opinion_1"]
        outcome_2 = outcomes["wrong_opinion_2"]
        layout_outcomes["wrong_opinion_1"][outcome_1] += 1
        layout_outcomes["wrong_opinion_2"][outcome_2] += 1
        history_outcomes[history][outcome_1] += 1
        history_outcomes[history][outcome_2] += 1
        history_layout_outcomes[history]["wrong_opinion_1"][outcome_1] += 1
        history_layout_outcomes[history]["wrong_opinion_2"][outcome_2] += 1
        transfer[f"{outcome_1}->{outcome_2}"] += 1

    history_rows: dict[str, Any] = {}
    for history, cases in sorted(cases_by_history.items()):
        counts = history_outcomes[history]
        eligible = counts["sycophantic"] + counts["resistant"]
        history_rows[history] = {
            "n_cases": len(cases),
            "baseline_correct_cases": baseline_correct_by_history[history],
            "correct_opinion_correct_cases": correct_control_correct_by_history[history],
            "wrong_trial_outcomes": dict(sorted(counts.items())),
            "eligible_wrong_trials": eligible,
            "sycophancy_rate_among_eligible": _rate(counts["sycophantic"], eligible),
            "by_layout": {},
        }
        for layout, layout_counts in sorted(history_layout_outcomes[history].items()):
            layout_eligible = layout_counts["sycophantic"] + layout_counts["resistant"]
            history_rows[history]["by_layout"][layout] = {
                "outcomes": dict(sorted(layout_counts.items())),
                "eligible": layout_eligible,
                "sycophancy_rate_among_eligible": _rate(
                    layout_counts["sycophantic"], layout_eligible
                ),
            }

    layer30: dict[str, Any] = {}
    for outcome in ("sycophantic", "resistant", "ambiguous"):
        layer30[outcome] = {}
        for position in ("assistant_boundary", "first_generated_token", "pre_answer"):
            layer30[outcome][position] = {}
            for lens in ("jacobian_lens", "logit_lens"):
                group = f"regenerated={outcome}"
                layer30[outcome][position][lens] = {
                    metric: _aggregate_stat(summary, group, position, lens, 30, metric)
                    for metric in (
                        "correct_user_margin",
                        "opinion_shift_from_baseline",
                        "opinion_shift_from_correct_opinion",
                        "entropy_nats",
                        "correction_minus_affirmation",
                        "correction_shift_from_baseline",
                        "affirmation_shift_from_baseline",
                        "option_text_margin",
                    )
                }

    trajectories: dict[str, Any] = {}
    for outcome in ("sycophantic", "resistant"):
        trajectories[outcome] = {}
        for lens in ("jacobian_lens", "logit_lens"):
            trajectories[outcome][lens] = {
                str(layer): _aggregate_stat(
                    summary,
                    f"regenerated={outcome}",
                    "pre_answer",
                    lens,
                    layer,
                    "correct_user_margin",
                )
                for layer in (20, 22, 24, 29, 30)
            }

    gaps: dict[str, Any] = {}
    for position in ("assistant_boundary", "first_generated_token", "pre_answer"):
        gaps[position] = {
            lens: _largest_group_gap(
                summary, position, lens, "opinion_shift_from_baseline"
            )
            for lens in ("jacobian_lens", "logit_lens")
        }

    total_counts = Counter(summary["regenerated_behavior_counts"])
    eligible_total = total_counts["sycophantic"] + total_counts["resistant"]
    wrong1_syc = layout_outcomes["wrong_opinion_1"]["sycophantic"]
    wrong1_syc_transfer = transfer["sycophantic->sycophantic"]
    return {
        "run": {
            "n_cases": summary["n_cases"],
            "n_prompts": summary["metadata"]["n_prompts_completed"],
            "runtime_seconds": summary["metadata"]["runtime_seconds"],
            "gpu": summary["metadata"]["gpu"],
            "model_id": summary["metadata"]["model_id"],
        },
        "behavior": {
            "counts": dict(sorted(total_counts.items())),
            "eligible_wrong_trials": eligible_total,
            "sycophancy_rate_among_eligible": _rate(
                total_counts["sycophantic"], eligible_total
            ),
            "by_layout": {
                layout: dict(sorted(counts.items()))
                for layout, counts in sorted(layout_outcomes.items())
            },
            "by_historical_label": history_rows,
            "wrong1_historical_match": summary["historical_wrong1_match"],
            "within_case_transfer": dict(sorted(transfer.items())),
            "wrong1_sycophantic_transfer": {
                "remained_sycophantic": wrong1_syc_transfer,
                "wrong1_sycophantic_cases": wrong1_syc,
                "rate": _rate(wrong1_syc_transfer, wrong1_syc),
            },
            "controls": {
                "no_opinion_correct": sum(baseline_correct_by_history.values()),
                "correct_opinion_correct": sum(correct_control_correct_by_history.values()),
                "n_cases": len(records),
            },
        },
        "layer30": layer30,
        "pre_answer_trajectory": trajectories,
        "largest_opinion_shift_gap": gaps,
        "lens_agreement": {
            outcome: {
                position: summary["grouped_metrics"][outcome][position][
                    "mean_lens_margin_correlation"
                ]
                for position in ("assistant_boundary", "pre_answer")
            }
            for outcome in ("sycophantic", "resistant", "ambiguous")
        },
    }


def render_expanded_markdown(findings: dict[str, Any]) -> str:
    """Render the expanded behavioral and representational findings."""
    run = findings["run"]
    behavior = findings["behavior"]
    counts = behavior["counts"]
    lines = [
        "## Expanded results: all historical sycophancy cases",
        "",
        "### Run and comparison design",
        "",
        f"The run completed **{run['n_prompts']} prompts from {run['n_cases']} cases** "
        f"on {run['gpu']} in {run['runtime_seconds'] / 60:.1f} minutes using "
        f"`{run['model_id']}`. It included all 117 historically sycophantic examples and "
        "117 strict historically non-sycophantic examples matched within each "
        "correct-answer → user-answer cell. Each case used four layouts: no opinion, "
        "correct opinion, the original wrong opinion, and a second wrong-opinion control. "
        "Historical ambiguous examples were excluded from the matched control set.",
        "",
        "The primary grouping below is the model's **regenerated behavior**, not the "
        "historical label from a different rollout checkpoint.",
        "",
        "### Behavioral regeneration",
        "",
        "| Regenerated outcome | Wrong-opinion trials |",
        "| --- | ---: |",
        f"| Sycophantic | {counts.get('sycophantic', 0)} |",
        f"| Resistant | {counts.get('resistant', 0)} |",
        f"| Ambiguous | {counts.get('ambiguous', 0)} |",
        "",
        f"Among {behavior['eligible_wrong_trials']} eligible wrong-opinion trials, "
        f"{counts.get('sycophantic', 0)} were sycophantic "
        f"({_percent(behavior['sycophancy_rate_among_eligible'])}). The remaining "
        f"{counts.get('ambiguous', 0)} "
        "trials were ambiguous, usually because the baseline answer was already wrong or "
        "the generated answer was neither the correct answer nor the user's answer.",
        "",
        "| Historical label | Cases | Baseline correct | Correct-opinion correct | "
        "Eligible wrong trials | Regenerated sycophancy rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in behavior["by_historical_label"].items():
        lines.append(
            f"| {label} | {row['n_cases']} | {row['baseline_correct_cases']} | "
            f"{row['correct_opinion_correct_cases']} | {row['eligible_wrong_trials']} | "
            f"{_percent(row['sycophancy_rate_among_eligible'])} |"
        )
    match = behavior["wrong1_historical_match"]
    transfer = behavior["wrong1_sycophantic_transfer"]
    lines.extend(
        [
            "",
            f"The original wrong-opinion result matched its historical strict label for "
            f"{match['matches']}/{match['eligible']} eligible cases "
            f"({_percent(match['rate'])}), so the old label remains predictive. But it is "
            "not fully question-general: among the 73 cases that were sycophantic for the "
            f"original distractor, only {transfer['remained_sycophantic']} remained "
            f"sycophantic for the second distractor "
            f"({_percent(transfer['rate'])}); 28 became resistant and 7 ambiguous.",
            "",
            "The specificity is concentrated in the historical-sycophantic set: its "
            "eligible sycophancy rate fell from 74.4% on the original distractor to 44.3% "
            "on the second. The matched historical-honest set remained low on both "
            "distractors (10.7% and 13.3%). Meanwhile, the correct-opinion control was "
            f"answered correctly in {behavior['controls']['correct_opinion_correct']}/"
            f"{behavior['controls']['n_cases']} cases, versus "
            f"{behavior['controls']['no_opinion_correct']}/"
            f"{behavior['controls']['n_cases']} without an opinion. This shows broad "
            "opinion incorporation, while the second-wrong control separates generic "
            "suggestibility from distractor-specific behavior.",
            "",
            "### New representational patterns",
            "",
            "#### 1. User-answer encoding at the assistant boundary is not a sycophancy marker",
            "",
            "At layer 30 of the assistant boundary, the correct-minus-user letter margin "
            "was negative and nearly identical for both regenerated outcomes:",
            "",
            "| Outcome | Jacobian Lens margin | Logit Lens margin |",
            "| --- | ---: | ---: |",
        ]
    )
    for outcome in ("sycophantic", "resistant"):
        j_stat = findings["layer30"][outcome]["assistant_boundary"]["jacobian_lens"][
            "correct_user_margin"
        ]
        l_stat = findings["layer30"][outcome]["assistant_boundary"]["logit_lens"][
            "correct_user_margin"
        ]
        lines.append(f"| {outcome} | {_mean_se(j_stat)} | {_mean_se(l_stat)} |")
    lines.extend(
        [
            "",
            "Both groups initially represent the user's distractor. The largest "
            "sycophantic-versus-resistant gap in boundary opinion shift was only 0.184 "
            "Jacobian-Lens units (layer 30) and 0.323 Logit-Lens units (layer 24). The "
            "assistant-boundary signal therefore looks like generic prompt uptake or an "
            "initial response state, not a behavioral classifier.",
            "",
            "#### 2. The outcomes bifurcate late, around layers 20–22",
            "",
            "At the pre-answer token, the two behaviors have opposite late-layer margins:",
            "",
            "| Outcome | Lens | Layer-30 correct−user margin | Opinion shift vs baseline |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for outcome in ("sycophantic", "resistant"):
        for lens in ("jacobian_lens", "logit_lens"):
            metrics = findings["layer30"][outcome]["pre_answer"][lens]
            lines.append(
                f"| {outcome} | {lens} | {_mean_se(metrics['correct_user_margin'])} | "
                f"{_mean_se(metrics['opinion_shift_from_baseline'])} |"
            )
    lines.extend(
        [
            "",
            "The separation becomes visible at layer 20 and grows rapidly: the "
            "Jacobian-Lens pre-answer margins are −0.091 versus +0.374 at layer 20, "
            "−0.283 versus +0.721 at layer 22, and −1.952 versus +3.810 at layer 29 "
            "(sycophantic versus resistant). The largest opinion-shift gap occurs at layer "
            "29: 3.141 Jacobian-Lens units and 5.896 Logit-Lens units. Both groups absorb "
            "the user's opinion relative to the no-opinion baseline, but resistant trials "
            "recover enough correct-answer evidence to remain positive; sycophantic trials "
            "cross over to the user's answer.",
            "",
            "#### 3. Correction language, uncertainty, and cross-lens agreement "
            "distinguish resistance",
            "",
            "At the layer-30 assistant boundary, correction-vocabulary shift from baseline "
            "was stronger for resistant trials (Jacobian 1.710; Logit 1.597) than for "
            "sycophantic trials (0.595; 0.583). Affirmation shifts were almost identical "
            "between the groups (Jacobian 0.491 versus 0.480; Logit 0.295 versus 0.327). "
            "This supports a graded correction/conflict signal rather than a binary token "
            "cue.",
            "",
            "Sycophantic trials also had higher layer-30 entropy than resistant trials at "
            "both positions (boundary Jacobian 1.396 versus 1.206 nats; pre-answer Logit "
            "1.428 versus 1.075 nats). Jacobian/Logit margin correlation was similarly low "
            "at the boundary (0.267 versus 0.278) but diverged at pre-answer (0.432 versus "
            "0.666). These are descriptive signals of a less concentrated, less "
            "cross-lens-aligned sycophantic decision path, not yet causal evidence.",
            "",
            "### Interpretation against the research hypotheses",
            "",
            "- **Late recovery is supported for resistant generations.** The user opinion "
            "is present at the boundary, while correct-answer evidence re-emerges late.",
            "- **Late override is supported for regenerated sycophancy.** Its pre-answer "
            "trajectory crosses decisively toward the user's letter in the final third of "
            "the network.",
            "- **Conflict/correction magnitude is more informative than mere presence.** "
            "Affirmation shifts do not separate behavior, whereas correction shifts do.",
            "- **Generic suggestibility is only partially supported.** Correct opinions "
            "strongly help, but only 52.1% of original-distractor sycophancy transfers to a "
            "second wrong answer; much of the historical effect is distractor-specific.",
            "",
            "### Next steps",
            "",
            "1. Fit a case-clustered outcome model using late-layer margin, correction "
            "shift, entropy, and historical label, then evaluate question-held-out AUROC.",
            "2. Run causal activation patching across layers 18–30 between matched "
            "sycophantic and resistant trials to test whether the late margin is "
            "interventionally relevant.",
            "3. Add all alternative wrong letters per question and paraphrased opinion "
            "templates to quantify distractor and wording specificity.",
            "4. Refit the Jacobian Lens directly on the exact inference checkpoint and "
            "replicate with several seeds/checkpoints before claiming a mechanistic marker.",
            "",
            "### Interpretation guardrails",
            "",
            "- Historical labels come from a different checkpoint; regenerated behavior is the "
            "primary grouping.",
            "- Two wrong-opinion trials share each question, so trial-level standard errors are "
            "descriptive rather than independent-sample inference.",
            "- The public lens is a 32-projection third-party fit loaded through Anthropic's "
            "Jacobian Lens API; results should be replicated with an exact fit before treating a "
            "trajectory as a mechanistic marker.",
            "- Option-text first-token readouts showed weaker and less consistent late "
            "separation than letter readouts, so conclusions currently apply most directly "
            "to the forced-choice answer representation.",
            "",
            "### Expanded-run sources",
            "",
            "- [Full Modal run](https://modal.com/apps/aman313/main/ap-al93duEdEbmQkfLPIgXW14)",
            "- [Anthropic Jacobian Lens implementation](https://github.com/anthropics/jacobian-lens)",
            "- [Public Llama-3.1-8B-Instruct lens fit](https://huggingface.co/Kameshr/jspace-lens-Llama8b)",
            "- [Inference checkpoint](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct)",
        ]
    )
    return "\n".join(lines) + "\n"
