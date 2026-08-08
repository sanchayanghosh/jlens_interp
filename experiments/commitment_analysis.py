"""Analyze the single-shot commitment detector and open J-space concepts."""

from __future__ import annotations

import math
import random
import re
from statistics import fmean
from typing import Any


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = 0.0
    for left in positive:
        for right in negative:
            wins += float(left > right) + 0.5 * float(left == right)
    return wins / (len(positive) * len(negative))


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return ordered[low]
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _score(
    example: dict[str, Any],
    position: str,
    lens: str,
    regime: str,
    family: str,
) -> float | None:
    points = (
        example.get("family_traces", {}).get(position, {}).get(lens, [])
    )
    values = []
    for point in points:
        if point["regime"] != regime:
            continue
        scores = point["family_log_probability"]
        if family in scores and "control" in scores:
            values.append(float(scores[family]) - float(scores["control"]))
    return _mean(values)


def _primary_metrics(
    examples: list[dict[str, Any]],
    position: str,
    lens: str,
    regime: str,
    family: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for example in examples:
        pair_id = example.get("primary_pair_id")
        if pair_id is None or not example["label_stable"]:
            continue
        by_pair.setdefault(pair_id, {})[example["condition"]] = example
    pairs: list[tuple[float, float]] = []
    for pair in by_pair.values():
        sycophantic = pair.get("sycophantic")
        warranted = pair.get("warranted_agreement")
        if sycophantic is None or warranted is None:
            continue
        syc_score = _score(sycophantic, position, lens, regime, family)
        warranted_score = _score(warranted, position, lens, regime, family)
        if syc_score is not None and warranted_score is not None:
            pairs.append((syc_score, warranted_score))
    positive = [pair[0] for pair in pairs]
    negative = [pair[1] for pair in pairs]
    differences = [left - right for left, right in pairs]
    result = {
        "n_pairs": len(pairs),
        "sycophantic_mean": _mean(positive),
        "warranted_mean": _mean(negative),
        "paired_mean_difference": _mean(differences),
        "paired_positive_rate": (
            sum(difference > 0 for difference in differences) / len(differences)
            if differences
            else None
        ),
        "auc": _auc(positive, negative),
    }
    if not pairs or bootstrap_samples <= 0:
        return result
    rng = random.Random(20260808)
    auc_samples: list[float] = []
    difference_samples: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        sampled_auc = _auc(
            [pair[0] for pair in sampled],
            [pair[1] for pair in sampled],
        )
        if sampled_auc is not None:
            auc_samples.append(sampled_auc)
        difference_samples.append(fmean(left - right for left, right in sampled))
    result["auc_95_ci"] = [
        _percentile(auc_samples, 0.025),
        _percentile(auc_samples, 0.975),
    ]
    result["paired_difference_95_ci"] = [
        _percentile(difference_samples, 0.025),
        _percentile(difference_samples, 0.975),
    ]
    return result


def _open_family_metrics(
    examples: list[dict[str, Any]],
    position: str,
    lens: str,
    regime: str,
    family: str,
) -> dict[str, Any]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for example in examples:
        pair_id = example.get("open_pair_id")
        if pair_id is None or not example["label_stable"]:
            continue
        by_pair.setdefault(pair_id, {})[example["condition"]] = example
    pairs: list[tuple[float, float]] = []
    for pair in by_pair.values():
        sycophantic = pair.get("sycophantic")
        resistant = pair.get("resistant")
        if sycophantic is None or resistant is None:
            continue
        syc_score = _score(sycophantic, position, lens, regime, family)
        resistant_score = _score(resistant, position, lens, regime, family)
        if syc_score is not None and resistant_score is not None:
            pairs.append((syc_score, resistant_score))
    sycophantic_scores = [pair[0] for pair in pairs]
    resistant_scores = [pair[1] for pair in pairs]
    differences = [left - right for left, right in pairs]
    return {
        "n_pairs": len(pairs),
        "sycophantic_mean": _mean(sycophantic_scores),
        "resistant_mean": _mean(resistant_scores),
        "paired_mean_difference": _mean(differences),
        "paired_positive_rate": (
            sum(difference > 0 for difference in differences) / len(differences)
            if differences
            else None
        ),
        "auc_sycophantic_positive": _auc(sycophantic_scores, resistant_scores),
    }


def _concept_entry(
    counts: dict[str, Any],
    split: str,
    condition: str,
    position: str,
    lens: str,
) -> dict[str, Any]:
    return (
        counts.get(split, {})
        .get(condition, {})
        .get(position, {})
        .get(lens, {"slots": 0, "hidden_counts": {}})
    )


def _open_concepts(
    raw: dict[str, Any],
    position: str,
    lens: str,
    n_tokens: int = 20,
) -> list[dict[str, Any]]:
    counts = raw["concept_counts"]
    token_text = raw["token_text"]
    discovery = {
        condition: _concept_entry(counts, "discovery", condition, position, lens)
        for condition in ("sycophantic", "resistant")
    }
    test = {
        condition: _concept_entry(counts, "test", condition, position, lens)
        for condition in ("sycophantic", "resistant")
    }
    canonical_counts: dict[str, dict[str, dict[str, int]]] = {
        split: {condition: {} for condition in ("sycophantic", "resistant")}
        for split in ("discovery", "test")
    }
    for split, entries in (("discovery", discovery), ("test", test)):
        for condition, entry in entries.items():
            for token_id, count in entry["hidden_counts"].items():
                text = token_text.get(token_id, "")
                stripped = text.strip()
                starts_word = text.startswith((" ", "\n")) or (
                    stripped and stripped[0].isupper()
                )
                if (
                    not starts_word
                    or not re.fullmatch(r"[A-Za-z][A-Za-z-]{1,29}", stripped)
                ):
                    continue
                canonical = stripped.lower()
                canonical_counts[split][condition][canonical] = (
                    canonical_counts[split][condition].get(canonical, 0) + int(count)
                )
    concepts = set(canonical_counts["discovery"]["sycophantic"]) | set(
        canonical_counts["discovery"]["resistant"]
    )
    candidates: list[tuple[float, str, float, float]] = []
    for concept in concepts:
        total = sum(
            canonical_counts["discovery"][condition].get(concept, 0)
            for condition in ("sycophantic", "resistant")
        )
        if total < 5:
            continue
        rates = {
            condition: (
                canonical_counts["discovery"][condition].get(concept, 0)
                / int(discovery[condition]["slots"])
                if discovery[condition]["slots"]
                else 0.0
            )
            for condition in ("sycophantic", "resistant")
        }
        gap = rates["sycophantic"] - rates["resistant"]
        candidates.append((abs(gap), concept, rates["sycophantic"], rates["resistant"]))
    rows: list[dict[str, Any]] = []
    for _absolute_gap, concept, syc_rate, resistant_rate in sorted(
        candidates, reverse=True
    )[:n_tokens]:
        test_rates = {
            condition: (
                canonical_counts["test"][condition].get(concept, 0)
                / int(test[condition]["slots"])
                if test[condition]["slots"]
                else 0.0
            )
            for condition in ("sycophantic", "resistant")
        }
        discovery_gap = syc_rate - resistant_rate
        test_gap = test_rates["sycophantic"] - test_rates["resistant"]
        rows.append(
            {
                "concept": concept,
                "discovery_sycophantic_rate": syc_rate,
                "discovery_resistant_rate": resistant_rate,
                "discovery_gap": discovery_gap,
                "test_sycophantic_rate": test_rates["sycophantic"],
                "test_resistant_rate": test_rates["resistant"],
                "test_gap": test_gap,
                "direction_replicated": discovery_gap * test_gap > 0,
            }
        )
    return rows


def analyze_commitment_results(
    raw: dict[str, Any],
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Return preregistered detector results and held-out open concept contrasts."""
    examples = raw["examples"]
    positions = raw["metadata"]["readout_positions"]
    primary: dict[str, Any] = {}
    for position in positions:
        primary[position] = {}
        for lens in ("jacobian_lens", "logit_lens"):
            primary[position][lens] = {}
            for regime in ("workspace", "motor"):
                primary[position][lens][regime] = {
                    family: _primary_metrics(
                        examples,
                        position,
                        lens,
                        regime,
                        family,
                        (
                            bootstrap_samples
                            if (
                                position,
                                lens,
                                regime,
                                family,
                            )
                            == (
                                "generated_answer",
                                "jacobian_lens",
                                "workspace",
                                "objection",
                            )
                            else 0
                        ),
                    )
                    for family in (
                        "objection",
                        "social_approval",
                        "epistemic",
                        "evaluation_awareness",
                    )
                }
    open_analysis = {
        position: {
            lens: _open_concepts(raw, position, lens)
            for lens in ("jacobian_lens", "logit_lens")
        }
        for position in positions
    }
    open_families: dict[str, Any] = {}
    for position in positions:
        open_families[position] = {}
        for lens in ("jacobian_lens", "logit_lens"):
            open_families[position][lens] = {}
            for regime in ("workspace", "motor"):
                open_families[position][lens][regime] = {
                    family: _open_family_metrics(
                        examples,
                        position,
                        lens,
                        regime,
                        family,
                    )
                    for family in (
                        "objection",
                        "social_approval",
                        "epistemic",
                        "evaluation_awareness",
                    )
                }
    stable = sum(example["label_stable"] for example in examples)
    return {
        "run": raw["metadata"],
        "design": raw["design"],
        "label_stability": {
            "stable": stable,
            "total": len(examples),
            "rate": stable / len(examples) if examples else None,
        },
        "preregistered": {
            "feature": (
                "workspace objection-family log-probability minus unrelated control-family "
                "log-probability at the generated answer token"
            ),
            "comparison": "sycophantic agreement versus warranted agreement",
            "result": primary["generated_answer"]["jacobian_lens"]["workspace"][
                "objection"
            ],
        },
        "primary_family_results": primary,
        "open_family_results": open_families,
        "open_concepts": open_analysis,
    }


def render_commitment_markdown(findings: dict[str, Any]) -> str:
    run = findings["run"]
    result = findings["preregistered"]["result"]
    opinion_objection = findings["open_family_results"]["user_opinion_letter"][
        "jacobian_lens"
    ]["workspace"]["objection"]
    boundary_objection = findings["open_family_results"]["assistant_boundary"][
        "jacobian_lens"
    ]["workspace"]["objection"]
    posthoc_primary = findings["primary_family_results"]["user_opinion_letter"][
        "jacobian_lens"
    ]["workspace"]["objection"]
    lines = [
        "## Single-shot commitment and open-concept experiment",
        "",
        f"The run completed {run['n_examples_completed']} examples on {run['gpu']} using "
        f"`{run['model_id']}`. Labels reproduced for "
        f"{findings['label_stability']['stable']}/{findings['label_stability']['total']} "
        "examples.",
        "",
        "### Preregistered post-commitment objection score",
        "",
        "The detector uses one opinion-prompt trace, no answer key, and no counterfactual at "
        "inference time. It compares sycophantic wrong agreement with warranted correct "
        "agreement on the same questions.",
        "",
        f"- Matched pairs: {result['n_pairs']}",
        f"- Sycophantic mean: {result['sycophantic_mean']:.3f}",
        f"- Warranted-agreement mean: {result['warranted_mean']:.3f}",
        f"- Paired difference: {result['paired_mean_difference']:.3f}",
        f"- AUROC: {result['auc']:.3f}",
        "- Cluster-bootstrap AUROC 95% CI: "
        f"[{result['auc_95_ci'][0]:.3f}, {result['auc_95_ci'][1]:.3f}]",
        "",
        "This is a modest answer-key-free signal, not a deployment-ready detector. A "
        "post-hoc scan found a stronger objection score at the user-opinion token "
        f"(AUROC {posthoc_primary['auc']:.3f}), but that estimate is exploratory because "
        "the position was selected after seeing the results.",
        "",
        "### Open J-space concepts",
        "",
        "The matched resistant comparison reveals a clearer semantic pattern. At the "
        "user-opinion token, sycophantic cases have a lower objection-family score than "
        f"resistant cases (paired difference {opinion_objection['paired_mean_difference']:.3f}; "
        f"AUROC with sycophancy positive {opinion_objection['auc_sycophantic_positive']:.3f}). "
        "The same direction is present at the assistant boundary "
        f"(difference {boundary_objection['paired_mean_difference']:.3f}; AUROC "
        f"{boundary_objection['auc_sycophantic_positive']:.3f}).",
        "",
        "Canonical word concepts below were selected by hidden top-25 prevalence "
        "differences in the 70% discovery split and checked on the held-out 30%. Positive "
        "gaps mean more common in sycophantic cases; negative gaps mean more common in "
        "resistant cases.",
    ]
    for position in (
        "user_opinion_letter",
        "assistant_boundary",
        "first_generated_token",
        "generated_answer",
    ):
        lines.extend(["", f"#### {position} — Jacobian Lens", ""])
        lines.extend(
            [
                "| Concept | Discovery gap | Test gap | Replicated |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for row in findings["open_concepts"][position]["jacobian_lens"][:10]:
            lines.append(
                f"| `{row['concept']}` | {row['discovery_gap']:.4f} | "
                f"{row['test_gap']:.4f} | {row['direction_replicated']} |"
            )
    lines.extend(
        [
            "",
            "The robust early contrast is best described as error recognition versus "
            "rationalization: resistant traces surface concepts such as `wrong`, "
            "`incorrect`, `but`, and `however`, while sycophantic traces more often surface "
            "causal or attributional concepts such as `because`, `based`, and `according`. "
            "The generated-answer vocabulary is substantially less coherent and should not "
            "be treated as a stable post-commitment marker.",
            "",
            "### Guardrails",
            "",
            "- The workspace band (layers 12–27) is a paper-motivated approximation for "
            "this 32-layer checkpoint, not a fitted boundary.",
            "- Open vocabulary findings are matched by layout and answer-letter mapping, but "
            "may still reflect difficulty, wording, or tokenization.",
            "- J-lens tokens are correlational projections and should not be described as "
            "literal thoughts without intervention evidence.",
        ]
    )
    return "\n".join(lines) + "\n"
