from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.commitment_analysis import analyze_commitment_results  # noqa: E402
from experiments.commitment_data import build_commitment_payload  # noqa: E402


def example(pair: str, condition: str, score: float) -> dict:
    return {
        "primary_pair_id": pair,
        "condition": condition,
        "label_stable": True,
        "family_traces": {
            "generated_answer": {
                "jacobian_lens": [{
                    "layer": 20,
                    "regime": "workspace",
                    "family_log_probability": {
                        "objection": score,
                        "social_approval": -2.0,
                        "epistemic": -2.0,
                        "evaluation_awareness": -2.0,
                        "control": -2.0,
                    },
                }],
                "logit_lens": [{
                    "layer": 20,
                    "regime": "workspace",
                    "family_log_probability": {
                        "objection": score,
                        "social_approval": -2.0,
                        "epistemic": -2.0,
                        "evaluation_awareness": -2.0,
                        "control": -2.0,
                    },
                }],
            }
        },
    }


class CommitmentDataTests(unittest.TestCase):
    def test_real_artifacts_build_expected_matched_design(self) -> None:
        cases = json.loads((REPO_ROOT / "experiments" / "expanded_cases.json").read_text())
        summary = json.loads(
            (REPO_ROOT / "experiments" / "results" / "expanded_summary.json").read_text()
        )
        payload = build_commitment_payload(cases, summary)
        self.assertEqual(payload["design"]["n_primary_pairs"], 85)
        self.assertEqual(payload["design"]["n_open_pairs"], 122)
        self.assertEqual(payload["design"]["conditions"]["sycophantic"], 123)
        self.assertEqual(payload["design"]["conditions"]["resistant"], 122)
        self.assertEqual(payload["design"]["conditions"]["warranted_agreement"], 85)


class CommitmentAnalysisTests(unittest.TestCase):
    def test_preregistered_score_separates_toy_pairs(self) -> None:
        raw = {
            "metadata": {"readout_positions": ["generated_answer"]},
            "design": {},
            "examples": [
                example("a", "sycophantic", -0.5),
                example("a", "warranted_agreement", -3.0),
                example("b", "sycophantic", -1.0),
                example("b", "warranted_agreement", -4.0),
            ],
            "concept_counts": {},
            "token_text": {},
        }
        findings = analyze_commitment_results(raw, bootstrap_samples=10)
        result = findings["preregistered"]["result"]
        self.assertEqual(result["n_pairs"], 2)
        self.assertEqual(result["auc"], 1.0)
        self.assertGreater(result["paired_mean_difference"], 0)


if __name__ == "__main__":
    unittest.main()
