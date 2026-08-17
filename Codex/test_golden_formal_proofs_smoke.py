"""CPU-only smoke checks for canonical proof generation and granularity metrics."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Training"))

from Codex.build_golden_formal_proofs import (  # noqa: E402
    prune_to_final_closure,
    validate_candidate,
)
from Test.Generation.metrics import (  # noqa: E402
    generated_proof_step_count,
    reference_step_count,
)
from recipe.formally_verifiable.rule_grounded_process_rl.reward import (  # noqa: E402
    RuleGroundedProcessRewardScorer,
)


class GoldenFormalProofSmokeTest(unittest.TestCase):
    def test_canonical_dataset_is_the_single_metric_scope(self) -> None:
        root = ROOT / "Data" / "ProverQA" / "test" / "golden"
        rows = [
            json.loads(line)
            for line in (root / "all.jsonl").read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 681)
        self.assertTrue(all(row.get("canonical_proofs") for row in rows))
        self.assertEqual(manifest["accepted_count"], 681)
        self.assertEqual(manifest["rejected_or_pending_count"], 129)
        self.assertEqual(
            manifest["metric_scope"], "all_metrics_use_the_same_canonical_subset"
        )

    def test_reference_length_prefers_canonical_metadata(self) -> None:
        problem = {
            "reasoning": "conclusion: one\nconclusion: two",
            "canonical_proof_reference": {"min_proof_length": 5},
        }
        self.assertEqual(reference_step_count(problem), 5)

    def test_generated_length_uses_non_goal_final_closure(self) -> None:
        summary = [
            {
                "id": "s1",
                "dependencies": ["h1"],
                "conclusion": "P A",
                "rule": "UNIVERSAL_ELIMINATION",
            },
            {
                "id": "s2",
                "dependencies": ["h2"],
                "conclusion": "Q A",
                "rule": "UNIVERSAL_ELIMINATION",
            },
            {
                "id": "h_goal_true",
                "dependencies": ["s1"],
                "conclusion": "P A",
                "rule": "GOAL_BINDING",
            },
        ]
        self.assertEqual(generated_proof_step_count(summary), 1)

    def test_closure_pruning_renumbers_actions(self) -> None:
        candidate = {
            "steps": [
                {"id": "s1", "dependencies": ["h1"], "conclusion": "P A", "rule": "UNIVERSAL_ELIMINATION"},
                {"id": "s2", "dependencies": ["h2"], "conclusion": "Q A", "rule": "UNIVERSAL_ELIMINATION"},
                {"id": "s3", "dependencies": ["s1"], "conclusion": "R A", "rule": "IMPLICATION_ELIMINATION"},
                {"id": "h_goal_true", "dependencies": ["s3"], "conclusion": "R A", "rule": "GOAL_BINDING"},
            ]
        }
        pruned = prune_to_final_closure(candidate)["steps"]
        self.assertEqual([step["id"] for step in pruned], ["s1", "s2", "h_goal_true"])
        self.assertEqual(pruned[-1]["dependencies"], ["s2"])

    def test_saved_golden_proof_still_passes_current_scorer(self) -> None:
        path = ROOT / "Data" / "ProverQA" / "test" / "golden" / "all.jsonl"
        problem = json.loads(path.read_text(encoding="utf-8-sig").splitlines()[0])
        candidate = {"steps": problem["canonical_proofs"][0]["steps"]}
        proof, detail = validate_candidate(
            problem, candidate, RuleGroundedProcessRewardScorer()
        )
        self.assertIsNotNone(proof, detail)


if __name__ == "__main__":
    unittest.main()
