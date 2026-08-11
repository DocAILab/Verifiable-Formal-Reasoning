"""CPU-only smoke tests for RuleGroundedProcessRL reward logic."""

from __future__ import annotations

import json
import unittest
from typing import Any

import torch

from recipe.formally_verifiable.rule_grounded_process_rl.advantages import assign_action_advantages


from recipe.formally_verifiable.rule_grounded_process_rl.reward import (
    FORMAT_BUCKET,
    ProcessRewardConfig,
    RuleGroundedProcessRewardScorer,
    dependency_closure,
    find_final_answer_step_index,
    gate_outcome_advantages,
    RuleEmaBaseline,
)


ARROW = "\u2192"


def response_for(steps: list[dict[str, Any]]) -> str:
    return (
        "<think>short proof</think>\n"
        "<summary>\n"
        f"{json.dumps(steps, ensure_ascii=False)}\n"
        "</summary>"
    )


class RuleGroundedRewardSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = {
            "id": "reward-smoke",
            "options": ["A) True", "B) False", "C) Uncertain"],
            "answer": "A",
            "nl2fol": {
                "A is blue.": "blue A",
                "If A is blue then A is happy.": f"blue A {ARROW} happy A",
            },
            "conclusion_fol": "happy(A)",
            "difficulty": "easy",
        }
        self.scorer = RuleGroundedProcessRewardScorer()

    def score_one(self, step: dict[str, Any]) -> dict[str, Any]:
        score = self.scorer.score_response(response_for([step]), self.problem)
        self.assertEqual(len(score["actions"]), 1)
        return score["actions"][0]

    def test_schema_error_gets_zero(self) -> None:
        action = self.score_one(
            {
                "id": "s1",
                "dependencies": "h1",
                "conclusion": "happy A",
                "rule": "IMPLICATION_ELIMINATION",
            }
        )
        self.assertFalse(action["schema_valid"])
        self.assertEqual(action["raw_reward"], 0.0)

    def test_schema_valid_but_z3_failure_gets_point_one(self) -> None:
        action = self.score_one(
            {
                "id": "s1",
                "dependencies": ["h1"],
                "conclusion": "happy A",
                "rule": "IMPLICATION_ELIMINATION",
            }
        )
        self.assertTrue(action["schema_valid"])
        self.assertFalse(action["semantic_verified"])
        self.assertEqual(action["raw_reward"], 0.1)

    def test_z3_verified_but_wrong_rule_gets_point_three(self) -> None:
        action = self.score_one(
            {
                "id": "s1",
                "dependencies": ["h2", "h1"],
                "conclusion": "happy A",
                "rule": "MODUS_TOLLENS",
            }
        )
        self.assertTrue(action["semantic_verified"])
        self.assertFalse(action["rule_grounded_verified"])
        self.assertFalse(action["rule_application_valid"])
        self.assertEqual(action["raw_reward"], 0.3)

    def test_rule_grounded_nontrivial_progress_gets_one(self) -> None:
        action = self.score_one(
            {
                "id": "s1",
                "dependencies": ["h2", "h1"],
                "conclusion": "happy A",
                "rule": "IMPLICATION_ELIMINATION",
            }
        )
        self.assertTrue(action["semantic_verified"])
        self.assertTrue(action["rule_grounded_verified"])
        self.assertTrue(action["nontrivial_verified_progress"])
        self.assertEqual(action["raw_reward"], 1.0)

    def test_format_scaffold_actions_reward_successful_format_only(self) -> None:
        scorer = RuleGroundedProcessRewardScorer(
            config=ProcessRewardConfig(enable_format_scaffold_actions=True)
        )
        response = response_for(
            [
                {
                    "id": "s1",
                    "dependencies": ["h2", "h1"],
                    "conclusion": "happy A",
                    "rule": "IMPLICATION_ELIMINATION",
                }
            ]
        )
        score = scorer.score_response(response, self.problem)
        scaffold_actions = [
            action for action in score["actions"] if action["kind"] == "format_scaffold"
        ]
        self.assertTrue(score["schema_valid"])
        self.assertEqual(len([action for action in score["actions"] if action["kind"] == "step"]), 1)
        self.assertGreaterEqual(len(scaffold_actions), 4)
        self.assertTrue(all(action["rule_bucket"] == FORMAT_BUCKET for action in scaffold_actions))
        self.assertTrue(all(action["raw_reward"] == 1.0 for action in scaffold_actions))
        scaffold_text = [response[action["char_start"] : action["char_end"]] for action in scaffold_actions]
        self.assertIn("<summary>", scaffold_text)
        self.assertIn("[", scaffold_text)
        self.assertIn("]", scaffold_text)
        self.assertIn("</summary>", scaffold_text)
        self.assertNotIn("<think>", scaffold_text)

    def test_format_scaffold_actions_skip_schema_failures(self) -> None:
        scorer = RuleGroundedProcessRewardScorer(
            config=ProcessRewardConfig(enable_format_scaffold_actions=True)
        )
        score = scorer.score_response(
            response_for(
                [
                    {
                        "id": "s1",
                        "dependencies": "h1",
                        "conclusion": "happy A",
                        "rule": "IMPLICATION_ELIMINATION",
                    }
                ]
            ),
            self.problem,
        )
        self.assertFalse(score["schema_valid"])
        self.assertFalse(any(action["kind"] == "format_scaffold" for action in score["actions"]))

    def test_object_with_nested_array_is_not_accepted_as_summary_array(self) -> None:
        response = (
            '<think>bad top level</think><summary>'
            '{"id":"s1","dependencies":[],"conclusion":"happy A",'
            '"rule":"GOAL_BINDING"}</summary>'
        )
        score = self.scorer.score_response(response, self.problem)

        self.assertFalse(score["parse_success"])
        self.assertEqual(score["parse_error"], "No JSON array found inside <summary>")
        self.assertEqual(len(score["actions"]), 1)
        self.assertEqual(score["actions"][0]["kind"], "parse_failure")
        self.assertEqual(score["actions"][0]["raw_reward"], 0.0)

    def test_empty_summary_array_has_a_parse_failure_action(self) -> None:
        response = "<think>empty</think><summary>[]</summary>"
        score = self.scorer.score_response(response, self.problem)

        self.assertFalse(score["parse_success"])
        self.assertEqual(score["parse_error"], "Empty summary")
        self.assertEqual(len(score["actions"]), 1)
        self.assertEqual(score["actions"][0]["kind"], "parse_failure")
        start, end = score["actions"][0]["char_start"], score["actions"][0]["char_end"]
        self.assertEqual(response[start:end], "[]")

    def test_outcome_gate_keeps_process_scores_but_masks_invalid_responses(self) -> None:
        valid_score = self.scorer.score_response(
            response_for(
                [
                    {
                        "id": "s1",
                        "dependencies": ["h2", "h1"],
                        "conclusion": "happy A",
                        "rule": "IMPLICATION_ELIMINATION",
                    },
                    {
                        "id": "h_goal_true",
                        "dependencies": ["s1"],
                        "conclusion": "happy A",
                        "rule": "GOAL_BINDING",
                    },
                ]
            ),
            self.problem,
        )
        invalid_score = self.scorer.score_response(
            response_for(
                [
                    {
                        "id": "h_goal_true",
                        "dependencies": "s1",
                        "conclusion": "happy A",
                        "rule": "GOAL_BINDING",
                    }
                ]
            ),
            self.problem,
        )
        parse_failure_score = self.scorer.score_response(
            "<think>bad json</think><summary>[{]</summary>",
            self.problem,
        )

        raw_advantages = [1.25, 0.75, -1.0]
        effective, eligible = gate_outcome_advantages(
            [valid_score, invalid_score, parse_failure_score],
            raw_advantages,
            require_valid_response=True,
        )

        self.assertEqual(eligible, [True, False, False])
        self.assertEqual(effective, [1.25, 0.0, 0.0])
        self.assertEqual(invalid_score["actions"][0]["raw_reward"], 0.0)
        self.assertTrue(invalid_score["actions"][0]["in_closure"])

    def test_outcome_gate_can_be_disabled_for_legacy_configs(self) -> None:
        scored = [
            {"parse_success": False, "summary_tag_present": False, "schema_valid": False},
            {"parse_success": True, "summary_tag_present": True, "schema_valid": True},
        ]
        effective, eligible = gate_outcome_advantages(
            scored,
            [0.5, -0.5],
            require_valid_response=False,
        )
        self.assertEqual(eligible, [True, True])
        self.assertEqual(effective, [0.5, -0.5])

    def test_goal_binding_restatement_gets_point_three_but_is_in_closure(self) -> None:
        score = self.scorer.score_response(
            response_for(
                [
                    {
                        "id": "s1",
                        "dependencies": ["h2", "h1"],
                        "conclusion": "happy A",
                        "rule": "IMPLICATION_ELIMINATION",
                    },
                    {
                        "id": "h_goal_true",
                        "dependencies": ["s1"],
                        "conclusion": "happy A",
                        "rule": "GOAL_BINDING",
                    },
                ]
            ),
            self.problem,
        )
        self.assertTrue(score["answer_correct"])
        self.assertEqual(score["final_answer_index"], 1)
        final_action = score["actions"][1]
        self.assertEqual(final_action["raw_reward"], 0.3)
        self.assertTrue(final_action["dependency_restatement"])
        self.assertTrue(final_action["in_closure"])
        self.assertTrue(score["actions"][0]["in_closure"])

    def test_no_final_answer_step_yields_empty_closure(self) -> None:
        score = self.scorer.score_response(
            response_for(
                [
                    {
                        "id": "s1",
                        "dependencies": ["h2", "h1"],
                        "conclusion": "happy A",
                        "rule": "IMPLICATION_ELIMINATION",
                    }
                ]
            ),
            self.problem,
        )
        self.assertIsNone(score["final_answer_index"])
        self.assertFalse(score["actions"][0]["in_closure"])

    def test_intermediate_claimed_goal_is_not_a_final_answer(self) -> None:
        steps = [
            {
                "id": "h_goal_true",
                "dependencies": ["h1"],
                "conclusion": "blue A",
                "rule": "GOAL_BINDING",
            },
            {
                "id": "s2",
                "dependencies": ["h2", "h1"],
                "conclusion": "happy A",
                "rule": "IMPLICATION_ELIMINATION",
            },
        ]
        self.assertIsNone(find_final_answer_step_index(steps))
        self.assertEqual(dependency_closure(steps, None), set())

    def test_dependency_closure_tracks_generated_steps_only(self) -> None:
        steps = [
            {"id": "s1", "dependencies": ["h2", "h1"], "conclusion": "happy A", "rule": "IMPLICATION_ELIMINATION"},
            {"id": "s2", "dependencies": ["s1", "h1"], "conclusion": "happy A", "rule": "CONJUNCTION_ELIMINATION"},
            {"id": "h_goal_true", "dependencies": ["s2", "h1"], "conclusion": "happy A", "rule": "GOAL_BINDING"},
        ]
        final_index = find_final_answer_step_index(steps)
        self.assertEqual(final_index, 2)
        self.assertEqual(dependency_closure(steps, final_index), {0, 1, 2})
        self.assertIsNone(find_final_answer_step_index(steps[:2]))
        self.assertEqual(dependency_closure(steps[:2], None), set())

    def test_nested_dependency_is_schema_invalid_without_crashing_closure(self) -> None:
        score = self.scorer.score_response(
            response_for(
                [
                    {
                        "id": "s1",
                        "dependencies": ["h2", "h1"],
                        "conclusion": "happy A",
                        "rule": "IMPLICATION_ELIMINATION",
                    },
                    {
                        "id": "h_goal_true",
                        "dependencies": [["s1"]],
                        "conclusion": "happy A",
                        "rule": "GOAL_BINDING",
                    },
                ]
            ),
            self.problem,
        )
        self.assertTrue(score["parse_success"])
        self.assertFalse(score["schema_valid"])
        self.assertEqual(score["final_answer_index"], 1)
        self.assertFalse(score["actions"][0]["in_closure"])
        self.assertTrue(score["actions"][1]["in_closure"])
        self.assertEqual(score["actions"][1]["raw_reward"], 0.0)

    def test_action_span_credit_is_token_length_normalized(self) -> None:
        baseline = RuleEmaBaseline(ProcessRewardConfig(baseline_mode="none"))
        rows = [
            {
                "uid": "p1",
                "problem_id": "p1",
                "trajectory_reward": 1.0,
                "scored_response": {
                    "parse_success": True,
                    "summary_tag_present": True,
                    "schema_valid": True,
                    "actions": [],
                },
                "actions": [
                    {
                        "raw_reward": 1.0,
                        "rule_bucket": "IMPLICATION_ELIMINATION",
                        "in_closure": True,
                        "span_mapping_ok": True,
                        "token_start": 2,
                        "token_end": 6,
                    }
                ],
            },
            {
                "uid": "p1",
                "problem_id": "p1",
                "trajectory_reward": 0.0,
                "scored_response": {
                    "parse_success": True,
                    "summary_tag_present": True,
                    "schema_valid": True,
                    "actions": [],
                },
                "actions": [
                    {
                        "raw_reward": 0.0,
                        "rule_bucket": "INVALID_RULE",
                        "in_closure": False,
                        "span_mapping_ok": True,
                        "token_start": 1,
                        "token_end": 2,
                    }
                ],
            },
        ]
        advantages, diagnostics = assign_action_advantages(
            rows,
            torch.ones((2, 8), dtype=torch.bool),
            baseline,
            lambda_process=1.0,
            lambda_outcome=1.0,
            failed_group_process_weight=0.25,
            outcome_requires_valid_response=True,
            eps=1e-6,
        )
        expected_action_advantage = 1.0 + 1.0
        self.assertAlmostEqual(float(advantages[0, 2:6].sum()), expected_action_advantage, places=5)
        self.assertEqual(float(advantages[0, :2].abs().sum()), 0.0)
        self.assertEqual(float(advantages[0, 6:].abs().sum()), 0.0)
        self.assertEqual(diagnostics["outcome_mode"], {"mixed": 1})

    def test_all_failed_group_keeps_only_weighted_process_credit(self) -> None:
        baseline = RuleEmaBaseline(ProcessRewardConfig(baseline_mode="none"))
        rows = [
            {
                "uid": "p2",
                "problem_id": "p2",
                "trajectory_reward": 0.0,
                "scored_response": {
                    "parse_success": True,
                    "summary_tag_present": True,
                    "schema_valid": True,
                    "actions": [],
                },
                "actions": [
                    {
                        "raw_reward": 1.0,
                        "rule_bucket": "IMPLICATION_ELIMINATION",
                        "in_closure": True,
                        "span_mapping_ok": True,
                        "token_start": 0,
                        "token_end": 2,
                    }
                ],
            }
        ]
        advantages, diagnostics = assign_action_advantages(
            rows,
            torch.ones((1, 4), dtype=torch.bool),
            baseline,
            lambda_process=1.0,
            lambda_outcome=1.0,
            failed_group_process_weight=0.25,
            outcome_requires_valid_response=True,
            eps=1e-6,
        )
        self.assertAlmostEqual(float(advantages.sum()), 0.25, places=6)
        self.assertEqual(diagnostics["outcome_advantages"], [0.0])
        self.assertEqual(diagnostics["retry_problem_ids"], ["p2"])

    def test_outcome_is_normalized_only_over_eligible_responses(self) -> None:
        baseline = RuleEmaBaseline(ProcessRewardConfig(baseline_mode="none"))

        def row(index: int, reward: float, *, valid: bool) -> dict[str, Any]:
            return {
                "uid": "eligible-first",
                "problem_id": "eligible-first",
                "trajectory_reward": reward,
                "scored_response": {
                    "parse_success": valid,
                    "summary_tag_present": valid,
                    "schema_valid": valid,
                    "actions": [],
                },
                "actions": [
                    {
                        "raw_reward": 0.0,
                        "rule_bucket": "INVALID_RULE",
                        "in_closure": True,
                        "span_mapping_ok": True,
                        "token_start": index,
                        "token_end": index + 1,
                    }
                ],
            }

        rows = [
            row(0, 1.0, valid=True),
            row(1, 1.0, valid=False),
            row(2, 0.0, valid=True),
            row(3, 0.0, valid=True),
        ]
        advantages, diagnostics = assign_action_advantages(
            rows,
            torch.ones((4, 4), dtype=torch.bool),
            baseline,
            lambda_process=0.0,
            lambda_outcome=1.0,
            failed_group_process_weight=0.25,
            outcome_requires_valid_response=True,
            eps=1e-6,
        )

        effective = diagnostics["outcome_advantages"]
        self.assertAlmostEqual(effective[0], 2**0.5, places=5)
        self.assertEqual(effective[1], 0.0)
        self.assertAlmostEqual(effective[2], -(2**-0.5), places=5)
        self.assertAlmostEqual(effective[3], -(2**-0.5), places=5)
        self.assertAlmostEqual(sum(effective), 0.0, places=5)
        self.assertAlmostEqual(float(advantages.sum()), 0.0, places=5)
        self.assertEqual(diagnostics["outcome_mode"], {"mixed": 1})
        self.assertEqual(diagnostics["outcome_success_count"], 1)
        self.assertEqual(diagnostics["outcome_raw_success_count"], 2)
        self.assertEqual(diagnostics["outcome_eligible_count"], 3)
        self.assertEqual(diagnostics["outcome_ineligible_count"], 1)
        self.assertEqual(
            [action["outcome_eligible"] for action in (row["actions"][0] for row in rows)],
            [True, False, True, True],
        )

    def test_group_without_eligible_response_retries_with_zero_outcome(self) -> None:
        baseline = RuleEmaBaseline(ProcessRewardConfig(baseline_mode="none"))
        rows = [
            {
                "uid": "no-eligible",
                "problem_id": "no-eligible",
                "trajectory_reward": 1.0,
                "scored_response": {
                    "parse_success": False,
                    "summary_tag_present": False,
                    "schema_valid": False,
                    "actions": [],
                },
                "actions": [
                    {
                        "raw_reward": 1.0,
                        "rule_bucket": "IMPLICATION_ELIMINATION",
                        "in_closure": False,
                        "span_mapping_ok": True,
                        "token_start": 0,
                        "token_end": 1,
                    }
                ],
            }
        ]
        advantages, diagnostics = assign_action_advantages(
            rows,
            torch.ones((1, 1), dtype=torch.bool),
            baseline,
            lambda_process=1.0,
            lambda_outcome=1.0,
            failed_group_process_weight=0.25,
            outcome_requires_valid_response=True,
            eps=1e-6,
        )

        self.assertAlmostEqual(float(advantages.sum()), 0.25, places=6)
        self.assertEqual(diagnostics["outcome_mode"], {"no_eligible": 1})
        self.assertEqual(diagnostics["outcome_advantages"], [0.0])
        self.assertEqual(diagnostics["retry_problem_ids"], ["no-eligible"])

    def test_goal_binding_keeps_unscaled_process_credit(self) -> None:
        baseline = RuleEmaBaseline(
            ProcessRewardConfig(
                baseline_mode="rule_ema_clipped",
                baseline_initial_value=0.4,
                baseline_clip_min=0.4,
                baseline_clip_max=0.8,
            )
        )
        def row(
            uid: str,
            success: bool,
            bucket: str,
            *,
            step_index: int = 0,
            final_answer_index: int | None = None,
            total_steps: int = 1,
            final_binding_ok: bool = True,
        ) -> dict:
            return {
                "uid": uid,
                "problem_id": uid,
                "trajectory_reward": float(success),
                "scored_response": {
                    "parse_success": True,
                    "summary_tag_present": True,
                    "schema_valid": True,
                    "final_answer_index": final_answer_index,
                    "total_steps": total_steps,
                    "actions": [],
                },
                "actions": [
                    {
                        "raw_reward": 0.3,
                        "rule_bucket": bucket,
                        "step_index": step_index,
                        "final_binding_ok": final_binding_ok,
                        "in_closure": True,
                        "span_mapping_ok": True,
                        "token_start": 0,
                        "token_end": 1,
                    }
                ],
            }

        rows = [
            row("ordinary", True, "IMPLICATION_ELIMINATION"),
            row("ordinary", False, "IMPLICATION_ELIMINATION"),
            row("goal", True, "GOAL_BINDING", final_answer_index=0),
            row("goal", False, "GOAL_BINDING", final_answer_index=0),
            row(
                "intermediate_goal",
                True,
                "GOAL_BINDING",
                step_index=0,
                final_answer_index=1,
                total_steps=2,
            ),
            row(
                "intermediate_goal",
                False,
                "GOAL_BINDING",
                step_index=0,
                final_answer_index=1,
                total_steps=2,
            ),
            row(
                "invalid_final_goal",
                True,
                "GOAL_BINDING",
                final_answer_index=0,
                final_binding_ok=False,
            ),
            row(
                "invalid_final_goal",
                False,
                "GOAL_BINDING",
                final_answer_index=0,
                final_binding_ok=False,
            ),
        ]
        advantages, diagnostics = assign_action_advantages(
            rows,
            torch.ones((8, 2), dtype=torch.bool),
            baseline,
            lambda_process=5.0,
            lambda_outcome=1.0,
            failed_group_process_weight=0.25,
            outcome_requires_valid_response=True,
            eps=1e-6,
            goal_binding_process_scale=1.0,
        )

        self.assertAlmostEqual(float(advantages[0].sum()), 0.5, places=5)
        self.assertAlmostEqual(float(advantages[2].sum()), 0.9, places=5)
        self.assertEqual(rows[0]["actions"][0]["process_scale"], 5.0)
        self.assertEqual(rows[2]["actions"][0]["process_scale"], 1.0)
        self.assertTrue(rows[2]["actions"][0]["terminal_goal_process_exempt"])
        self.assertEqual(rows[4]["actions"][0]["process_scale"], 5.0)
        self.assertFalse(rows[4]["actions"][0]["terminal_goal_process_exempt"])
        self.assertEqual(rows[6]["actions"][0]["process_scale"], 5.0)
        self.assertFalse(rows[6]["actions"][0]["terminal_goal_process_exempt"])
        self.assertEqual(diagnostics["goal_binding_process_scale"], 1.0)

    def test_uneven_action_counts_reproduce_global_action_mean(self) -> None:
        baseline = RuleEmaBaseline(ProcessRewardConfig(baseline_mode="none"))
        valid_score = {
            "parse_success": True,
            "summary_tag_present": True,
            "schema_valid": True,
            "actions": [],
        }
        action = lambda start: {
            "raw_reward": 1.0,
            "rule_bucket": "IMPLICATION_ELIMINATION",
            "in_closure": False,
            "span_mapping_ok": True,
            "token_start": start,
            "token_end": start + 2,
        }
        rows = [
            {
                "uid": "p3",
                "problem_id": "p3",
                "trajectory_reward": 0.0,
                "scored_response": valid_score,
                "actions": [action(0), action(2)],
            },
            {
                "uid": "p3",
                "problem_id": "p3",
                "trajectory_reward": 0.0,
                "scored_response": valid_score,
                "actions": [action(0)],
            },
        ]
        advantages, diagnostics = assign_action_advantages(
            rows,
            torch.ones((2, 4), dtype=torch.bool),
            baseline,
            lambda_process=1.0,
            lambda_outcome=0.0,
            failed_group_process_weight=1.0,
            outcome_requires_valid_response=True,
            eps=1e-6,
        )
        # Verl later averages these two sequence sums. The result must equal
        # the mean of the three unit action advantages: 1.0.
        self.assertAlmostEqual(float(advantages.sum(dim=1).mean()), 1.0, places=6)
        self.assertEqual(diagnostics["mapped_action_count"], 3)
        self.assertAlmostEqual(diagnostics["action_batch_scale"], 2.0 / 3.0)

    def test_process_negative_mass_is_capped_by_positive_mass_per_group(self) -> None:
        baseline = RuleEmaBaseline(
            ProcessRewardConfig(
                baseline_initial_value=0.3,
                baseline_clip_min=0.3,
                baseline_clip_max=0.7,
            )
        )

        def row(index: int, raw_reward: float, trajectory_reward: float) -> dict[str, Any]:
            return {
                "uid": "balanced",
                "problem_id": "balanced",
                "trajectory_reward": trajectory_reward,
                "scored_response": {
                    "parse_success": True,
                    "summary_tag_present": True,
                    "schema_valid": True,
                    "actions": [],
                },
                "actions": [
                    {
                        "raw_reward": raw_reward,
                        "rule_bucket": "IMPLICATION_ELIMINATION",
                        "in_closure": False,
                        "span_mapping_ok": True,
                        "token_start": index,
                        "token_end": index + 1,
                    }
                ],
            }

        rows = [row(0, 1.0, 1.0)] + [row(index, 0.0, 0.0) for index in range(1, 5)]
        advantages, diagnostics = assign_action_advantages(
            rows,
            torch.ones((5, 5), dtype=torch.bool),
            baseline,
            lambda_process=3.0,
            lambda_outcome=0.0,
            failed_group_process_weight=0.25,
            outcome_requires_valid_response=True,
            eps=1e-6,
            negative_mass_ratio=1.0,
            no_positive_process_weight=0.1,
        )

        self.assertAlmostEqual(diagnostics["process_positive_mass"], 2.1, places=6)
        self.assertAlmostEqual(diagnostics["process_negative_mass"], 3.6, places=6)
        self.assertAlmostEqual(diagnostics["process_negative_scale"], 2.1 / 3.6, places=5)
        self.assertAlmostEqual(float(advantages[0].sum()), 2.1, places=5)
        self.assertAlmostEqual(float(advantages[1:].sum()), -2.1, places=5)
        self.assertEqual(diagnostics["process_no_positive_group_count"], 0)

    def test_process_only_negative_group_uses_low_weight(self) -> None:
        baseline = RuleEmaBaseline(
            ProcessRewardConfig(
                baseline_initial_value=0.3,
                baseline_clip_min=0.3,
                baseline_clip_max=0.7,
            )
        )
        valid_score = {
            "parse_success": True,
            "summary_tag_present": True,
            "schema_valid": True,
            "actions": [],
        }
        rows = [
            {
                "uid": "negative-only",
                "problem_id": "negative-only",
                "trajectory_reward": float(index == 0),
                "scored_response": valid_score,
                "actions": [
                    {
                        "raw_reward": reward,
                        "rule_bucket": "INVALID_RULE",
                        "in_closure": False,
                        "span_mapping_ok": True,
                        "token_start": index,
                        "token_end": index + 1,
                    }
                ],
            }
            for index, reward in enumerate((0.1, 0.3))
        ]
        advantages, diagnostics = assign_action_advantages(
            rows,
            torch.ones((2, 2), dtype=torch.bool),
            baseline,
            lambda_process=3.0,
            lambda_outcome=0.0,
            failed_group_process_weight=0.25,
            outcome_requires_valid_response=True,
            eps=1e-6,
            negative_mass_ratio=1.0,
            no_positive_process_weight=0.1,
        )

        self.assertAlmostEqual(float(advantages[0].sum()), -0.06, places=6)
        self.assertAlmostEqual(float(advantages[1].sum()), 0.0, places=6)
        self.assertAlmostEqual(diagnostics["process_negative_scale"], 0.1, places=6)
        self.assertEqual(diagnostics["process_no_positive_group_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
