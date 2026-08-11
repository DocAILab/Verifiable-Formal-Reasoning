"""Action-span credit assignment for RuleGroundedProcessRL."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping

import torch

from recipe.formally_verifiable.rule_grounded_process_rl.reward import (
    INVALID_RULE_BUCKET,
    RuleEmaBaseline,
    outcome_eligibility,
    per_rule_bucket_summary,
    summarize_scored_responses,
)


# 仅在合格 response 中按组标准化终局奖励。
def _normalize_outcomes(values: list[float], eps: float) -> tuple[list[float], str]:
    successes = sum(value > 0.5 for value in values)
    if successes == 0:
        return [0.0] * len(values), "all_failed"
    if successes == len(values):
        return [0.0] * len(values), "all_success"
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return [(value - mean) / (std + eps) for value in values], "mixed"


# 计算 action 级 process/outcome 优势并映射到对应 token span。
def assign_action_advantages(
    rows: list[dict[str, Any]],
    response_mask: torch.Tensor,
    baseline: RuleEmaBaseline,
    *,
    lambda_process: float,
    lambda_outcome: float,
    failed_group_process_weight: float,
    outcome_requires_valid_response: bool,
    eps: float,
    goal_binding_process_scale: float | None = None,
    negative_mass_ratio: float | None = None,
    no_positive_process_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return token advantages with each action normalized by its own span length."""
    if negative_mass_ratio is not None and negative_mass_ratio < 0:
        raise ValueError("negative_mass_ratio must be non-negative or None")
    if not 0.0 <= no_positive_process_weight <= 1.0:
        raise ValueError("no_positive_process_weight must be in [0, 1]")

    token_advantages = torch.zeros_like(response_mask, dtype=torch.float32)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["uid"])].append(index)

    retry_problem_ids: list[str] = []
    outcome_modes: Counter[str] = Counter()
    all_actions: list[dict[str, Any]] = []
    mapping_failures = 0
    mapping_total = 0
    outcome_success_count = 0
    outcome_raw_success_count = 0
    outcome_eligible_count = 0
    outcome_ineligible_count = 0
    outcome_rewards: list[float] = []
    outcome_advantages: list[float] = []
    closure_masks: list[list[bool]] = []
    process_positive_mass = 0.0
    process_negative_mass = 0.0
    process_negative_scales: list[float] = []
    process_no_positive_group_count = 0
    process_terms_before_balance: list[float] = []
    process_terms_after_balance: list[float] = []
    mapped_action_count = sum(
        bool(action.get("span_mapping_ok"))
        for row in rows
        for action in row["actions"]
    )
    # Verl averages seq-mean-token-sum over responses. This scale converts
    # that response mean into the current implementation's mapped-action mean.
    action_batch_scale = len(rows) / max(1, mapped_action_count)
    goal_binding_scale = (
        lambda_process if goal_binding_process_scale is None else goal_binding_process_scale
    )

    for uid, indices in grouped.items():
        rewards = [float(rows[index]["trajectory_reward"]) for index in indices]
        scored = [rows[index]["scored_response"] for index in indices]
        eligible = outcome_eligibility(
            scored,
            require_valid_response=outcome_requires_valid_response,
        )
        eligible_local_indices = [
            local_index for local_index, is_eligible in enumerate(eligible) if is_eligible
        ]
        effective_outcome_adv = [0.0] * len(rewards)
        if eligible_local_indices:
            eligible_rewards = [rewards[local_index] for local_index in eligible_local_indices]
            eligible_advantages, mode = _normalize_outcomes(eligible_rewards, eps)
            for local_index, advantage in zip(
                eligible_local_indices,
                eligible_advantages,
                strict=True,
            ):
                effective_outcome_adv[local_index] = advantage
        else:
            mode = "no_eligible"

        outcome_modes[mode] += 1
        outcome_raw_success_count += sum(value > 0.5 for value in rewards)
        outcome_success_count += sum(
            rewards[local_index] > 0.5 for local_index in eligible_local_indices
        )
        outcome_eligible_count += len(eligible_local_indices)
        outcome_ineligible_count += len(rewards) - len(eligible_local_indices)
        outcome_rewards.extend(rewards)
        outcome_advantages.extend(effective_outcome_adv)
        if mode in {"all_failed", "no_eligible"}:
            retry_problem_ids.append(str(rows[indices[0]].get("problem_id") or uid))

        process_weight = (
            failed_group_process_weight
            if mode in {"all_failed", "no_eligible"}
            else 1.0
        )
        group_entries: list[dict[str, Any]] = []
        for local_index, row_index in enumerate(indices):
            row = rows[row_index]
            scored_response = row.get("scored_response") or {}
            final_answer_index = scored_response.get("final_answer_index")
            total_steps = int(scored_response.get("total_steps") or 0)
            row_closure: list[bool] = []
            for action in row["actions"]:
                mapping_total += 1
                bucket = str(action.get("rule_bucket") or INVALID_RULE_BUCKET)
                process_advantage = float(action.get("raw_reward", 0.0)) - baseline.value(bucket)
                closure_credit = bool(action.get("in_closure")) and bool(eligible[local_index])
                step_index = action.get("step_index")
                terminal_goal_process_exempt = bool(
                    bucket == "GOAL_BINDING"
                    and step_index is not None
                    and step_index == final_answer_index
                    and step_index == total_steps - 1
                    and action.get("final_binding_ok")
                )
                process_scale = (
                    goal_binding_scale if terminal_goal_process_exempt else lambda_process
                )
                process_term = process_scale * process_weight * process_advantage
                outcome_term = (
                    lambda_outcome * float(closure_credit) * effective_outcome_adv[local_index]
                )
                action["process_advantage"] = process_advantage
                action["process_scale"] = process_scale
                action["terminal_goal_process_exempt"] = terminal_goal_process_exempt
                action["outcome_advantage"] = effective_outcome_adv[local_index]
                action["outcome_eligible"] = bool(eligible[local_index])
                action["process_term_before_balance"] = process_term
                row_closure.append(bool(action.get("in_closure")))
                all_actions.append(action)

                if not action.get("span_mapping_ok"):
                    mapping_failures += 1
                    group_entries.append(
                        {
                            "action": action,
                            "process_term": process_term,
                            "outcome_term": outcome_term,
                            "mapped": False,
                        }
                    )
                    continue
                start, end = int(action["token_start"]), int(action["token_end"])
                end = min(end, int(response_mask.shape[1]))
                valid = response_mask[row_index, start:end].bool()
                span_length = int(valid.sum().item())
                if start < 0 or end <= start or span_length == 0:
                    mapping_failures += 1
                    group_entries.append(
                        {
                            "action": action,
                            "process_term": process_term,
                            "outcome_term": outcome_term,
                            "mapped": False,
                        }
                    )
                    continue
                group_entries.append(
                    {
                        "action": action,
                        "process_term": process_term,
                        "outcome_term": outcome_term,
                        "mapped": True,
                        "row_index": row_index,
                        "start": start,
                        "end": end,
                        "valid": valid,
                        "span_length": span_length,
                    }
                )
            closure_masks.append(row_closure)

        mapped_process_terms = [
            float(entry["process_term"])
            for entry in group_entries
            if entry["mapped"]
        ]
        positive_mass = sum(max(value, 0.0) for value in mapped_process_terms)
        negative_mass = sum(max(-value, 0.0) for value in mapped_process_terms)
        negative_scale = 1.0
        if negative_mass_ratio is not None and negative_mass > eps:
            if positive_mass > eps:
                negative_scale = min(
                    1.0,
                    negative_mass_ratio * positive_mass / (negative_mass + eps),
                )
            elif mapped_process_terms:
                negative_scale = no_positive_process_weight
                process_no_positive_group_count += 1

        process_positive_mass += positive_mass
        process_negative_mass += negative_mass
        process_negative_scales.append(negative_scale)
        for entry in group_entries:
            process_term = float(entry["process_term"])
            balanced_process_term = (
                process_term * negative_scale if process_term < 0.0 else process_term
            )
            total_advantage = balanced_process_term + float(entry["outcome_term"])
            action = entry["action"]
            action["process_negative_scale"] = negative_scale
            action["process_term_after_balance"] = balanced_process_term
            action["total_advantage"] = total_advantage
            if not entry["mapped"]:
                continue

            process_terms_before_balance.append(process_term)
            process_terms_after_balance.append(balanced_process_term)
            # Verl's seq-mean-token-sum now sums to one action-level mean.
            token_advantages[entry["row_index"], entry["start"] : entry["end"]] += entry[
                "valid"
            ].float() * (total_advantage * action_batch_scale / entry["span_length"])

    baseline_snapshot_before = baseline.snapshot(action.get("rule_bucket") for action in all_actions)
    per_rule = per_rule_bucket_summary(all_actions, baselines=baseline)
    baseline.update(all_actions)
    scored_responses = [row["scored_response"] for row in rows]
    diagnostics = {
        **summarize_scored_responses(scored_responses),
        "outcome_mode": dict(outcome_modes),
        "outcome_success_count": outcome_success_count,
        "outcome_raw_success_count": outcome_raw_success_count,
        "outcome_eligible_count": outcome_eligible_count,
        "outcome_ineligible_count": outcome_ineligible_count,
        "outcome_rewards": outcome_rewards,
        "outcome_advantages": outcome_advantages,
        "retry_buffer_size": len(retry_problem_ids),
        "retry_problem_ids": retry_problem_ids,
        "failed_group_process_weight": failed_group_process_weight,
        "lambda_process": lambda_process,
        "goal_binding_process_scale": goal_binding_scale,
        "negative_mass_ratio": negative_mass_ratio,
        "no_positive_process_weight": no_positive_process_weight,
        "process_positive_mass": process_positive_mass,
        "process_negative_mass": process_negative_mass,
        "process_negative_scale": (
            sum(process_negative_scales) / max(1, len(process_negative_scales))
        ),
        "process_negative_scale_min": min(process_negative_scales, default=1.0),
        "process_negative_scale_max": max(process_negative_scales, default=1.0),
        "process_no_positive_group_count": process_no_positive_group_count,
        "process_advantage_mean_before_balance": (
            sum(process_terms_before_balance) / max(1, len(process_terms_before_balance))
        ),
        "process_advantage_mean_after_balance": (
            sum(process_terms_after_balance) / max(1, len(process_terms_after_balance))
        ),
        "action_span_mapping_failure_rate": mapping_failures / max(1, mapping_total),
        "mapped_action_count": mapped_action_count,
        "action_batch_scale": action_batch_scale,
        "closure_mask": closure_masks,
        "per_rule_buckets": per_rule,
        "baseline_snapshot": baseline_snapshot_before,
    }
    return token_advantages, diagnostics
