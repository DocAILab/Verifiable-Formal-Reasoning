"""Verl V1 trainer integration for action-span RuleGroundedProcessRL."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transfer_queue as tq
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import apply_kl_penalty
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.workers.utils.padding import response_to_nested

from recipe.formally_verifiable.rule_grounded_process_rl.advantages import assign_action_advantages
from recipe.formally_verifiable.rule_grounded_process_rl.reward import ProcessRewardConfig, RuleEmaBaseline


# 将张量或序列统一转换为 Python 列表。
def _list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


# 把 VERL batch 字段整理为逐轨迹字典。
def _rows(uids, extras) -> list[dict[str, Any]]:
    rows = []
    for uid, extra in zip(_list(uids), _list(extras), strict=False):
        extra = extra or {}
        rows.append(
            {
                "uid": str(uid),
                "problem_id": extra.get("problem_id"),
                "trajectory_reward": float(extra.get("trajectory_reward", 0.0)),
                "scored_response": extra.get("scored_response") or {},
                "response_text": extra.get("response_text", ""),
                "actions": [dict(action) for action in (extra.get("actions") or [])],
            }
        )
    return rows


# 递归展开诊断字典中的标量，供 VERL logger 记录。
def _flatten_scalars(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, bool):
        output[prefix] = float(value)
    elif isinstance(value, (int, float)):
        output[prefix] = float(value)
    elif isinstance(value, dict):
        for key, nested in value.items():
            _flatten_scalars(f"{prefix}/{key}", nested, output)


@register_trainer("rule_grounded_process_rl_sync")
class RuleGroundedProcessRLTrainer(PPOTrainerSync):
    # 初始化自定义 VERL trainer、规则 baseline 和 retry buffer。
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config.algorithm.get("rule_grounded_process_rl", {})
        reward_cfg = cfg.get("reward", {})
        process_cfg = ProcessRewardConfig(
            **{key: reward_cfg[key] for key in ProcessRewardConfig.__dataclass_fields__ if key in reward_cfg}
        )
        self.rule_baseline = RuleEmaBaseline(process_cfg)
        snapshot = cfg.get("initial_baseline_snapshot") or {}
        if snapshot:
            self.rule_baseline.load_snapshot(snapshot)
        self.retry_buffer = deque(maxlen=int(cfg.get("retry_buffer_max_size", 4096)))

    # 按配置周期保存 response 与逐 action 奖励明细。
    def _write_sampled_responses(self, rows: list[dict[str, Any]], diagnostics: dict[str, Any]) -> None:
        cfg = self.config.algorithm.get("rule_grounded_process_rl", {})
        frequency = int(cfg.get("sampled_response_log_steps", 10))
        if frequency <= 0 or self.global_steps % frequency:
            return
        output_dir = Path(str(self.config.trainer.default_local_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "global_step": int(self.global_steps),
            "outcome_mode": diagnostics.get("outcome_mode"),
            "outcome_rewards": diagnostics.get("outcome_rewards"),
            "outcome_advantages": diagnostics.get("outcome_advantages"),
            "baseline_snapshot": diagnostics.get("baseline_snapshot"),
            "samples": [
                {
                    "uid": row["uid"],
                    "problem_id": row.get("problem_id"),
                    "response": row.get("response_text"),
                    "trajectory_reward": row.get("trajectory_reward"),
                    "scored_response": row.get("scored_response"),
                    "actions": row.get("actions"),
                }
                for row in rows
            ],
        }
        with (output_dir / "sampled_responses.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # 计算并回写 action-span token advantages 与 returns。
    def _compute_advantage(self, batch, metrics: dict):
        fields = [
            "uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs",
            "ref_log_prob", "values", "extra_fields",
        ]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        response_mask_nested = data["response_mask"]
        padded = data.to_padded_tensor()
        response_mask = padded["response_mask"]
        rows = _rows(data["uid"], data["extra_fields"])

        proto = DataProto(batch=padded)
        proto.non_tensor_batch["uid"] = np.array([row["uid"] for row in rows], dtype=object)
        if self.config.algorithm.use_kl_in_reward:
            proto.batch["token_level_scores"] = proto.batch["rm_scores"]
            proto, kl_metrics = apply_kl_penalty(
                proto, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)

        cfg = self.config.algorithm.get("rule_grounded_process_rl", {})
        negative_mass_ratio = cfg.get("negative_mass_ratio")
        advantages, diagnostics = assign_action_advantages(
            rows,
            response_mask,
            self.rule_baseline,
            lambda_process=float(cfg.get("lambda_process", 1.0)),
            lambda_outcome=float(cfg.get("lambda_outcome", 1.0)),
            failed_group_process_weight=float(cfg.get("failed_group_process_weight", 0.25)),
            outcome_requires_valid_response=bool(cfg.get("outcome_requires_valid_response", True)),
            eps=float(cfg.get("advantage_eps", 1e-6)),
            goal_binding_process_scale=float(cfg.get("goal_binding_process_scale", 1.0)),
            negative_mass_ratio=(
                float(negative_mass_ratio) if negative_mass_ratio is not None else None
            ),
            no_positive_process_weight=float(cfg.get("no_positive_process_weight", 1.0)),
        )
        self.retry_buffer.extend(diagnostics.pop("retry_problem_ids", []))
        diagnostics["retry_buffer_size"] = len(self.retry_buffer)
        self._write_sampled_responses(rows, diagnostics)
        scalar_metrics: dict[str, float] = {}
        _flatten_scalars("rule_grounded_process_rl", diagnostics, scalar_metrics)
        metrics.update(scalar_metrics)

        proto.batch["advantages"] = advantages
        proto.batch["returns"] = advantages.clone()
        output = {
            "advantages": response_to_nested(proto.batch["advantages"], response_mask_nested),
            "returns": response_to_nested(proto.batch["returns"], response_mask_nested),
        }
        if self.config.algorithm.use_kl_in_reward:
            output["token_level_rewards"] = response_to_nested(
                proto.batch["token_level_rewards"], response_mask_nested
            )
        return tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(output, batch_size=len(batch)),
        )
