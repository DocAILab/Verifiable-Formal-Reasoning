"""One-shot full-trajectory rollout for RuleGroundedProcessRL."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register

from recipe.formally_verifiable.rule_grounded_process_rl.reward import (
    ProcessRewardConfig,
    RuleGroundedProcessRewardScorer,
)
from recipe.formally_verifiable.rule_grounded_process_rl.structured_prompt import build_chat_prompt


# 将张量标量等包装值转换为普通 Python 值。
def _plain(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


# 整理 rollout 附带的全局训练步数范围。
def _global_step_fields(extra_fields: Any, fallback: Any = None) -> dict[str, int]:
    extra_fields = extra_fields if isinstance(extra_fields, dict) else {}
    global_steps = _plain(extra_fields.get("global_steps", fallback))
    global_steps = 0 if global_steps is None else int(global_steps)
    min_steps = _plain(extra_fields.get("min_global_steps", global_steps))
    max_steps = _plain(extra_fields.get("max_global_steps", global_steps))
    return {
        "global_steps": global_steps,
        "min_global_steps": global_steps if min_steps is None else int(min_steps),
        "max_global_steps": global_steps if max_steps is None else int(max_steps),
    }


# 把 JSON object 的字符区间映射到生成 token 区间。
def _token_span(tokenizer, text: str, token_ids: list[int], start: int, end: int) -> tuple[int, int] | None:
    """Map a decoded character span back to rollout tokens, failing visibly on ambiguity."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) == token_ids:
        indices = [
            index
            for index, (token_start, token_end) in enumerate(encoded["offset_mapping"])
            if token_end > start and token_start < end
        ]
        return (indices[0], indices[-1] + 1) if indices else None

    # Some tokenizers clean spaces while decoding. Prefix decoding uses the
    # exact sampled ids and is slower, but preserves the rollout token index.
    boundaries = [len(tokenizer.decode(token_ids[:index], skip_special_tokens=True)) for index in range(len(token_ids) + 1)]
    indices = [
        index
        for index in range(len(token_ids))
        if boundaries[index + 1] > start and boundaries[index] < end
    ]
    return (indices[0], indices[-1] + 1) if indices else None


@register("rule_grounded_process_rl")
class RuleGroundedProcessRLAgentLoop(AgentLoopBase):
    # 初始化完整轨迹 rollout 与 process reward scorer。
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config.algorithm.get("rule_grounded_process_rl", {})
        reward_cfg = cfg.get("reward", {})
        process_cfg = ProcessRewardConfig(
            **{key: reward_cfg[key] for key in ProcessRewardConfig.__dataclass_fields__ if key in reward_cfg}
        )
        self.scorer = RuleGroundedProcessRewardScorer(config=process_cfg)
        self.response_length = int(self.rollout_config.response_length)

    # 从 rollout 参数中提取当前形式化问题。
    def _problem(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        extra = _plain(kwargs.get("extra_info")) or {}
        if isinstance(extra, dict) and isinstance(extra.get("problem"), dict):
            return extra["problem"]
        reward_model = _plain(kwargs.get("reward_model")) or {}
        if isinstance(reward_model, dict) and isinstance(reward_model.get("ground_truth"), dict):
            return reward_model["ground_truth"]
        raise ValueError("rule_grounded_process_rl requires extra_info.problem or reward_model.ground_truth")

    # 一次性生成完整轨迹、评分并记录 action token span。
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        problem = self._problem(kwargs)
        prompt_text = build_chat_prompt(self.tokenizer, problem)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        started = time.perf_counter()
        token_output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            priority=int(_plain(kwargs.get("priority", 0)) or 0),
        )
        elapsed = time.perf_counter() - started
        response_ids = list(token_output.token_ids[: self.response_length])
        if not response_ids:
            fallback = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0
            response_ids = [int(fallback)]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        scored = self.scorer.score_response(response_text, problem)
        global_step_fields = _global_step_fields(
            getattr(token_output, "extra_fields", None), kwargs.get("global_steps")
        )

        action_records: list[dict[str, Any]] = []
        mapping_failures: list[dict[str, Any]] = []
        for action in scored.get("actions", []):
            record = dict(action)
            char_span = (int(action["char_start"]), int(action["char_end"]))
            token_span = _token_span(self.tokenizer, response_text, response_ids, *char_span)
            if token_span is None:
                record["span_mapping_ok"] = False
                mapping_failures.append(
                    {"step_index": action.get("step_index"), "step_id": action.get("step_id"), "char_span": char_span}
                )
            else:
                record.update(
                    {"span_mapping_ok": True, "token_start": token_span[0], "token_end": token_span[1]}
                )
            action_records.append(record)

        response_logprobs = None
        if token_output.log_probs and len(token_output.log_probs) >= len(response_ids):
            response_logprobs = token_output.log_probs[: len(response_ids)]
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            reward_score=0.0,
            num_turns=2,
            metrics=AgentLoopMetrics(
                generate_sequences=elapsed,
                num_preempted=token_output.num_preempted or 0,
            ),
            extra_fields={
                "problem_id": problem.get("id"),
                "ground_truth": problem.get("answer"),
                "scored_response": scored,
                "response_text": response_text,
                "actions": action_records,
                "action_span_mapping_failures": mapping_failures,
                "trajectory_reward": float(bool(scored.get("answer_correct"))),
                **global_step_fields,
                "turn_scores": [],
                "tool_rewards": [],
            },
        )
