"""Map formally verifiable recipe YAML files to verl Hydra overrides."""

from __future__ import annotations

from pathlib import Path
import os
import re
from typing import Any

import yaml


_OC_ENV_PATTERN = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)\}")


# 递归展开配置值中的环境变量占位符。
def _resolve_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    # 将单个环境变量占位符替换为当前环境中的值。
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return os.environ.get(name, match.group(0))

    return _OC_ENV_PATTERN.sub(replace, value)


# 读取 YAML recipe 配置并解析环境变量。
def load_recipe_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Recipe config must be a mapping: {path}")
    return _resolve_environment(config)


# 将 Python 配置值序列化为 Hydra override 字符串。
def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, list):
        return "[" + ",".join(_value(item) for item in value) + "]"
    if value is None:
        return "null"
    return str(value)


# 向 Hydra override 列表追加一个配置项。
def _append(overrides: list[str], key: str, value: Any, *, create: bool = False) -> None:
    prefix = "+" if create else ""
    overrides.append(f"{prefix}{key}={_value(value)}")


# 将单个数据路径统一规范为路径列表。
def _data_files(value: Any) -> Any:
    return value if isinstance(value, list) else [value]


# 生成模型、数据、rollout、actor 和运行时的公共 VERL 配置。
def _append_common_overrides(overrides: list[str], config: dict[str, Any]) -> None:
    model = config["model"]
    data = config["data"]
    rollout = config["rollout"]
    actor = config["actor"]
    ref = config.get("ref", {})
    reward = config["reward"]
    trainer = config["trainer"]
    runtime = config.get("runtime", {})

    _append(overrides, "trainer.use_v1", True)
    _append(overrides, "algorithm.adv_estimator", "grpo")
    _append(overrides, "algorithm.use_kl_in_reward", False)
    _append(overrides, "data.train_files", _data_files(data["train_files"]))
    _append(overrides, "data.val_files", _data_files(data["val_files"]))
    _append(overrides, "data.train_batch_size", data["train_batch_size"])
    _append(overrides, "data.max_prompt_length", data["max_prompt_length"])
    _append(overrides, "data.max_response_length", data["max_response_length"])
    _append(overrides, "data.prompt_key", "prompt")
    _append(overrides, "data.reward_fn_key", "data_source")
    _append(overrides, "data.filter_overlong_prompts", True)
    _append(overrides, "data.truncation", "error")

    _append(overrides, "actor_rollout_ref.model.path", model["path"])
    _append(overrides, "actor_rollout_ref.model.use_remove_padding", True)
    _append(overrides, "actor_rollout_ref.model.enable_gradient_checkpointing", True)
    _append(overrides, "actor_rollout_ref.model.lora_rank", model.get("lora_rank", 0))
    _append(overrides, "actor_rollout_ref.model.lora_alpha", model.get("lora_alpha", 16))
    _append(overrides, "actor_rollout_ref.model.target_modules", model.get("target_modules", "all-linear"))
    if model.get("init_adapter_path"):
        _append(overrides, "actor_rollout_ref.model.lora_adapter_path", model["init_adapter_path"])

    _append(overrides, "actor_rollout_ref.rollout.name", config.get("infer_backend", "vllm"))
    _append(overrides, "actor_rollout_ref.rollout.n", rollout["n"])
    _append(overrides, "actor_rollout_ref.rollout.temperature", rollout["temperature"])
    _append(overrides, "actor_rollout_ref.rollout.top_p", rollout["top_p"])
    _append(overrides, "actor_rollout_ref.rollout.response_length", data["max_response_length"])
    if rollout.get("tensor_model_parallel_size") is not None:
        _append(overrides, "actor_rollout_ref.rollout.tensor_model_parallel_size", rollout["tensor_model_parallel_size"])
    if rollout.get("gpu_memory_utilization") is not None:
        _append(overrides, "actor_rollout_ref.rollout.gpu_memory_utilization", rollout["gpu_memory_utilization"])
    if rollout.get("enforce_eager") is not None:
        _append(overrides, "actor_rollout_ref.rollout.enforce_eager", rollout["enforce_eager"])
    if rollout.get("layered_summon") is not None:
        _append(overrides, "actor_rollout_ref.rollout.layered_summon", rollout["layered_summon"])
    if rollout.get("max_model_len") is not None:
        _append(overrides, "actor_rollout_ref.rollout.max_model_len", rollout["max_model_len"])
    if rollout.get("max_num_batched_tokens") is not None:
        _append(overrides, "actor_rollout_ref.rollout.max_num_batched_tokens", rollout["max_num_batched_tokens"])
    if rollout.get("load_format") is not None:
        _append(overrides, "actor_rollout_ref.rollout.load_format", rollout["load_format"])
    if rollout.get("log_prob_use_dynamic_bsz") is not None:
        _append(overrides, "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz", rollout["log_prob_use_dynamic_bsz"])
    if rollout.get("log_prob_max_token_len_per_gpu") is not None:
        _append(
            overrides,
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
            rollout["log_prob_max_token_len_per_gpu"],
        )
    _append(overrides, "actor_rollout_ref.rollout.calculate_log_probs", True)

    _append(overrides, "actor_rollout_ref.actor.optim.lr", actor["learning_rate"])
    _append(overrides, "actor_rollout_ref.actor.ppo_epochs", actor["ppo_epochs"])
    _append(overrides, "actor_rollout_ref.actor.ppo_mini_batch_size", actor["ppo_mini_batch_size"])
    if actor.get("use_dynamic_bsz") is not None:
        _append(overrides, "actor_rollout_ref.actor.use_dynamic_bsz", actor["use_dynamic_bsz"])
    if actor.get("ppo_max_token_len_per_gpu") is not None:
        _append(overrides, "actor_rollout_ref.actor.ppo_max_token_len_per_gpu", actor["ppo_max_token_len_per_gpu"])
    if actor.get("fsdp_param_offload") is not None:
        _append(overrides, "actor_rollout_ref.actor.fsdp_config.param_offload", actor["fsdp_param_offload"])
    if actor.get("fsdp_optimizer_offload") is not None:
        _append(overrides, "actor_rollout_ref.actor.fsdp_config.optimizer_offload", actor["fsdp_optimizer_offload"])
    if actor.get("fsdp_size") is not None:
        _append(overrides, "actor_rollout_ref.actor.fsdp_config.fsdp_size", actor["fsdp_size"])
    if actor.get("checkpoint_save_contents") is not None:
        _append(overrides, "actor_rollout_ref.actor.checkpoint.save_contents", actor["checkpoint_save_contents"])
    if actor.get("checkpoint_load_contents") is not None:
        _append(overrides, "actor_rollout_ref.actor.checkpoint.load_contents", actor["checkpoint_load_contents"])
    _append(overrides, "actor_rollout_ref.actor.clip_ratio_low", actor["clip_ratio_low"])
    _append(overrides, "actor_rollout_ref.actor.clip_ratio_high", actor["clip_ratio_high"])
    _append(overrides, "actor_rollout_ref.actor.loss_agg_mode", actor["loss_agg_mode"])
    _append(overrides, "actor_rollout_ref.actor.use_kl_loss", True)
    _append(overrides, "actor_rollout_ref.actor.kl_loss_coef", reward["kl_beta"])
    _append(overrides, "actor_rollout_ref.actor.kl_loss_type", "low_var_kl")
    if ref.get("log_prob_use_dynamic_bsz") is not None:
        _append(overrides, "actor_rollout_ref.ref.log_prob_use_dynamic_bsz", ref["log_prob_use_dynamic_bsz"])
    if ref.get("log_prob_max_token_len_per_gpu") is not None:
        _append(overrides, "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu", ref["log_prob_max_token_len_per_gpu"])
    if ref.get("fsdp_param_offload") is not None:
        _append(overrides, "actor_rollout_ref.ref.fsdp_config.param_offload", ref["fsdp_param_offload"])

    _append(overrides, "critic.enable", False)
    if runtime.get("balance_batch") is not None:
        _append(overrides, "trainer.balance_batch", runtime["balance_batch"])
    if runtime.get("critic_warmup") is not None:
        _append(overrides, "trainer.critic_warmup", runtime["critic_warmup"])
    if runtime.get("logger") is not None:
        _append(overrides, "trainer.logger", runtime["logger"])
    if runtime.get("nnodes") is not None:
        _append(overrides, "trainer.nnodes", runtime["nnodes"])
    if runtime.get("n_gpus_per_node") is not None:
        _append(overrides, "trainer.n_gpus_per_node", runtime["n_gpus_per_node"])
    if runtime.get("default_local_dir") is not None:
        _append(overrides, "trainer.default_local_dir", runtime["default_local_dir"])
    if runtime.get("max_actor_ckpt_to_keep") is not None:
        _append(overrides, "trainer.max_actor_ckpt_to_keep", runtime["max_actor_ckpt_to_keep"])
    if runtime.get("max_critic_ckpt_to_keep") is not None:
        _append(overrides, "trainer.max_critic_ckpt_to_keep", runtime["max_critic_ckpt_to_keep"])
    if runtime.get("export_lora_adapter_dir") is not None:
        _append(
            overrides,
            "actor_rollout_ref.actor.checkpoint.export_lora_adapter_dir",
            runtime["export_lora_adapter_dir"],
            create=True,
        )
    _append(overrides, "trainer.project_name", trainer["project_name"])
    _append(overrides, "trainer.experiment_name", trainer["experiment_name"])
    _append(overrides, "trainer.total_epochs", trainer["total_epochs"])
    _append(overrides, "trainer.save_freq", trainer["save_freq"])
    _append(overrides, "trainer.test_freq", trainer["test_freq"])
    _append(overrides, "actor_rollout_ref.actor.optim.total_training_steps", trainer["total_training_steps"])


# 追加 RuleGroundedProcessRL 专属 trainer、agent 与奖励配置。
def _append_rule_grounded_overrides(overrides: list[str], config: dict[str, Any]) -> None:
    _append(overrides, "trainer.v1.trainer_mode", "rule_grounded_process_rl_sync")
    _append(
        overrides,
        "trainer.v1.custom_trainer_module",
        "recipe.formally_verifiable.rule_grounded_process_rl.trainer",
        create=True,
    )
    _append(overrides, "actor_rollout_ref.rollout.agent.default_agent_loop", "rule_grounded_process_rl")
    _append(
        overrides,
        "actor_rollout_ref.rollout.agent.agent_loop_config_path",
        "recipe/formally_verifiable/rule_grounded_process_rl/agent_loop.yaml",
    )
    base = "algorithm.rule_grounded_process_rl"
    method_cfg = config["rule_grounded_process_rl"]
    for key, value in method_cfg.items():
        if key in {"reward", "initial_baseline_snapshot"}:
            continue
        _append(overrides, f"{base}.{key}", value, create=True)
    for key, value in config["reward"].items():
        if key == "kl_beta":
            continue
        _append(overrides, f"{base}.reward.{key}", value, create=True)
    for bucket, state in (method_cfg.get("initial_baseline_snapshot") or {}).items():
        for key, value in state.items():
            _append(overrides, f"{base}.initial_baseline_snapshot.{bucket}.{key}", value, create=True)


# 把最终方法 YAML 转换为完整的 VERL Hydra overrides。
def build_verl_overrides(config: dict[str, Any]) -> list[str]:
    method = config.get("method")
    if method != "rule_grounded_process_rl":
        raise ValueError(f"Unsupported method: {method}")
    overrides: list[str] = []
    _append_common_overrides(overrides, config)
    _append_rule_grounded_overrides(overrides, config)
    return overrides
