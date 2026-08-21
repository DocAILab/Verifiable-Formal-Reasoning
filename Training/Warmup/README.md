# Warmup

This directory contains the reproducible LoRA SFT warmup stage shared by the
Qwen2.5 and Qwen3 RuleGroundedProcessRL experiments. It is intentionally
separate from baseline methods.

## Modes

| Config | Model | Thinking behavior |
| --- | --- | --- |
| `configs/qwen3_8b_native.yaml` | Qwen3-8B | Native chat-template thinking (`enable_thinking=True`) |
| `configs/qwen25_7b_explicit.yaml` | Qwen2.5-7B-Instruct | Explicit `<think>...</think>` output without native thinking |

Both modes use the same structured prompt implementation as RL training:

```text
Training/recipe/formally_verifiable/rule_grounded_process_rl/structured_prompt.py
```

The Qwen3 chat template already opens the native `<think>` block. The collator
therefore removes only the duplicate leading `<think>` from the supervised
target. The reasoning content, closing `</think>`, and `<summary>` are still
trained. Qwen2.5 keeps the complete explicit response unchanged.

## Data

The checked-in warmup set contains 56 verified, final-answer trajectories:

```text
Data/Warmup/clean_trajectory_warmup56.jsonl
```

It can be rebuilt from clean trajectory records with:

```bash
python Training/Warmup/build_clean_trajectory_data.py \
  --input Data/StepSFT/cleaned/rule_grounded_v2/step_sft_clean307_trajectories.jsonl \
  --output Data/Warmup/clean_trajectory_warmup56.jsonl \
  --max_records 56 \
  --exclude_uncertain
```

## Train

```bash
export FVCODE_ROOT=/path/to/FVCode

# Qwen3-8B native thinking
export QWEN3_MODEL_PATH=/path/to/Qwen3-8B
python Training/Warmup/train_lora.py \
  --config Training/Warmup/configs/qwen3_8b_native.yaml

# Qwen2.5-7B explicit thinking tags, no native thinking mode
export QWEN25_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
python Training/Warmup/train_lora.py \
  --config Training/Warmup/configs/qwen25_7b_explicit.yaml
```

Adapters, tokenizer files, logs, and resolved configs are written under
`Output/Warmup/`.

