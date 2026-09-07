# RuleGroundedProcessRL

This repository contains the final RuleGroundedProcessRL implementation and
its aligned evaluation code. Historical baselines, superseded configurations,
experiment outputs, adapters, and local scratch scripts are deliberately not
included.

## Repository layout

```text
Data/      Raw problem records, legacy VERL training records, and test data
Training/  Final VERL recipe, configuration, and tested VERL integration overlay
Test/      Generation evaluator, metrics, shard merging, and CPU smoke tests
```

## Data formats and paths

**The two files named `train.jsonl` below have different schemas. Do not feed
the already converted VERL file to the raw-data preprocessor.**

| Path relative to this repository | Records | Format and role |
|---|---:|---|
| `Data/ProverQA/datasets/official_ab_fol_verified_v1/train.jsonl` | 2,935 | Raw problems; use as `data.raw_train_files` |
| `Data/ProverQA/datasets/official_ab_fol_verified_v1/dev.jsonl` | 150 | Existing raw training-validation split; use as `data.raw_val_files` |
| `Data/ProverQA/train.jsonl` | 2,935 | Already converted VERL training records; retained for compatibility, not raw input |
| `Data/ProverQA/test/golden/all.jsonl` | 681 | Current canonical A/B test set; report overall and Easy/Medium/Hard results |

The raw training set contains 1,488 A and 1,447 B answers, with no Uncertain
examples. It is identical, in content and order, to `extra_info.problem` in the
legacy VERL training file. No labels or formal annotations were regenerated.
The 150-record validation split is the existing split referenced by the training
configs (36 A, 59 B, 55 C), not a replacement for the current 681-record test set.
The restored raw files are copies of the original local dataset, not new splits.

A **raw problem** has `id`, `question`, `nl2fol`, `options`, `answer`, and
`conclusion_fol` directly at the top level. A **VERL record** instead has
`data_source`, `prompt`, `ability`, `reward_model`, and `extra_info`; its raw
problem is stored once in `extra_info.problem` and `reward_model.ground_truth`.

The recommended path is to preprocess the checked-in raw files exactly once.
The default config writes the generated VERL files to:

```text
Data/Verl/rule_grounded_process_rl_verified2935/train.jsonl
Data/Verl/rule_grounded_process_rl_verified2935/dev.jsonl
```

Do not change `raw_train_files` to `Data/ProverQA/train.jsonl`. If intentionally
using an already converted file, assign it to `train_files` and skip training-data
conversion; prepare a compatible `val_files` separately. Keep the prompt mode
aligned with the selected model. The default quick start below uses the raw-file
route, not this alternative. Reinstall the recipe after editing it or edit the
installed YAML under the VERL checkout, which is the copy used by the commands.

Preprocessing rejects already converted records and missing/empty problem fields
with the input path and line number, rather than silently creating empty prompts.

## Training

Install the method into the tested VERL revision:

```bash
git clone https://github.com/verl-project/verl.git
git -C verl checkout 91666d99
python Training/install_into_verl.py --verl-root /path/to/verl
```

Set `FVCODE_ROOT` to the absolute path of **this repository**, where `Data/` and
`Training/` live. Then preprocess the raw data from the VERL checkout:

```bash
cd /path/to/verl
export PYTHONPATH="$PWD"
export FVCODE_ROOT=/absolute/path/to/Verifiable-Formal-Reasoning
export QWEN_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct

python -m recipe.formally_verifiable.data_preprocess \
  --recipe-config recipe/formally_verifiable/config/rule_grounded_process_rl.yaml && \
bash recipe/formally_verifiable/rule_grounded_process_rl/run_rule_grounded_process_rl.sh
```

The YAML's raw-data paths now exist in this repository; do not substitute the
legacy VERL training file merely because it has the same basename. Create the
warmup adapter referenced by the YAML before starting RL; see
`Training/Warmup/README.md`. Hydra overrides may be appended to the launch command
when paths differ. This quick start uses the Qwen2.5 explicit-thinking config;
changing only model/adapter paths is not a complete native-thinking adaptation.
The `&&` prevents RL from starting if preprocessing fails.

## Evaluation

Evaluation uses the exact training-side formal logic implementation. See
`Test/README.md` for single-adapter evaluation and uncertainty-subset summaries.

## Result

| Method | Avg@3 ↑ | AccPass@3 ↑ | Format ↑ | Formal Verification ↑ | RGD ↓ |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct (Base) | 34.74 | 56.84 | 77.54 | 68.26 | 0.841 |
| SFT | 36.49 | 57.89 | 76.84 | 66.56 | 0.870 |
| **RuleGroundedProcessRL (Ours)** | **65.61** | **86.32** | **96.49** | **71.60** | **0.659** |
