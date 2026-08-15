# RuleGroundedProcessRL

This repository contains the final RuleGroundedProcessRL implementation and
its aligned evaluation code. Historical baselines, superseded configurations,
experiment outputs, adapters, and local scratch scripts are deliberately not
included.

## Repository layout

```text
Training/  Final VERL recipe, configuration, and tested VERL integration overlay
Test/      Generation evaluator, metrics, shard merging, and CPU smoke tests
```

## Training

Install the method into the tested VERL revision:

```bash
git clone https://github.com/verl-project/verl.git
git -C verl checkout 91666d99
python Training/install_into_verl.py --verl-root /path/to/verl
```

Set paths and preprocess the data from the VERL checkout:

```bash
cd /path/to/verl
export PYTHONPATH="$PWD"
export FVCODE_ROOT=/path/to/project-data-root
export QWEN_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct

python -m recipe.formally_verifiable.data_preprocess \
  --recipe-config recipe/formally_verifiable/config/rule_grounded_process_rl.yaml

bash recipe/formally_verifiable/rule_grounded_process_rl/run_rule_grounded_process_rl.sh
```

`FVCODE_ROOT` must contain the data and warmup adapter paths referenced by the
YAML. Hydra overrides may be appended to the launch command when paths differ.

## Evaluation

Evaluation uses the exact training-side formal logic implementation. See
`Test/README.md` for single-adapter evaluation and uncertainty-subset summaries.

## Result

| Method | Avg@3 ↑ | AccPass@3 ↑ | Format ↑ | Formal Verification ↑ | RGD ↓ |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct (Base) | 34.74 | 56.84 | 77.54 | 68.26 | 0.841 |
| SFT | 36.49 | 57.89 | 76.84 | 66.56 | 0.870 |
| **RuleGroundedProcessRL (Ours)** | **65.61** | **86.32** | **96.49** | **71.60** | **0.659** |
