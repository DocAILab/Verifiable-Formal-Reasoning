# Training

This directory contains only the final RuleGroundedProcessRL implementation.
It is an overlay for VERL commit `91666d99`, rather than a vendored copy of
the complete upstream framework.

## Contents

- `recipe/formally_verifiable/rule_grounded_process_rl/`: rollout, reward,
  RuleChecker, action-span credit, advantages, and trainer integration.
- `recipe/formally_verifiable/config/rule_grounded_process_rl.yaml`: the final
  2,935-problem, 1,098-update experiment configuration.
- `recipe/formally_verifiable/data_preprocess.py`: conversion to VERL's RLHF
  JSONL schema.
- `verl_overlay/`: the exact upstream integration files used by the experiment,
  including custom trainer loading, FSDP LoRA initialization/export, and vLLM
  LoRA synchronization.
- `install_into_verl.py`: revision-checked installer for the recipe and overlay.

## Install

```bash
git clone https://github.com/verl-project/verl.git
git -C verl checkout 91666d99
python Training/install_into_verl.py --verl-root /path/to/verl
```

The overlay is intentionally tied to the tested VERL revision. Review the
integration changes before using `--force` with another revision.

