#!/usr/bin/env bash
set -xeuo pipefail

RECIPE_CONFIG=${RECIPE_CONFIG:-recipe/formally_verifiable/config/rule_grounded_process_rl.yaml}

python3 -m recipe.formally_verifiable.main_ppo \
  --recipe-config "${RECIPE_CONFIG}" \
  "$@"
