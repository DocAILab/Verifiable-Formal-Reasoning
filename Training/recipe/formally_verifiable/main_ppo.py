"""Recipe entry point that loads YAML settings before delegating to verl main_ppo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from recipe.formally_verifiable.config_utils import build_verl_overrides, load_recipe_config


# 解析命令行参数并启动当前模块的主流程。
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--recipe-config", required=True)
    args, hydra_overrides = parser.parse_known_args(argv)

    recipe_config = load_recipe_config(args.recipe_config)
    if recipe_config.get("method") != "rule_grounded_process_rl":
        raise ValueError("This repository only supports rule_grounded_process_rl")
    import recipe.formally_verifiable.rule_grounded_process_rl.trainer  # noqa: F401

    overrides = build_verl_overrides(recipe_config)
    sys.argv = [sys.argv[0], *overrides, *hydra_overrides]
    from verl.trainer.main_ppo import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()
