"""Install the method recipe and tested integration overlay into a VERL checkout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


TESTED_VERL_COMMIT = "91666d99"
TRAINING_ROOT = Path(__file__).resolve().parent


def git_revision(verl_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(verl_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def copy_tree(source: Path, destination: Path) -> None:
    for source_file in source.rglob("*"):
        if not source_file.is_file() or "__pycache__" in source_file.parts:
            continue
        target = destination / source_file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-root", required=True, type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Install even when the VERL revision differs from the tested commit.",
    )
    args = parser.parse_args()

    verl_root = args.verl_root.resolve()
    if not (verl_root / "verl" / "__init__.py").exists():
        raise FileNotFoundError(f"Not a VERL checkout: {verl_root}")
    revision = git_revision(verl_root)
    if not revision.startswith(TESTED_VERL_COMMIT) and not args.force:
        raise RuntimeError(
            f"Expected VERL {TESTED_VERL_COMMIT}, found {revision[:8]}. "
            "Checkout the tested revision or pass --force after reviewing compatibility."
        )

    copy_tree(TRAINING_ROOT / "recipe", verl_root / "recipe")
    copy_tree(TRAINING_ROOT / "verl_overlay", verl_root)
    print(f"Installed RuleGroundedProcessRL into {verl_root}")
    print(f"VERL revision: {revision[:8]}")


if __name__ == "__main__":
    main()

