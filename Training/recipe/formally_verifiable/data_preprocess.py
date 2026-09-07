"""Convert formal-verification JSONL records to verl RLHF schema."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
import sys
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from recipe.formally_verifiable.config_utils import load_recipe_config
from recipe.formally_verifiable.rule_grounded_process_rl.structured_prompt import (
    build_messages as build_rule_grounded_messages,
)

METHOD = "rule_grounded_process_rl"
DATA_SOURCE = "formally_verifiable/rule_grounded_process_rl"


def validate_raw_problem(problem: Any) -> None:
    if not isinstance(problem, dict):
        raise ValueError("Expected a raw problem object, not a JSON scalar or array.")
    if any(key in problem for key in ("prompt", "extra_info", "reward_model")):
        raise ValueError(
            "Expected a raw problem, but received an already converted VERL record. "
            "Use Data/ProverQA/datasets/official_ab_fol_verified_v1/train.jsonl "
            "as raw_train_files. Data/ProverQA/train.jsonl is already converted; "
            "use it as train_files without preprocessing it again."
        )
    missing = [
        key for key in ("id", "question", "nl2fol", "options", "answer", "conclusion_fol")
        if key not in problem or problem[key] is None
    ]
    if missing:
        raise ValueError(f"Raw problem is missing required fields: {missing}")
    if not str(problem["id"]).strip():
        raise ValueError("Raw problem id must not be empty.")
    for key in ("question", "conclusion_fol"):
        if not isinstance(problem[key], str) or not problem[key].strip():
            raise ValueError(f"Raw problem {key} must be a nonempty string.")
    premises = problem["nl2fol"]
    if not isinstance(premises, dict) or not premises or any(
        not isinstance(value, str) or not value.strip()
        for pair in premises.items() for value in pair
    ):
        raise ValueError("Raw problem nl2fol must map nonempty text to nonempty formulas.")
    options = problem["options"]
    if not isinstance(options, list) or not options or any(
        not isinstance(option, str) or not option.strip() for option in options
    ):
        raise ValueError("Raw problem options must be a nonempty list of strings.")
    if problem["answer"] not in ("A", "B", "C"):
        raise ValueError("Raw problem answer must be A, B, or C.")


# 将单条形式化问题转换为 VERL RLHF 数据格式。
def convert_problem_record(problem: dict[str, Any], *, method: str) -> dict[str, Any]:
    if method != METHOD:
        raise ValueError(f"Unsupported method: {method}")
    validate_raw_problem(problem)
    prompt = build_rule_grounded_messages(problem)
    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": "formal_logic",
        "reward_model": {"style": "rule", "ground_truth": problem},
        "extra_info": {
            "problem": problem,
            "problem_id": problem.get("id"),
            "method": method,
        },
    }


# 逐行读取原始数据，并按配置过滤和转换样本。
def iter_converted_records(
    input_path: str | Path,
    *,
    method: str,
    exclude_uncertain: bool = True,
) -> Iterator[dict[str, Any]]:
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                problem = json.loads(line)
                converted = convert_problem_record(problem, method=method)
            except ValueError as exc:
                raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
            if exclude_uncertain and str(problem.get("answer", "")).strip().upper() == "C":
                continue
            yield converted


# 将转换后的样本流写入 JSONL 文件。
def write_jsonl(records: Iterator[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# 把单路径或路径列表统一转换为字符串列表。
def _as_file_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


# 成对处理原始数据文件与目标 VERL 数据文件。
def _preprocess_file_pairs(
    *,
    raw_files: Any,
    output_files: Any,
    method: str,
    exclude_uncertain: bool,
) -> None:
    raw_paths = _as_file_list(raw_files)
    output_paths = _as_file_list(output_files)
    if len(raw_paths) != len(output_paths):
        raise ValueError(
            "raw and processed file lists must have the same length: "
            f"{len(raw_paths)} raw vs {len(output_paths)} processed"
        )
    for raw_path, output_path in zip(raw_paths, output_paths, strict=True):
        write_jsonl(
            iter_converted_records(raw_path, method=method, exclude_uncertain=exclude_uncertain),
            output_path,
        )


# 依据 recipe 配置预处理训练集和验证集。
def preprocess_from_config(config_path: str | Path) -> None:
    config = load_recipe_config(config_path)

    method = config["method"]
    data = config["data"]
    unresolved = [
        str(data[key])
        for key in ("raw_train_files", "raw_val_files", "train_files", "val_files")
        if "${oc.env:" in str(data[key])
    ]
    if unresolved:
        raise ValueError("Set FVCODE_ROOT before preprocessing; unresolved paths: " + ", ".join(unresolved))
    default_exclude_uncertain = bool(data.get("exclude_uncertain", True))
    exclude_uncertain_train = bool(
        data.get("exclude_uncertain_train", default_exclude_uncertain)
    )
    exclude_uncertain_val = bool(
        data.get("exclude_uncertain_val", default_exclude_uncertain)
    )
    _preprocess_file_pairs(
        raw_files=data["raw_train_files"],
        output_files=data["train_files"],
        method=method,
        exclude_uncertain=exclude_uncertain_train,
    )
    _preprocess_file_pairs(
        raw_files=data["raw_val_files"],
        output_files=data["val_files"],
        method=method,
        exclude_uncertain=exclude_uncertain_val,
    )


# 解析命令行参数并启动当前模块的主流程。
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-config")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--method", choices=[METHOD], default=METHOD)
    parser.add_argument("--include-uncertain", action="store_true")
    args = parser.parse_args()
    if args.recipe_config:
        preprocess_from_config(args.recipe_config)
        return
    if not args.input or not args.output or not args.method:
        parser.error("--input, --output, and --method are required unless --recipe-config is set")
    write_jsonl(
        iter_converted_records(args.input, method=args.method, exclude_uncertain=not args.include_uncertain),
        args.output,
    )


if __name__ == "__main__":
    main()
