"""Merge sharded generation-evaluation outputs into one report directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FVCODE_ROOT = Path(__file__).resolve().parents[2]
if str(FVCODE_ROOT) not in sys.path:
    sys.path.insert(0, str(FVCODE_ROOT))

from Test.Generation.metrics import (  # noqa: E402
    load_jsonl,
    plot_metrics_svg,
    summarize_by_difficulty,
    summarize_records,
    write_difficulty_summary,
    write_json,
    write_jsonl,
    write_report,
)


def _record_sort_key(record: dict[str, Any]) -> tuple[str, int, str, int]:
    problem_id = str(record.get("problem_id", ""))
    try:
        numeric_id = int(problem_id)
    except ValueError:
        numeric_id = 10**12
    return (
        str(record.get("difficulty") or "unknown"),
        numeric_id,
        problem_id,
        int(record.get("sample_index", 0) or 0),
    )


def _load_records(shard_dir: Path) -> list[dict[str, Any]]:
    records_path = shard_dir / "records.jsonl"
    if not records_path.exists():
        records_path = shard_dir / "records" / "incremental_records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"No records found in shard: {shard_dir}")
    return load_jsonl(records_path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "records").mkdir(parents=True, exist_ok=True)
    (output_dir / "responses" / "text").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    shard_dirs = [Path(item) for item in args.shard_dirs]
    for shard_dir in shard_dirs:
        for record in _load_records(shard_dir):
            merged = dict(record)
            merged["run_name"] = args.name
            records.append(merged)
        response_dir = shard_dir / "responses" / "text"
        if response_dir.exists():
            for text_file in response_dir.glob("*.txt"):
                shutil.copy2(text_file, output_dir / "responses" / "text" / text_file.name)

    records.sort(key=_record_sort_key)
    input_path = Path(args.input).resolve()
    input_rows = load_jsonl(input_path)
    expected_problems = {
        (
            str(problem.get("difficulty") or "unknown").strip().lower(),
            str(problem.get("id")),
        )
        for problem in input_rows
    }
    observed_problems = {
        (
            str(record.get("difficulty") or "unknown").strip().lower(),
            str(record.get("problem_id")),
        )
        for record in records
    }
    if observed_problems != expected_problems:
        missing = sorted(expected_problems - observed_problems)[:20]
        extra = sorted(observed_problems - expected_problems)[:20]
        raise RuntimeError(
            "Shard records do not exactly cover the evaluation dataset: "
            f"missing={missing}, extra={extra}"
        )
    expected_record_keys = {
        (difficulty, problem_id, sample_index)
        for difficulty, problem_id in expected_problems
        for sample_index in range(args.num_samples)
    }
    observed_record_keys = {
        (
            str(record.get("difficulty") or "unknown").strip().lower(),
            str(record.get("problem_id")),
            int(record.get("sample_index", 0) or 0),
        )
        for record in records
    }
    if observed_record_keys != expected_record_keys or len(records) != len(expected_record_keys):
        raise RuntimeError(
            "Shard records do not contain exactly the requested number of samples "
            f"per problem: expected={len(expected_record_keys)}, actual={len(records)}"
        )
    k = min(3, args.num_samples)
    summary = {
        "run_name": args.name,
        "method_type": "generation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "adapter": args.adapter,
        "input": str(input_path),
        "dataset_sha256": _file_sha256(input_path),
        "dataset_problem_count": len(expected_problems),
        "output_dir": str(output_dir),
        "split": args.split,
        "num_samples": args.num_samples,
        "shard_dirs": [str(item) for item in shard_dirs],
        "metrics": summarize_records(records, k=k),
        "by_difficulty": summarize_by_difficulty(records, k=k),
    }
    write_jsonl(output_dir / "records" / "incremental_records.jsonl", records)
    write_jsonl(output_dir / "records.jsonl", records)
    write_json(output_dir / "summary.json", summary)
    write_difficulty_summary(output_dir, summary)
    write_json(output_dir / "run_config.json", vars(args))
    write_report(output_dir / "report.md", summary)
    plot_metrics_svg(output_dir / "plots" / "metrics.svg", summary)
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
