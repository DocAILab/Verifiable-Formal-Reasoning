"""Write paired generation metrics with and without Uncertain problems."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FVCODE_ROOT = Path(__file__).resolve().parents[2]
if str(FVCODE_ROOT) not in sys.path:
    sys.path.insert(0, str(FVCODE_ROOT))

from Test.Generation.metrics import (  # noqa: E402
    load_jsonl,
    summarize_by_difficulty,
    summarize_records,
    write_json,
)


def _load_summary(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing merged summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records = load_jsonl(output_dir / "records.jsonl")
    full_summary = _load_summary(output_dir)
    certain_records = [
        record
        for record in records
        if str(record.get("ground_truth", "")).strip().upper() != "C"
    ]

    including = dict(full_summary)
    including["uncertain_policy"] = "include"
    including["ground_truth_labels"] = ["A", "B", "C"]
    excluding = {
        key: value
        for key, value in full_summary.items()
        if key not in {"metrics", "by_difficulty"}
    }
    excluding.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "uncertain_policy": "exclude",
            "ground_truth_labels": ["A", "B"],
            "metrics": summarize_records(certain_records, k=args.k),
            "by_difficulty": summarize_by_difficulty(certain_records, k=args.k),
        }
    )
    paired = {
        "run_name": full_summary.get("run_name"),
        "split": full_summary.get("split"),
        "including_uncertain": including["metrics"],
        "excluding_uncertain": excluding["metrics"],
    }

    write_json(output_dir / "summary_including_uncertain.json", including)
    write_json(output_dir / "summary_excluding_uncertain.json", excluding)
    write_json(output_dir / "summary_uncertainty_subsets.json", paired)
    print(json.dumps(paired, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
