"""Recompute summaries from saved generation records without loading a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


FVCODE_ROOT = Path(__file__).resolve().parents[2]
if str(FVCODE_ROOT) not in sys.path:
    sys.path.insert(0, str(FVCODE_ROOT))

from Test.Generation.metrics import (  # noqa: E402
    load_jsonl,
    plot_metrics_svg,
    summarize_by_difficulty,
    summarize_records,
    write_json,
    write_jsonl,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute generation evaluation metrics.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records_path = output_dir / "records.jsonl"
    if not records_path.exists():
        records_path = output_dir / "records" / "incremental_records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"No records found under {output_dir}")

    old_summary = {}
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        old_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    records = load_jsonl(records_path)
    k = args.k or int((old_summary.get("metrics") or {}).get("k", 3) or 3)
    summary = dict(old_summary)
    summary["metrics"] = summarize_records(records, k=k)
    summary["by_difficulty"] = summarize_by_difficulty(records, k=k)
    summary["recomputed_from"] = str(records_path)

    write_jsonl(output_dir / "records.jsonl", records)
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "report.md", summary)
    plot_metrics_svg(output_dir / "plots" / "metrics.svg", summary)
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
