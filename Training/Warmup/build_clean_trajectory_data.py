"""Build warmup data from verified clean proof trajectories."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": step["id"],
        "dependencies": list(step.get("dependencies") or []),
        "conclusion": step["conclusion"],
        "rule": step["rule"],
    }


def convert_problem(
    clean_problem: dict[str, Any],
    *,
    problem_id: Any,
    difficulty: Any,
    answer: str,
) -> dict[str, Any]:
    nl2fol = {
        str(premise.get("natural_language", "")).strip(): str(
            premise.get("formal", "")
        ).strip()
        for premise in clean_problem.get("premises") or []
        if premise.get("natural_language") and premise.get("formal")
    }
    options = [
        f"{option.get('letter')}) {option.get('text')}"
        for option in clean_problem.get("answer_options") or []
        if option.get("letter") and option.get("text")
    ]
    return {
        "id": problem_id,
        "difficulty": difficulty,
        "question": clean_problem.get("question", ""),
        "nl2fol": nl2fol,
        "options": options,
        "answer": answer,
        "conclusion_fol": clean_problem.get("target_conclusion", ""),
    }


def build_reasoning(row: dict[str, Any], derivation: list[dict[str, Any]]) -> str:
    descriptions = []
    for step in derivation:
        dependencies = ", ".join(step.get("dependencies") or [])
        descriptions.append(
            f"{step.get('id')}: apply {step.get('rule')} to {dependencies} "
            f"to derive {step.get('conclusion')}."
        )
    body = "\n".join(descriptions)
    return (
        f"Use the verified formal derivation for problem {row.get('problem_id')}.\n"
        f"{body}\n"
        "The final structured step binds the derivation to the selected answer option."
    )


def build_response(row: dict[str, Any], derivation: list[dict[str, Any]]) -> str:
    reasoning = build_reasoning(row, derivation)
    summary = json.dumps(derivation, ensure_ascii=False, indent=2)
    return f"<think>\n{reasoning}\n</think>\n<summary>{summary}</summary>"


def has_final_answer(row: dict[str, Any], derivation: list[dict[str, Any]]) -> bool:
    if not derivation or not row.get("has_final_answer"):
        return False
    return str(derivation[-1].get("id", "")).startswith("h_goal_")


def build_records(
    rows: list[dict[str, Any]],
    *,
    require_final_answer: bool,
    exclude_uncertain: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for source_index, row in enumerate(rows):
        answer = str(row.get("gold_answer", "")).strip().upper()
        if exclude_uncertain and answer == "C":
            skipped["uncertain"] += 1
            continue
        derivation = [
            clean_step(step) for step in row.get("accepted_derivation") or []
        ]
        if require_final_answer and not has_final_answer(row, derivation):
            skipped["no_final_answer"] += 1
            continue
        if not derivation:
            skipped["empty_derivation"] += 1
            continue

        problem = convert_problem(
            row.get("problem") or {},
            problem_id=row.get("problem_id"),
            difficulty=row.get("difficulty"),
            answer=answer,
        )
        records.append(
            {
                "id": len(records),
                "source_index": source_index,
                "record_index": row.get("record_index"),
                "problem_id": row.get("problem_id"),
                "difficulty": row.get("difficulty"),
                "answer": answer,
                "problem": problem,
                "response": build_response(row, derivation),
                "purpose": "clean_trajectory_warmup",
                "source": "rule_grounded_clean_trajectory",
                "accepted_step_count": len(derivation),
                "rule_schema_version": row.get("rule_schema_version"),
            }
        )
    return records, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert verified clean trajectories into warmup SFT data."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_uncertain", action="store_true")
    parser.add_argument("--allow_non_final", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    records, skipped = build_records(
        load_jsonl(input_path),
        require_final_answer=not args.allow_non_final,
        exclude_uncertain=args.exclude_uncertain,
    )
    random.Random(args.seed).shuffle(records)
    if args.max_records is not None:
        records = records[: args.max_records]
    for index, record in enumerate(records):
        record["id"] = index

    output_path = Path(args.output)
    write_jsonl(output_path, records)
    difficulty = Counter(str(record.get("difficulty")) for record in records)
    answers = Counter(str(record.get("answer")) for record in records)
    step_counts = [int(record.get("accepted_step_count", 0)) for record in records]
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "num_records": len(records),
        "require_final_answer": not args.allow_non_final,
        "exclude_uncertain": bool(args.exclude_uncertain),
        "max_records": args.max_records,
        "seed": args.seed,
        "skipped": dict(skipped),
        "difficulty": dict(difficulty),
        "answers": dict(answers),
        "avg_accepted_step_count": (
            round(sum(step_counts) / len(step_counts), 4) if step_counts else 0.0
        ),
        "min_accepted_step_count": min(step_counts) if step_counts else 0,
        "max_accepted_step_count": max(step_counts) if step_counts else 0,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
