#!/usr/bin/env python3
"""Validate ProofWriter A/B data with the repository's FOL and Z3 stack."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


REQUIRED_FIELDS = (
    "id",
    "dataset",
    "split",
    "context",
    "question",
    "options",
    "answer",
    "difficulty",
    "nl2fol",
    "conclusion_fol",
    "reasoning",
    "evaluation_metadata",
)
BINARY_OPTIONS = ["A) True", "B) False"]


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def opposite_formula(formula: str) -> str:
    formula = formula.strip()
    if formula.startswith("¬(") and formula.endswith(")"):
        return formula[2:-1].strip()
    return f"¬({formula})"


def reasoning_formulas(problem: dict[str, Any]) -> list[str]:
    lines = []
    for line in str(problem["reasoning"]).splitlines():
        if not line.startswith("Conclusion: "):
            raise ValueError(f"invalid reasoning line: {line!r}")
        lines.append(line.removeprefix("Conclusion: ").strip())
    formulas = [
        str(formula).strip()
        for formula in problem["evaluation_metadata"].get("reference_reasoning_fol", [])
    ]
    if not lines or not formulas:
        raise ValueError("empty reasoning chain")
    if len(lines) != len(formulas):
        raise ValueError("reasoning text and reference FOL have different step counts")
    return formulas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--report", type=Path)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo_root.resolve() / "Training"))
    from recipe.formally_verifiable.common.fol_converter import FOLToZ3Converter
    from z3 import Not, Solver, unsat

    reports: dict[str, Any] = {}
    all_valid = True

    for path in args.files:
        converter = FOLToZ3Converter()
        theory_cache: dict[tuple[str, ...], tuple[list[Any], str]] = {}
        counts: Counter[str] = Counter()
        ids: set[str] = set()
        errors: list[str] = []

        for line_number, problem in read_jsonl(path):
            counts["records"] += 1
            try:
                missing = [field for field in REQUIRED_FIELDS if field not in problem]
                if missing:
                    raise ValueError(f"missing fields {missing}")
                if problem["options"] != BINARY_OPTIONS:
                    raise ValueError(f"options must be exactly {BINARY_OPTIONS}")
                if problem["answer"] not in {"A", "B"}:
                    raise ValueError("answer must be A or B")
                if problem["difficulty"] not in {"easy", "medium", "hard"}:
                    raise ValueError("difficulty must be easy, medium or hard")

                problem_id = str(problem["id"])
                if not problem_id or problem_id in ids:
                    raise ValueError(f"empty or duplicate id {problem_id!r}")
                ids.add(problem_id)

                metadata = problem["evaluation_metadata"]
                if problem["dataset"] != "proofwriter" or problem["split"] != "test":
                    raise ValueError("top-level dataset/split must be proofwriter/test")
                if metadata.get("dataset") != "ProofWriter":
                    raise ValueError("evaluation_metadata.dataset must be ProofWriter")
                if metadata.get("proof_source") != "proofsWithIntermediates":
                    raise ValueError("reasoning proof source is not proofsWithIntermediates")

                formulas = tuple(str(value).strip() for value in problem["nl2fol"].values())
                if not formulas or any(not formula for formula in formulas):
                    raise ValueError("empty premise set or formula")
                if formulas not in theory_cache:
                    premises = [converter.convert(formula) for formula in formulas]
                    solver = Solver()
                    solver.set("timeout", args.timeout_ms)
                    solver.add(*premises)
                    theory_cache[formulas] = (premises, str(solver.check()))
                premises, premise_status = theory_cache[formulas]
                if premise_status != "sat":
                    raise ValueError(f"premises must be SAT, got {premise_status}")

                conclusion_text = str(problem["conclusion_fol"]).strip()
                conclusion = converter.convert(conclusion_text)
                solver = Solver()
                solver.set("timeout", args.timeout_ms)
                solver.add(*premises)
                solver.add(Not(conclusion) if problem["answer"] == "A" else conclusion)
                if solver.check() != unsat:
                    raise ValueError(f"gold answer {problem['answer']} is not entailed")

                proof_formulas = reasoning_formulas(problem)
                for proof_formula in proof_formulas:
                    converter.convert(proof_formula)
                expected_last = (
                    conclusion_text
                    if problem["answer"] == "A"
                    else opposite_formula(conclusion_text)
                )
                if proof_formulas[-1] != expected_last:
                    raise ValueError(
                        "proof chain does not end in the gold literal: "
                        f"expected {expected_last!r}, got {proof_formulas[-1]!r}"
                    )

                counts[f"answer_{problem['answer']}"] += 1
                counts[f"difficulty_{problem['difficulty']}"] += 1
                counts["proofs_from_official_intermediates"] += 1
            except Exception as exc:
                errors.append(f"line {line_number}: {exc}")
                if len(errors) >= 50:
                    break

        valid = not errors
        all_valid &= valid
        reports[path.name] = {
            "valid": valid,
            "counts": dict(sorted(counts.items())),
            "unique_ids": len(ids),
            "unique_theories_checked": len(theory_cache),
            "errors": errors,
        }

    report = {
        "valid": all_valid,
        "z3_timeout_ms": args.timeout_ms,
        "checks": [
            "required schema and unique ids",
            "A/B labels and exactly two options",
            "project FOL parser compatibility",
            "satisfiable premise theories",
            "Z3 entailment of every gold label",
            "reasoning sourced from proofsWithIntermediates",
            "proof chain ends in the gold literal",
        ],
        "files": reports,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
