#!/usr/bin/env python3
"""Convert official ProofWriter OWA test data to the project A/B schema."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence


BINARY_OPTIONS = ["A) True", "B) False"]
DEPTH_PARTITIONS = ("depth-0", "depth-1", "depth-2", "depth-3", "depth-5")
ATOM_RE = re.compile(
    r'\(\s*"([^"]+)"\s+"([^"]+)"\s+"([^"]+)"\s+"([+\-~])"\s*\)'
)
VARIABLE_ALIASES = {
    "something",
    "someone",
    "somebody",
    "it",
    "they",
    "anything",
    "anyone",
}
DIFFICULTY_RANK = {"easy": 0, "medium": 1, "hard": 2}


@dataclass(frozen=True, order=True)
class Literal:
    predicate: str
    arguments: tuple[str, ...]
    negative: bool = False

    def opposite(self) -> "Literal":
        return Literal(self.predicate, self.arguments, not self.negative)


@dataclass(frozen=True)
class Rule:
    antecedents: tuple[Literal, ...]
    consequent: Literal


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)
    )


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)
            count += 1
    return count, digest.hexdigest()


def rows_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_identifier(value: str, *, fallback: str) -> str:
    value = value.strip().replace("'", "")
    value = re.sub(r"[^0-9A-Za-z_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = fallback
    if value[0].isdigit():
        value = f"{fallback}_{value}"
    # The project parser treats a single lower-case letter as a variable.
    if len(value) == 1 and value.islower():
        value = f"{fallback}_{value}"
    return value


def is_variable(value: str) -> bool:
    return value.strip().lower() in VARIABLE_ALIASES


def atom_to_literal(atom: tuple[str, str, str, str]) -> Literal:
    subject, relation, obj, polarity = atom
    if relation.strip().lower() == "is":
        predicate = sanitize_identifier(obj.lower(), fallback="predicate")
        arguments = (subject.strip(),)
    else:
        predicate = sanitize_identifier(relation.lower(), fallback="predicate")
        arguments = (subject.strip(), obj.strip())
    return Literal(predicate, arguments, polarity in {"-", "~"})


def parse_fact_representation(text: str) -> Literal:
    match = ATOM_RE.fullmatch(text.strip())
    if not match:
        raise ValueError(f"Unsupported atom representation: {text}")
    return atom_to_literal(match.groups())


def parse_rule_representation(text: str) -> Rule:
    if "->" not in text:
        raise ValueError(f"Rule has no implication arrow: {text}")
    left, right = text.split("->", 1)
    antecedents = [atom_to_literal(match) for match in ATOM_RE.findall(left)]
    consequents = [atom_to_literal(match) for match in ATOM_RE.findall(right)]
    if not antecedents or len(consequents) != 1:
        raise ValueError(f"Unsupported rule representation: {text}")
    return Rule(tuple(antecedents), consequents[0])


def variable_names(rule: Rule) -> dict[str, str]:
    aliases: list[str] = []
    for literal in (*rule.antecedents, rule.consequent):
        for argument in literal.arguments:
            lowered = argument.strip().lower()
            if is_variable(argument) and lowered not in aliases:
                aliases.append(lowered)
    names = ("x", "y", "z", "w", "u", "v")
    if len(aliases) > len(names):
        raise ValueError(f"Too many variables in rule: {rule}")
    return dict(zip(aliases, names))


def argument_to_fol(argument: str, variables: dict[str, str]) -> str:
    lowered = argument.strip().lower()
    if lowered in variables:
        return variables[lowered]
    return sanitize_identifier(argument, fallback="entity")


def literal_to_fol(literal: Literal, variables: dict[str, str] | None = None) -> str:
    """Serialize a literal using the prefix predicate syntax used by the parser."""

    variables = variables or {}
    arguments = " ".join(argument_to_fol(arg, variables) for arg in literal.arguments)
    atom = f"{literal.predicate} {arguments}"
    return f"¬({atom})" if literal.negative else atom


def rule_to_fol(rule: Rule) -> str:
    variables = variable_names(rule)
    left_parts = [literal_to_fol(literal, variables) for literal in rule.antecedents]
    left = left_parts[0] if len(left_parts) == 1 else f"({' ∧ '.join(left_parts)})"
    body = f"({left} → {literal_to_fol(rule.consequent, variables)})"
    for variable in reversed(tuple(variables.values())):
        body = f"∀{variable} {body}"
    return body


def ground_literal(literal: Literal, assignment: dict[str, str]) -> Literal:
    arguments = tuple(
        assignment.get(argument.strip().lower(), argument.strip())
        if is_variable(argument)
        else argument.strip()
        for argument in literal.arguments
    )
    return Literal(literal.predicate, arguments, literal.negative)


def collect_constants(facts: Sequence[Literal], rules: Sequence[Rule]) -> tuple[str, ...]:
    constants: set[str] = set()
    for literal in facts:
        constants.update(arg for arg in literal.arguments if not is_variable(arg))
    for rule in rules:
        for literal in (*rule.antecedents, rule.consequent):
            constants.update(arg for arg in literal.arguments if not is_variable(arg))
    return tuple(sorted(constants))


def forward_closure(facts: Sequence[Literal], rules: Sequence[Rule]) -> set[Literal]:
    closure = set(facts)
    constants = collect_constants(facts, rules)
    changed = True
    while changed:
        changed = False
        for rule in rules:
            aliases = tuple(variable_names(rule))
            assignments = (
                (dict(zip(aliases, values)) for values in itertools.product(constants, repeat=len(aliases)))
                if aliases
                else ({},)
            )
            for assignment in assignments:
                antecedents = [ground_literal(item, assignment) for item in rule.antecedents]
                if all(item in closure for item in antecedents):
                    conclusion = ground_literal(rule.consequent, assignment)
                    if conclusion not in closure:
                        closure.add(conclusion)
                        changed = True
    return closure


def source_answer(answer: Any) -> str:
    if answer is True:
        return "A"
    if answer is False:
        return "B"
    if isinstance(answer, str) and answer.strip().lower() == "unknown":
        return "C"
    raise ValueError(f"Unsupported source answer: {answer!r}")


def closure_answer(query: Literal, closure: set[Literal]) -> tuple[str | None, str]:
    positive = query in closure
    negative = query.opposite() in closure
    if positive and negative:
        return None, "conflict"
    if positive:
        return "A", "query_provable"
    if negative:
        return "B", "negation_provable"
    return "C", "neither_provable"


def difficulty_band(value: Any, *, easy_max: int, medium_max: int) -> str:
    number = int(value)
    if number <= easy_max:
        return "easy"
    if number <= medium_max:
        return "medium"
    return "hard"


def joint_difficulty(question: dict[str, Any]) -> str:
    """Use the harder of the QLen and QDep bands."""

    qlen = difficulty_band(question["QLen"], easy_max=2, medium_max=5)
    qdep = difficulty_band(question["QDep"], easy_max=1, medium_max=3)
    return max((qlen, qdep), key=DIFFICULTY_RANK.__getitem__)


def build_nl2fol(
    triples: dict[str, Any], rules: dict[str, Any]
) -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    context: list[str] = []

    def add(source_id: str, text: str, formula: str) -> None:
        key = text.strip()
        if key in mapping:
            key = f"{key} [{source_id}]"
        mapping[key] = formula
        context.append(text.strip())

    for source_id, item in sorted(triples.items(), key=lambda pair: natural_key(pair[0])):
        add(source_id, item["text"], literal_to_fol(parse_fact_representation(item["representation"])))
    for source_id, item in sorted(rules.items(), key=lambda pair: natural_key(pair[0])):
        add(source_id, item["text"], rule_to_fol(parse_rule_representation(item["representation"])))
    return mapping, context


def proof_reasoning(
    question: dict[str, Any],
    answer: str,
    triples: dict[str, Any],
) -> tuple[str, list[str]]:
    """Extract one source proof chain and verify that its last item proves the label."""

    proof_items = question.get("proofsWithIntermediates") or []
    if not proof_items:
        raise ValueError("A/B example has no proofsWithIntermediates")

    target = parse_fact_representation(question["representation"])
    if answer == "B":
        target = target.opposite()

    proof = proof_items[0]
    intermediates = proof.get("intermediates") or {}
    steps: list[tuple[str, Literal]] = []
    if isinstance(intermediates, dict) and intermediates:
        # In ProofWriter, intN is the earliest derived item and int1 is the goal.
        for _, item in sorted(
            intermediates.items(), key=lambda pair: natural_key(pair[0]), reverse=True
        ):
            steps.append(
                (item["text"].strip(), parse_fact_representation(item["representation"]))
            )
    else:
        # Direct facts have an empty intermediate map and name the source triple.
        proof_text = str(proof.get("representation") or question.get("proofs") or "")
        seen: set[str] = set()
        for triple_id in re.findall(r"triple\d+", proof_text):
            if triple_id in seen:
                continue
            seen.add(triple_id)
            item = triples[triple_id]
            steps.append(
                (item["text"].strip(), parse_fact_representation(item["representation"]))
            )

    if not steps:
        raise ValueError("Could not extract a proof chain")
    if steps[-1][1] != target:
        raise ValueError(
            "Extracted proof does not end in the gold literal: "
            f"expected {target}, got {steps[-1][1]}"
        )

    deduplicated: list[tuple[str, Literal]] = []
    seen_text: set[str] = set()
    for text, literal in steps:
        if text and text not in seen_text:
            seen_text.add(text)
            deduplicated.append((text, literal))
    formulas = [literal_to_fol(literal) for _, literal in deduplicated]
    reasoning = "\n".join(f"Conclusion: {text}" for text, _ in deduplicated)
    return reasoning, formulas


def source_files(root: Path, split: str) -> list[tuple[str, Path]]:
    files = [(depth, root / depth / f"meta-{split}.jsonl") for depth in DEPTH_PARTITIONS]
    missing = [str(path) for _, path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ProofWriter source files: " + ", ".join(missing))
    return files


def distribution(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        counts[f"answer_{row['answer']}"] += 1
        counts[f"difficulty_{row['difficulty']}"] += 1
    counts["records"] = len(rows)
    return dict(sorted(counts.items()))


def balanced_sample(
    rows: Sequence[dict[str, Any]], size: int, *, seed: int
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((row["answer"], row["difficulty"]), []).append(row)
    keys = [(answer, difficulty) for answer in ("A", "B") for difficulty in ("easy", "medium", "hard")]
    if size % len(keys):
        raise ValueError(f"Sample size must be divisible by {len(keys)}")
    per_bucket = size // len(keys)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in keys:
        candidates = list(buckets.get(key, []))
        if len(candidates) < per_bucket:
            raise ValueError(f"Not enough rows in bucket {key}: {len(candidates)}")
        selected.extend(rng.sample(candidates, per_bucket))
    rng.shuffle(selected)
    return selected


def convert(
    root: Path,
    *,
    split: str,
    theory_status: Callable[[dict[str, str]], str],
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    sources: list[dict[str, str]] = []

    for depth, path in source_files(root, split):
        sources.append({"path": f"{depth}/{path.name}", "sha256": file_sha256(path)})
        for theory in read_jsonl(path):
            counts["source_theories"] += 1
            triples = theory["triples"]
            rules_source = theory["rules"]
            facts = [
                parse_fact_representation(item["representation"])
                for _, item in sorted(triples.items(), key=lambda pair: natural_key(pair[0]))
            ]
            rules = [
                parse_rule_representation(item["representation"])
                for _, item in sorted(rules_source.items(), key=lambda pair: natural_key(pair[0]))
            ]
            nl2fol, context = build_nl2fol(triples, rules_source)
            closure = forward_closure(facts, rules)
            premise_status = theory_status(nl2fol)
            counts[f"z3_theory_{premise_status}"] += 1

            for question_id, question in sorted(
                theory["questions"].items(), key=lambda pair: natural_key(pair[0])
            ):
                counts["source_questions"] += 1
                source_label = source_answer(question["answer"])
                if source_label == "C":
                    counts["excluded_unknown"] += 1
                    continue
                if premise_status != "sat":
                    counts[f"excluded_{premise_status}_theory_questions"] += 1
                    counts[f"excluded_{premise_status}_answer_{source_label}"] += 1
                    continue
                query = parse_fact_representation(question["representation"])
                derived_label, status = closure_answer(query, closure)
                if derived_label != source_label:
                    counts[f"excluded_label_mismatch_{status}"] += 1
                    continue

                reasoning, reasoning_fol = proof_reasoning(question, source_label, triples)

                record_id = sanitize_identifier(
                    f"proofwriter_owa_{depth}_{theory['id']}_{question_id}",
                    fallback="problem",
                )
                record = {
                    "id": record_id,
                    "dataset": "proofwriter",
                    "split": split,
                    "context": " ".join(context),
                    "question": str(question["question"]).strip(),
                    "options": BINARY_OPTIONS.copy(),
                    "answer": source_label,
                    "difficulty": joint_difficulty(question),
                    "nl2fol": nl2fol,
                    "conclusion_fol": literal_to_fol(query),
                    "reasoning": reasoning,
                    "evaluation_metadata": {
                        "dataset": "ProofWriter",
                        "version": "V2020.12.3",
                        "world_assumption": "OWA",
                        "split": split,
                        "depth_partition": depth,
                        "theory_id": str(theory["id"]),
                        "question_id": str(question_id),
                        "QLen": int(question["QLen"]),
                        "QDep": int(question["QDep"]),
                        "proof_source": "proofsWithIntermediates",
                        "reference_reasoning_fol": reasoning_fol,
                    },
                }
                records.append(record)
                counts[f"answer_{source_label}"] += 1
                counts[f"difficulty_{record['difficulty']}"] += 1

    if any(key.startswith("excluded_label_mismatch") for key in counts):
        raise ValueError(f"Source/closure label mismatch detected: {dict(counts)}")
    return records, counts, sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--pilot-size", default=300, type=int)
    parser.add_argument("--benchmark-size", default=1500, type=int)
    parser.add_argument("--seed", default=20260821, type=int)
    parser.add_argument("--write-full", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--z3-timeout-ms", default=5000, type=int)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo_root.resolve() / "Training"))
    from recipe.formally_verifiable.common.fol_converter import FOLToZ3Converter
    from z3 import Solver

    converter = FOLToZ3Converter()

    def theory_status(nl2fol: dict[str, str]) -> str:
        try:
            premises = [converter.convert(formula) for formula in nl2fol.values()]
            solver = Solver()
            solver.set("timeout", args.z3_timeout_ms)
            solver.add(*premises)
            return str(solver.check())
        except Exception:
            return "parse_error"

    records, counts, sources = convert(
        args.source_root, split=args.split, theory_status=theory_status
    )
    pilot = balanced_sample(records, args.pilot_size, seed=args.seed)
    benchmark = balanced_sample(records, args.benchmark_size, seed=args.seed + 1)

    outputs: dict[str, Any] = {}
    for name, rows in (
        ("pilot_300.jsonl", pilot),
        ("benchmark_1500.jsonl", benchmark),
    ):
        count, digest = write_jsonl(args.output_dir / name, rows)
        outputs[name] = {
            "records": count,
            "sha256": digest,
            "checked_in": True,
            "distribution": distribution(rows),
        }
    if args.write_full:
        count, digest = write_jsonl(args.output_dir / "full.jsonl", records)
        outputs["full.jsonl"] = {
            "records": count,
            "sha256": digest,
            "checked_in": False,
            "distribution": distribution(records),
        }

    manifest = {
        "format_version": "proofwriter_owa_binary_v1",
        "source": {
            "dataset": "ProofWriter",
            "version": "V2020.12.3",
            "world_assumption": "OWA",
            "split": args.split,
            "files": sources,
        },
        "policy": {
            "unknown": "exclude all source Unknown/C rows; never relabel them",
            "options": BINARY_OPTIONS,
            "labels": {"A": "query derivable", "B": "explicit negation derivable"},
            "fol_serialization": "project-native prefix predicate syntax",
            "premises": "exclude theories that are not SAT in the project Z3 parser",
            "reasoning": "first official proofsWithIntermediates chain",
            "difficulty": {
                "QLen": "easy <= 2; medium 3-5; hard >= 6",
                "QDep": "easy <= 1; medium 2-3; hard >= 4",
                "combination": "take the harder of the QLen and QDep bands",
            },
        },
        "seed": args.seed,
        "conversion_counts": dict(sorted(counts.items())),
        "full_records": len(records),
        "full_sha256": rows_sha256(records),
        "full_distribution": distribution(records),
        "outputs": outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
