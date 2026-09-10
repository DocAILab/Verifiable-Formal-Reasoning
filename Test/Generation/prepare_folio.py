"""Build a conservative, label-verified FOLIO A/B subset without changing raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "Training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import z3

from recipe.formally_verifiable.common.fol_converter import (
    AppNode, BinOpNode, ConstNode, FOLParser, FOLToZ3Converter, FOLToken,
    NotNode, QuantNode, VarNode, collect_bound_vars, tokenize_fol,
)

PREPARATION_VERSION = "folio_verified_ab_v1"
OPTIONS = ["A) True", "B) False", "C) Uncertain"]


class Excluded(ValueError):
    def __init__(self, reason: str, detail: str):
        self.reason = reason
        super().__init__(detail)


def normalize_formula(text: str, signatures: dict[str, int] | None = None) -> str:
    """Normalize atomic applications only; never repair operators or parentheses."""
    signatures = signatures if signatures is not None else {}
    try:
        tokens = tokenize_fol(text)
        normalized = []
        index = 0
        while index < len(tokens):
            kind, name = tokens[index]
            # P(a, b) becomes (P a b), but the binder in forall x (...) is not P.
            is_binder = index > 0 and tokens[index - 1][0] in (
                FOLToken.FORALL, FOLToken.EXISTS,
            )
            if kind == FOLToken.IDENT and not is_binder and tokens[index + 1:index + 2] == [(FOLToken.LPAREN, "(")]:
                cursor = index + 2
                arguments = []
                while cursor < len(tokens) and tokens[cursor][0] == FOLToken.IDENT:
                    arguments.append(tokens[cursor])
                    cursor += 1
                    if cursor < len(tokens) and tokens[cursor][0] == FOLToken.COMMA:
                        cursor += 1
                        if cursor >= len(tokens) or tokens[cursor][0] != FOLToken.IDENT:
                            raise ValueError("Missing predicate argument after comma")
                    else:
                        break
                if arguments and tokens[cursor:cursor + 1] == [(FOLToken.RPAREN, ")")]:
                    normalized.extend([(FOLToken.LPAREN, "("), tokens[index], *arguments, (FOLToken.RPAREN, ")")])
                    index = cursor + 1
                    continue
            normalized.append(tokens[index])
            index += 1

        # Isolate each explicitly grouped quantifier body before using the shared
        # prefix parser, whose quantifier otherwise consumes the rest of an expression.
        for index in range(len(normalized) - 1, -1, -1):
            if normalized[index][0] not in (FOLToken.FORALL, FOLToken.EXISTS):
                continue
            cursor = index + 1
            if cursor >= len(normalized) or normalized[cursor][0] != FOLToken.IDENT:
                raise ValueError("Missing quantifier variable")
            cursor += 1
            if cursor < len(normalized) and normalized[cursor][0] == FOLToken.COMMA:
                cursor += 1
            if cursor >= len(normalized) or normalized[cursor][0] != FOLToken.LPAREN:
                raise ValueError("Quantifier body must be explicitly grouped")
            depth = 0
            for end in range(cursor, len(normalized)):
                depth += int(normalized[end][0] == FOLToken.LPAREN)
                depth -= int(normalized[end][0] == FOLToken.RPAREN)
                if depth == 0:
                    normalized[index:end + 1] = [(FOLToken.LPAREN, "("), *normalized[index:end + 1], (FOLToken.RPAREN, ")")]
                    break
            else:
                raise ValueError("Unclosed quantifier body")

        # The shared parser's implication associativity is not a source-data policy.
        # Require explicit grouping for chains instead of guessing their meaning.
        arrows = [0]
        for kind, _ in normalized:
            if kind == FOLToken.LPAREN:
                arrows.append(0)
            elif kind == FOLToken.RPAREN:
                if len(arrows) == 1:
                    raise ValueError("Unbalanced closing parenthesis")
                arrows.pop()
            elif kind == FOLToken.ARROW:
                arrows[-1] += 1
                if arrows[-1] > 1:
                    raise ValueError("Unparenthesized implication chain is unsupported")
        parser = FOLParser(normalized, collect_bound_vars(normalized))
        ast = parser.parse_expr()
        if parser.peek()[0] != "EOF":
            raise ValueError(f"Trailing tokens: {parser.tokens[parser.pos:]}")

        def render(node, bound: frozenset[str] = frozenset()) -> str:
            if isinstance(node, AppNode):
                arity = len(node.args)
                if node.func in signatures and signatures[node.func] != arity:
                    raise ValueError(f"Predicate arity conflict: {node.func}")
                signatures[node.func] = arity
                terms = []
                for arg in node.args:
                    if not isinstance(arg, (ConstNode, VarNode)):
                        raise ValueError("Function terms / formula-valued arguments are unsupported")
                    if isinstance(arg, VarNode) and arg.name not in bound:
                        raise ValueError(f"Unbound variable: {arg.name}")
                    terms.append(arg.name)
                return "(" + " ".join([node.func, *terms]) + ")"
            if isinstance(node, NotNode):
                return f"(\u00ac{render(node.body, bound)})"
            if isinstance(node, BinOpNode):
                op = {"and": "\u2227", "or": "\u2228", "xor": "\u2295", "implies": "\u2192"}[node.op]
                return f"({render(node.left, bound)} {op} {render(node.right, bound)})"
            if isinstance(node, QuantNode):
                if node.var in bound:
                    raise ValueError(f"Shadowed quantifier variable is unsupported: {node.var}")
                op = "\u2200" if node.quant == "forall" else "\u2203"
                return f"({op}{node.var}, {render(node.body, bound | {node.var})})"
            raise ValueError("A formula must be Boolean, not a bare term")

        result = render(ast)
        if not z3.is_bool(FOLToZ3Converter().convert(result)):
            raise ValueError("Non-Boolean formula")
        return result
    except (ValueError, KeyError, z3.Z3Exception) as exc:
        raise Excluded("unsupported_or_malformed_formula", str(exc)) from exc


def check_label(premises: list[str], target: str, answer: str, timeout_ms: int) -> None:
    converter = FOLToZ3Converter()
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(*[converter.convert(value) for value in premises])
    status = solver.check()
    if status == z3.unknown:
        raise Excluded("solver_unknown", f"Premise consistency: {solver.reason_unknown()}")
    if status == z3.unsat:
        raise Excluded("inconsistent_premises", "Premises are UNSAT; explosion is not a valid label check")
    conclusion = converter.convert(target)
    solver.add(z3.Not(conclusion) if answer == "A" else conclusion)
    status = solver.check()
    if status == z3.unknown:
        raise Excluded("solver_unknown", f"Label entailment: {solver.reason_unknown()}")
    if status != z3.unsat:
        raise Excluded("label_not_entailed", f"Consistent premises do not entail gold label {answer}")


def prepare_problem(raw: dict[str, Any], *, split: str, timeout_ms: int = 5000) -> dict[str, Any]:
    answer = str(raw.get("answer", "")).strip().upper()
    if answer == "C":
        raise Excluded("uncertain", "Gold label C excluded; three answer options are retained")
    if answer not in {"A", "B"}:
        raise Excluded("source_schema", f"Unexpected answer: {answer!r}")
    context = raw.get("context")
    question = raw.get("question")
    mapping = raw.get("nl2fol")
    target = raw.get("conclusion_fol")
    if not all(isinstance(value, str) and value.strip() for value in (context, question, target)) or not isinstance(mapping, dict):
        raise Excluded("source_schema", "Missing context, question, conclusion_fol or nl2fol")
    if raw.get("id") is None or raw.get("options") != OPTIONS:
        raise Excluded("source_schema", "Missing id or unexpected answer-option mapping")
    if context not in mapping or not isinstance(mapping[context], str):
        raise Excluded("source_schema", "Cannot identify the context-only premise block")
    if set(mapping) - {context, question}:
        raise Excluded("source_schema", "Unexpected nl2fol keys; cannot safely infer premise mapping")
    if question in mapping and context != question and (
        not isinstance(mapping[question], str) or mapping[question].strip() != target.strip()
    ):
        raise Excluded("source_schema", "Question mapping and conclusion_fol disagree")
    nl_premises = [line.strip() for line in context.splitlines() if line.strip()]
    fol_premises = [line.strip() for line in mapping[context].splitlines() if line.strip()]
    if not nl_premises or len(nl_premises) != len(fol_premises):
        raise Excluded("premise_alignment", f"NL/FOL premise counts: {len(nl_premises)}/{len(fol_premises)}")
    if len(set(nl_premises)) != len(nl_premises):
        raise Excluded("premise_alignment", "Duplicate NL keys cannot preserve one-to-one mapping")
    signatures: dict[str, int] = {}
    normalized = []
    for index, formula in enumerate([*fol_premises, target]):
        try:
            normalized.append(normalize_formula(formula, signatures))
        except Excluded as exc:
            location = f"h{index + 1}" if index < len(fol_premises) else "goal"
            raise Excluded(exc.reason, f"{location}: {exc}; formula={formula}") from exc
    check_label(normalized[:-1], normalized[-1], answer, timeout_ms)
    result = {
        "id": str(raw["id"]), "dataset": "folio", "split": split,
        "context": context, "question": question, "options": OPTIONS.copy(),
        "answer": answer, "nl2fol": dict(zip(nl_premises, normalized[:-1])),
        "conclusion_fol": normalized[-1],
        "evaluation_metadata": {
            "preparation_version": PREPARATION_VERSION,
            "subset": "AB_supported_label_verified",
            "premises_satisfiable": True, "gold_label_entailed": True,
            "reference_proof_available": False, "difficulty_available": False,
            "has_existential": any("\u2203" in value for value in normalized),
            "rule_metric_scope": "training_closed_ontology_not_complete_for_FOLIO",
        },
    }
    return result


def validate_prepared_problem(problem: dict[str, Any]) -> None:
    metadata = problem.get("evaluation_metadata") or {}
    if metadata.get("preparation_version") != PREPARATION_VERSION:
        raise ValueError("Raw/unrecognized FOLIO conversion is unsafe. Run Test/Generation/prepare_folio.py first.")
    if problem.get("answer") not in {"A", "B"} or problem.get("options") != OPTIONS:
        raise ValueError("Prepared FOLIO evaluation requires A/B labels and the unchanged A/B/C options")
    if not metadata.get("premises_satisfiable") or not metadata.get("gold_label_entailed"):
        raise ValueError("FOLIO row has not passed premise consistency and label verification")
    mapping = problem.get("nl2fol")
    premises = [line.strip() for line in str(problem.get("context", "")).splitlines() if line.strip()]
    if not isinstance(mapping, dict) or not premises or list(mapping) != premises:
        raise ValueError("FOLIO nl2fol must contain exactly the ordered context premises, not the query")
    signatures: dict[str, int] = {}
    for formula in [*mapping.values(), problem.get("conclusion_fol")]:
        if not isinstance(formula, str) or "\n" in formula or normalize_formula(formula, signatures) != formula:
            raise ValueError("FOLIO formulas must be individually normalized by prepare_folio.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, default=ROOT / "Data/FOLIO")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "Data/FOLIO/verified_ab_v1")
    parser.add_argument("--splits", nargs="+", choices=["train", "validation"], default=["train", "validation"])
    parser.add_argument("--z3_timeout_ms", type=int, default=5000)
    args = parser.parse_args()
    if args.z3_timeout_ms <= 0:
        parser.error("--z3_timeout_ms must be positive")
    from Test.Generation.metrics import write_json, write_jsonl

    manifest = {"preparation_version": PREPARATION_VERSION, "z3_version": z3.get_version_string(),
                "z3_timeout_ms": args.z3_timeout_ms, "evaluation_split": "validation",
                "notes": ["Curated supported A/B subset, not the full official FOLIO benchmark.",
                          "No label changes, inferred proofs, guessed syntax repairs or train/validation merge.",
                          "Z3 checks formula-label consistency, not NL-to-FOL translation fidelity.",
                          "RGD is unavailable; rule scores use the unchanged, incomplete training ontology."],
                "splits": {}}
    for split in args.splits:
        source = args.source_dir / f"converted_{split}.jsonl"
        accepted, rejected, labels, seen = [], [], Counter(), set()
        with source.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                identity = str(raw.get("id"))
                if identity in seen:
                    raise ValueError(f"Duplicate id {identity} at {source}:{line_number}")
                seen.add(identity)
                labels[str(raw.get("answer"))] += 1
                try:
                    row = prepare_problem(raw, split=split, timeout_ms=args.z3_timeout_ms)
                    row["evaluation_metadata"].update(source_file=source.name, source_line=line_number)
                    validate_prepared_problem(row)
                    accepted.append(row)
                except Excluded as exc:
                    rejected.append({"id": identity, "source_line": line_number, "answer": raw.get("answer"),
                                     "reason": exc.reason, "detail": str(exc)})
        destination = args.output_dir / f"{split}.jsonl"
        write_jsonl(destination, accepted)
        write_jsonl(args.output_dir / f"excluded_{split}.jsonl", rejected)
        manifest["splits"][split] = {
            "source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output": destination.name, "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "source_count": len(seen), "source_labels": dict(labels), "accepted_count": len(accepted),
            "accepted_labels": dict(Counter(row["answer"] for row in accepted)),
            "excluded_counts": dict(Counter(row["reason"] for row in rejected)),
            "existential_problem_count": sum(row["evaluation_metadata"]["has_existential"] for row in accepted),
        }
        print(json.dumps({split: manifest["splits"][split]}, ensure_ascii=True), flush=True)
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
