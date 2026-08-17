"""Offline metrics for single-shot structured generation."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from recipe.formally_verifiable.rule_grounded_process_rl.structured_parser import (
    match_answer_to_option,
    parse_model_output,
)
from recipe.formally_verifiable.rule_grounded_process_rl.structured_prompt import fol_infix_to_prefix
from recipe.formally_verifiable.rule_grounded_process_rl.rule_checker import RuleChecker
from recipe.formally_verifiable.rule_grounded_process_rl.reward import (
    dependency_closure,
    find_final_answer_step_index,
)
from recipe.formally_verifiable.common.verifier.z3_verifier import Z3Verifier


def load_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_premises_fol(problem: dict[str, Any]) -> dict[str, str]:
    return {
        f"h{index}": str(fol).strip()
        for index, fol in enumerate((problem.get("nl2fol") or {}).values(), start=1)
    }


def build_answer_options(problem: dict[str, Any]) -> list[dict[str, str]]:
    """Build formal answer-option bindings for GOAL_BINDING rule checks."""
    target = fol_infix_to_prefix(str(problem.get("conclusion_fol", "")).strip())
    if not target:
        return []

    answer_options: list[dict[str, str]] = []
    for option in problem.get("options", []):
        if ")" not in str(option):
            continue
        letter, text = str(option).split(")", 1)
        option_text = text.strip().lower()
        if "true" in option_text:
            answer_id = "h_goal_true"
            formal = target
        elif "false" in option_text:
            answer_id = "h_goal_false"
            formal = f"\u00ac({target})"
        elif "uncertain" in option_text:
            answer_id = "h_goal_uncertain"
            formal = target
        else:
            continue
        answer_options.append(
            {
                "letter": letter.strip().upper(),
                "text": option_text,
                "answer_id": answer_id,
                "formal": formal,
            }
        )
    return answer_options


def validate_step_schema(step: Any) -> dict[str, Any]:
    required_fields = {"id", "dependencies", "conclusion", "rule"}
    if not isinstance(step, dict):
        return {"valid": False, "error": "step is not a JSON object"}
    missing = sorted(required_fields - set(step))
    if missing:
        return {"valid": False, "error": f"missing fields: {missing}"}
    if not isinstance(step.get("id"), str) or not step.get("id", "").strip():
        return {"valid": False, "error": "id must be a non-empty string"}
    if not isinstance(step.get("dependencies"), list):
        return {"valid": False, "error": "dependencies must be a list"}
    if not all(isinstance(dep, str) and dep.strip() for dep in step["dependencies"]):
        return {"valid": False, "error": "dependencies must contain non-empty strings"}
    if len(step["dependencies"]) >= 5:
        return {"valid": False, "error": "too many dependencies (>=5)"}
    if not isinstance(step.get("conclusion"), str) or not step["conclusion"].strip():
        return {"valid": False, "error": "conclusion must be a non-empty string"}
    if not isinstance(step.get("rule"), str) or not step["rule"].strip():
        return {"valid": False, "error": "rule must be a non-empty string"}
    return {"valid": True, "error": None}


def verifier_safe_summary(summary: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace schema-invalid steps before calling verifiers so evaluation never crashes."""
    safe_summary: list[dict[str, Any]] = []
    schemas: list[dict[str, Any]] = []
    for index, step in enumerate(summary):
        schema = validate_step_schema(step)
        schemas.append(schema)
        if schema["valid"] and isinstance(step, dict):
            safe_summary.append(step)
            continue
        safe_summary.append(
            {
                "id": f"__invalid_step_{index}",
                "dependencies": [],
                "conclusion": "",
                "rule": "INVALID_RULE",
            }
        )
    return safe_summary, schemas


def reference_step_count(problem: dict[str, Any]) -> int:
    reference = problem.get("canonical_proof_reference")
    if isinstance(reference, dict):
        length = reference.get("min_proof_length")
        if isinstance(length, int) and length >= 0:
            return length

    proof_lengths = []
    for proof in problem.get("canonical_proofs") or []:
        if not isinstance(proof, dict):
            continue
        length = proof.get("proof_length")
        if isinstance(length, int) and length >= 0:
            proof_lengths.append(length)
    if proof_lengths:
        return min(proof_lengths)

    reasoning = str(problem.get("reasoning", "") or "")
    conclusion_steps = re.findall(r"(?im)^\s*conclusion\s*:", reasoning)
    return max(1, len(conclusion_steps) if conclusion_steps else 1)


def generated_proof_step_count(summary: list[Any]) -> int:
    final_index = find_final_answer_step_index(summary)
    closure = dependency_closure(summary, final_index)
    return sum(1 for index in closure if index != final_index)


def is_missing_dependency_error(error: Any) -> bool:
    return str(error or "").strip().lower().startswith("missing dependencies")


def var_without_cascade(
    metrics: dict[str, Any],
    *,
    verified_key: str = "verified",
    cascade_key: str = "cascade_failure",
) -> dict[str, Any]:
    steps = metrics.get("step_metrics") or []
    eligible = [
        step
        for step in steps
        if isinstance(step, dict) and not bool(step.get(cascade_key))
    ]
    if not eligible:
        return {"value": 0.0, "numerator": 0, "denominator": 0}
    numerator = sum(1 for step in eligible if step.get(verified_key))
    denominator = len(eligible)
    return {
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def granularity_error(metrics: dict[str, Any], problem: dict[str, Any]) -> float:
    generated_steps = max(1, int(metrics.get("generated_proof_step_count", 0) or 0))
    reference_steps = max(1, reference_step_count(problem))
    return abs(math.log(generated_steps / reference_steps))


def evaluate_response(
    response_text: str,
    problem: dict[str, Any],
    *,
    verifier: Z3Verifier | None = None,
    rule_checker: RuleChecker | None = None,
) -> dict[str, Any]:
    verifier = verifier or Z3Verifier()
    rule_checker = rule_checker or RuleChecker(timeout_ms=getattr(verifier, "timeout_ms", 5000))
    parsed = parse_model_output(response_text)
    result: dict[str, Any] = {
        "problem_id": problem.get("id"),
        "format_correct": False,
        "strict_summary": bool(parsed.get("summary_tag_present")),
        "parse_error": parsed.get("parse_error"),
        "answer_correct": False,
        "parsed_answer": None,
        "final_answer_id": None,
        "total_steps": 0,
        "generated_proof_step_count": 0,
        "verified_count": 0,
        "verified_step_fraction": 0.0,
        "all_steps_verified": False,
        "z3_positive_response": False,
        "fully_correct": False,
        "rule_recognized_count": 0,
        "rule_application_valid_count": 0,
        "rule_grounded_verified_count": 0,
        "rule_recognized_fraction": 0.0,
        "rule_application_valid_fraction": 0.0,
        "rule_grounded_step_fraction": 0.0,
        "rule_grounded_positive_response": False,
        "all_rules_recognized": False,
        "all_rules_valid": False,
        "all_steps_rule_grounded": False,
        "fully_correct_rule_grounded": False,
        "semantic_verified_but_rule_invalid_count": 0,
        "semantic_verified_but_rule_invalid_fraction": 0.0,
        "first_failure_index": None,
        "first_failure_position": None,
        "first_rule_failure_index": None,
        "first_rule_failure_position": None,
        "cascade_failure_count": 0,
        "root_failure_count": 0,
        "rule_cascade_failure_count": 0,
        "rule_root_failure_count": 0,
        "step_metrics": [],
    }
    if parsed.get("parse_error"):
        return result

    summary = parsed.get("summary") or []
    if not summary:
        result["parse_error"] = "Empty summary"
        return result

    result["format_correct"] = True
    result["total_steps"] = len(summary)
    result["generated_proof_step_count"] = generated_proof_step_count(summary)
    last_step = summary[-1]
    result["final_answer_id"] = last_step.get("id")
    parsed_answer = match_answer_to_option(
        last_step.get("id", ""),
        problem,
        last_step.get("conclusion", ""),
    )
    answer_correct = parsed_answer == str(problem.get("answer", "")).strip().upper()
    result["parsed_answer"] = parsed_answer
    result["answer_correct"] = bool(answer_correct)

    premises_fol = build_premises_fol(problem)
    verifier_summary, schemas = verifier_safe_summary(summary)
    semantic_results = verifier.batch_verify(verifier_summary, premises_fol)
    rule_results = rule_checker.batch_check(
        verifier_summary,
        premises_fol,
        answer_options=build_answer_options(problem),
    )

    verified_count = sum(bool(item.get("verified")) for item in semantic_results)
    rule_recognized_count = sum(bool(item.get("rule_recognized")) for item in rule_results)
    rule_application_valid_count = sum(
        bool(item.get("rule_application_valid")) for item in rule_results
    )
    rule_grounded_count = sum(bool(item.get("verified")) for item in rule_results)
    semantic_verified_but_rule_invalid_count = 0
    first_failure_index: int | None = None
    first_rule_failure_index: int | None = None
    cascade_failure_count = 0
    root_failure_count = 0
    rule_cascade_failure_count = 0
    rule_root_failure_count = 0
    step_metrics: list[dict[str, Any]] = []

    for index, (step, semantic_result, rule_result) in enumerate(
        zip(summary, semantic_results, rule_results)
    ):
        schema = schemas[index]
        if not schema["valid"]:
            semantic_result = {
                "verified": False,
                "error": f"schema invalid: {schema['error']}",
            }
            rule_result = {
                "verified": False,
                "rule_recognized": False,
                "rule_application_valid": False,
                "claimed_rule": step.get("rule") if isinstance(step, dict) else None,
                "canonical_rule": "INVALID_RULE",
                "error": f"schema invalid: {schema['error']}",
            }
        semantic_verified = bool(semantic_result.get("verified"))
        rule_grounded_verified = bool(rule_result.get("verified"))
        semantic_error = semantic_result.get("error")
        rule_error = rule_result.get("error")
        cascade_failure = is_missing_dependency_error(semantic_error)
        rule_cascade_failure = is_missing_dependency_error(rule_error)

        if not semantic_verified and first_failure_index is None:
            first_failure_index = index
        if not rule_grounded_verified and first_rule_failure_index is None:
            first_rule_failure_index = index
        if not semantic_verified and cascade_failure:
            cascade_failure_count += 1
        if not semantic_verified and not cascade_failure:
            root_failure_count += 1
        if not rule_grounded_verified and rule_cascade_failure:
            rule_cascade_failure_count += 1
        if not rule_grounded_verified and not rule_cascade_failure:
            rule_root_failure_count += 1
        if semantic_verified and not rule_grounded_verified:
            semantic_verified_but_rule_invalid_count += 1

        step_metrics.append(
            {
                "index": index,
                "id": step.get("id") if isinstance(step, dict) else None,
                "schema_valid": bool(schema["valid"]),
                "schema_error": schema["error"],
                "verified": semantic_verified,
                "semantic_verified": semantic_verified,
                "cascade_failure": cascade_failure,
                "semantic_verification_error": semantic_error,
                "verification_error": semantic_error,
                "rule_recognized": bool(rule_result.get("rule_recognized")),
                "rule_application_valid": bool(rule_result.get("rule_application_valid")),
                "rule_grounded_verified": rule_grounded_verified,
                "claimed_rule": rule_result.get("claimed_rule"),
                "canonical_rule": rule_result.get("canonical_rule"),
                "rule_cascade_failure": rule_cascade_failure,
                "rule_error": rule_error,
                "rule_details": rule_result.get("details") or {},
            }
        )

    total = len(semantic_results)
    all_steps_verified = bool(total) and verified_count == total
    all_rules_recognized = bool(total) and rule_recognized_count == total
    all_rules_valid = bool(total) and rule_application_valid_count == total
    all_steps_rule_grounded = bool(total) and rule_grounded_count == total
    result.update(
        {
            "verified_count": verified_count,
            "verified_step_fraction": round(verified_count / total, 6) if total else 0.0,
            "all_steps_verified": all_steps_verified,
            "z3_positive_response": verified_count > 0,
            "fully_correct": bool(answer_correct and all_steps_verified),
            "rule_recognized_count": rule_recognized_count,
            "rule_application_valid_count": rule_application_valid_count,
            "rule_grounded_verified_count": rule_grounded_count,
            "rule_recognized_fraction": round(rule_recognized_count / total, 6)
            if total
            else 0.0,
            "rule_application_valid_fraction": round(
                rule_application_valid_count / total, 6
            )
            if total
            else 0.0,
            "rule_grounded_step_fraction": round(rule_grounded_count / total, 6)
            if total
            else 0.0,
            "rule_grounded_positive_response": rule_grounded_count > 0,
            "all_rules_recognized": all_rules_recognized,
            "all_rules_valid": all_rules_valid,
            "all_steps_rule_grounded": all_steps_rule_grounded,
            "fully_correct_rule_grounded": bool(answer_correct and all_steps_rule_grounded),
            "semantic_verified_but_rule_invalid_count": semantic_verified_but_rule_invalid_count,
            "semantic_verified_but_rule_invalid_fraction": round(
                semantic_verified_but_rule_invalid_count / total, 6
            )
            if total
            else 0.0,
            "first_failure_index": first_failure_index,
            "first_failure_position": round((first_failure_index + 1) / total, 6)
            if first_failure_index is not None and total
            else None,
            "first_rule_failure_index": first_rule_failure_index,
            "first_rule_failure_position": round((first_rule_failure_index + 1) / total, 6)
            if first_rule_failure_index is not None and total
            else None,
            "cascade_failure_count": cascade_failure_count,
            "root_failure_count": root_failure_count,
            "rule_cascade_failure_count": rule_cascade_failure_count,
            "rule_root_failure_count": rule_root_failure_count,
            "step_metrics": step_metrics,
        }
    )
    return result


def attach_extended_metrics(record: dict[str, Any], problem: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") or {}
    semantic_var = var_without_cascade(metrics)
    rule_var = var_without_cascade(
        metrics,
        verified_key="rule_grounded_verified",
        cascade_key="rule_cascade_failure",
    )
    record["extended_metrics"] = {
        "reference_step_count": reference_step_count(problem),
        "generated_step_count": int(metrics.get("generated_proof_step_count", 0) or 0),
        "granularity_error": granularity_error(metrics, problem),
        "var_no_cascade": semantic_var["value"],
        "var_no_cascade_numerator": semantic_var["numerator"],
        "var_no_cascade_denominator": semantic_var["denominator"],
        "rule_var_no_cascade": rule_var["value"],
        "rule_var_no_cascade_numerator": rule_var["numerator"],
        "rule_var_no_cascade_denominator": rule_var["denominator"],
    }
    return record


def problem_group_key(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("difficulty") or "unknown").strip().lower(),
        str(record.get("problem_id")),
    )


def summarize_records(records: list[dict[str, Any]], *, k: int = 3) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[problem_group_key(record)].append(record)
    for items in groups.values():
        items.sort(key=lambda item: int(item.get("sample_index", 0) or 0))

    problem_count = len(groups)
    response_count = 0
    pass_hits = 0
    answer_hits = 0
    format_hits = 0
    all_verified_hits = 0
    z3_positive_hits = 0
    fully_correct_hits = 0
    rule_positive_hits = 0
    all_rules_recognized_hits = 0
    all_rules_valid_hits = 0
    all_rule_grounded_hits = 0
    fully_correct_rule_grounded_hits = 0
    total_steps = 0
    verified_steps = 0
    rule_recognized_steps = 0
    rule_valid_steps = 0
    rule_grounded_steps = 0
    semantic_verified_but_rule_invalid_steps = 0
    cascade_failures = 0
    root_failures = 0
    rule_cascade_failures = 0
    rule_root_failures = 0
    var_macro: list[float] = []
    var_numerator = 0
    var_denominator = 0
    rule_var_macro: list[float] = []
    rule_var_numerator = 0
    rule_var_denominator = 0
    granularity_values: list[float] = []

    for items in groups.values():
        topk = items[:k]
        if any((item.get("metrics") or {}).get("answer_correct") for item in topk):
            pass_hits += 1
        for item in topk:
            metrics = item.get("metrics") or {}
            extended = item.get("extended_metrics") or {}
            response_count += 1
            answer_hits += int(bool(metrics.get("answer_correct")))
            format_hits += int(bool(metrics.get("format_correct")))
            all_verified_hits += int(bool(metrics.get("all_steps_verified")))
            z3_positive_hits += int(bool(metrics.get("z3_positive_response")))
            fully_correct_hits += int(bool(metrics.get("fully_correct")))
            rule_positive_hits += int(bool(metrics.get("rule_grounded_positive_response")))
            all_rules_recognized_hits += int(bool(metrics.get("all_rules_recognized")))
            all_rules_valid_hits += int(bool(metrics.get("all_rules_valid")))
            all_rule_grounded_hits += int(bool(metrics.get("all_steps_rule_grounded")))
            fully_correct_rule_grounded_hits += int(
                bool(metrics.get("fully_correct_rule_grounded"))
            )
            total_steps += int(metrics.get("total_steps", 0) or 0)
            verified_steps += int(metrics.get("verified_count", 0) or 0)
            rule_recognized_steps += int(metrics.get("rule_recognized_count", 0) or 0)
            rule_valid_steps += int(metrics.get("rule_application_valid_count", 0) or 0)
            rule_grounded_steps += int(metrics.get("rule_grounded_verified_count", 0) or 0)
            semantic_verified_but_rule_invalid_steps += int(
                metrics.get("semantic_verified_but_rule_invalid_count", 0) or 0
            )
            cascade_failures += int(metrics.get("cascade_failure_count", 0) or 0)
            root_failures += int(metrics.get("root_failure_count", 0) or 0)
            rule_cascade_failures += int(metrics.get("rule_cascade_failure_count", 0) or 0)
            rule_root_failures += int(metrics.get("rule_root_failure_count", 0) or 0)
            var_macro.append(float(extended.get("var_no_cascade", 0.0) or 0.0))
            var_numerator += int(extended.get("var_no_cascade_numerator", 0) or 0)
            var_denominator += int(extended.get("var_no_cascade_denominator", 0) or 0)
            rule_var_macro.append(
                float(extended.get("rule_var_no_cascade", 0.0) or 0.0)
            )
            rule_var_numerator += int(
                extended.get("rule_var_no_cascade_numerator", 0) or 0
            )
            rule_var_denominator += int(
                extended.get("rule_var_no_cascade_denominator", 0) or 0
            )
            granularity_values.append(float(extended.get("granularity_error", 0.0) or 0.0))

    return {
        "num_problems": problem_count,
        "num_responses": response_count,
        "k": k,
        f"avg_at_{k}": answer_hits / max(1, response_count),
        f"pass_at_{k}": pass_hits / max(1, problem_count),
        "format_correct_rate": format_hits / max(1, response_count),
        "answer_accuracy": answer_hits / max(1, response_count),
        "all_steps_verified_rate": all_verified_hits / max(1, response_count),
        "z3_positive_response_rate": z3_positive_hits / max(1, response_count),
        "fully_correct_rate": fully_correct_hits / max(1, response_count),
        "verified_step_fraction_micro": verified_steps / total_steps if total_steps else 0.0,
        "rule_grounded_positive_response_rate": rule_positive_hits / max(1, response_count),
        "all_rules_recognized_rate": all_rules_recognized_hits / max(1, response_count),
        "all_rules_valid_rate": all_rules_valid_hits / max(1, response_count),
        "all_steps_rule_grounded_rate": all_rule_grounded_hits / max(1, response_count),
        "fully_correct_rule_grounded_rate": fully_correct_rule_grounded_hits
        / max(1, response_count),
        "rule_recognized_step_fraction_micro": rule_recognized_steps / total_steps
        if total_steps
        else 0.0,
        "rule_application_valid_step_fraction_micro": rule_valid_steps / total_steps
        if total_steps
        else 0.0,
        "rule_grounded_step_fraction_micro": rule_grounded_steps / total_steps
        if total_steps
        else 0.0,
        "semantic_verified_but_rule_invalid_fraction_micro": (
            semantic_verified_but_rule_invalid_steps / total_steps if total_steps else 0.0
        ),
        "avg_total_steps": total_steps / max(1, response_count),
        "avg_verified_steps": verified_steps / max(1, response_count),
        "avg_rule_grounded_steps": rule_grounded_steps / max(1, response_count),
        "avg_cascade_failures": cascade_failures / max(1, response_count),
        "avg_root_failures": root_failures / max(1, response_count),
        "avg_rule_cascade_failures": rule_cascade_failures / max(1, response_count),
        "avg_rule_root_failures": rule_root_failures / max(1, response_count),
        "var_no_cascade_macro": sum(var_macro) / max(1, len(var_macro)),
        "var_no_cascade_micro": var_numerator / var_denominator if var_denominator else 0.0,
        "var_no_cascade_numerator": var_numerator,
        "var_no_cascade_denominator": var_denominator,
        "rule_var_no_cascade_macro": sum(rule_var_macro)
        / max(1, len(rule_var_macro)),
        "rule_var_no_cascade_micro": rule_var_numerator / rule_var_denominator
        if rule_var_denominator
        else 0.0,
        "rule_var_no_cascade_numerator": rule_var_numerator,
        "rule_var_no_cascade_denominator": rule_var_denominator,
        "granularity_error_mean": sum(granularity_values)
        / max(1, len(granularity_values)),
    }


def summarize_by_difficulty(records: list[dict[str, Any]], *, k: int = 3) -> dict[str, Any]:
    result: dict[str, Any] = {}
    present = {
        str(record.get("difficulty") or "unknown").strip().lower() for record in records
    }
    difficulties = [name for name in ("easy", "medium", "hard") if name in present]
    difficulties.extend(sorted(present - set(difficulties)))
    for difficulty in difficulties:
        subset = [
            record
            for record in records
            if str(record.get("difficulty") or "unknown").strip().lower() == difficulty
        ]
        result[difficulty] = summarize_records(subset, k=k)
    return result


def write_difficulty_summary(output_dir: str | Path, summary: dict[str, Any]) -> None:
    """Write the canonical Overall/Easy/Medium/Hard evaluation view."""
    output = Path(output_dir)
    overall = summary.get("metrics") or {}
    by_difficulty = summary.get("by_difficulty") or {}
    k = int(overall.get("k", 3) or 3)
    metric_keys = (
        f"avg_at_{k}",
        f"pass_at_{k}",
        "format_correct_rate",
        "var_no_cascade_micro",
        "rule_var_no_cascade_micro",
        "rule_grounded_step_fraction_micro",
        "granularity_error_mean",
    )
    rows: list[dict[str, Any]] = []
    for label, metrics in (
        ("overall", overall),
        ("easy", by_difficulty.get("easy") or {}),
        ("medium", by_difficulty.get("medium") or {}),
        ("hard", by_difficulty.get("hard") or {}),
    ):
        rows.append(
            {
                "scope": label,
                "num_problems": int(metrics.get("num_problems", 0) or 0),
                "num_responses": int(metrics.get("num_responses", 0) or 0),
                **{key: float(metrics.get(key, 0.0) or 0.0) for key in metric_keys},
            }
        )
    write_json(
        output / "difficulty_summary.json",
        {"run_name": summary.get("run_name"), "input": summary.get("input"), "k": k, "rows": rows},
    )
    lines = [
        f"# {summary.get('run_name')} Difficulty Summary",
        "",
        "| Scope | Problems | Avg@{k} | AccPass@{k} | Format | VAR(no cascade) | Rule VAR(no cascade) | Rule-grounded steps | RGD |".format(k=k),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scope} | {num_problems} | {avg:.6f} | {passed:.6f} | {fmt:.6f} | "
            "{var:.6f} | {rule_var:.6f} | {rule_steps:.6f} | {rgd:.6f} |".format(
                scope=str(row["scope"]).title(), num_problems=row["num_problems"],
                avg=row[f"avg_at_{k}"], passed=row[f"pass_at_{k}"],
                fmt=row["format_correct_rate"], var=row["var_no_cascade_micro"],
                rule_var=row["rule_var_no_cascade_micro"],
                rule_steps=row["rule_grounded_step_fraction_micro"],
                rgd=row["granularity_error_mean"],
            )
        )
    (output / "difficulty_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: str | Path, summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics") or {}
    by_difficulty = summary.get("by_difficulty") or {}
    lines = [
        f"# {summary.get('run_name')} 测试报告",
        "",
        "## 协议",
        "",
        "- 推理方式：一次性完整生成 `<think>...</think><summary>[...]</summary>`。",
        "- Prompt：复用训练阶段的统一 structured generation prompt。",
        "- LoRA：可选；未传入 adapter 时直接评估 base model。",
        "- 评估：生成结束后离线运行 parser、Z3 semantic verifier、rule checker 和答案匹配器。",
        "- Rule correctness：`rule_grounded` 要求该 step 同时通过 Z3 语义验证，并且声明的推理规则应用正确。",
        "",
        "## 总体结果",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| `{key}` | {rendered} |")
    lines.extend(["", "## 分难度结果", ""])
    lines.append(
        "| 难度 | Answer | Pass | Format | All Z3 | All Rule | Rule VAR(no cascade) | Avg Steps |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    k = int(metrics.get("k", 3) or 3)
    for difficulty, item in by_difficulty.items():
        lines.append(
            "| {difficulty} | {answer:.6f} | {passed:.6f} | {fmt:.6f} | "
            "{all_verified:.6f} | {all_rule:.6f} | {rule_var:.6f} | {steps:.3f} |".format(
                difficulty=difficulty,
                answer=float(item.get(f"avg_at_{k}", 0.0) or 0.0),
                passed=float(item.get(f"pass_at_{k}", 0.0) or 0.0),
                fmt=float(item.get("format_correct_rate", 0.0) or 0.0),
                all_verified=float(item.get("all_steps_verified_rate", 0.0) or 0.0),
                all_rule=float(item.get("all_steps_rule_grounded_rate", 0.0) or 0.0),
                rule_var=float(item.get("rule_var_no_cascade_micro", 0.0) or 0.0),
                steps=float(item.get("avg_total_steps", 0.0) or 0.0),
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metrics_svg(path: str | Path, summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics") or {}
    bars = [
        ("Answer", float(metrics.get("answer_accuracy", 0.0) or 0.0)),
        ("Format", float(metrics.get("format_correct_rate", 0.0) or 0.0)),
        ("Z3+", float(metrics.get("z3_positive_response_rate", 0.0) or 0.0)),
        ("All Z3", float(metrics.get("all_steps_verified_rate", 0.0) or 0.0)),
        ("Rule", float(metrics.get("rule_grounded_step_fraction_micro", 0.0) or 0.0)),
        ("All Rule", float(metrics.get("all_steps_rule_grounded_rate", 0.0) or 0.0)),
        ("Full", float(metrics.get("fully_correct_rate", 0.0) or 0.0)),
        ("Rule Full", float(metrics.get("fully_correct_rule_grounded_rate", 0.0) or 0.0)),
        ("VAR", float(metrics.get("var_no_cascade_micro", 0.0) or 0.0)),
        ("Rule VAR", float(metrics.get("rule_var_no_cascade_micro", 0.0) or 0.0)),
    ]
    width = 1120
    height = 430
    margin_left = 70
    chart_top = 55
    chart_height = 270
    bar_width = 58
    gap = 42
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="70" y="32" font-family="Arial, sans-serif" font-size="22" font-weight="700">Generation Evaluation Metrics</text>',
        f'<line x1="{margin_left}" y1="{chart_top + chart_height}" x2="{width - 45}" y2="{chart_top + chart_height}" stroke="#222" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        value = tick / 5
        y = chart_top + chart_height - value * chart_height
        svg.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - 45}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="28" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#555">{value:.1f}</text>'
        )
    for index, (label, value) in enumerate(bars):
        x = margin_left + 28 + index * (bar_width + gap)
        bar_height = max(0.0, min(1.0, value)) * chart_height
        y = chart_top + chart_height - bar_height
        color = "#2563eb" if "Rule" not in label else "#7c3aed"
        svg.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="4" fill="{color}"/>'
        )
        svg.append(
            f'<text x="{x + bar_width / 2}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111">{value:.3f}</text>'
        )
        svg.append(
            f'<text x="{x + bar_width / 2}" y="{chart_top + chart_height + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#222">{label}</text>'
        )
    svg.append("</svg>")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg), encoding="utf-8")
