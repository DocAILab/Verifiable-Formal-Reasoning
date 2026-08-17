"""Build RuleChecker-aligned golden formal proofs for the ProverQA test set.

DeepSeek proposes structured proofs. A proposal is accepted only when the
project's own schema, Z3, RuleChecker, answer-binding, nontrivial-progress, and
dependency-closure checks all pass. The source test files are never modified.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import requests
from z3 import Not, Solver, is_bool, sat, unknown, unsat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Training"))

from recipe.formally_verifiable.common.verifier.z3_verifier import Z3Verifier  # noqa: E402
from recipe.formally_verifiable.rule_grounded_process_rl.reward import (  # noqa: E402
    RuleGroundedProcessRewardScorer,
    build_answer_options,
    build_premises_fol,
    dependency_closure,
    find_final_answer_step_index,
)
from recipe.formally_verifiable.rule_grounded_process_rl.rule_ontology import (  # noqa: E402
    GOAL_BINDING,
    RULE_PROMPT_GUIDANCE,
    canonicalize_rule,
)
from recipe.formally_verifiable.rule_grounded_process_rl.structured_prompt import (  # noqa: E402
    fol_infix_to_prefix,
)


STEP_FIELDS = {"id", "dependencies", "conclusion", "rule"}
FINAL_IDS = {"A": "h_goal_true", "B": "h_goal_false"}


SYSTEM_PROMPT = f"""You construct minimal formal proofs for ProverQA.
Return one JSON object only: {{"steps": [<step objects>]}}.

Each step object must have exactly four fields:
- id: s1, s2, ... for proof actions; the final id supplied by the user for the last action.
- dependencies: fewer than 5 premise ids or earlier generated-step ids.
- conclusion: exactly one formal formula, using the notation in the premises.
- rule: exactly one canonical rule name.

Canonical rule semantics:
{RULE_PROMPT_GUIDANCE}

Proof policy:
- Use only the supplied formal premises. The natural-language reference is a fallible hint.
- Every proof action must add nontrivial progress: no tautologies, repeated conclusions, or dependency restatements.
- Keep only actions in the dependency closure of the final answer.
- The final action must be GOAL_BINDING, with exactly the supplied final id and formula.
- GOAL_BINDING must depend on exactly one generated proof action that semantically establishes the supplied final formula. Logically equivalent surface forms such as P and NOT NOT P need no extra action. GOAL_BINDING must not perform substantive reasoning directly from premises.
- Premises are leaves and are not copied into generated steps.
- Minimize the number of non-GOAL actions.
- The checker permits implication-activated XOR/OR/AND elimination as one action when the other dependencies establish the implication antecedent and select the resulting branch.
- There is no disjunction-introduction rule. To activate an implication whose antecedent is A OR B, a dependency proving either A or B is sufficient; do not generate an intermediate A OR B action.
- For MODUS_TOLLENS on an implication with a compound antecedent, first derive the negation of the entire antecedent from the implication and the negated consequent. A separate elimination action may then derive a component; do not skip the whole-compound negation.
- There is no double-negation introduction or elimination rule. Never generate P -> NOT NOT P as an action or label it as another rule; use P directly wherever its logically equivalent double-negated form is required.
- A universally quantified implication may be grounded while applying IMPLICATION_ELIMINATION. Use UNIVERSAL_ELIMINATION only when the instantiated formula itself is the conclusion.
- Do not include markdown, commentary, <think>, or <summary> tags."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: Mapping[str, Any], lock: threading.Lock) -> None:
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_json_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"(^|\s)//.*?$", r"\1", text, flags=re.MULTILINE)
    return re.sub(r",\s*([}\]])", r"\1", text)


def vscode_claude_environment(settings_path: Path | None = None) -> dict[str, str]:
    path = settings_path or (
        Path(os.environ.get("APPDATA", "")) / "Code" / "User" / "settings.json"
    )
    if not path.is_file():
        return {}
    data = json.loads(strip_json_comments(path.read_text(encoding="utf-8-sig")))
    configured = data.get("claudeCode.environmentVariables", [])
    if isinstance(configured, Mapping):
        return {str(key): str(value) for key, value in configured.items()}
    if not isinstance(configured, list):
        return {}
    return {
        str(item.get("name")): str(item.get("value"))
        for item in configured
        if isinstance(item, Mapping) and item.get("name") and item.get("value")
    }


def resolve_api_config(args: argparse.Namespace) -> tuple[str, str, str]:
    configured = vscode_claude_environment(
        Path(args.vscode_settings) if args.vscode_settings else None
    )
    base_url = (
        args.base_url
        or os.environ.get("ANTHROPIC_BASE_URL")
        or configured.get("ANTHROPIC_BASE_URL")
    )
    token = (
        os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("ANTHROPIC_API_KEY")
        or configured.get("ANTHROPIC_AUTH_TOKEN")
        or configured.get("ANTHROPIC_API_KEY")
    )
    model = (
        args.model
        or os.environ.get("ANTHROPIC_MODEL")
        or configured.get("ANTHROPIC_MODEL")
        or "deepseek-v4-pro"
    )
    if not base_url or not token:
        raise RuntimeError(
            "Claude Code API configuration was not found in the environment or VS Code settings"
        )
    return base_url, token, model


def extract_json_value(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    for candidate in (cleaned,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    object_start, object_end = cleaned.find("{"), cleaned.rfind("}")
    if object_start >= 0 and object_end > object_start:
        return json.loads(cleaned[object_start : object_end + 1])
    array_start, array_end = cleaned.find("["), cleaned.rfind("]")
    if array_start >= 0 and array_end > array_start:
        return json.loads(cleaned[array_start : array_end + 1])
    raise ValueError("API response contains no JSON object or array")


class AnthropicCompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        model: str,
        timeout: int,
        max_tokens: int,
        temperature: float,
        thinking: bool,
    ):
        self.url = base_url.rstrip("/") + "/v1/messages"
        self.token = token
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking = thinking
        self.local = threading.local()

    def session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def propose(self, user_prompt: str) -> tuple[Any, dict[str, Any]]:
        request_body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "enabled" if self.thinking else "disabled"},
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if not self.thinking:
            request_body["temperature"] = self.temperature
        response = self.session().post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "x-api-key": self.token,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=request_body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("content")
        if isinstance(content, list):
            text = "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text"
            )
        else:
            text = str(payload.get("choices", [{}])[0].get("message", {}).get("content", ""))
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        try:
            value = extract_json_value(text)
        except Exception as exc:
            content_types = [
                str(item.get("type"))
                for item in content
                if isinstance(content, list) and isinstance(item, Mapping)
            ]
            raise ValueError(
                f"{exc}; content_types={content_types}; text_excerpt={text[:500]!r}"
            ) from exc
        return value, dict(usage)


def expected_goal(problem: Mapping[str, Any]) -> dict[str, str]:
    answer = str(problem.get("answer", "")).strip().upper()
    options = build_answer_options(problem)
    matches = [option for option in options if option.get("letter") == answer]
    if answer not in FINAL_IDS or len(matches) != 1:
        raise ValueError(f"problem must have one A/B formal answer option, got {answer!r}")
    return {
        "letter": answer,
        "id": FINAL_IDS[answer],
        "formal": str(matches[0]["formal"]),
    }


def build_generation_prompt(
    problem: Mapping[str, Any],
    *,
    previous_failure: Mapping[str, Any] | None = None,
    attempt: int = 1,
) -> str:
    premises = build_premises_fol(problem)
    goal = expected_goal(problem)
    lines = ["Formal premises:"]
    for premise_id, formula in premises.items():
        lines.append(f"{premise_id}: {fol_infix_to_prefix(formula)}")
    lines.extend(
        [
            "",
            f"Question: {problem.get('question', '')}",
            f"Gold option: {goal['letter']}",
            f"Required final id: {goal['id']}",
            f"Required final formula: {goal['formal']}",
            f"Generation attempt: {attempt}",
            "",
            "Fallible natural-language reference (use only as a hint):",
            str(problem.get("reasoning", "") or "<none>"),
        ]
    )
    if previous_failure:
        lines.extend(
            [
                "",
                "The previous proposal was rejected. Return a revised proof, not the same proof. Correct these verifier findings:",
                json.dumps(previous_failure, ensure_ascii=False),
            ]
        )
    return "\n".join(lines)


def candidate_steps(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("steps"), list):
        return list(value["steps"])
    raise ValueError("candidate must be a JSON array or an object containing a steps array")


def prune_to_final_closure(value: Any) -> dict[str, list[dict[str, Any]]]:
    """Drop off-closure actions and restore sequential generated-step ids."""
    steps = candidate_steps(value)
    final_index = find_final_answer_step_index(steps)
    closure = dependency_closure(steps, final_index)
    if final_index is None or not closure:
        raise ValueError("candidate has no explicit final-answer closure")

    kept_indices = sorted(closure)
    id_map: dict[str, str] = {}
    action_number = 0
    for index in kept_indices:
        step = steps[index]
        if not isinstance(step, Mapping) or not isinstance(step.get("id"), str):
            raise ValueError(f"closure step {index} has no valid id")
        old_id = str(step["id"])
        if index == final_index:
            id_map[old_id] = old_id
        else:
            action_number += 1
            id_map[old_id] = f"s{action_number}"

    pruned: list[dict[str, Any]] = []
    for index in kept_indices:
        source = dict(steps[index])
        source["id"] = id_map[str(source["id"])]
        dependencies = source.get("dependencies")
        if isinstance(dependencies, list):
            source["dependencies"] = [id_map.get(str(dep), str(dep)) for dep in dependencies]
        pruned.append(source)
    return {"steps": pruned}


def validate_problem_label(
    problem: Mapping[str, Any], verifier: Z3Verifier, timeout_ms: int
) -> dict[str, Any]:
    premises = build_premises_fol(problem)
    goal = expected_goal(problem)
    try:
        premise_exprs = [verifier.converter.convert(formula) for formula in premises.values()]
        target = verifier.converter.convert(goal["formal"])
    except Exception as exc:
        return {"valid": False, "reason": "fol_parse", "error": str(exc)}
    if not all(is_bool(expr) for expr in [*premise_exprs, target]):
        return {"valid": False, "reason": "non_boolean_formula"}

    def solve(*extra: Any) -> Any:
        solver = Solver()
        solver.set("timeout", timeout_ms)
        solver.add(*premise_exprs, *extra)
        return solver.check()

    consistency = solve()
    target_counterexample = solve(Not(target))
    opposite_counterexample = solve(target)
    if unknown in (consistency, target_counterexample, opposite_counterexample):
        return {"valid": False, "reason": "z3_unknown"}
    if consistency != sat:
        return {"valid": False, "reason": "inconsistent_premises"}
    if target_counterexample != unsat:
        return {"valid": False, "reason": "gold_target_not_entailed"}
    if opposite_counterexample != sat:
        return {"valid": False, "reason": "opposite_also_entailed"}
    return {
        "valid": True,
        "premises_consistent": True,
        "gold_target_entailed": True,
        "opposite_not_entailed": True,
    }


def validate_candidate(
    problem: Mapping[str, Any], value: Any, scorer: RuleGroundedProcessRewardScorer
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        steps = candidate_steps(value)
    except Exception as exc:
        return None, {"reason": "candidate_schema", "error": str(exc)}
    if not steps:
        return None, {"reason": "empty_steps"}

    expected = expected_goal(problem)
    seen: set[str] = set()
    prefix: set[str] = set(build_premises_fol(problem))
    generated_conclusions: dict[str, str] = {}
    structural_errors: list[str] = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            structural_errors.append(f"step {index} is not an object")
            continue
        if set(step) != STEP_FIELDS:
            structural_errors.append(f"step {index} fields are {sorted(step)}")
        step_id = step.get("id")
        expected_id = expected["id"] if index == len(steps) - 1 else f"s{index + 1}"
        if step_id != expected_id:
            structural_errors.append(
                f"step {index} id must be {expected_id!r}, got {step_id!r}"
            )
        if not isinstance(step_id, str) or step_id in seen:
            structural_errors.append(f"step {index} has invalid or duplicate id")
        dependencies = step.get("dependencies")
        if isinstance(dependencies, list):
            missing = [dep for dep in dependencies if dep not in prefix]
            if missing:
                structural_errors.append(
                    f"step {index} has non-premise or non-prefix dependencies: {missing}"
                )
        if isinstance(step_id, str):
            seen.add(step_id)
            prefix.add(step_id)
            generated_conclusions[step_id] = str(step.get("conclusion", ""))

    final = steps[-1] if isinstance(steps[-1], Mapping) else {}
    if canonicalize_rule(final.get("rule")) != GOAL_BINDING:
        structural_errors.append("last step must claim GOAL_BINDING")
    if final.get("conclusion") != expected["formal"]:
        structural_errors.append("last conclusion does not exactly match the gold option")
    final_dependencies = final.get("dependencies")
    proof_ids = set(generated_conclusions) - {expected["id"]}
    if (
        not isinstance(final_dependencies, list)
        or len(final_dependencies) != 1
        or final_dependencies[0] not in proof_ids
    ):
        structural_errors.append(
            "GOAL_BINDING must depend on exactly one generated proof action"
        )
    if structural_errors:
        return None, {"reason": "structure", "errors": structural_errors[:20]}

    response = (
        "<think>golden proof candidate</think>\n<summary>\n"
        + json.dumps(steps, ensure_ascii=False)
        + "\n</summary>"
    )
    score = scorer.score_response(response, dict(problem))
    final_index = find_final_answer_step_index(steps)
    closure = dependency_closure(steps, final_index)
    action_errors: list[dict[str, Any]] = []
    for action in score.get("actions", []):
        if action.get("kind") != "step":
            continue
        is_goal = action.get("canonical_rule") == GOAL_BINDING
        valid = (
            action.get("schema_valid")
            and action.get("semantic_verified")
            and action.get("rule_application_valid")
            and action.get("rule_grounded_verified")
            and action.get("in_closure")
            and (is_goal or action.get("nontrivial_verified_progress"))
            and (not is_goal or action.get("final_binding_ok"))
        )
        if not valid:
            action_errors.append(
                {
                    "step_index": action.get("step_index"),
                    "step_id": action.get("step_id"),
                    "canonical_rule": action.get("canonical_rule"),
                    "semantic_verified": action.get("semantic_verified"),
                    "semantic_error": action.get("semantic_error"),
                    "rule_application_valid": action.get("rule_application_valid"),
                    "rule_error": action.get("rule_error"),
                    "nontrivial_verified_progress": action.get(
                        "nontrivial_verified_progress"
                    ),
                    "tautology": action.get("tautology"),
                    "repeated_conclusion": action.get("repeated_conclusion"),
                    "dependency_restatement": action.get("dependency_restatement"),
                    "final_binding_ok": action.get("final_binding_ok"),
                    "in_closure": action.get("in_closure"),
                }
            )

    expected_closure = set(range(len(steps)))
    if not score.get("parse_success") or not score.get("schema_valid"):
        return None, {
            "reason": "parse_or_schema",
            "parse_error": score.get("parse_error"),
        }
    if not score.get("answer_correct"):
        return None, {
            "reason": "answer_binding",
            "parsed_answer": score.get("parsed_answer"),
        }
    if final_index != len(steps) - 1:
        return None, {"reason": "missing_explicit_final"}
    if closure != expected_closure:
        return None, {
            "reason": "off_closure_steps",
            "closure": sorted(closure),
            "expected": sorted(expected_closure),
        }
    if action_errors:
        return None, {"reason": "step_validation", "actions": action_errors}

    non_goal_ids = [
        str(step["id"])
        for step in steps
        if canonicalize_rule(step.get("rule")) != GOAL_BINDING
    ]
    proof = {
        "proof_id": "canonical_0",
        "steps": steps,
        "closure_step_ids": [str(step["id"]) for step in steps],
        "closure_non_goal_step_ids": non_goal_ids,
        "proof_length": len(non_goal_ids),
        "validation": {
            "schema_valid": True,
            "all_steps_semantic_verified": True,
            "all_steps_rule_application_valid": True,
            "all_non_goal_steps_nontrivial": True,
            "answer_binding_valid": True,
            "all_steps_in_final_closure": True,
        },
    }
    return proof, {"reason": "accepted", "proof_length": len(non_goal_ids)}


def completed_indices(*paths: Path) -> set[int]:
    completed: set[int] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "source_index" in row:
                    completed.add(int(row["source_index"]))
    return completed


def assemble_outputs(
    source_rows: list[dict[str, Any]],
    accepted_path: Path,
    output_root: Path,
    model: str,
    rejected_path: Path | None = None,
) -> dict[str, Any]:
    accepted: dict[int, dict[str, Any]] = {}
    if accepted_path.exists():
        for line in accepted_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                row = json.loads(line)
                accepted[int(row["source_index"])] = row

    output_rows: list[dict[str, Any]] = []
    by_difficulty: dict[str, list[dict[str, Any]]] = {
        "easy": [],
        "medium": [],
        "hard": [],
    }
    proof_lengths: list[int] = []
    for index, problem in enumerate(source_rows):
        annotation = accepted.get(index)
        if annotation is None:
            continue
        enriched = dict(problem)
        proof = annotation["proof"]
        enriched["canonical_proofs"] = [proof]
        enriched["canonical_proof_reference"] = {
            "version": "rule_checker_v1",
            "source": model,
            "proof_count": 1,
            "min_proof_length": int(proof["proof_length"]),
            "length_excludes_goal_binding": True,
            "length_uses_final_dependency_closure": True,
        }
        output_rows.append(enriched)
        difficulty = str(problem.get("difficulty", "")).lower()
        if difficulty in by_difficulty:
            by_difficulty[difficulty].append(enriched)
        proof_lengths.append(int(proof["proof_length"]))

    latest_rejections: dict[int, dict[str, Any]] = {}
    if rejected_path is not None and rejected_path.exists():
        for line in rejected_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                rejection = json.loads(line)
                latest_rejections[int(rejection["source_index"])] = rejection
    excluded_rows: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    for index, problem in enumerate(source_rows):
        if index in accepted:
            continue
        rejection = latest_rejections.get(index, {})
        status = str(rejection.get("status") or "pending")
        exclusion_counts[status] += 1
        excluded_rows.append(
            {
                "source_index": index,
                "problem_id": problem.get("id"),
                "difficulty": problem.get("difficulty"),
                "answer": problem.get("answer"),
                "status": status,
                "validation": rejection.get("validation"),
                "last_validation": rejection.get("last_validation"),
            }
        )

    write_jsonl(output_root / "all.jsonl", output_rows)
    for difficulty, rows in by_difficulty.items():
        write_jsonl(output_root / f"{difficulty}.jsonl", rows)
    write_jsonl(output_root / "excluded.jsonl", excluded_rows)
    manifest = {
        "source": "Data/ProverQA/test/all.jsonl",
        "source_count": len(source_rows),
        "accepted_count": len(output_rows),
        "canonical_coverage_rate": len(output_rows) / max(1, len(source_rows)),
        "recommended_evaluation_input": "Data/ProverQA/test/golden/all.jsonl",
        "metric_scope": "all_metrics_use_the_same_canonical_subset",
        "rejected_or_pending_count": len(source_rows) - len(output_rows),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "by_difficulty": {key: len(value) for key, value in by_difficulty.items()},
        "generator_model": model,
        "ontology_version": "rule_checker_v1",
        "proof_length_excludes_goal_binding": True,
        "proof_length_uses_final_dependency_closure": True,
        "proof_length_min": min(proof_lengths) if proof_lengths else None,
        "proof_length_max": max(proof_lengths) if proof_lengths else None,
        "proof_length_mean": (
            sum(proof_lengths) / len(proof_lengths) if proof_lengths else None
        ),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Data/ProverQA/test/all.jsonl")
    parser.add_argument("--output-root", default="Data/ProverQA/test/golden")
    parser.add_argument(
        "--work-root", default="Codex/golden_formal_proofs"
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--vscode-settings")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--api-max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--z3-timeout-ms", type=int, default=5000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument("--stop-on-first-valid", action="store_true")
    parser.add_argument("--revalidate-candidates-only", action="store_true")
    parser.add_argument("--rebuild-outputs-only", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    input_path = Path(args.input)
    output_root = Path(args.output_root)
    work_root = Path(args.work_root)
    accepted_path = work_root / "accepted.jsonl"
    rejected_path = work_root / "rejected.jsonl"
    candidates_path = work_root / "candidates.jsonl"
    api_errors_path = work_root / "api_errors.jsonl"
    source_rows = load_jsonl(input_path)

    base_url, token, model = resolve_api_config(args)
    if args.rebuild_outputs_only:
        print(json.dumps(assemble_outputs(source_rows, accepted_path, output_root, model, rejected_path), ensure_ascii=False, indent=2))
        return

    completed_paths = [accepted_path]
    if not args.retry_rejected:
        completed_paths.append(rejected_path)
    completed = completed_indices(*completed_paths)
    pending = [
        (index, row)
        for index, row in enumerate(source_rows)
        if index >= args.start_index and index not in completed
    ]
    if args.max_records is not None:
        pending = pending[: args.max_records]

    client = AnthropicCompatibleClient(
        base_url=base_url,
        token=token,
        model=model,
        timeout=args.timeout,
        max_tokens=args.api_max_tokens,
        temperature=args.temperature,
        thinking=args.thinking,
    )
    verifier = Z3Verifier(timeout_ms=args.z3_timeout_ms)
    scorer = RuleGroundedProcessRewardScorer(verifier=verifier)
    io_lock = threading.Lock()
    z3_lock = threading.Lock()
    stats: Counter[str] = Counter()
    started = time.time()

    if args.revalidate_candidates_only:
        accepted_indices = completed_indices(accepted_path)
        recovered: set[int] = set()
        tried: Counter[int] = Counter()
        candidate_rows = []
        if candidates_path.exists():
            candidate_rows = [
                json.loads(line)
                for line in candidates_path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
        for row in reversed(candidate_rows):
            index = int(row.get("source_index", -1))
            if index < 0 or index in accepted_indices or index in recovered:
                continue
            if tried[index] >= args.max_attempts:
                continue
            tried[index] += 1
            problem = source_rows[index]
            value = row.get("candidate")
            proof, detail = validate_candidate(problem, value, scorer)
            normalized = False
            if proof is None and detail.get("reason") == "off_closure_steps":
                try:
                    value = prune_to_final_closure(value)
                except Exception:
                    pass
                else:
                    proof, detail = validate_candidate(problem, value, scorer)
                    normalized = proof is not None
            if proof is None:
                continue
            label_validation = validate_problem_label(
                problem, verifier, args.z3_timeout_ms
            )
            if not label_validation.get("valid"):
                continue
            append_jsonl(
                accepted_path,
                {
                    "source_index": index,
                    "problem_id": problem.get("id"),
                    "difficulty": problem.get("difficulty"),
                    "answer": problem.get("answer"),
                    "generator_model": model,
                    "problem_validation": label_validation,
                    "valid_candidate_count": 1,
                    "revalidated_from_candidate_attempt": row.get("attempt"),
                    "closure_pruned_and_renumbered": normalized,
                    "proof": proof,
                },
                io_lock,
            )
            recovered.add(index)
        manifest = assemble_outputs(source_rows, accepted_path, output_root, model, rejected_path)
        manifest["revalidated_candidate_count"] = sum(tried.values())
        manifest["recovered_count"] = len(recovered)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    def process(index: int, problem: dict[str, Any]) -> tuple[int, str]:
        with z3_lock:
            label_validation = validate_problem_label(
                problem, verifier, args.z3_timeout_ms
            )
        if not label_validation.get("valid"):
            append_jsonl(
                rejected_path,
                {
                    "source_index": index,
                    "problem_id": problem.get("id"),
                    "difficulty": problem.get("difficulty"),
                    "status": "problem_validation_failed",
                    "validation": label_validation,
                },
                io_lock,
            )
            return index, "problem_validation_failed"

        previous_failure: dict[str, Any] | None = None
        valid_proofs: list[dict[str, Any]] = []
        for attempt in range(1, args.max_attempts + 1):
            try:
                value, usage = client.propose(
                    build_generation_prompt(
                        problem,
                        previous_failure=previous_failure,
                        attempt=attempt,
                    )
                )
            except Exception as exc:
                append_jsonl(
                    api_errors_path,
                    {
                        "source_index": index,
                        "problem_id": problem.get("id"),
                        "attempt": attempt,
                        "error": str(exc),
                    },
                    io_lock,
                )
                previous_failure = {"reason": "api_error", "error": str(exc)}
                continue
            with z3_lock:
                proof, detail = validate_candidate(problem, value, scorer)
            append_jsonl(
                candidates_path,
                {
                    "source_index": index,
                    "problem_id": problem.get("id"),
                    "difficulty": problem.get("difficulty"),
                    "attempt": attempt,
                    "status": detail.get("reason"),
                    "validation": detail,
                    "usage": usage,
                    "candidate": value,
                },
                io_lock,
            )
            if proof is not None:
                valid_proofs.append(proof)
                if args.stop_on_first_valid or int(proof["proof_length"]) <= 1:
                    break
                previous_failure = {
                    "reason": "valid_but_seek_shorter",
                    "current_best_proof_length": min(
                        int(item["proof_length"]) for item in valid_proofs
                    ),
                    "instruction": "Return a strictly shorter proof if one exists.",
                }
            else:
                previous_failure = {
                    "verifier": detail,
                    "previous_steps": value,
                }

        if valid_proofs:
            best = min(
                valid_proofs,
                key=lambda item: (int(item["proof_length"]), len(json.dumps(item))),
            )
            append_jsonl(
                accepted_path,
                {
                    "source_index": index,
                    "problem_id": problem.get("id"),
                    "difficulty": problem.get("difficulty"),
                    "answer": problem.get("answer"),
                    "generator_model": model,
                    "problem_validation": label_validation,
                    "valid_candidate_count": len(valid_proofs),
                    "proof": best,
                },
                io_lock,
            )
            return index, "accepted"

        append_jsonl(
            rejected_path,
            {
                "source_index": index,
                "problem_id": problem.get("id"),
                "difficulty": problem.get("difficulty"),
                "status": "proof_attempts_exhausted",
                "last_validation": previous_failure,
            },
            io_lock,
        )
        return index, "proof_attempts_exhausted"

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.workers)
    ) as pool:
        futures = [pool.submit(process, index, row) for index, row in pending]
        for done, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            index, status = future.result()
            stats[status] += 1
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"done={done}/{len(futures)} accepted={stats['accepted']} "
                f"rate={done / elapsed:.3f}/s last_index={index} status={status}",
                flush=True,
            )

    manifest = assemble_outputs(source_rows, accepted_path, output_root, model, rejected_path)
    manifest["run_stats"] = dict(stats)
    manifest["work_root"] = str(work_root)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
