"""Rule-grounded process reward and trajectory-credit utilities.

The scorer keeps semantic entailment, rule correctness, and final-answer credit
separate.  Step rewards are assigned only to structured JSON actions; final
answer trajectory reward is masked by the generated-step dependency closure.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from recipe.formally_verifiable.common.verifier.z3_verifier import Z3Verifier
from recipe.formally_verifiable.rule_grounded_process_rl.rule_checker import RuleChecker
from recipe.formally_verifiable.rule_grounded_process_rl.rule_ontology import GOAL_BINDING, canonicalize_rule
from recipe.formally_verifiable.rule_grounded_process_rl.structured_parser import match_answer_to_option
from recipe.formally_verifiable.rule_grounded_process_rl.structured_prompt import fol_infix_to_prefix


INVALID_RULE_BUCKET = "INVALID_RULE"
FORMAT_BUCKET = "FORMAT"
REWARD_LEVELS = (0.0, 0.1, 0.3, 1.0)
FINAL_ANSWER_IDS = {"h_goal_true", "h_goal_false", "h_goal_uncertain"}


@dataclass(frozen=True)
class ParsedSummary:
    summary: list[Any]
    step_spans: list[tuple[int, int]]
    parse_error: str | None
    summary_tag_present: bool
    failure_span: tuple[int, int] | None
    format_scaffold_spans: list[tuple[int, int]]


@dataclass
class ProcessRewardConfig:
    require_summary_tag: bool = True
    reward_parse_failures: bool = True
    reward_empty_summary: bool = True
    enable_format_scaffold_actions: bool = False
    format_scaffold_reward: float = 1.0
    baseline_mode: str = "rule_ema_clipped"
    baseline_initial_value: float = 0.5
    baseline_ema_beta: float = 0.9
    baseline_clip_min: float = 0.4
    baseline_clip_max: float = 0.8


class RuleEmaBaseline:
    """Historical per-rule EMA used only for advantage centering."""

    # 初始化按 rule bucket 维护的 EMA baseline。
    def __init__(self, config: ProcessRewardConfig):
        self.config = config
        self.ema: dict[str, float] = {}
        self.counts: Counter[str] = Counter()

    # 读取指定 rule bucket 当前裁剪后的 EMA baseline。
    def value(self, bucket: str) -> float:
        mode = self.config.baseline_mode
        if mode == "none":
            return 0.0
        raw = self.ema.get(bucket, float(self.config.baseline_initial_value))
        if mode.endswith("_clipped"):
            raw = min(
                float(self.config.baseline_clip_max),
                max(float(self.config.baseline_clip_min), raw),
            )
        return float(raw)

    # 使用本批 action 原始奖励更新各 rule bucket 的 EMA。
    def update(self, actions: Iterable[Mapping[str, Any]]) -> None:
        by_bucket: dict[str, list[float]] = defaultdict(list)
        for action in actions:
            bucket = str(action.get("rule_bucket") or INVALID_RULE_BUCKET)
            by_bucket[bucket].append(float(action.get("raw_reward", 0.0) or 0.0))
        beta = float(self.config.baseline_ema_beta)
        for bucket, rewards in by_bucket.items():
            mean_reward = sum(rewards) / max(1, len(rewards))
            previous = self.ema.get(bucket, float(self.config.baseline_initial_value))
            self.ema[bucket] = beta * previous + (1.0 - beta) * mean_reward
            self.counts[bucket] += len(rewards)

    # 从 checkpoint 日志快照恢复 rule EMA 状态。
    def load_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        for bucket, state in snapshot.items():
            if not isinstance(bucket, str) or not isinstance(state, Mapping):
                raise ValueError("baseline snapshot must map rule names to state objects")
            ema = state.get("ema")
            count = state.get("count", 0)
            if not isinstance(ema, (int, float)) or not isinstance(count, int):
                raise ValueError(f"invalid baseline snapshot state for {bucket!r}")
            self.ema[bucket] = float(ema)
            self.counts[bucket] = int(count)

    # 导出指定 rule bucket 的 EMA、baseline 与计数。
    def snapshot(self, buckets: Iterable[str] = ()) -> dict[str, dict[str, float | int]]:
        keys = set(self.ema) | set(buckets)
        return {
            key: {
                "ema": float(self.ema.get(key, float(self.config.baseline_initial_value))),
                "baseline": float(self.value(key)),
                "count": int(self.counts.get(key, 0)),
            }
            for key in sorted(keys)
        }


# 去除字符区间首尾空白并返回有效范围。
def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


# 在解析失败时选择可记录和映射的回退字符区间。
def _fallback_failure_span(text: str) -> tuple[int, int] | None:
    summary = re.search(r"<summary>(.*?)</summary>", text, flags=re.DOTALL | re.IGNORECASE)
    if summary:
        return _trim_span(text, summary.start(1), summary.end(1))
    think = re.search(r"<think>.*?</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if think and think.end() < len(text):
        return _trim_span(text, think.end(), len(text))
    return _trim_span(text, 0, len(text))


# 从成对分隔符中定位外层平衡区间。
def _balanced_span(text: str, start: int, open_char: str, close_char: str) -> tuple[int, int] | None:
    depth = 0
    in_string = False
    escaped = False
    begin: int | None = None
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == open_char:
            if depth == 0:
                begin = idx
            depth += 1
        elif char == close_char and depth:
            depth -= 1
            if depth == 0 and begin is not None:
                return begin, idx + 1
    return None


# 定位 summary 中顶层 JSON 数组的字符范围。
def _json_array_span(text: str) -> tuple[int, int] | None:
    start = len(text) - len(text.lstrip())
    if start >= len(text) or text[start] != "[":
        return None
    return _balanced_span(text, start, "[", "]")


# 扫描 JSON 数组中每个顶层 value 的字符范围。
def _top_level_value_spans(array_text: str) -> list[tuple[int, int]]:
    """Return spans of top-level JSON array elements, relative to array_text."""
    if not array_text.strip().startswith("["):
        return []
    spans: list[tuple[int, int]] = []
    depth = 0
    in_string = False
    escaped = False
    value_start: int | None = None
    end_limit = len(array_text) - 1
    for idx in range(1, end_limit):
        char = array_text[idx]
        if value_start is None and not char.isspace() and char != ",":
            value_start = idx
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "[{":
            depth += 1
            continue
        if char in "]}":
            depth = max(0, depth - 1)
            continue
        if char == "," and depth == 0 and value_start is not None:
            trimmed = _trim_span(array_text, value_start, idx)
            if trimmed:
                spans.append(trimmed)
            value_start = None
    if value_start is not None:
        trimmed = _trim_span(array_text, value_start, end_limit)
        if trimmed:
            spans.append(trimmed)
    return spans


# 定位顶层数组括号和逗号等格式 token 区间。
def _top_level_separator_spans(array_text: str) -> list[tuple[int, int]]:
    """Return spans of top-level commas between JSON array elements."""
    spans: list[tuple[int, int]] = []
    depth = 0
    in_string = False
    escaped = False
    for idx, char in enumerate(array_text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "[{":
            depth += 1
            continue
        if char in "]}":
            depth = max(0, depth - 1)
            continue
        if char == "," and depth == 1:
            spans.append((idx, idx + 1))
    return spans


# 严格解析 summary，并保留每个 action 的字符区间。
def parse_summary_with_spans(text: str) -> ParsedSummary:
    summary_match = re.search(r"<summary>(.*?)</summary>", text, flags=re.DOTALL | re.IGNORECASE)
    if summary_match is None:
        return ParsedSummary([], [], "Missing <summary> tag", False, _fallback_failure_span(text), [])

    summary_text = summary_match.group(1)
    summary_offset = summary_match.start(1)
    code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", summary_text, flags=re.DOTALL | re.IGNORECASE)
    if code_block:
        json_text = code_block.group(1)
        json_offset = summary_offset + code_block.start(1)
    else:
        json_text = summary_text
        json_offset = summary_offset

    array_span = _json_array_span(json_text)
    if array_span is None:
        return ParsedSummary([], [], "No JSON array found inside <summary>", True, _fallback_failure_span(text), [])
    if json_text[array_span[1] :].strip():
        return ParsedSummary(
            [],
            [],
            "Unexpected content after the summary JSON array",
            True,
            _fallback_failure_span(text),
            [],
        )

    array_text = json_text[array_span[0] : array_span[1]]
    array_global_start = json_offset + array_span[0]
    array_global_end = array_global_start + len(array_text)
    format_scaffold_spans = [
        (summary_match.start(), summary_match.start(1)),
        (array_global_start, array_global_start + 1),
        (array_global_end - 1, array_global_end),
        (summary_match.end(1), summary_match.end()),
    ]
    format_scaffold_spans.extend(
        (array_global_start + start, array_global_start + end)
        for start, end in _top_level_separator_spans(array_text)
    )

    try:
        summary = json.loads(array_text)
    except json.JSONDecodeError as exc:
        failure = _trim_span(text, array_global_start, array_global_start + len(array_text))
        return ParsedSummary([], [], f"JSON decode error: {exc}", True, failure, [])

    if not isinstance(summary, list):
        failure = _trim_span(text, array_global_start, array_global_start + len(array_text))
        return ParsedSummary([], [], "Summary is not a JSON array", True, failure, [])

    value_spans = _top_level_value_spans(array_text)
    step_spans = [
        (array_global_start + start, array_global_start + end)
        for start, end in value_spans[: len(summary)]
    ]
    if len(step_spans) != len(summary):
        failure = _trim_span(text, array_global_start, array_global_start + len(array_text))
        return ParsedSummary(
            summary,
            step_spans,
            f"Could not locate all JSON element spans ({len(step_spans)}/{len(summary)})",
            True,
            failure,
            [],
        )
    return ParsedSummary(summary, step_spans, None, True, None, format_scaffold_spans)


# 将问题中的 nl2fol 条目编号为可引用的形式化前提。
def build_premises_fol(problem: Mapping[str, Any]) -> dict[str, str]:
    return {
        f"h{index}": str(fol).strip()
        for index, fol in enumerate((problem.get("nl2fol") or {}).values(), start=1)
    }


# 构造 True、False、Uncertain 对应的终局形式化选项。
def build_answer_options(problem: Mapping[str, Any]) -> list[dict[str, str]]:
    target = fol_infix_to_prefix(str(problem.get("conclusion_fol", "")).strip())
    if not target:
        return []
    options: list[dict[str, str]] = []
    for option in problem.get("options", []):
        option_text = str(option)
        if ")" not in option_text:
            continue
        letter, text = option_text.split(")", 1)
        normalized = text.strip().lower()
        if "true" in normalized:
            answer_id = "h_goal_true"
            formal = target
        elif "false" in normalized:
            answer_id = "h_goal_false"
            formal = f"\u00ac({target})"
        elif "uncertain" in normalized:
            answer_id = "h_goal_uncertain"
            formal = target
        else:
            continue
        options.append(
            {
                "letter": letter.strip().upper(),
                "text": normalized,
                "answer_id": answer_id,
                "formal": formal,
            }
        )
    return options


# 检查 step 是否具有合法 id、依赖、结论和 rule 字段。
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


# 规范化公式文本，供重复和答案绑定比较使用。
def _normalize_formula(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


# 检查 GOAL_BINDING 的 id 与结论是否匹配同一答案选项。
def _goal_binding_matches_answer_option(step: Mapping[str, Any], answer_options: Sequence[Mapping[str, Any]]) -> bool:
    step_id = step.get("id")
    conclusion = _normalize_formula(step.get("conclusion"))
    for option in answer_options:
        if option.get("answer_id") == step_id and _normalize_formula(option.get("formal")) == conclusion:
            return True
    return False


# 检查结论在空前提下是否成立。
def _is_tautology(verifier: Z3Verifier, conclusion: str) -> bool:
    result = verifier.verify_step(
        dependencies=[],
        conclusion=conclusion,
        premises_fol={},
        all_steps_fol={},
    )
    return bool(result.get("verified"))


# 判定步骤是否满足非重言、非重复和非复述进展。
def nontrivial_verified_progress(
    *,
    step: Mapping[str, Any],
    verifier: Z3Verifier,
    dependency_formulas: Mapping[str, str],
    prefix_conclusions: set[str],
    answer_options: Sequence[Mapping[str, Any]],
    canonical_rule: str | None,
) -> dict[str, Any]:
    conclusion = _normalize_formula(step.get("conclusion"))
    dependency_restatement = any(
        conclusion == _normalize_formula(dependency_formulas.get(dep))
        for dep in step.get("dependencies", [])
    )
    repeated = conclusion in prefix_conclusions
    tautology = _is_tautology(verifier, str(step.get("conclusion", "")))
    final_binding_ok = True
    if canonical_rule == GOAL_BINDING:
        final_binding_ok = _goal_binding_matches_answer_option(step, answer_options)
    ok = not tautology and not repeated and not dependency_restatement and final_binding_ok
    return {
        "nontrivial_verified_progress": bool(ok),
        "tautology": bool(tautology),
        "repeated_conclusion": bool(repeated),
        "dependency_restatement": bool(dependency_restatement),
        "final_binding_ok": bool(final_binding_ok),
    }


# 仅在最后一步明确绑定答案时返回终局步骤索引。
def find_final_answer_step_index(steps: Sequence[Any]) -> int | None:
    """Return the last step only when it explicitly represents the final answer."""
    if not steps:
        return None
    index = len(steps) - 1
    step = steps[index]
    if not isinstance(step, Mapping):
        return None
    step_id = step.get("id")
    canonical = canonicalize_rule(step.get("rule"))
    if canonical == GOAL_BINDING or (
        isinstance(step_id, str) and step_id in FINAL_ANSWER_IDS
    ):
        return index
    return None


# 从终局步骤递归计算仅含生成步骤的证明依赖闭包。
def dependency_closure(steps: Sequence[Any], final_index: int | None) -> set[int]:
    """Trace generated-step dependencies from the explicit final answer step."""
    if final_index is None:
        return set()
    id_to_index = {
        step.get("id"): index
        for index, step in enumerate(steps)
        if isinstance(step, Mapping) and isinstance(step.get("id"), str)
    }
    closure: set[int] = set()

    # 递归访问当前节点并收集其可达依赖。
    def visit(index: int) -> None:
        if index in closure or index < 0 or index >= len(steps):
            return
        step = steps[index]
        if not isinstance(step, Mapping):
            return
        closure.add(index)
        dependencies = step.get("dependencies")
        if not isinstance(dependencies, list):
            return
        for dep in dependencies:
            if not isinstance(dep, str):
                continue
            dep_index = id_to_index.get(dep)
            if dep_index is not None:
                visit(dep_index)

    visit(final_index)
    return closure


# 按 schema、Z3、RuleChecker 和非平凡进展分配离散奖励。
def _reward_for_step(
    *,
    schema_valid: bool,
    semantic_verified: bool,
    rule_application_valid: bool,
    nontrivial_progress: bool,
) -> float:
    if not schema_valid:
        return 0.0
    if not semantic_verified:
        return 0.1
    if not rule_application_valid or not nontrivial_progress:
        return 0.3
    return 1.0


class RuleGroundedProcessRewardScorer:
    # 初始化 process reward scorer 与形式化检查器。
    def __init__(
        self,
        *,
        verifier: Z3Verifier | None = None,
        rule_checker: RuleChecker | None = None,
        config: ProcessRewardConfig | None = None,
    ) -> None:
        self.verifier = verifier or Z3Verifier()
        self.rule_checker = rule_checker or RuleChecker(timeout_ms=getattr(self.verifier, "timeout_ms", 5000))
        self.config = config or ProcessRewardConfig()

    # 把 summary 解析失败表示为可记录的零奖励 action。
    def _parse_failure_action(self, parsed: ParsedSummary) -> list[dict[str, Any]]:
        if not self.config.reward_parse_failures or parsed.failure_span is None:
            return []
        start, end = parsed.failure_span
        return [
            {
                "kind": "parse_failure",
                "step_index": None,
                "step_id": None,
                "char_start": start,
                "char_end": end,
                "schema_valid": False,
                "schema_error": parsed.parse_error,
                "parse_error": parsed.parse_error,
                "raw_reward": 0.0,
                "rule_bucket": INVALID_RULE_BUCKET,
                "claimed_rule": None,
                "canonical_rule": None,
                "semantic_verified": False,
                "rule_recognized": False,
                "rule_application_valid": False,
                "rule_grounded_verified": False,
                "semantic_verified_but_rule_invalid": False,
                "nontrivial_verified_progress": False,
                "in_closure": False,
            }
        ]

    # 在启用时构造格式骨架 token 的奖励 action。
    def _format_scaffold_actions(self, parsed: ParsedSummary) -> list[dict[str, Any]]:
        if not self.config.enable_format_scaffold_actions:
            return []
        actions: list[dict[str, Any]] = []
        for index, (start, end) in enumerate(parsed.format_scaffold_spans):
            actions.append(
                {
                    "kind": "format_scaffold",
                    "step_index": None,
                    "step_id": f"format_scaffold_{index}",
                    "char_start": start,
                    "char_end": end,
                    "schema_valid": True,
                    "schema_error": None,
                    "parse_error": None,
                    "raw_reward": float(self.config.format_scaffold_reward),
                    "rule_bucket": FORMAT_BUCKET,
                    "claimed_rule": None,
                    "canonical_rule": None,
                    "semantic_verified": False,
                    "semantic_error": None,
                    "rule_recognized": False,
                    "rule_application_valid": False,
                    "rule_grounded_verified": False,
                    "rule_error": None,
                    "rule_details": {},
                    "semantic_verified_but_rule_invalid": False,
                    "nontrivial_verified_progress": False,
                    "tautology": False,
                    "repeated_conclusion": False,
                    "dependency_restatement": False,
                    "final_binding_ok": False,
                    "in_closure": False,
                }
            )
        return actions

    # 解析完整 response，并计算逐步骤奖励、闭包与诊断指标。
    def score_response(self, response_text: str, problem: Mapping[str, Any]) -> dict[str, Any]:
        parsed = parse_summary_with_spans(response_text)
        parse_error = parsed.parse_error
        if self.config.require_summary_tag and not parsed.summary_tag_present:
            parse_error = parse_error or "Missing required <summary> tag"

        result: dict[str, Any] = {
            "parse_success": False,
            "summary_tag_present": bool(parsed.summary_tag_present),
            "parse_error": parse_error,
            "schema_valid": False,
            "answer_correct": False,
            "parsed_answer": None,
            "final_answer_index": None,
            "final_answer_id": None,
            "total_steps": 0,
            "actions": [],
        }
        if parse_error:
            result["actions"] = self._parse_failure_action(parsed)
            return result
        if not parsed.summary:
            empty = ParsedSummary(
                parsed.summary,
                parsed.step_spans,
                "Empty summary",
                parsed.summary_tag_present,
                parsed.failure_span or _fallback_failure_span(response_text),
                [],
            )
            result["parse_error"] = "Empty summary"
            result["actions"] = self._parse_failure_action(empty) if self.config.reward_empty_summary else []
            return result

        steps = parsed.summary
        result["parse_success"] = True
        result["total_steps"] = len(steps)
        premises_fol = build_premises_fol(problem)
        answer_options = build_answer_options(problem)
        semantic_prefix: dict[str, str] = {}
        rule_prefix: dict[str, str] = {}
        all_previous_steps: dict[str, str] = {}
        prefix_conclusions: set[str] = set()
        actions: list[dict[str, Any]] = []

        final_index = find_final_answer_step_index(steps)
        closure = dependency_closure(steps, final_index)

        for index, step in enumerate(steps):
            char_start, char_end = parsed.step_spans[index]
            schema = validate_step_schema(step)
            canonical = canonicalize_rule(step.get("rule")) if isinstance(step, Mapping) else None
            bucket = canonical or INVALID_RULE_BUCKET
            semantic_result: dict[str, Any] = {"verified": False, "error": None, "details": {}}
            rule_result: dict[str, Any] = {
                "rule_recognized": canonical is not None,
                "rule_application_valid": False,
                "verified": False,
                "error": None,
                "details": {},
            }
            nontrivial = {
                "nontrivial_verified_progress": False,
                "tautology": False,
                "repeated_conclusion": False,
                "dependency_restatement": False,
                "final_binding_ok": canonical != GOAL_BINDING,
            }

            if schema["valid"] and isinstance(step, Mapping):
                semantic_result = self.verifier.verify_step(
                    dependencies=list(step.get("dependencies", [])),
                    conclusion=str(step.get("conclusion", "")),
                    premises_fol=premises_fol,
                    all_steps_fol=semantic_prefix,
                )
                rule_result = self.rule_checker.check_step(
                    step,
                    premises_fol,
                    rule_prefix,
                    answer_options=answer_options,
                )
                if semantic_result.get("verified"):
                    nontrivial = nontrivial_verified_progress(
                        step=step,
                        verifier=self.verifier,
                        dependency_formulas={**premises_fol, **all_previous_steps},
                        prefix_conclusions=prefix_conclusions,
                        answer_options=answer_options,
                        canonical_rule=canonical,
                    )

            semantic_verified = bool(semantic_result.get("verified"))
            rule_application_valid = bool(rule_result.get("rule_application_valid"))
            rule_grounded_verified = bool(semantic_verified and rule_application_valid)
            raw_reward = _reward_for_step(
                schema_valid=bool(schema["valid"]),
                semantic_verified=semantic_verified,
                rule_application_valid=rule_application_valid,
                nontrivial_progress=bool(nontrivial["nontrivial_verified_progress"]),
            )
            action = {
                "kind": "step",
                "step_index": index,
                "step_id": step.get("id") if isinstance(step, Mapping) else None,
                "char_start": char_start,
                "char_end": char_end,
                "schema_valid": bool(schema["valid"]),
                "schema_error": schema["error"],
                "parse_error": None,
                "raw_reward": float(raw_reward),
                "rule_bucket": bucket,
                "claimed_rule": step.get("rule") if isinstance(step, Mapping) else None,
                "canonical_rule": canonical,
                "semantic_verified": semantic_verified,
                "semantic_error": semantic_result.get("error"),
                "rule_recognized": bool(rule_result.get("rule_recognized")),
                "rule_application_valid": rule_application_valid,
                "rule_grounded_verified": rule_grounded_verified,
                "rule_error": rule_result.get("error"),
                "rule_details": rule_result.get("details") or {},
                "semantic_verified_but_rule_invalid": bool(semantic_verified and not rule_application_valid),
                "in_closure": index in closure,
                **nontrivial,
            }
            actions.append(action)

            if isinstance(step, Mapping) and isinstance(step.get("id"), str):
                conclusion = str(step.get("conclusion", ""))
                all_previous_steps[step["id"]] = conclusion
                prefix_conclusions.add(_normalize_formula(conclusion))
                if semantic_verified:
                    semantic_prefix[step["id"]] = conclusion
                if bool(rule_result.get("verified")):
                    rule_prefix[step["id"]] = conclusion

        if final_index is not None and isinstance(steps[final_index], Mapping):
            final_step = steps[final_index]
            result["final_answer_index"] = final_index
            result["final_answer_id"] = final_step.get("id")
            parsed_answer = match_answer_to_option(
                final_step.get("id", ""),
                dict(problem),
                final_step.get("conclusion", ""),
            )
            result["parsed_answer"] = parsed_answer
            result["answer_correct"] = parsed_answer == str(problem.get("answer", "")).strip().upper()

        result["schema_valid"] = all(bool(action.get("schema_valid")) for action in actions)
        if result["schema_valid"]:
            actions.extend(self._format_scaffold_actions(parsed))
        result["actions"] = actions
        return result


# 统计四个离散 process reward 档位的数量和比例。
def reward_distribution(actions: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    total = max(1, len(actions))
    counts = Counter(round(float(action.get("raw_reward", 0.0) or 0.0), 2) for action in actions)
    return {
        f"{level:.2f}": {
            "count": int(counts.get(level, 0)),
            "fraction": float(counts.get(level, 0) / total),
        }
        for level in REWARD_LEVELS
    }


# 判断哪些 response 可参与终局奖励的组内归一化。
def outcome_eligibility(
    scored: Sequence[Mapping[str, Any]],
    *,
    require_valid_response: bool,
) -> list[bool]:
    if require_valid_response:
        return [
            bool(item.get("parse_success"))
            and bool(item.get("summary_tag_present"))
            and bool(item.get("schema_valid"))
            for item in scored
        ]
    return [True for _ in scored]


# 将终局优势仅保留给格式和 schema 合格的 response。
def gate_outcome_advantages(
    scored: Sequence[Mapping[str, Any]],
    outcome_advantages: Sequence[float],
    *,
    require_valid_response: bool,
) -> tuple[list[float], list[bool]]:
    if len(scored) != len(outcome_advantages):
        raise ValueError(
            "scored responses and outcome advantages must have the same length: "
            f"{len(scored)} != {len(outcome_advantages)}"
        )

    eligible = outcome_eligibility(
        scored,
        require_valid_response=require_valid_response,
    )

    effective = [
        float(advantage) if is_eligible else 0.0
        for advantage, is_eligible in zip(outcome_advantages, eligible)
    ]
    return effective, eligible


# 汇总每个 canonical rule bucket 的奖励与优势统计。
def per_rule_bucket_summary(
    actions: Sequence[Mapping[str, Any]],
    *,
    baselines: RuleEmaBaseline | None = None,
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for action in actions:
        grouped[str(action.get("rule_bucket") or INVALID_RULE_BUCKET)].append(action)
    result: dict[str, dict[str, float | int]] = {}
    for bucket, items in sorted(grouped.items()):
        raw_rewards = [float(item.get("raw_reward", 0.0) or 0.0) for item in items]
        advantages = [
            float(item.get("process_advantage", 0.0) or 0.0)
            for item in items
            if "process_advantage" in item
        ]
        entry: dict[str, float | int] = {
            "count": len(items),
            "raw_reward_mean": sum(raw_rewards) / max(1, len(raw_rewards)),
            "advantage_mean": sum(advantages) / max(1, len(advantages)) if advantages else 0.0,
        }
        if baselines is not None:
            entry["baseline"] = baselines.value(bucket)
        result[bucket] = entry
    return result


# 聚合一批 response 的解析、验证、规则和奖励指标。
def summarize_scored_responses(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    response_count = max(1, len(scored))
    actions = [action for item in scored for action in item.get("actions", [])]
    schema_actions = [action for action in actions if action.get("kind") == "step"]
    process_actions = [action for action in actions if action.get("kind") != "format_scaffold"]
    format_actions = [action for action in actions if action.get("kind") == "format_scaffold"]
    step_denominator = max(1, len(schema_actions))
    semantic_verified = sum(bool(action.get("semantic_verified")) for action in schema_actions)
    rule_recognized = sum(bool(action.get("rule_recognized")) for action in schema_actions)
    rule_valid = sum(bool(action.get("rule_application_valid")) for action in schema_actions)
    rule_grounded = sum(bool(action.get("rule_grounded_verified")) for action in schema_actions)
    nontrivial = sum(bool(action.get("nontrivial_verified_progress")) for action in schema_actions)
    semantic_rule_invalid = sum(
        bool(action.get("semantic_verified_but_rule_invalid")) for action in schema_actions
    )
    return {
        "parse_success_rate": sum(bool(item.get("parse_success")) for item in scored) / response_count,
        "schema_valid_rate": sum(bool(action.get("schema_valid")) for action in schema_actions) / step_denominator,
        "z3_verified_step_fraction": semantic_verified / step_denominator,
        "rule_recognized_step_fraction": rule_recognized / step_denominator,
        "rule_application_valid_step_fraction": rule_valid / step_denominator,
        "rule_grounded_step_fraction": rule_grounded / step_denominator,
        "nontrivial_verified_progress_rate": nontrivial / step_denominator,
        "semantic_verified_but_rule_invalid_fraction": semantic_rule_invalid / step_denominator,
        "answer_correct_rate": sum(bool(item.get("answer_correct")) for item in scored) / response_count,
        "num_actions": len(actions),
        "num_step_actions": len(schema_actions),
        "num_format_scaffold_actions": len(format_actions),
        "format_scaffold_response_rate": sum(
            any(action.get("kind") == "format_scaffold" for action in item.get("actions", []))
            for item in scored
        )
        / response_count,
        "reward_distribution": reward_distribution(process_actions),
    }
