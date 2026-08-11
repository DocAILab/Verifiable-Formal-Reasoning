"""Shared parser for full structured generation responses."""

from __future__ import annotations

import json
import re
from typing import Any

from recipe.formally_verifiable.rule_grounded_process_rl.structured_prompt import fol_infix_to_prefix


# 提取 response 中 think 与 summary 标签内容。
def extract_tags(text: str, tag_name: str) -> str | None:
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


# 严格解析模型输出并返回 summary、答案和错误信息。
def parse_model_output(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "think": None,
        "summary": None,
        "raw": text,
        "parse_error": None,
        "summary_tag_present": False,
    }

    think = extract_tags(text, "think")
    if think is None:
        summary_match = re.search(r"<summary>", text, re.IGNORECASE)
        think = text[: summary_match.start()].strip() if summary_match else text.strip()
    result["think"] = think

    summary_text = extract_tags(text, "summary")
    if summary_text is None:
        summary_text = text
    else:
        result["summary_tag_present"] = True

    code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", summary_text, re.DOTALL)
    json_text = code_block_match.group(1).strip() if code_block_match else summary_text

    start_idx = json_text.find("[")
    end_idx = json_text.rfind("]")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        result["parse_error"] = "No JSON array found inside <summary>"
        return result

    try:
        summary = json.loads(json_text[start_idx : end_idx + 1])
    except json.JSONDecodeError as exc:
        result["parse_error"] = f"JSON decode error: {exc}"
        return result

    if not isinstance(summary, list):
        result["parse_error"] = "Summary is not a JSON array"
        return result

    required_fields = {"id", "dependencies", "conclusion", "rule"}
    for index, step in enumerate(summary):
        if not isinstance(step, dict):
            result["parse_error"] = f"Step {index} is not a dict"
            return result
        missing = required_fields - set(step.keys())
        if missing:
            result["parse_error"] = f"Step {index} missing fields: {sorted(missing)}"
            return result
        if not isinstance(step["dependencies"], list):
            result["parse_error"] = f"Step {index} dependencies is not a list"
            return result
        if len(step["dependencies"]) >= 5:
            result["parse_error"] = f"Step {index} has too many dependencies (>=5)"
            return result

    result["summary"] = summary
    return result


# 根据终局 step id 和结论匹配答案选项。
def match_answer_to_option(
    answer_id: Any,
    problem: dict[str, Any],
    conclusion: Any = None,
) -> str | None:
    answer_id_text = answer_id if isinstance(answer_id, str) else ""
    conclusion_text = conclusion if isinstance(conclusion, str) else ""
    answer_id_lower = answer_id_text.lower()

    option_map: dict[str, str] = {}
    for option in problem.get("options", []):
        if ")" in option:
            letter, text = option.split(")", 1)
            option_map[letter.strip()] = text.strip().lower()

    if "true" in answer_id_lower:
        return next((letter for letter, text in option_map.items() if "true" in text), None)
    if "false" in answer_id_lower:
        return next((letter for letter, text in option_map.items() if "false" in text), None)
    if "uncertain" in answer_id_lower:
        return next((letter for letter, text in option_map.items() if "uncertain" in text), None)
    if answer_id_text.upper() in option_map:
        return answer_id_text.upper()

    if conclusion_text:
        normalized = " ".join(conclusion_text.split())
        conclusion_fol = fol_infix_to_prefix(str(problem.get("conclusion_fol", "")).strip())
        has_negation = (
            "卢" in conclusion_text
            or "¬" in conclusion_text
            or "not" in conclusion_text.lower()
        )
        for letter, text in option_map.items():
            if "false" in text and has_negation:
                expected = f"¬({conclusion_fol})"
                if normalized == " ".join(expected.split()):
                    return letter
            if "true" in text and not has_negation:
                if normalized == " ".join(conclusion_fol.split()):
                    return letter
            if "uncertain" in text and not has_negation:
                if normalized == " ".join(conclusion_fol.split()):
                    return letter
    return None
