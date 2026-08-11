"""Shared prompt helpers for full structured generation."""

from __future__ import annotations

import re
from typing import Any

from recipe.formally_verifiable.rule_grounded_process_rl.rule_ontology import RULE_PROMPT_GUIDANCE


SYSTEM_PROMPT = f"""You are a math reasoner. For each problem, you will receive a set of formal premises and a multiple-choice logical reasoning question.

Answer with exactly two parts:
1. Put brief natural-language reasoning inside <think>...</think>.
2. Put a JSON array of structured formal intermediate steps inside <summary>...</summary>.

Each JSON object in <summary> must have exactly these fields:
- id: a unique identifier for this step.
- dependencies: a list of premise ids or earlier step ids.
- conclusion: exactly one formal formula.
- rule: the single inference rule applied, chosen from the canonical rule options below.

Canonical rule options:
{RULE_PROMPT_GUIDANCE}

Use GOAL_BINDING for the final JSON object whose id is one of h_goal_true, h_goal_false, or h_goal_uncertain.

Critical requirements:
- Use "s1", "s2", "s3", etc. for intermediate steps.
- The last JSON object must be the final answer.
- The final answer id must be exactly one of the option ids shown in the problem, such as h_goal_true, h_goal_false, or h_goal_uncertain.
- The final answer conclusion must match the formal statement of that option.
- The JSON array must be valid and parsable.
- Each step must use fewer than 5 dependencies.
- Do not output any additional text after </summary>."""


# 将数据中的中缀 FOL 运算符转换为统一前缀格式。
def fol_infix_to_prefix(fol: str) -> str:
    pattern = re.compile(r"(\w+)\(([^()]+)\)")

    # 把单个正则匹配到的二元中缀表达式改写为前缀表达式。
    def replacer(match: re.Match[str]) -> str:
        func = match.group(1)
        args = " ".join(arg.strip() for arg in match.group(2).split(","))
        return f"{func} {args}"

    previous = None
    result = fol
    while previous != result:
        previous = result
        result = pattern.sub(replacer, result)
    return result


# 把问题前提、查询和答案选项组装为用户提示。
def build_user_prompt(problem: dict[str, Any]) -> str:
    lines: list[str] = ["Context:"]

    nl2fol = problem.get("nl2fol", {}) or {}
    for index, (nl_text, fol_text) in enumerate(nl2fol.items(), start=1):
        formal = fol_infix_to_prefix(str(fol_text).strip())
        lines.append(
            f"{index}. {str(nl_text).strip()} Formal statement: 'h{index} : {formal}'."
        )

    lines.extend(["", f"Question: {problem.get('question', '')}", "", "Options:"])

    conclusion_fol = fol_infix_to_prefix(str(problem.get("conclusion_fol", "")).strip())
    option_texts: dict[str, str] = {}
    for option in problem.get("options", []):
        if ")" in option:
            letter, text = option.split(")", 1)
            option_texts[letter.strip()] = text.strip().lower()

    option_fols: dict[str, str] = {}
    for letter, text in option_texts.items():
        if "true" in text:
            option_fols[letter] = f"h_goal_true: {conclusion_fol}"
        elif "false" in text:
            option_fols[letter] = f"h_goal_false: ¬({conclusion_fol})"
        elif "uncertain" in text:
            option_fols[letter] = f"h_goal_uncertain: {conclusion_fol}"

    for letter in ("A", "B", "C"):
        if letter not in option_texts:
            continue
        fol_line = option_fols.get(letter, "")
        option_id, _, formal_statement = fol_line.partition(":")
        lines.append(
            f"{letter}) {option_texts[letter]}. "
            f"Answer id: '{option_id.strip()}'. "
            f"Formal statement: '{formal_statement.strip()}'."
        )

    lines.extend(["", "The correct option is:"])
    return "\n".join(lines)


# 构造训练和推理共用的 system/user 消息列表。
def build_messages(problem: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(problem)},
    ]


# 使用 tokenizer chat template 生成最终模型输入文本。
def build_chat_prompt(tokenizer, problem: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        build_messages(problem),
        tokenize=False,
        add_generation_prompt=True,
    )
