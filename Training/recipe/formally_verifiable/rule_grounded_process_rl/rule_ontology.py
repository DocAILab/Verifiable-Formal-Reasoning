"""Closed inference-rule ontology shared by prompts, data, and verifiers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


IMPLICATION_ELIMINATION = "IMPLICATION_ELIMINATION"
MODUS_TOLLENS = "MODUS_TOLLENS"
EXCLUSIVE_DISJUNCTION_INTRODUCTION = "EXCLUSIVE_DISJUNCTION_INTRODUCTION"
EXCLUSIVE_DISJUNCTION_ELIMINATION = "EXCLUSIVE_DISJUNCTION_ELIMINATION"
DISJUNCTIVE_SYLLOGISM = "DISJUNCTIVE_SYLLOGISM"
CONJUNCTION_INTRODUCTION = "CONJUNCTION_INTRODUCTION"
CONJUNCTION_ELIMINATION = "CONJUNCTION_ELIMINATION"
UNIVERSAL_ELIMINATION = "UNIVERSAL_ELIMINATION"
GOAL_BINDING = "GOAL_BINDING"


CANONICAL_RULES: Tuple[str, ...] = (
    IMPLICATION_ELIMINATION,
    MODUS_TOLLENS,
    EXCLUSIVE_DISJUNCTION_INTRODUCTION,
    EXCLUSIVE_DISJUNCTION_ELIMINATION,
    DISJUNCTIVE_SYLLOGISM,
    CONJUNCTION_INTRODUCTION,
    CONJUNCTION_ELIMINATION,
    UNIVERSAL_ELIMINATION,
    GOAL_BINDING,
)


RULE_ALIASES: Dict[str, str] = {
    "mp": IMPLICATION_ELIMINATION,
    "modus_ponens": IMPLICATION_ELIMINATION,
    "implication_elimination": IMPLICATION_ELIMINATION,
    "implies_elimination": IMPLICATION_ELIMINATION,
    "mt": MODUS_TOLLENS,
    "modus_tollens": MODUS_TOLLENS,
    "xor_introduction": EXCLUSIVE_DISJUNCTION_INTRODUCTION,
    "exclusive_or_introduction": EXCLUSIVE_DISJUNCTION_INTRODUCTION,
    "exclusive_disjunction_introduction": EXCLUSIVE_DISJUNCTION_INTRODUCTION,
    "xor_elimination": EXCLUSIVE_DISJUNCTION_ELIMINATION,
    "exclusive_or_elimination": EXCLUSIVE_DISJUNCTION_ELIMINATION,
    "exclusive_disjunction_elimination": EXCLUSIVE_DISJUNCTION_ELIMINATION,
    "ds": DISJUNCTIVE_SYLLOGISM,
    "disjunctive_syllogism": DISJUNCTIVE_SYLLOGISM,
    "and_introduction": CONJUNCTION_INTRODUCTION,
    "conjunction_introduction": CONJUNCTION_INTRODUCTION,
    "and_elimination": CONJUNCTION_ELIMINATION,
    "conjunction_elimination": CONJUNCTION_ELIMINATION,
    "forall_elimination": UNIVERSAL_ELIMINATION,
    "universal_elimination": UNIVERSAL_ELIMINATION,
    "universal_instantiation": UNIVERSAL_ELIMINATION,
    "final": GOAL_BINDING,
    "final_answer": GOAL_BINDING,
    "goal_binding": GOAL_BINDING,
}


# 规范化模型声明的规则名称以便本体映射。
def normalize_rule_token(rule: str) -> str:
    return rule.strip().lower().replace("-", "_").replace(" ", "_")


# 将规则别名映射到 canonical rule ontology。
def canonicalize_rule(rule: Any) -> Optional[str]:
    """Map legacy spellings to one canonical enum value."""
    if not isinstance(rule, str):
        return None
    normalized = normalize_rule_token(rule)
    if rule.strip().upper() in CANONICAL_RULES:
        return rule.strip().upper()
    return RULE_ALIASES.get(normalized)


RULE_PROMPT_GUIDANCE = """Use exactly one of these canonical rule names:
- IMPLICATION_ELIMINATION: from A -> B and support for A, derive the entire consequent B.
- MODUS_TOLLENS: from A -> B and support for not B, derive not A (including a justified component of a compound A).
- EXCLUSIVE_DISJUNCTION_INTRODUCTION: from A and not B, or from not A and B, derive A XOR B.
- EXCLUSIVE_DISJUNCTION_ELIMINATION: from A XOR B plus one known branch or its negation, derive the forced other branch or its negation.
- DISJUNCTIVE_SYLLOGISM: from A OR B and not A derive B, or from A OR B and not B derive A.
- CONJUNCTION_INTRODUCTION: from A and B derive A AND B.
- CONJUNCTION_ELIMINATION: from A AND B derive A or derive B.
- UNIVERSAL_ELIMINATION: instantiate a universally quantified formula for one concrete entity.
- GOAL_BINDING: only for a final_answer action whose id and conclusion exactly match an answer option.

The rule must describe the operation that produces the declared conclusion. If an implication first activates a compound consequent and the same action then selects one XOR/OR/AND branch, use the corresponding elimination rule, not IMPLICATION_ELIMINATION. Do not invent rule names, aliases, abbreviations, or alternative capitalization."""
