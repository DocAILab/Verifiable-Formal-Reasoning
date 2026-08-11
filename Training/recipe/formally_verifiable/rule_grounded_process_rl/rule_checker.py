"""Rule-grounded verification for ProverQA-style derivation steps.

The existing Z3 verifier answers whether the declared dependencies entail the
conclusion.  This module adds a separate question: does that entailment have
the shape of the inference rule claimed by the model?
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from recipe.formally_verifiable.common.fol_converter import (
    ASTNode,
    AppNode,
    BinOpNode,
    ConstNode,
    FOLParser,
    NotNode,
    QuantNode,
    VarNode,
    collect_bound_vars,
    tokenize_fol,
)
from recipe.formally_verifiable.rule_grounded_process_rl.rule_ontology import (
    CANONICAL_RULES,
    CONJUNCTION_ELIMINATION,
    CONJUNCTION_INTRODUCTION,
    DISJUNCTIVE_SYLLOGISM,
    EXCLUSIVE_DISJUNCTION_ELIMINATION,
    EXCLUSIVE_DISJUNCTION_INTRODUCTION,
    GOAL_BINDING,
    IMPLICATION_ELIMINATION,
    MODUS_TOLLENS,
    UNIVERSAL_ELIMINATION,
    canonicalize_rule,
)
from recipe.formally_verifiable.common.verifier.z3_verifier import Z3Verifier


SUPPORTED_RULES: Tuple[str, ...] = CANONICAL_RULES

RULE_HANDLERS = {
    IMPLICATION_ELIMINATION: "_check_modus_ponens",
    MODUS_TOLLENS: "_check_modus_tollens",
    EXCLUSIVE_DISJUNCTION_INTRODUCTION: "_check_xor_introduction",
    EXCLUSIVE_DISJUNCTION_ELIMINATION: "_check_xor_elimination",
    DISJUNCTIVE_SYLLOGISM: "_check_disjunctive_syllogism",
    CONJUNCTION_INTRODUCTION: "_check_conjunction_introduction",
    CONJUNCTION_ELIMINATION: "_check_conjunction_elimination",
    UNIVERSAL_ELIMINATION: "_check_universal_instantiation",
    GOAL_BINDING: "_check_final_answer",
}


@dataclass(frozen=True)
class Formula:
    """A parsed formula together with the dependency that supplied it."""

    dependency_id: str
    text: str
    ast: ASTNode


@dataclass(frozen=True)
class RuleMatch:
    """Successful match of one inference-rule schema."""

    principal_dependency: Optional[str]
    used_dependencies: Tuple[str, ...]
    mode: str
    substitution: Mapping[str, str]


# 将 FOL 文本解析为 RuleChecker 使用的结构化 AST。
def parse_fol_ast(text: str) -> ASTNode:
    tokens = tokenize_fol(text)
    parser = FOLParser(tokens, collect_bound_vars(tokens))
    node = parser.parse_expr()
    if parser.peek()[0] != "EOF":
        remaining = " ".join(token[1] for token in parser.tokens[parser.pos :])
        raise ValueError(f"Unexpected trailing tokens: {remaining}")
    return node


# 将结构化 FOL AST 还原为规范文本。
def ast_to_fol(node: ASTNode) -> str:
    """Serialize an AST conservatively; parentheses preserve the parse tree."""
    if isinstance(node, VarNode):
        return node.name
    if isinstance(node, ConstNode):
        return node.name
    if isinstance(node, AppNode):
        return " ".join([node.func, *(ast_to_fol(arg) for arg in node.args)])
    if isinstance(node, NotNode):
        return "\u00ac(" + ast_to_fol(node.body) + ")"
    if isinstance(node, BinOpNode):
        operators = {
            "and": "\u2227",
            "or": "\u2228",
            "implies": "\u2192",
            "xor": "\u2295",
        }
        return f"({ast_to_fol(node.left)} {operators[node.op]} {ast_to_fol(node.right)})"
    if isinstance(node, QuantNode):
        quantifier = "\u2200" if node.quant == "forall" else "\u2203"
        return f"{quantifier}{node.var} ({ast_to_fol(node.body)})"
    raise TypeError(f"Unsupported AST node: {type(node).__name__}")


# 在 AST 中按映射替换变量或项。
def _substitute(node: ASTNode, substitutions: Mapping[str, ASTNode]) -> ASTNode:
    if isinstance(node, VarNode):
        return substitutions.get(node.name, VarNode(node.name))
    if isinstance(node, ConstNode):
        return ConstNode(node.name)
    if isinstance(node, AppNode):
        return AppNode(node.func, [_substitute(arg, substitutions) for arg in node.args])
    if isinstance(node, NotNode):
        return NotNode(_substitute(node.body, substitutions))
    if isinstance(node, BinOpNode):
        return BinOpNode(
            node.op,
            _substitute(node.left, substitutions),
            _substitute(node.right, substitutions),
        )
    if isinstance(node, QuantNode):
        nested = dict(substitutions)
        nested.pop(node.var, None)
        return QuantNode(node.quant, node.var, _substitute(node.body, nested))
    raise TypeError(f"Unsupported AST node: {type(node).__name__}")


# 收集公式中可用于量词实例化的 ground terms。
def _collect_ground_terms(node: ASTNode) -> List[ASTNode]:
    """Collect constants and compound terms used as predicate arguments."""
    terms: List[ASTNode] = []

    # 递归访问当前节点并收集其可达依赖。
    def visit(current: ASTNode, inside_term: bool = False) -> None:
        if isinstance(current, ConstNode):
            if inside_term:
                terms.append(current)
            return
        if isinstance(current, VarNode):
            return
        if isinstance(current, AppNode):
            for arg in current.args:
                visit(arg, True)
            return
        if isinstance(current, NotNode):
            visit(current.body)
            return
        if isinstance(current, BinOpNode):
            visit(current.left)
            visit(current.right)
            return
        if isinstance(current, QuantNode):
            visit(current.body)

    visit(node)
    unique: Dict[str, ASTNode] = {}
    for term in terms:
        unique.setdefault(ast_to_fol(term), term)
    return list(unique.values())


# 剥离外层全称量词并返回变量与量词体。
def _unwrap_forall(node: ASTNode) -> Tuple[List[str], ASTNode]:
    variables: List[str] = []
    while isinstance(node, QuantNode) and node.quant == "forall":
        variables.append(node.var)
        node = node.body
    return variables, node


# 构造或消去一层逻辑否定。
def _not(node: ASTNode) -> NotNode:
    return NotNode(node)


class RuleChecker:
    """Check semantic entailment and the model-declared inference rule."""

    # 初始化规则检查器、Z3 验证器和超时设置。
    def __init__(self, timeout_ms: int = 5000, max_groundings: int = 128):
        self.timeout_ms = timeout_ms
        self.max_groundings = max_groundings
        self.semantic_verifier = Z3Verifier(timeout_ms=timeout_ms)

    # 验证 claimed rule 是否被识别且真实匹配依赖到结论的推理形态。
    def check_step(
        self,
        step: Mapping[str, Any],
        premises_fol: Mapping[str, str],
        all_steps_fol: Mapping[str, str],
        answer_options: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Check one step without mutating the accepted proof prefix."""
        claimed_rule = step.get("rule")
        canonical_rule = canonicalize_rule(claimed_rule)
        dependencies = step.get("dependencies", [])
        conclusion_text = step.get("conclusion", "")

        base_result: Dict[str, Any] = {
            "verified": False,
            "semantic_verified": False,
            "rule_recognized": canonical_rule is not None,
            "rule_application_valid": False,
            "claimed_rule": claimed_rule,
            "canonical_rule": canonical_rule,
            "error": None,
            "details": {},
        }
        if not isinstance(dependencies, list) or not all(
            isinstance(dep, str) for dep in dependencies
        ):
            base_result["error"] = "dependencies must be a list of IDs"
            return base_result
        if not isinstance(conclusion_text, str) or not conclusion_text.strip():
            base_result["error"] = "conclusion must be a non-empty FOL string"
            return base_result
        formulas: List[Formula] = []
        missing: List[str] = []
        try:
            for dependency_id in dependencies:
                if dependency_id in premises_fol:
                    text = premises_fol[dependency_id]
                elif dependency_id in all_steps_fol:
                    text = all_steps_fol[dependency_id]
                else:
                    missing.append(dependency_id)
                    continue
                formulas.append(Formula(dependency_id, text, parse_fol_ast(text)))
            conclusion = parse_fol_ast(conclusion_text)
        except Exception as exc:
            base_result["error"] = f"formula parse failed: {exc}"
            return base_result
        if missing:
            base_result["error"] = f"missing dependencies: {missing}"
            base_result["details"] = {"missing_dependencies": missing}
            return base_result

        semantic = self.semantic_verifier.verify_step(
            dependencies=dependencies,
            conclusion=conclusion_text,
            premises_fol=dict(premises_fol),
            all_steps_fol=dict(all_steps_fol),
        )
        base_result["semantic_verified"] = bool(semantic["verified"])
        base_result["details"]["semantic"] = semantic

        # Keep semantic verification independent from rule recognition.  This
        # lets reward code distinguish "Z3-valid but rule-invalid" steps.
        if canonical_rule is None:
            rule_error = f"unknown rule: {claimed_rule!r}"
            base_result["details"]["rule_error"] = rule_error
            base_result["error"] = (
                semantic.get("error") if not semantic["verified"] else rule_error
            )
            return base_result

        matcher = getattr(self, RULE_HANDLERS[canonical_rule])
        match = matcher(
            formulas=formulas,
            conclusion=conclusion,
            step=step,
            answer_options=answer_options or [],
        )
        rule_valid = match is not None
        base_result["rule_application_valid"] = rule_valid
        if match is not None:
            used = list(match.used_dependencies)
            base_result["details"].update(
                {
                    "principal_dependency": match.principal_dependency,
                    "used_dependencies": used,
                    "unused_dependencies": [
                        dep for dep in dependencies if dep not in match.used_dependencies
                    ],
                    "match_mode": match.mode,
                    "substitution": dict(match.substitution),
                }
            )
        else:
            base_result["details"]["rule_error"] = (
                f"dependencies and conclusion do not match {canonical_rule}"
            )

        base_result["verified"] = bool(semantic["verified"] and rule_valid)
        if not semantic["verified"]:
            base_result["error"] = semantic.get("error") or "semantic entailment failed"
        elif not rule_valid:
            base_result["error"] = base_result["details"]["rule_error"]
        return base_result

    # 批量检查多个步骤的 rule application validity。
    def batch_check(
        self,
        steps: Sequence[Mapping[str, Any]],
        premises_fol: Mapping[str, str],
        answer_options: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Check steps sequentially and append only fully verified steps."""
        accepted: Dict[str, str] = {}
        results: List[Dict[str, Any]] = []
        for step in steps:
            result = self.check_step(step, premises_fol, accepted, answer_options)
            result["step_id"] = step.get("id")
            results.append(result)
            if result["verified"] and isinstance(step.get("id"), str):
                accepted[step["id"]] = step["conclusion"]
        return results

    # 判断文本前提是否蕴含文本结论。
    @lru_cache(maxsize=65536)
    def _entails_text(self, premises: Tuple[str, ...], conclusion: str) -> bool:
        premise_map = {f"p{i}": text for i, text in enumerate(premises)}
        result = self.semantic_verifier.verify_step(
            dependencies=list(premise_map),
            conclusion=conclusion,
            premises_fol=premise_map,
            all_steps_fol={},
        )
        return bool(result["verified"])

    # 判断 AST 前提是否蕴含 AST 结论。
    def _entails(self, premises: Iterable[ASTNode], conclusion: ASTNode) -> bool:
        premise_texts = tuple(sorted(ast_to_fol(node) for node in premises))
        return self._entails_text(premise_texts, ast_to_fol(conclusion))

    # 检查两个 AST 公式是否逻辑等价。
    def _equivalent(self, left: ASTNode, right: ASTNode) -> bool:
        return self._entails([left], right) and self._entails([right], left)

    # 枚举量词公式在候选 ground terms 上的实例。
    def _ground_instances(
        self, node: ASTNode, context: Sequence[ASTNode]
    ) -> Iterator[Tuple[ASTNode, Dict[str, str]]]:
        variables, body = _unwrap_forall(node)
        if not variables:
            yield body, {}
            return

        terms: Dict[str, ASTNode] = {}
        for context_node in context:
            for term in _collect_ground_terms(context_node):
                terms.setdefault(ast_to_fol(term), term)
        if not terms:
            return

        emitted = 0
        term_values = list(terms.values())
        for values in product(term_values, repeat=len(variables)):
            substitutions = dict(zip(variables, values))
            yield (
                _substitute(body, substitutions),
                {name: ast_to_fol(value) for name, value in substitutions.items()},
            )
            emitted += 1
            if emitted >= self.max_groundings:
                break

    # 枚举依赖中可用的蕴含公式及其实例。
    def _iter_implications(
        self, formulas: Sequence[Formula], conclusion: ASTNode
    ) -> Iterator[Tuple[int, Formula, BinOpNode, Dict[str, str]]]:
        context = [formula.ast for formula in formulas] + [conclusion]
        for index, formula in enumerate(formulas):
            for body, substitution in self._ground_instances(formula.ast, context):
                if isinstance(body, BinOpNode) and body.op == "implies":
                    yield index, formula, body, substitution

    # 枚举依赖中可作为连接词规则来源的公式。
    def _iter_connective_sources(
        self,
        formulas: Sequence[Formula],
        conclusion: ASTNode,
        op: str,
        allow_negated: bool = False,
    ) -> Iterator[
        Tuple[int, Formula, ASTNode, bool, str, Dict[str, str]]
    ]:
        """Yield direct or implication-activated principal connectives."""
        context = [formula.ast for formula in formulas] + [conclusion]
        for index, formula in enumerate(formulas):
            supports = [f.ast for i, f in enumerate(formulas) if i != index]
            for body, substitution in self._ground_instances(formula.ast, context):
                connective, polarity = self._extract_connective(body, op, allow_negated)
                if connective is not None:
                    yield index, formula, connective, polarity, "direct", substitution
                    continue
                if isinstance(body, BinOpNode) and body.op == "implies":
                    consequent, polarity = self._extract_connective(
                        body.right, op, allow_negated
                    )
                    if consequent is not None and self._entails(supports, body.left):
                        yield (
                            index,
                            formula,
                            consequent,
                            polarity,
                            "implication_activated",
                            substitution,
                        )

    # 从公式中提取指定逻辑连接词的操作数。
    @staticmethod
    def _extract_connective(
        node: ASTNode, op: str, allow_negated: bool
    ) -> Tuple[Optional[ASTNode], bool]:
        if isinstance(node, BinOpNode) and node.op == op:
            return node, True
        if (
            allow_negated
            and isinstance(node, NotNode)
            and isinstance(node.body, BinOpNode)
            and node.body.op == op
        ):
            return node.body, False
        return None, True

    # 匹配公式模板并维护一致的变量绑定。
    @staticmethod
    def _match(
        principal: Optional[str], formulas: Sequence[Formula], mode: str,
        substitution: Mapping[str, str], used: Optional[Iterable[str]] = None,
    ) -> RuleMatch:
        used_dependencies = tuple(
            dict.fromkeys(used if used is not None else (f.dependency_id for f in formulas))
        )
        return RuleMatch(principal, used_dependencies, mode, substitution)

    # 检查步骤是否构成有效的肯定前件推理。
    def _check_modus_ponens(
        self, formulas: Sequence[Formula], conclusion: ASTNode, **_: Any
    ) -> Optional[RuleMatch]:
        for index, principal, implication, substitution in self._iter_implications(
            formulas, conclusion
        ):
            supports = [f.ast for i, f in enumerate(formulas) if i != index]
            if supports and self._entails(supports, implication.left) and self._equivalent(
                conclusion, implication.right
            ):
                return self._match(
                    principal.dependency_id,
                    formulas,
                    "implication_elimination",
                    substitution,
                )
        return None

    # 检查步骤是否构成有效的否定后件推理。
    def _check_modus_tollens(
        self, formulas: Sequence[Formula], conclusion: ASTNode, **_: Any
    ) -> Optional[RuleMatch]:
        for index, principal, implication, substitution in self._iter_implications(
            formulas, conclusion
        ):
            supports = [f.ast for i, f in enumerate(formulas) if i != index]
            if not supports or not self._entails(supports, _not(implication.right)):
                continue
            negated_antecedent = _not(implication.left)
            if self._equivalent(conclusion, negated_antecedent):
                return self._match(
                    principal.dependency_id,
                    formulas,
                    "contraposition",
                    substitution,
                )
            # ProverQA often projects one component from a negated compound
            # antecedent.  The principal formula must add information; otherwise
            # arbitrary consequences could be mislabeled as modus tollens.
            if (
                self._entails([*supports, negated_antecedent], conclusion)
                and not self._entails(supports, conclusion)
            ):
                return self._match(
                    principal.dependency_id,
                    formulas,
                    "contraposition_with_compound_projection",
                    substitution,
                )
        return None

    # 检查步骤是否正确应用异或消去规则。
    def _check_xor_elimination(
        self, formulas: Sequence[Formula], conclusion: ASTNode, **_: Any
    ) -> Optional[RuleMatch]:
        for index, principal, source, polarity, mode, substitution in self._iter_connective_sources(
            formulas, conclusion, "xor", allow_negated=True
        ):
            assert isinstance(source, BinOpNode)
            supports = [f.ast for i, f in enumerate(formulas) if i != index]
            left, right = source.left, source.right
            if polarity:
                cases = (
                    (left, _not(right)),
                    (_not(left), right),
                    (right, _not(left)),
                    (_not(right), left),
                )
            else:
                cases = (
                    (left, right),
                    (_not(left), _not(right)),
                    (right, left),
                    (_not(right), _not(left)),
                )
            for discriminator, expected in cases:
                if self._entails(supports, discriminator) and self._equivalent(
                    conclusion, expected
                ):
                    return self._match(
                        principal.dependency_id,
                        formulas,
                        f"{mode}_{'xor_true' if polarity else 'xor_false'}",
                        substitution,
                    )
        return None

    # 检查步骤是否正确构造异或结论。
    def _check_xor_introduction(
        self, formulas: Sequence[Formula], conclusion: ASTNode, **_: Any
    ) -> Optional[RuleMatch]:
        if not isinstance(conclusion, BinOpNode) or conclusion.op != "xor":
            return None
        supports = [formula.ast for formula in formulas]
        cases = (
            (conclusion.left, _not(conclusion.right)),
            (_not(conclusion.left), conclusion.right),
        )
        for left_support, right_support in cases:
            if self._entails(supports, left_support) and self._entails(
                supports, right_support
            ):
                return self._match(
                    None,
                    formulas,
                    "xor_introduction",
                    {},
                )
        return None

    # 检查步骤是否构成有效的析取三段论。
    def _check_disjunctive_syllogism(
        self, formulas: Sequence[Formula], conclusion: ASTNode, **_: Any
    ) -> Optional[RuleMatch]:
        for index, principal, source, _, mode, substitution in self._iter_connective_sources(
            formulas, conclusion, "or"
        ):
            assert isinstance(source, BinOpNode)
            supports = [f.ast for i, f in enumerate(formulas) if i != index]
            cases = ((_not(source.left), source.right), (_not(source.right), source.left))
            for eliminated, expected in cases:
                if self._entails(supports, eliminated) and self._equivalent(
                    conclusion, expected
                ):
                    return self._match(
                        principal.dependency_id,
                        formulas,
                        mode,
                        substitution,
                    )
        return None

    # 检查步骤是否由依赖正确引入合取。
    def _check_conjunction_introduction(
        self, formulas: Sequence[Formula], conclusion: ASTNode, **_: Any
    ) -> Optional[RuleMatch]:
        if not isinstance(conclusion, BinOpNode) or conclusion.op != "and":
            return None
        for left_index, left_formula in enumerate(formulas):
            if not self._equivalent(left_formula.ast, conclusion.left):
                continue
            for right_index, right_formula in enumerate(formulas):
                if left_index == right_index:
                    continue
                if self._equivalent(right_formula.ast, conclusion.right):
                    return self._match(
                        None,
                        formulas,
                        "and_introduction",
                        {},
                        (left_formula.dependency_id, right_formula.dependency_id),
                    )
        return None

    # 检查步骤是否从合取中正确消去分量。
    def _check_conjunction_elimination(
        self, formulas: Sequence[Formula], conclusion: ASTNode, **_: Any
    ) -> Optional[RuleMatch]:
        for _, formula, source, _, mode, substitution in self._iter_connective_sources(
            formulas, conclusion, "and"
        ):
            assert isinstance(source, BinOpNode)
            if self._equivalent(conclusion, source.left) or self._equivalent(
                conclusion, source.right
            ):
                used = None if mode == "implication_activated" else (formula.dependency_id,)
                return self._match(
                    formula.dependency_id,
                    formulas,
                    f"{mode}_and_elimination",
                    substitution,
                    used,
                )
        return None

    # 检查步骤是否为有效的全称量词实例化。
    def _check_universal_instantiation(
        self, formulas: Sequence[Formula], conclusion: ASTNode, **_: Any
    ) -> Optional[RuleMatch]:
        context = [formula.ast for formula in formulas] + [conclusion]
        for formula in formulas:
            variables, _ = _unwrap_forall(formula.ast)
            if not variables:
                continue
            for instance, substitution in self._ground_instances(formula.ast, context):
                if self._equivalent(conclusion, instance):
                    return self._match(
                        formula.dependency_id,
                        formulas,
                        "forall_elimination",
                        substitution,
                        (formula.dependency_id,),
                    )
        return None

    # 检查 GOAL_BINDING 是否严格匹配答案 id 与结论。
    def _check_final_answer(
        self,
        formulas: Sequence[Formula],
        conclusion: ASTNode,
        step: Mapping[str, Any],
        answer_options: Sequence[Mapping[str, Any]],
        **_: Any,
    ) -> Optional[RuleMatch]:
        step_id = step.get("id")
        if not isinstance(step_id, str) or not step_id.startswith("h_goal_"):
            return None
        matching_options = [option for option in answer_options if option.get("answer_id") == step_id]
        if not matching_options:
            return None
        option_formula = matching_options[0].get("formal")
        if not isinstance(option_formula, str):
            return None
        try:
            target = parse_fol_ast(option_formula)
        except Exception:
            return None
        if not self._equivalent(conclusion, target):
            return None
        return self._match(
            None,
            formulas,
            "answer_option_binding",
            {},
        )
