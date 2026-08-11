"""Convert Lean4-style FOL expressions to Z3 Python API."""
import re
from typing import List, Set, Tuple, Union
from z3 import (
    And, Or, Not, Implies, ForAll, Exists, Xor,
    BoolSort, DeclareSort, Const, Function,
    Solver, unsat, is_expr,
)


class FOLToken:
    IDENT = "IDENT"
    FORALL = "FORALL"      # ∀
    EXISTS = "EXISTS"      # ∃
    ARROW = "ARROW"        # →
    AND = "AND"            # ∧
    OR = "OR"              # ∨
    NOT = "NOT"            # ¬
    XOR = "XOR"            # ⊕
    COMMA = "COMMA"        # ,
    LPAREN = "LPAREN"      # (
    RPAREN = "RPAREN"      # )


# 将 FOL 公式切分为解析器可消费的词法 token。
def tokenize_fol(text: str) -> List[Tuple[str, str]]:
    """Tokenize a Lean4-style FOL string."""
    tokens = []
    i = 0
    text = text.strip()
    while i < len(text):
        c = text[i]
        # Skip whitespace
        if c.isspace():
            i += 1
            continue
        # Multi-char operators / Unicode
        if c == '∀':
            tokens.append((FOLToken.FORALL, c)); i += 1; continue
        if c == '∃':
            tokens.append((FOLToken.EXISTS, c)); i += 1; continue
        if c == '→':
            tokens.append((FOLToken.ARROW, c)); i += 1; continue
        if c == '∧':
            tokens.append((FOLToken.AND, c)); i += 1; continue
        if c == '∨':
            tokens.append((FOLToken.OR, c)); i += 1; continue
        if c == '¬':
            tokens.append((FOLToken.NOT, c)); i += 1; continue
        if c == '⊕':
            tokens.append((FOLToken.XOR, c)); i += 1; continue
        if c == ',':
            tokens.append((FOLToken.COMMA, c)); i += 1; continue
        if c == '(':
            tokens.append((FOLToken.LPAREN, c)); i += 1; continue
        if c == ')':
            tokens.append((FOLToken.RPAREN, c)); i += 1; continue
        # Identifier: letters, digits, underscores
        if c.isalpha() or c == '_':
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                j += 1
            tokens.append((FOLToken.IDENT, text[i:j]))
            i = j
            continue
        # Unknown char, skip or raise
        raise ValueError(f"Unexpected character '{c}' at position {i} in: {text}")
    return tokens


# AST node types
class ASTNode:
    pass

class VarNode(ASTNode):
    # 初始化 VarNode 所需状态。
    def __init__(self, name: str):
        self.name = name

class ConstNode(ASTNode):
    # 初始化 ConstNode 所需状态。
    def __init__(self, name: str):
        self.name = name

class AppNode(ASTNode):
    # 初始化 AppNode 所需状态。
    def __init__(self, func: str, args: List[ASTNode]):
        self.func = func
        self.args = args

class NotNode(ASTNode):
    # 初始化 NotNode 所需状态。
    def __init__(self, body: ASTNode):
        self.body = body

class BinOpNode(ASTNode):
    # 初始化 BinOpNode 所需状态。
    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        self.op = op  # 'and', 'or', 'implies', 'xor'
        self.left = left
        self.right = right

class QuantNode(ASTNode):
    # 初始化 QuantNode 所需状态。
    def __init__(self, quant: str, var: str, body: ASTNode):
        self.quant = quant  # 'forall', 'exists'
        self.var = var
        self.body = body


class FOLParser:
    """Recursive descent parser for Lean4-style FOL."""

    # 初始化 FOLParser 所需状态。
    def __init__(self, tokens: List[Tuple[str, str]], bound_vars: Set[str]):
        self.tokens = tokens
        self.pos = 0
        self.bound_vars = bound_vars

    # 查看当前位置的 token，但不推进解析游标。
    def peek(self) -> Tuple[str, str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ("EOF", "")

    # 消费并校验当前位置的 token，然后推进解析游标。
    def consume(self, expected_type: str = None) -> Tuple[str, str]:
        tok = self.peek()
        if expected_type and tok[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {tok}")
        self.pos += 1
        return tok

    # 从当前游标解析一个完整的 FOL 表达式。
    def parse_expr(self) -> ASTNode:
        """Top-level expression."""
        tok = self.peek()
        if tok[0] == FOLToken.FORALL:
            return self.parse_quant()
        if tok[0] == FOLToken.EXISTS:
            return self.parse_quant()
        return self.parse_implies()

    # 解析全称量词或存在量词表达式。
    def parse_quant(self) -> ASTNode:
        tok = self.consume()
        quant = "forall" if tok[0] == FOLToken.FORALL else "exists"
        var_tok = self.consume(FOLToken.IDENT)
        var_name = var_tok[1]
        # Comma after quantifier variable is optional (ProverQA uses ∀x (...))
        if self.peek()[0] == FOLToken.COMMA:
            self.consume()
        self.bound_vars.add(var_name)
        body = self.parse_expr()
        self.bound_vars.discard(var_name)
        return QuantNode(quant, var_name, body)

    # 按右结合规则解析蕴含表达式。
    def parse_implies(self) -> ASTNode:
        left = self.parse_or()
        while self.peek()[0] == FOLToken.ARROW:
            self.consume()
            right = self.parse_or()
            left = BinOpNode("implies", left, right)
        return left

    # 解析析取表达式及其连续操作数。
    def parse_or(self) -> ASTNode:
        left = self.parse_xor()
        while self.peek()[0] == FOLToken.OR:
            self.consume()
            right = self.parse_xor()
            left = BinOpNode("or", left, right)
        return left

    # 解析异或表达式及其连续操作数。
    def parse_xor(self) -> ASTNode:
        left = self.parse_and()
        while self.peek()[0] == FOLToken.XOR:
            self.consume()
            right = self.parse_and()
            left = BinOpNode("xor", left, right)
        return left

    # 解析合取表达式及其连续操作数。
    def parse_and(self) -> ASTNode:
        left = self.parse_not()
        while self.peek()[0] == FOLToken.AND:
            self.consume()
            right = self.parse_not()
            left = BinOpNode("and", left, right)
        return left

    # 解析否定表达式并递归处理其操作数。
    def parse_not(self) -> ASTNode:
        if self.peek()[0] == FOLToken.NOT:
            self.consume()
            body = self.parse_not()
            return NotNode(body)
        return self.parse_app()

    # 解析谓词调用或函数应用。
    def parse_app(self) -> ASTNode:
        primary = self.parse_primary()
        args = []
        # Keep consuming primaries while next token starts a primary
        while True:
            tok = self.peek()
            if tok[0] in (FOLToken.IDENT, FOLToken.LPAREN, FOLToken.NOT,
                          FOLToken.FORALL, FOLToken.EXISTS):
                # But stop if the next token is actually a binary operator at expr level
                # In our grammar, binary operators are only at specific levels.
                # However, a quantifier like ∀x, ... should only appear at expr start.
                # For safety, we don't allow nested quantifiers without parens here.
                if tok[0] in (FOLToken.FORALL, FOLToken.EXISTS):
                    break
                arg = self.parse_primary()
                args.append(arg)
            else:
                break

        if args:
            # The first primary is the function/predicate name
            if isinstance(primary, VarNode):
                return AppNode(primary.name, args)
            elif isinstance(primary, ConstNode):
                return AppNode(primary.name, args)
            else:
                # If primary is complex (e.g., parenthesized expr), it can't be applied
                # This shouldn't happen in valid FOL
                raise ValueError(f"Cannot apply arguments to non-identifier: {primary}")
        return primary

    # 解析括号、原子项或谓词等基础表达式。
    def parse_primary(self) -> ASTNode:
        tok = self.peek()
        if tok[0] == FOLToken.LPAREN:
            self.consume()
            expr = self.parse_expr()
            self.consume(FOLToken.RPAREN)
            return expr
        if tok[0] == FOLToken.IDENT:
            self.consume()
            name = tok[1]
            if name in self.bound_vars:
                return VarNode(name)
            # Heuristic: single lowercase letters are likely variables if not bound
            # But in ProverQA, variables are typically x,y,z,w
            if len(name) == 1 and name.islower():
                return VarNode(name)
            return ConstNode(name)
        raise ValueError(f"Unexpected token in primary: {tok}")


# 收集公式中由量词绑定的变量名。
def collect_bound_vars(tokens: List[Tuple[str, str]]) -> Set[str]:
    """Pre-collect all variables bound by quantifiers."""
    bound = set()
    i = 0
    while i < len(tokens):
        if tokens[i][0] in (FOLToken.FORALL, FOLToken.EXISTS):
            i += 1
            if i < len(tokens) and tokens[i][0] == FOLToken.IDENT:
                bound.add(tokens[i][1])
                i += 1
            continue
        i += 1
    return bound


class FOLToZ3Converter:
    """Converts parsed FOL AST to Z3 expressions."""

    # 初始化 FOL 到 Z3 的符号缓存与转换状态。
    def __init__(self):
        self.Entity = DeclareSort('Entity')
        self.bound_vars: dict = {}
        self.consts: dict = {}
        self.preds: dict = {}
        self.funcs: dict = {}

    # 清空一次公式转换过程中缓存的 Z3 符号。
    def reset(self):
        self.bound_vars = {}
        self.consts = {}
        self.preds = {}
        self.funcs = {}

    # 获取或创建给定名称的 Z3 常量。
    def get_or_create_const(self, name: str):
        if name not in self.consts:
            self.consts[name] = Const(name, self.Entity)
        return self.consts[name]

    # 获取或创建给定名称的 Z3 变量。
    def get_or_create_var(self, name: str):
        if name not in self.bound_vars:
            self.bound_vars[name] = Const(name, self.Entity)
        return self.bound_vars[name]

    # 按名称和元数获取或创建 Z3 谓词。
    def get_or_create_pred(self, name: str, arity: int):
        key = (name, arity)
        if key not in self.preds:
            sorts = [self.Entity] * arity + [BoolSort()]
            self.preds[key] = Function(name, *sorts)
        return self.preds[key]

    # 将解析后的 FOL AST 递归转换为 Z3 表达式。
    def to_z3(self, node: ASTNode) -> is_expr:
        if isinstance(node, VarNode):
            return self.get_or_create_var(node.name)
        if isinstance(node, ConstNode):
            return self.get_or_create_const(node.name)
        if isinstance(node, AppNode):
            pred = self.get_or_create_pred(node.func, len(node.args))
            z3_args = [self.to_z3(a) for a in node.args]
            return pred(*z3_args)
        if isinstance(node, NotNode):
            return Not(self.to_z3(node.body))
        if isinstance(node, BinOpNode):
            left = self.to_z3(node.left)
            right = self.to_z3(node.right)
            if node.op == "and":
                return And(left, right)
            if node.op == "or":
                return Or(left, right)
            if node.op == "implies":
                return Implies(left, right)
            if node.op == "xor":
                return Xor(left, right)
            raise ValueError(f"Unknown binop: {node.op}")
        if isinstance(node, QuantNode):
            var = self.get_or_create_var(node.var)
            body = self.to_z3(node.body)
            if node.quant == "forall":
                return ForAll([var], body)
            if node.quant == "exists":
                return Exists([var], body)
            raise ValueError(f"Unknown quant: {node.quant}")
        raise ValueError(f"Unknown AST node: {node}")

    # 将文本形式的 FOL 公式解析并转换为 Z3 表达式。
    def convert(self, text: str) -> is_expr:
        """Parse and convert a FOL string to Z3 expression."""
        self.reset()
        tokens = tokenize_fol(text)
        bound_vars = collect_bound_vars(tokens)
        parser = FOLParser(tokens, bound_vars)
        ast = parser.parse_expr()
        if parser.peek()[0] != "EOF":
            remaining = " ".join([t[1] for t in parser.tokens[parser.pos:]])
            raise ValueError(f"Unexpected trailing tokens: {remaining}")
        return self.to_z3(ast)


# 检查一组前提是否能够在 Z3 中推出结论。
def verify_implication(premises_z3: List, conclusion_z3, timeout_ms: int = 5000) -> Tuple[bool, str]:
    """
    Check if premises entail conclusion using Z3.
    Returns (verified, message).
    """
    s = Solver()
    s.set("timeout", timeout_ms)
    for p in premises_z3:
        s.add(p)
    s.add(Not(conclusion_z3))
    result = s.check()
    if result == unsat:
        return True, "UNSAT: conclusion follows from premises"
    if result == unsat:
        return True, "UNSAT"
    # Check if result is unknown due to timeout
    if str(result) == "unknown":
        return False, "UNKNOWN (possibly timeout)"
    return False, f"SAT: counterexample exists ({result})"
