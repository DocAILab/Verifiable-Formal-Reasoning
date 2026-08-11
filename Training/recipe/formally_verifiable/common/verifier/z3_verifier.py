"""Z3-based FOL step verifier."""
from typing import List, Dict, Any
from .base import BaseVerifier
from ..fol_converter import FOLToZ3Converter, verify_implication


class Z3Verifier(BaseVerifier):
    """Verify reasoning steps using Z3 SMT solver."""

    # 初始化带超时配置的 Z3 语义验证器。
    def __init__(self, timeout_ms: int = 5000):
        self.converter = FOLToZ3Converter()
        self.timeout_ms = timeout_ms

    # 验证当前推理步骤的依赖是否语义蕴含其结论。
    def verify_step(
        self,
        dependencies: List[str],
        conclusion: str,
        premises_fol: Dict[str, str],
        all_steps_fol: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Verify a single step: do the dependencies entail the conclusion?
        """
        # Gather premises
        premises_z3 = []
        missing = []
        for dep in dependencies:
            if dep in premises_fol:
                fol_str = premises_fol[dep]
            elif dep in all_steps_fol:
                fol_str = all_steps_fol[dep]
            else:
                missing.append(dep)
                continue
            try:
                z3_expr = self.converter.convert(fol_str)
                premises_z3.append(z3_expr)
            except Exception as e:
                return {
                    "verified": False,
                    "error": f"Failed to parse dependency '{dep}': {e}",
                    "details": {"dependency": dep, "fol": fol_str},
                }

        if missing:
            return {
                "verified": False,
                "error": f"Missing dependencies: {missing}",
                "details": {"missing": missing},
            }

        # Parse conclusion
        try:
            conclusion_z3 = self.converter.convert(conclusion)
        except Exception as e:
            return {
                "verified": False,
                "error": f"Failed to parse conclusion: {e}",
                "details": {"conclusion": conclusion},
            }

        # Verify. Malformed model outputs can parse into non-Boolean Z3 terms;
        # keep those as verifier failures instead of aborting preference mining.
        try:
            verified, msg = verify_implication(premises_z3, conclusion_z3, self.timeout_ms)
        except Exception as e:
            return {
                "verified": False,
                "error": f"Failed to verify implication: {e}",
                "details": {"conclusion": conclusion},
            }
        return {
            "verified": verified,
            "error": None if verified else msg,
            "details": {
                "premises_count": len(premises_z3),
                "message": msg,
            },
        }

    # 批量执行形式化步骤验证并返回逐项结果。
    def batch_verify(
        self,
        steps: List[Dict[str, Any]],
        premises_fol: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Verify steps sequentially, accumulating verified step conclusions."""
        all_steps_fol: Dict[str, str] = {}
        results = []
        for step in steps:
            res = self.verify_step(
                dependencies=step.get("dependencies", []),
                conclusion=step.get("conclusion", ""),
                premises_fol=premises_fol,
                all_steps_fol=all_steps_fol,
            )
            res["step_id"] = step.get("id")
            results.append(res)
            # If verified, make this step available for later steps
            if res["verified"]:
                all_steps_fol[step.get("id", "")] = step.get("conclusion", "")
        return results
