"""Base class for formal verifiers."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseVerifier(ABC):
    """Abstract base class for FOL step verifiers."""

    # 验证当前推理步骤的依赖是否语义蕴含其结论。
    @abstractmethod
    def verify_step(
        self,
        dependencies: List[str],
        conclusion: str,
        premises_fol: Dict[str, str],
        all_steps_fol: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Verify a single reasoning step.

        Args:
            dependencies: List of dependency IDs (e.g., ['h1', 's1'])
            conclusion: FOL string of the conclusion
            premises_fol: Mapping from premise ID to FOL string (e.g., {'h1': '...'})
            all_steps_fol: Mapping from step ID to FOL string for previously verified steps

        Returns:
            dict with keys:
                - 'verified': bool
                - 'error': str or None
                - 'details': dict with extra info
        """
        pass

    # 批量执行形式化步骤验证并返回逐项结果。
    @abstractmethod
    def batch_verify(
        self,
        steps: List[Dict[str, Any]],
        premises_fol: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Verify multiple steps sequentially.

        Args:
            steps: List of step dicts with keys 'id', 'dependencies', 'conclusion'
            premises_fol: Premise ID -> FOL string

        Returns:
            List of result dicts (same order as steps)
        """
        pass
