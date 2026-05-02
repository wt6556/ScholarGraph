"""CriticAgent - 评估循环 Agent"""

import logging
from typing import List

from .base import BaseAgent, AgentResult
from ..shared_env import SharedEnvironment
from ..memory.pipeline_state import PipelineState
from ..models.query import SynthesizedAnswer, CritiqueResult

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """
    CriticAgent: 答案质量评估 + 重试

    维护 PipelineState，记录失败检查项
    """

    def __init__(self, memory: PipelineState):
        """
        初始化 CriticAgent

        Args:
            memory: PipelineState 实例
        """
        super().__init__("CriticAgent", memory=memory)

    def execute(self, env: SharedEnvironment,
                answer: SynthesizedAnswer,
                query: str) -> AgentResult:
        """
        评估答案

        Args:
            env: SharedEnvironment
            answer: 合成答案
            query: 原始查询

        Returns:
            AgentResult.data = CritiqueResult
        """
        try:
            # 执行评估
            critique = self._evaluate(answer, query)

            # 更新 Memory
            if not critique.passed:
                self.memory.add_failed_check(", ".join(critique.failed_checks))
                self.memory.increment_retry()

            return AgentResult(
                success=True,
                data=critique.to_dict(),
                state_update={"critique": critique.to_dict()}
            )

        except Exception as e:
            logger.error(f"Critique failed: {e}")
            return AgentResult(success=False, error=str(e))

    def _evaluate(self, answer: SynthesizedAnswer, query: str) -> CritiqueResult:
        """
        评估答案质量

        检查项：
        1. 覆盖率（coverage）：是否引用了多篇论文？
        2. 对比性（comparison）：是否包含维度/实验对比？
        3. 局限性（limitations）：是否识别了局限性？
        4. 依据性（grounding）：声明是否有来源支撑？

        Args:
            answer: 合成答案
            query: 原始查询

        Returns:
            CritiqueResult
        """
        failed_checks: List[str] = []
        needs_retrieval = False

        # 1. 检查覆盖率
        coverage_score = min(answer.papers_used / 3.0, 1.0) if answer.papers_used > 0 else 0.0
        if answer.papers_used < 2:
            failed_checks.append("coverage_low")
            needs_retrieval = True

        # 2. 检查对比性
        comparison_score = 1.0 if answer.method_groups else 0.5
        if not answer.method_groups and "比较" in query:
            failed_checks.append("comparison_missing")

        # 3. 检查局限性
        limitation_score = 1.0 if answer.limitations else 0.5
        if not answer.limitations:
            failed_checks.append("limitation_missing")

        # 4. 检查依据性
        grounding_score = min(len(answer.evidence) / 3.0, 1.0) if answer.evidence else 0.0
        if not answer.evidence:
            failed_checks.append("grounding_missing")

        # 判断是否通过
        passed = (
            coverage_score >= 0.5 and
            grounding_score >= 0.3 and
            len(failed_checks) <= 1
        )

        return CritiqueResult(
            passed=passed,
            coverage_score=coverage_score,
            comparison_score=comparison_score,
            limitation_score=limitation_score,
            grounding_score=grounding_score,
            failed_checks=failed_checks,
            needs_retrieval=needs_retrieval
        )
