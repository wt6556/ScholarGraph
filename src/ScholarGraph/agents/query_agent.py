"""QueryAgent - 查询分析 Agent"""

import logging
from typing import Optional, List

from .base import BaseAgent, AgentResult
from ..shared_env import SharedEnvironment
from ..memory.conversation_history import ConversationHistory
from ..models.query import QueryAnalysisResult

logger = logging.getLogger(__name__)


class QueryAgent(BaseAgent):
    """
    QueryAgent: 查询分析 + 路由决策

    维护 ConversationHistory，支持多轮对话
    """

    def __init__(self, memory: ConversationHistory):
        """
        初始化 QueryAgent

        Args:
            memory: ConversationHistory 实例
        """
        super().__init__("QueryAgent", memory=memory)

    def execute(self, env: SharedEnvironment, query: str) -> AgentResult:
        """
        分析查询

        Args:
            env: SharedEnvironment
            query: 用户查询

        Returns:
            AgentResult.data = QueryAnalysisResult
        """
        try:
            # 1. 记录对话历史
            if self.memory:
                self.memory.append("user", query)

            # 2. LLM 分析查询
            analysis = self._analyze_query(query)

            # 3. 记录助手回复
            if self.memory:
                self.memory.append("assistant", f"Query type: {analysis.query_type}")

            return AgentResult(success=True, data=analysis.to_dict())

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return AgentResult(success=False, error=str(e))

    def _analyze_query(self, query: str) -> QueryAnalysisResult:
        """
        分析查询（简单实现）

        Args:
            query: 用户查询

        Returns:
            QueryAnalysisResult
        """
        # TODO: 使用 LLM 进行更准确的分析
        # 目前使用简单规则

        query_lower = query.lower()
        keywords = self._extract_keywords(query)

        # 判断查询类型
        query_type = "factual_qa"
        target_function = "retrieval"

        if any(word in query_lower for word in ["哪些方法", "改进", "领域"]):
            query_type = "method_topology"
            target_function = "relation"
        elif any(word in query_lower for word in ["比较", "对比", "优缺点"]):
            query_type = "comparison"
        elif any(word in query_lower for word in ["趋势", "进展", "年来"]):
            query_type = "trend_analysis"
        elif any(word in query_lower for word in ["局限", "问题", "缺点"]):
            query_type = "limitation_analysis"
        elif any(word in query_lower for word in ["总结", "概括"]):
            query_type = "summarization"
        elif any(word in query_lower for word in ["ccf", "a类", "b类", "会议"]):
            query_type = "structured_query"
            target_function = "sql"

        # 提取约束条件
        constraints = self._extract_constraints(query)

        return QueryAnalysisResult(
            query_type=query_type,
            original_query=query,
            keywords=keywords,
            constraints=constraints,
            target_function=target_function
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 简单实现：分词并过滤停用词
        stopwords = {"的", "了", "是", "在", "和", "与", "或", "等", "什么", "如何", "怎样"}
        words = query.split()
        return [w for w in words if w not in stopwords and len(w) > 1]

    def _extract_constraints(self, query: str) -> dict:
        """提取约束条件"""
        constraints = {}

        # 提取年份约束
        import re
        year_pattern = r'(20\d{2})年?'
        years = re.findall(year_pattern, query)
        if years:
            constraints['year_range'] = [int(y) for y in years]

        # 提取 CCF 约束
        if 'CCF-A' in query or 'ccf-a' in query.lower():
            constraints['ccf_rating'] = 'A'
        elif 'CCF-B' in query or 'ccf-b' in query.lower():
            constraints['ccf_rating'] = 'B'

        return constraints
