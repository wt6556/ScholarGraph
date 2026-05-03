"""Synthesis - 答案合成"""

import logging
import json
import re
from typing import List, Optional, Dict, Any

from .base import BaseFunction, FunctionResult
from ..shared_env import SharedEnvironment
from ..models.paper import Paper
from ..models.query import QueryAnalysisResult, SynthesizedAnswer
from ..llm import get_llm_client, Message

logger = logging.getLogger(__name__)


# 不同查询类型的合成提示词
SYNTHESIS_PROMPTS = {
    "factual_qa": """You are an academic paper Q&A assistant. Answer the user's question based on the following paper content.

User Question: {query}

Related Papers:
{papers_context}

Return the answer in JSON format:
{{
    "conclusion": "Final answer/conclusion",
    "summary": "Brief summary",
    "evidence": [
        {{"statement": "Statement supporting conclusion", "source": "paper_id"}},
        ...
    ],
    "method_groups": [
        {{"method": "Method name", "papers": ["paper_id list"]}},
        ...
    ],
    "limitations": ["Limitation 1", "Limitation 2"],
    "confidence": 0.85,
    "papers_used": count
}}
""",
    "comparison": """You are an academic paper comparison assistant. Compare the methods, performance, and other aspects of the following papers.

User Question: {query}

Related Papers:
{papers_context}

Return the comparison result in JSON format:
{{
    "conclusion": "Comparison conclusion",
    "summary": "Brief summary",
    "comparison": [
        {{"aspect": "Comparison dimension", "paper1": "Content from paper 1", "paper2": "Content from paper 2"}},
        ...
    ],
    "evidence": [
        {{"statement": "Statement supporting conclusion", "source": "paper_id"}},
        ...
    ],
    "limitations": ["Limitation 1"],
    "confidence": 0.85,
    "papers_used": count
}}
""",
    "method_topology": """You are an academic paper topology analysis assistant. Analyze the method evolution relationships in the following papers.

User Question: {query}

Related Papers:
{papers_context}

Return the method topology in JSON format:
{{
    "conclusion": "Method evolution conclusion",
    "summary": "Brief summary",
    "method_groups": [
        {{"method": "Method name", "papers": ["paper_id"], "parent_methods": ["Parent method names"]}},
        ...
    ],
    "evidence": [
        {{"statement": "Statement supporting conclusion", "source": "paper_id"}},
        ...
    ],
    "limitations": ["Limitation 1"],
    "confidence": 0.85,
    "papers_used": count
}}
""",
    "default": """You are an academic paper analysis assistant. Answer the user's question based on the following papers.

User Question: {query}

Related Papers:
{papers_context}

Return the answer in JSON format:
{{
    "conclusion": "Final answer/conclusion",
    "summary": "Brief summary",
    "evidence": [
        {{"statement": "Statement supporting conclusion", "source": "paper_id"}},
        ...
    ],
    "limitations": [],
    "confidence": 0.85,
    "papers_used": count
}}
"""
}


def _build_papers_context(papers: List[Paper]) -> str:
    """Build paper context string"""
    context_parts = []
    for i, paper in enumerate(papers):
        context_parts.append(
            f"Paper {i+1} (ID: {paper.id}):\n"
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors) if paper.authors else 'N/A'}\n"
            f"Year: {paper.year or 'N/A'}\n"
            f"Venue: {paper.venue or 'N/A'}\n"
            f"Method: {paper.method or 'N/A'}\n"
            f"Task: {paper.task or 'N/A'}\n"
            f"Core Idea: {paper.core_idea or 'N/A'}\n"
            f"Results: {paper.improvements or 'N/A'}\n"
            f"Limitations: {paper.limitation or 'N/A'}\n"
            f"---"
        )
    return "\n".join(context_parts)


class Synthesis(BaseFunction):
    """
    Synthesis: 多论文 → 结构化答案

    输入: List[Paper] + QueryAnalysisResult
    输出: SynthesizedAnswer
    """

    def __init__(self):
        super().__init__("Synthesis")

    def execute(self, env: SharedEnvironment,
                papers: List[Paper],
                query_analysis: QueryAnalysisResult) -> FunctionResult:
        """
        合成答案

        Args:
            env: SharedEnvironment
            papers: 论文列表
            query_analysis: 查询分析结果

        Returns:
            FunctionResult.data = SynthesizedAnswer
        """
        try:
            if not papers:
                answer = SynthesizedAnswer(
                    conclusion="No relevant papers found",
                    summary="",
                    evidence=[],
                    papers_used=0
                )
                return FunctionResult(success=True, data=answer)

            # 使用 LLM 合成答案
            query_type = query_analysis.query_type or "factual_qa"
            synthesized = self._synthesize_with_llm(
                query=query_analysis.original_query,
                papers=papers,
                query_type=query_type
            )

            if synthesized:
                answer = SynthesizedAnswer(
                    conclusion=synthesized.get("conclusion", ""),
                    summary=synthesized.get("summary", ""),
                    evidence=synthesized.get("evidence", []),
                    method_groups=synthesized.get("method_groups", []),
                    limitations=synthesized.get("limitations", []),
                    confidence=synthesized.get("confidence", 0.5),
                    papers_used=max(synthesized.get("papers_used", 0), len(papers))
                )
            else:
                # LLM 合成失败，返回基本信息
                answer = SynthesizedAnswer(
                    conclusion=f"找到 {len(papers)} 篇相关论文",
                    summary=f"检索到 {len(papers)} 篇与查询相关的论文",
                    evidence=[],
                    papers_used=len(papers)
                )

            logger.info(f"Synthesized answer from {len(papers)} papers")
            return FunctionResult(success=True, data=answer)

        except Exception as e:
            logger.error(f"Failed to synthesize answer: {e}")
            return FunctionResult(success=False, error=str(e))

    def _synthesize_with_llm(
        self,
        query: str,
        papers: List[Paper],
        query_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        使用 LLM 合成答案

        Args:
            query: 用户查询
            papers: 论文列表
            query_type: 查询类型

        Returns:
            合成结果字典
        """
        try:
            client = get_llm_client()

            # 选择合适的提示词
            prompt_template = SYNTHESIS_PROMPTS.get(query_type, SYNTHESIS_PROMPTS["default"])
            papers_context = _build_papers_context(papers)

            prompt = prompt_template.format(
                query=query,
                papers_context=papers_context
            )

            response = client.chat(
                messages=[Message(role="user", content=prompt)],
                max_tokens=4096,
                temperature=0.7
            )

            content = response.content.strip()

            # 提取 JSON
            json_match = re.search(r'\{[^{}]*"[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    # 尝试找到完整的 JSON 对象
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start >= 0 and end > start:
                        return json.loads(content[start:end])
                except json.JSONDecodeError:
                    pass

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse synthesis response: {content[:200]}")
                return None

        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return None
