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
    "factual_qa": """你是一个学术论文问答助手。请根据以下论文内容回答用户问题。

用户问题: {query}

相关论文:
{papers_context}

请以JSON格式返回答案：
{{
    "conclusion": "最终答案/结论",
    "summary": "简要总结",
    "evidence": [
        {{"statement": "支持结论的陈述", "source": "论文ID"}},
        ...
    ],
    "method_groups": [
        {{"method": "方法名称", "papers": ["论文ID列表"]}},
        ...
    ],
    "limitations": ["局限性1", "局限性2"],
    "confidence": 0.85,
    "papers_used": 数量
}}
""",
    "comparison": """你是一个学术论文比较助手。请比较以下论文的方法、性能等方面。

用户问题: {query}

相关论文:
{papers_context}

请以JSON格式返回比较结果：
{{
    "conclusion": "比较结论",
    "summary": "简要总结",
    "comparison": [
        {{"aspect": "比较维度", "paper1": "论文1的内容", "paper2": "论文2的内容"}},
        ...
    ],
    "evidence": [
        {{"statement": "支持结论的陈述", "source": "论文ID"}},
        ...
    ],
    "limitations": ["局限性1"],
    "confidence": 0.85,
    "papers_used": 数量
}}
""",
    "method_topology": """你是一个学术论文拓扑分析助手。请分析以下论文中方法的演进关系。

用户问题: {query}

相关论文:
{papers_context}

请以JSON格式返回方法拓扑：
{{
    "conclusion": "方法演进结论",
    "summary": "简要总结",
    "method_groups": [
        {{"method": "方法名称", "papers": ["论文ID"], "parent_methods": ["父方法名称"]}},
        ...
    ],
    "evidence": [
        {{"statement": "支持结论的陈述", "source": "论文ID"}},
        ...
    ],
    "limitations": ["局限性1"],
    "confidence": 0.85,
    "papers_used": 数量
}}
""",
    "default": """你是一个学术论文分析助手。请根据以下论文回答用户问题。

用户问题: {query}

相关论文:
{papers_context}

请以JSON格式返回答案：
{{
    "conclusion": "最终答案/结论",
    "summary": "简要总结",
    "evidence": [
        {{"statement": "支持结论的陈述", "source": "论文ID"}},
        ...
    ],
    "limitations": [],
    "confidence": 0.85,
    "papers_used": 数量
}}
"""
}


def _build_papers_context(papers: List[Paper]) -> str:
    """构建论文上下文字符串"""
    context_parts = []
    for i, paper in enumerate(papers):
        context_parts.append(
            f"论文 {i+1} (ID: {paper.id}):\n"
            f"标题: {paper.title}\n"
            f"作者: {', '.join(paper.authors) if paper.authors else 'N/A'}\n"
            f"年份: {paper.year or 'N/A'}\n"
            f"会议/期刊: {paper.venue or 'N/A'}\n"
            f"方法: {paper.method or 'N/A'}\n"
            f"任务: {paper.task or 'N/A'}\n"
            f"核心思想: {paper.core_idea or 'N/A'}\n"
            f"实验结果: {paper.improvements or 'N/A'}\n"
            f"局限性: {paper.limitation or 'N/A'}\n"
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
                    conclusion="没有找到相关论文",
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
