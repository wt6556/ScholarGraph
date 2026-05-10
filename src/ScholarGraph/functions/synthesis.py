"""Synthesis - 答案合成"""

import logging
import json
import os
import re
from typing import List, Optional, Dict, Any, Tuple

from .base import BaseFunction, FunctionResult
from ..shared_env import SharedEnvironment
from ..models.paper import Paper, Chunk
from ..models.query import QueryAnalysisResult, SynthesizedAnswer
from ..llm import get_llm_client, Message

logger = logging.getLogger(__name__)


# =============================================================================
# System Prompt - 定义 Synthesis 的操作
# =============================================================================
SYNTHESIS_SYSTEM_PROMPT = """You are an academic paper Q&A assistant.

## Operations

You support EXACTLY 1 operation:

---

### Operation: SYNTHESIS

When user asks a question about papers or academic topics, answer based on provided reference materials.

**Input**:
- Question: the user's question
- Reference Materials: chunks from relevant papers

**Your task**:
1. Answer the question based on the reference materials provided
2. Cite sources using paper IDs
3. Acknowledge limitations if information is insufficient

**Return JSON** (no additional text):
{
    "conclusion": "Final answer/conclusion",
    "evidence": [
        {"statement": "Supporting statement", "source": "paper_id"},
        ...
    ],
    "limitations": [],
    "confidence": 0.85,
    "papers_used": count
}

---

IMPORTANT: Only return the JSON object, no additional text."""

SYNTHESIS_OPERATION = "SYNTHESIS"

SYNTHESIS_USER_PROMPT = """Operation: {operation}

Question: {query}

Reference Materials:
{papers_context}

Return JSON:"""


def _build_papers_context(papers: List[Paper]) -> str:
    """Build paper context string"""
    context_parts = []
    for i, paper in enumerate(papers):
        try:
            safe_title = (paper.title or 'N/A').encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            safe_method = (paper.method or 'N/A').encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            safe_task = (paper.task or 'N/A').encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            safe_core_idea = (paper.core_idea or 'N/A').encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            safe_improvements = (paper.improvements or 'N/A').encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            safe_limitation = (paper.limitation or 'N/A').encode('utf-8', errors='replace').decode('utf-8', errors='replace')

            context_parts.append(
                f"Paper {i+1} (ID: {paper.id}):\n"
                f"Title: {safe_title}\n"
                f"Authors: {', '.join(paper.authors) if paper.authors else 'N/A'}\n"
                f"Year: {paper.year or 'N/A'}\n"
                f"Venue: {paper.venue or 'N/A'}\n"
                f"Method: {safe_method}\n"
                f"Task: {safe_task}\n"
                f"Core Idea: {safe_core_idea}\n"
                f"Results: {safe_improvements}\n"
                f"Limitations: {safe_limitation}\n"
                f"---"
            )
        except Exception as e:
            safe_error = str(e).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            logger.warning(f"Failed to build paper context: {safe_error}")
            continue
    return "\n".join(context_parts)


def _build_chunks_context(chunks: List[Chunk], scores: List[float], env: SharedEnvironment = None) -> str:
    """Build chunks context string (Chunk级别上下文)"""
    context_parts = []
    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        try:
            # 获取关联的 Paper 信息
            paper = None
            if chunk.paper_id and env:
                paper = env.get_paper(chunk.paper_id)

            paper_info = ""
            if paper:
                paper_info = (
                    f"Paper: {paper.title}\n"
                    f"Authors: {', '.join(paper.authors) if paper.authors else 'N/A'}\n"
                    f"Year: {paper.year or 'N/A'}\n"
                    f"Venue: {paper.venue or 'N/A'}\n"
                )

            # 使用 safe 编码避免 GBK 错误
            safe_content = chunk.content[:500].encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            context_parts.append(
                f"Chunk {i+1} (ID: {chunk.id}, Type: {chunk.chunk_type}, Score: {score:.4f}):\n"
                f"{paper_info}"
                f"Content: {safe_content}{'...' if len(chunk.content) > 500 else ''}\n"
                f"---"
            )
        except Exception as e:
            safe_error = str(e).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            logger.warning(f"Failed to build chunk context: {safe_error}")
            continue
    return "\n".join(context_parts)


class Synthesis(BaseFunction):
    """
    Synthesis: Chunk 级别 → 结构化答案

    输入: List[Chunk] + List[scores] + QueryAnalysisResult
    输出: SynthesizedAnswer
    """

    def __init__(self):
        super().__init__("Synthesis")

    def _is_english(self, text: str) -> bool:
        """检测文本是否全为英文（ASCII 可打印字符）"""
        import re
        return bool(re.match(r'^[\x20-\x7E\s]+$', text))

    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        if self._is_english(text):
            return "English"
        import re
        if re.search(r'[一-鿿]', text):
            return "Chinese"
        return "other language"

    def execute(self, env: SharedEnvironment,
                chunks: List[Chunk],
                scores: List[float],
                query_analysis: QueryAnalysisResult,
                original_query: str = "",
                translated_query: str = "") -> FunctionResult:
        """
        合成答案（Chunk 级别）

        Args:
            env: SharedEnvironment
            chunks: Chunk 列表
            scores: 对应的分数列表
            query_analysis: 查询分析结果
            original_query: 原始查询（非英文语言，用于控制回答语言）
            translated_query: 翻译后的英文查询（用于检索）
        """
        try:
            if not chunks:
                answer = SynthesizedAnswer(
                    conclusion="No relevant chunks found",
                    summary="",
                    evidence=[],
                    papers_used=0
                )
                return FunctionResult(success=True, data=answer)

            # 按 paper_id 分组，用于统计论文数
            paper_ids = set(chunk.paper_id for chunk in chunks if chunk.paper_id)

            # 使用 LLM 合成答案
            # query_type = query_analysis.query_type or "factual_qa"
            synthesized = self._synthesize_with_llm(
                query=translated_query or query_analysis.original_query,
                chunks=chunks,
                scores=scores,
                # query_type=query_type,
                original_query=original_query or query_analysis.original_query,
                env=env
            )

            if synthesized:
                answer = SynthesizedAnswer(
                    conclusion=synthesized.get("conclusion", ""),
                    summary=synthesized.get("summary", ""),
                    evidence=synthesized.get("evidence", []),
                    method_groups=synthesized.get("method_groups", []),
                    limitations=synthesized.get("limitations", []),
                    confidence=synthesized.get("confidence", 0.5),
                    papers_used=max(synthesized.get("papers_used", 0), len(paper_ids))
                )
            else:
                # LLM 合成失败，返回基本信息
                answer = SynthesizedAnswer(
                    conclusion=f"找到 {len(paper_ids)} 篇相关论文，共 {len(chunks)} 个相关 chunk",
                    summary=f"检索到 {len(paper_ids)} 篇论文中的 {len(chunks)} 个相关片段",
                    evidence=[],
                    papers_used=len(paper_ids)
                )

            logger.info(f"Synthesized answer from {len(chunks)} chunks ({len(paper_ids)} papers)")
            return FunctionResult(success=True, data=answer)

        except Exception as e:
            logger.error(f"Failed to synthesize answer: {e}")
            return FunctionResult(success=False, error=str(e))

    def _synthesize_with_llm(
        self,
        query: str,
        chunks: List[Chunk],
        scores: List[float],
        # _query_type: str,
        original_query: str = "",
        env: SharedEnvironment = None
    ) -> Optional[Dict[str, Any]]:
        """
        使用 LLM 合成答案（Chunk 级别）

        Args:
            query: 用户查询（可能是翻译后的英文）
            chunks: Chunk 列表
            scores: 分数列表
            query_type: 查询类型
            original_query: 原始查询（非英文语言，用于控制回答语言）

        Returns:
            合成结果字典
        """
        try:
            client = get_llm_client()

            # 构建 chunks 上下文
            chunks_context = _build_chunks_context(chunks, scores, env)

            # 如果有原始查询（非英文），在开头添加语言控制指令
            if original_query and original_query != query:
                # 使用 safe 编码避免 GBK 错误
                safe_original = original_query.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                lang_instruction = f'\n\nIMPORTANT: The user\'s original question is in {self._detect_language(original_query)}. Original question: "{safe_original}". Please answer in the same language as the original question.'
                query_display = query + lang_instruction
            else:
                query_display = query

            # 拼接 prompt：首先说明操作类型，再给输入
            prompt = SYNTHESIS_USER_PROMPT.format(
                operation=SYNTHESIS_OPERATION,
                query=query_display,
                papers_context=chunks_context
            )

            # 打印发送给 LLM 的完整 prompt
            if os.environ.get("SCHOLARGRAPH_DEBUG") == "1":
                print("\n" + "="*80)
                print("【发送给 LLM 的 Prompt】")
                print("="*80)
                print(prompt.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
                print("="*80 + "\n")

            response = client.chat(
                messages=[
                    Message(role="system", content=SYNTHESIS_SYSTEM_PROMPT),
                    Message(role="user", content=prompt)
                ],
                max_tokens=4096,
                temperature=0.7
            )

            content = response.content.strip()

            # 打印 LLM 返回的原始字符串
            if os.environ.get("SCHOLARGRAPH_DEBUG") == "1":
                print("\n" + "="*80)
                print("【LLM 返回的原始字符串】")
                print("="*80)
                # 使用 replace 错误处理避免 GBK 编码错误
                safe_content = content.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                print(safe_content)
                print("="*80 + "\n")

            # 提取 JSON - 支持多行嵌套 JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                safe_content = content.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                logger.warning(f"Failed to parse synthesis response: {safe_content[:200]}")
                return None

        except Exception as e:
            safe_error = str(e).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            logger.warning(f"LLM synthesis failed: {safe_error}")
            return None
