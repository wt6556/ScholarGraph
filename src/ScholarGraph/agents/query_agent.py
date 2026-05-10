"""QueryAgent - 查询分析 Agent"""

import logging
import os
from typing import Optional, List

from .base import BaseAgent, AgentResult
from ..shared_env import SharedEnvironment
from ..memory.conversation_history import ConversationHistory
from ..models.query import QueryAnalysisResult
from ..llm import get_llm_client, Message

logger = logging.getLogger(__name__)


# =============================================================================
# System Prompt - 在 __init__ 时设置，定义 Agent 的两个操作
# =============================================================================
SYSTEM_PROMPT = """You are a query preprocessing assistant for an academic paper search system.

## Operations

You support EXACTLY 2 operations. Each operation has a specific input format and output format.

---

### Operation 1: EXPAND_QUERY

When user asks to preprocess/expand a query, perform:

**Input**: A query string to expand

**Your task**:
1. **Translate to English**: If the original query is non-English, translate it to English
2. **Expand references**: Replace pronouns/vague references using conversation context
3. **Enhance for retrieval**:
   - `expanded_query`: Generate a concise, keyword-rich query for embedding vector similarity search. Keep it under 100 characters, focus on core technical terms.
   - `keywords`: Select the 3 most relevant keywords for BM25 keyword retrieval. Choose terms that appear in academic papers.

**Return JSON** (no additional text):
{
    "translated_query": "English translation or original English query",
    "expanded_query": "Concise query for embedding (max 100 chars, core technical terms)",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "query_type": "factual_qa|comparison|method_topology|trend_analysis|limitation_analysis|summarization|structured_query"
}

---

### Operation 2: ANSWER_WITH_CONTEXT

When user asks a question about papers or academic topics, answer based on provided materials and memory.

**Input**: Question + (optional) context from retrieved papers

**Your task**:
1. Answer the question based on the context provided
2. Cite sources using paper IDs
3. Acknowledge limitations if information is insufficient

**Return JSON**:
{
    "conclusion": "Final answer/conclusion",
    "evidence": [
        {"statement": "Supporting statement", "source": "paper_id"},
        ...
    ],
    "limitations": ["limitation1", "limitation2"],
    "confidence": 0.85,
    "papers_used": 3
}

---

IMPORTANT: Only return the JSON object, no additional text."""


OPERATION_EXPAND_QUERY = "EXPAND_QUERY"
OPERATION_ANSWER_WITH_CONTEXT = "ANSWER_WITH_CONTEXT"


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
        # 在初始化时设置 system prompt
        self.system_prompt = SYSTEM_PROMPT

    def execute(self, env: SharedEnvironment, query: str, operation: str = OPERATION_EXPAND_QUERY, context: Optional[dict] = None) -> AgentResult:
        """
        执行 QueryAgent

        Args:
            env: SharedEnvironment
            query: 用户查询
            operation: 操作类型 - OPERATION_EXPAND_QUERY 或 OPERATION_ANSWER_WITH_CONTEXT
            context: 可选的上下文信息（用于 ANSWER_WITH_CONTEXT 操作）

        Returns:
            AgentResult
        """
        try:
            # 1. 记录对话历史
            if self.memory:
                self.memory.append("user", query)

            # 2. 根据操作类型执行
            if operation == OPERATION_EXPAND_QUERY:
                result = self._expand_query(query)
            elif operation == OPERATION_ANSWER_WITH_CONTEXT:
                result = self._answer_with_context(query, context)
            else:
                return AgentResult(success=False, error=f"Unknown operation: {operation}")

            # 3. 记录助手回复
            if self.memory and operation == OPERATION_EXPAND_QUERY:
                self.memory.append("assistant", f"Query type: {result.query_type}, expanded: {result.original_query}")

            return AgentResult(success=True, data=result)

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return AgentResult(success=False, error=str(e))

    def _expand_query(self, query: str) -> QueryAnalysisResult:
        """执行查询扩展操作"""
        client = get_llm_client()

        # 构建对话历史上下文
        conversation_history = ""
        if self.memory:
            history = self.memory.get_history_for_llm()
            if history:
                conversation_history = "\n".join([
                    f"User: {h.get('content', '')}" if h.get('role') == 'user' else f"Assistant: {h.get('content', '')}"
                    for h in history[-6:]
                ])

        # 拼接 prompt: 首先说明操作类型，再给输入
        prompt = f"""Operation: {OPERATION_EXPAND_QUERY}

Conversation context (for resolving references):
{conversation_history or "(no previous conversation)"}

Original query: "{query}"

Return JSON:"""

        return self._call_llm_and_parse(query, prompt, client)

    def _answer_with_context(self, query: str, context: Optional[dict] = None) -> dict:
        """执行基于上下文的回答操作"""
        client = get_llm_client()

        # 构建 prompt
        context_str = ""
        if context:
            context_str = "\n\nContext from retrieved papers:\n"
            if "chunks" in context:
                for chunk in context["chunks"][:5]:  # 最多使用5个chunk
                    context_str += f"- [{chunk.get('paper_id', 'unknown')}]: {chunk.get('content', '')[:200]}...\n"

        prompt = f"""Operation: {OPERATION_ANSWER_WITH_CONTEXT}

Question: {query}
{context_str}

Return JSON:"""

        response = client.chat(
            messages=[
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=prompt)
            ],
            max_tokens=2048,
            temperature=0.3
        )

        content = response.content.strip()

        # 打印调试信息
        if os.environ.get("SCHOLARGRAPH_DEBUG") == "1":
            print("\n" + "="*80)
            print("【QueryAgent - ANSWER_WITH_CONTEXT 返回】")
            print("="*80)
            print(content)
            print("="*80 + "\n")

        import json
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "conclusion": "Failed to parse LLM response",
                "evidence": [],
                "limitations": ["JSON parsing failed"],
                "confidence": 0.0,
                "papers_used": 0
            }

    def _call_llm_and_parse(self, query: str, prompt: str, client) -> QueryAnalysisResult:
        """调用 LLM 并解析结果"""
        # 打印发送给 LLM 的 prompt（调试模式）
        if os.environ.get("SCHOLARGRAPH_DEBUG") == "1":
            print("\n" + "="*80)
            print("【QueryAgent - 发送给 LLM 的 Prompt】")
            print("="*80)
            print(prompt)
            print("="*80 + "\n")

        response = client.chat(
            messages=[
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=prompt)
            ],
            max_tokens=2048,
            temperature=0.3
        )

        content = response.content.strip()

        # 打印 LLM 返回（调试模式）
        if os.environ.get("SCHOLARGRAPH_DEBUG") == "1":
            print("\n" + "="*80)
            print("【QueryAgent - LLM 返回】")
            print("="*80)
            print(content)
            print("="*80 + "\n")

        # 解析 JSON 响应
        import json
        try:
            result = json.loads(content)
            translated_query = result.get("translated_query", query)
            expanded_query = result.get("expanded_query", translated_query)
            keywords = result.get("keywords", [])
            query_type = result.get("query_type", "factual_qa")
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response: {content[:200]}")
            translated_query = query
            expanded_query = query
            keywords = self._extract_keywords(query)
            query_type = self._classify_query(query)

        # 判断目标函数
        target_function = "retrieval"
        if query_type == "method_topology":
            target_function = "relation"
        elif query_type == "structured_query":
            target_function = "sql"

        # 提取约束条件
        constraints = self._extract_constraints(query)

        return QueryAnalysisResult(
            query_type=query_type,
            original_query=query,  # 原始用户输入
            translated_query=translated_query,  # 翻译后的查询
            expanded_query=expanded_query,  # 扩展后的查询（用于检索）
            keywords=keywords,
            constraints=constraints,
            target_function=target_function
        )

    def _classify_query(self, query: str) -> str:
        """简单规则分类"""
        query_lower = query.lower()
        if any(word in query_lower for word in ["哪些方法", "改进", "领域"]):
            return "method_topology"
        elif any(word in query_lower for word in ["比较", "对比", "优缺点"]):
            return "comparison"
        elif any(word in query_lower for word in ["趋势", "进展", "年来"]):
            return "trend_analysis"
        elif any(word in query_lower for word in ["局限", "问题", "缺点"]):
            return "limitation_analysis"
        elif any(word in query_lower for word in ["总结", "概括"]):
            return "summarization"
        elif any(word in query_lower for word in ["ccf", "a类", "b类", "会议"]):
            return "structured_query"
        return "factual_qa"

    def _analyze_query(self, query: str) -> QueryAnalysisResult:
        """
        分析查询（简单规则实现，备用）

        Args:
            query: 用户查询

        Returns:
            QueryAnalysisResult
        """
        query_lower = query.lower()
        keywords = self._extract_keywords(query)

        # 判断查询类型
        query_type = self._classify_query(query)

        target_function = "retrieval"
        if query_type == "method_topology":
            target_function = "relation"
        elif query_type == "structured_query":
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
