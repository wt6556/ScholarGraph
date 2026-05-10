"""Retrieval - 混合检索"""

import logging
import os
from typing import List, Dict, Any

from .base import BaseFunction, FunctionResult
from ..shared_env import SharedEnvironment
from ..models.paper import Chunk
from ..models.query import QueryAnalysisResult
from ..rag.reranker import Reranker

logger = logging.getLogger(__name__)


# =============================================================================
# System Prompt - 定义 Retrieval 的操作
# =============================================================================
RETRIEVAL_SYSTEM_PROMPT = """You are a query translation assistant for an academic paper search system.

## Operations

You support EXACTLY 1 operation:

---

### Operation: TRANSLATE

Translate the given text to English.

**Input**: Text to translate

**Your task**: Translate the text to English accurately

**Return JSON** (no additional text):
{
    "translated_text": "English translation"
}

---

IMPORTANT: Only return the JSON object, no additional text."""

RETRIEVAL_OPERATION_TRANSLATE = "TRANSLATE"

RETRIEVAL_USER_PROMPT = """Operation: {operation}

Text to translate: {text}

Return JSON:"""


class RetrievalResult:
    """检索结果（Chunk 级别，带分数）"""

    def __init__(self, chunks: List[Chunk], scores: List[float], original_query: str = "", translated_query: str = ""):
        """
        Args:
            chunks: Chunk 列表
            scores: 对应的分数列表
            original_query: 原始查询（非英文语言）
            translated_query: 翻译后的英文查询（用于检索）
        """
        self.chunks = chunks
        self.scores = scores
        self.original_query = original_query
        self.translated_query = translated_query

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(zip(self.chunks, self.scores))

    def top_k(self, k: int) -> 'RetrievalResult':
        """返回 top_k 结果"""
        if k >= len(self.chunks):
            return self
        return RetrievalResult(self.chunks[:k], self.scores[:k], self.original_query, self.translated_query)


class Retrieval(BaseFunction):
    """
    Retrieval: 混合检索（vector + BM25）→ Reranker 重排

    输入: QueryAnalysisResult
    输出: RetrievalResult (List[Chunk] + scores)
    """

    def __init__(self):
        super().__init__("Retrieval")

    def _get_embedding_model_and_tokenizer(self):
        """获取 embedding 模型和 tokenizer"""
        from ..config import get_config
        config = get_config()
        from transformers import AutoTokenizer, AutoModel
        import torch

        model = AutoModel.from_pretrained(config.embedding.model, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(config.embedding.model, local_files_only=True)
        device = config.embedding.device
        pooling_strategy = getattr(config.embedding, 'pooling_strategy', 'mean')
        model.to(device)
        model.eval()

        return model, tokenizer, device, pooling_strategy

    def _translate_to_english(self, query: str) -> str:
        """将查询翻译为英文（用于检索）"""
        from ..llm import get_llm_client, Message

        try:
            client = get_llm_client()

            # 拼接 prompt：首先说明操作类型，再给输入
            prompt = RETRIEVAL_USER_PROMPT.format(
                operation=RETRIEVAL_OPERATION_TRANSLATE,
                text=query
            )

            response = client.chat(
                messages=[
                    Message(role="system", content=RETRIEVAL_SYSTEM_PROMPT),
                    Message(role="user", content=prompt)
                ],
                max_tokens=512,
                temperature=0.0
            )

            content = response.content.strip()

            # 解析 JSON 返回
            import json
            try:
                result = json.loads(content)
                english_query = result.get("translated_text", query)
            except json.JSONDecodeError:
                english_query = query

            if os.environ.get("SCHOLARGRAPH_DEBUG") == "1":
                print(f"\n[DEBUG] 翻译: '{query}' -> '{english_query}'")
            return english_query

        except Exception as e:
            if os.environ.get("SCHOLARGRAPH_DEBUG") == "1":
                print(f"\n[DEBUG] 翻译失败，使用原文: {e}")
            return query

    def _is_english(self, text: str) -> bool:
        """检测文本是否主要为英文（简单检测：包含非ASCII字符则视为非英文）"""
        import re
        # 如果包含非ASCII字符（中文、日文等），视为非英文
        return bool(re.match(r'^[a-zA-Z0-9\s\.,\?\!\-\'\"]+$', text))

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """生成文本 embeddings"""
        import torch
        import torch.nn.functional as F

        model, tokenizer, device, pooling_strategy = self._get_embedding_model_and_tokenizer()
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        if pooling_strategy == 'last_token':
            # Last-token pooling
            attention_mask = encoded['attention_mask']
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                embeddings = outputs.last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = outputs.last_hidden_state.shape[0]
                embeddings = outputs.last_hidden_state[
                    torch.arange(batch_size, device=outputs.last_hidden_state.device),
                    sequence_lengths
                ]
        else:
            # Mean pooling
            attention_mask = encoded['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().tolist()

    def _vector_search(self, env: SharedEnvironment, query: str, top_k: int = 20) -> List[tuple]:
        """向量检索（返回 Chunk + 分数）"""
        try:
            query_embedding = self._encode_texts([query])[0]

            # 从 FAISS 检索（返回 List[(Chunk, score)]）
            chunk_scores = env.chunk_store.search_with_scores(query_embedding, top_k)

            return chunk_scores
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    def _bm25_search(self, env: SharedEnvironment, query: str, keywords: List[str], top_k: int = 20) -> List[tuple]:
        """BM25 关键词检索（返回 Chunk + 分数）"""
        from collections import Counter
        import math

        # 获取所有 chunks 的文本用于 BM25
        sql = f"SELECT id, paper_id, chunk_type, content FROM {env.chunk_store.CHUNKS_TABLE}"
        rows = env.chunk_store.db.fetch_all(sql)
        if not rows:
            if os.environ.get("SCHOLARGRAPH_DEBUG") == "1":
                print("[DEBUG] BM25: 没有从数据库获取到任何 chunks")
            return []

        chunks_info = []
        for row in rows:
            chunks_info.append({
                "id": row["id"],
                "paper_id": row["paper_id"],
                "chunk_type": row["chunk_type"],
                "content": row["content"]
            })

        # 计算 BM25
        k1 = 1.5
        b = 0.75

        # 使用 jieba 分词（支持中英文混合）
        import jieba
        doc_tokenized = []
        for info in chunks_info:
            tokens = list(jieba.cut(info["content"].lower()))
            doc_tokenized.append(tokens)

        doc_lengths = [len(tokens) for tokens in doc_tokenized]
        avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1

        # 词频统计
        term_doc_freq: Dict[str, int] = {}
        term_freqs: List[Counter] = []
        for tokens in doc_tokenized:
            tf = Counter(tokens)
            term_freqs.append(tf)
            for term in tf.keys():
                term_doc_freq[term] = term_doc_freq.get(term, 0) + 1

        # 查询词 - 如果有关键词则使用关键词，否则从 query 提取
        if keywords:
            query_words = [w.lower() for w in keywords if w]
        else:
            raw_words = list(jieba.cut(query.lower()))
            import re
            query_words = [w for w in raw_words if re.match(r'^[a-z]+$', w)]

        # 调试：打印查询词
        if os.environ.get("SCHOLARGRAPH_DEBUG") == "1":
            print(f"\n[DEBUG] BM25 查询词: {query_words}")

        # 计算分数
        scores = []
        matched_docs = 0
        for i, info in enumerate(chunks_info):
            score = 0.0
            doc_len = doc_lengths[i]
            for word in query_words:
                if word in term_freqs[i]:
                    tf = term_freqs[i][word]
                    df = term_doc_freq.get(word, 0)
                    idf = math.log((len(chunks_info) - df + 0.5) / (df + 0.5) + 1)
                    idf = max(0, idf)
                    score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
            if score > 0:
                matched_docs += 1
                # 获取 Chunk 对象
                chunk = env.chunk_store.get(info["id"])
                if chunk:
                    scores.append((chunk, score))

        if os.environ.get("SCHOLARGRAPH_DEBUG") == "1":
            print(f"[DEBUG] BM25: {matched_docs}/{len(chunks_info)} 个文档匹配到查询词")

        scores.sort(key=lambda x: x[1], reverse=True)
        if os.environ.get("SCHOLARGRAPH_DEBUG") == "1":
            print(f"[DEBUG] BM25: 获取到 {len(scores)} 个结果")
        return scores[:top_k]

    def execute(self, env: SharedEnvironment,
                query_analysis: QueryAnalysisResult) -> FunctionResult:
        """
        检索 Chunk（带分数）

        Args:
            env: SharedEnvironment
            query_analysis: 查询分析结果

        Returns:
            FunctionResult.data = RetrievalResult
        """
        try:
            from ..config import get_config
            config = get_config()

            original_query = query_analysis.original_query
            expanded_query = query_analysis.expanded_query or query_analysis.original_query
            translated_query = query_analysis.translated_query or expanded_query
            keywords = query_analysis.keywords or []

            # 使用 expanded_query 进行检索
            search_query = expanded_query

            # 1. 向量检索（使用英文查询）
            vector_results = self._vector_search(env, search_query, top_k=config.rag.retrieval.top_k)

            # 打印向量检索结果
            if os.environ.get("SCHOLARGRAPH_DEBUG") == "1":
                print("\n" + "="*80)
                print("【向量检索返回的 Chunks】")
                print("="*80)
                for i, (chunk, score) in enumerate(vector_results):
                    print(f"\n--- 向量检索结果 #{i+1} ---")
                    print(f"chunk_id: {chunk.id}")
                    print(f"paper_id: {chunk.paper_id}")
                    print(f"score: {score}")
                    print(f"content:\n{chunk.content}")
                print("\n" + "="*80)

            # 2. BM25 检索（使用英文查询）
            bm25_results = self._bm25_search(env, search_query, keywords, top_k=config.rag.retrieval.top_k)

            # 打印 BM25 检索结果
            if os.environ.get("SCHOLARGRAPH_DEBUG") == "1":
                print("\n" + "="*80)
                print("【BM25 检索返回的 Chunks】")
                print("="*80)
                for i, (chunk, score) in enumerate(bm25_results):
                    print(f"\n--- BM25 检索结果 #{i+1} ---")
                    print(f"chunk_id: {chunk.id}")
                    print(f"paper_id: {chunk.paper_id}")
                    print(f"score: {score}")
                    print(f"content:\n{chunk.content}")
                print("\n" + "="*80)

            # 3. 合并结果（加权融合）
            chunk_scores: Dict[str, float] = {}
            for chunk, score in vector_results:
                chunk_scores[chunk.id] = chunk_scores.get(chunk.id, 0) + score * 0.7
            for chunk, score in bm25_results:
                chunk_scores[chunk.id] = chunk_scores.get(chunk.id, 0) + score * 0.3

            # 排序
            sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

            # 构建 Chunk + 分数列表
            all_chunks = []
            all_scores = []
            for chunk_id, score in sorted_chunks:
                chunk = env.chunk_store.get(chunk_id)
                if chunk:
                    all_chunks.append(chunk)
                    all_scores.append(score)

            # 4. Reranker 重排（使用英文查询）
            reranker = Reranker()  # 按需创建，符合无状态原则
            candidates = [
                {"chunk": chunk, "score": score}
                for chunk, score in zip(all_chunks, all_scores)
            ]
            reranked = reranker.rerank(
                query=search_query,
                candidates=candidates,
                top_k=config.rag.retrieval.rerank_top_k
            )

            # 打印重排结果
            if os.environ.get("SCHOLARGRAPH_DEBUG") == "1":
                print("\n" + "="*80)
                print("【重排返回的 Chunks】")
                print("="*80)
                for i, c in enumerate(reranked):
                    chunk = c["chunk"]
                    score = c["score"]
                    print(f"\n--- 重排结果 #{i+1} ---")
                    print(f"chunk_id: {chunk.id}")
                    print(f"paper_id: {chunk.paper_id}")
                    print(f"score: {score}")
                    print(f"content:\n{chunk.content}")
                print("\n" + "="*80 + "\n")

            # 提取重排后的 Chunk 和分数
            result_chunks = [c["chunk"] for c in reranked]
            result_scores = [c["score"] for c in reranked]

            result = RetrievalResult(result_chunks, result_scores, original_query=original_query, translated_query=search_query)

            logger.info(f"Retrieved {len(result.chunks)} chunks for query: {original_query}")
            return FunctionResult(success=True, data=result)

        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}")
            return FunctionResult(success=False, error=str(e))
