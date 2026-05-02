"""Retrieval - 混合检索"""

import logging
from typing import List, Dict, Any

from .base import BaseFunction, FunctionResult
from ..shared_env import SharedEnvironment
from ..models.paper import Paper, Chunk
from ..models.query import QueryAnalysisResult

logger = logging.getLogger(__name__)


class Retrieval(BaseFunction):
    """
    Retrieval: 混合检索（vector + BM25）

    输入: QueryAnalysisResult
    输出: List[Paper]
    """

    def __init__(self):
        super().__init__("Retrieval")
        self._embedding_model = None
        self._tokenizer = None

    def _get_embedding_model(self):
        """获取 embedding 模型"""
        if self._embedding_model is None:
            from ..config import get_config
            config = get_config()
            from transformers import AutoTokenizer, AutoModel
            import torch

            self._embedding_model = AutoModel.from_pretrained(config.embedding.model)
            self._tokenizer = AutoTokenizer.from_pretrained(config.embedding.model)
            self._device = config.embedding.device
            self._embedding_model.to(self._device)
            self._embedding_model.eval()

        return self._embedding_model

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """生成文本 embeddings"""
        import torch
        import torch.nn.functional as F

        model = self._get_embedding_model()
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        attention_mask = encoded['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().tolist()

    def _vector_search(self, env: SharedEnvironment, query: str, top_k: int = 20) -> List[tuple]:
        """向量检索"""
        try:
            query_embedding = self._encode_texts([query])[0]

            # 从 FAISS 检索（返回 List[Chunk]）
            chunks = env.chunk_store.search(query_embedding, top_k)

            # 收集 paper_id（每篇论文取最高的 chunk 分数）
            paper_best_score: Dict[str, float] = {}
            paper_chunks: Dict[str, List] = {}

            for i, chunk in enumerate(chunks):
                paper_id = chunk.paper_id
                # FAISS 返回的 score 是距离，需要转换
                score = 1.0 / (1.0 + i)  # 简单排名分数
                if paper_id not in paper_best_score or score > paper_best_score[paper_id]:
                    paper_best_score[paper_id] = score

            return [(pid, score) for pid, score in paper_best_score.items()]
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    def _bm25_search(self, env: SharedEnvironment, query: str, keywords: List[str], top_k: int = 20) -> List[tuple]:
        """BM25 关键词检索"""
        papers = env.get_all_papers(limit=1000)

        # 简单 BM25 实现
        k1 = 1.5
        b = 0.75
        avgdl = 0
        doc_lengths = {}
        doc_freq = {}

        # 统计
        for paper in papers:
            text = self._get_paper_text(paper)
            words = text.lower().split()
            doc_len = len(words)
            doc_lengths[paper.id] = doc_len
            avgdl += doc_len

            for word in set(words):
                doc_freq[word] = doc_freq.get(word, 0) + 1

        avgdl = avgdl / len(papers) if papers else 1

        # 计算查询词在每篇论文中的分数
        query_words = query.lower().split()
        query_words.extend([w.lower() for w in keywords])

        paper_scores = {}
        for paper in papers:
            words = self._get_paper_text(paper).lower().split()
            doc_len = doc_lengths.get(paper.id, 1)

            score = 0
            for word in query_words:
                if word in doc_freq:
                    tf = words.count(word)
                    df = doc_freq[word]
                    idf = (len(papers) - df + 0.5) / (df + 0.5)
                    idf = max(0, idf) if idf > 0 else 0
                    score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))

            if score > 0:
                paper_scores[paper.id] = score

        # 排序返回
        sorted_papers = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_papers[:top_k]

    def _get_paper_text(self, paper: Paper) -> str:
        """获取论文的文本表示"""
        parts = [
            paper.title or "",
            paper.abstract or "",
            paper.task or "",
            paper.method or "",
            paper.contribution or ""
        ]
        if paper.datasets:
            parts.append(", ".join(paper.datasets))
        if paper.baselines:
            parts.append(", ".join(paper.baselines))
        return " ".join(parts)

    def execute(self, env: SharedEnvironment,
                query_analysis: QueryAnalysisResult) -> FunctionResult:
        """
        检索论文

        Args:
            env: SharedEnvironment
            query_analysis: 查询分析结果

        Returns:
            FunctionResult.data = List[Paper]
        """
        try:
            query = query_analysis.original_query
            keywords = query_analysis.keywords or []

            # 1. 向量检索
            vector_results = self._vector_search(env, query, top_k=20)

            # 2. BM25 检索
            bm25_results = self._bm25_search(env, query, keywords, top_k=20)

            # 3. 合并结果（简单加权）
            paper_scores: Dict[str, float] = {}
            for pid, score in vector_results:
                paper_scores[pid] = paper_scores.get(pid, 0) + score * 0.7
            for pid, score in bm25_results:
                paper_scores[pid] = paper_scores.get(pid, 0) + score * 0.3

            # 排序并获取论文
            sorted_papers = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)
            top_papers = [env.get_paper(pid) for pid, _ in sorted_papers[:10] if env.get_paper(pid)]

            papers = [p for p in top_papers if p is not None]

            logger.info(f"Retrieved {len(papers)} papers for query: {query}")
            return FunctionResult(success=True, data=papers)

        except Exception as e:
            logger.error(f"Failed to retrieve papers: {e}")
            return FunctionResult(success=False, error=str(e))
