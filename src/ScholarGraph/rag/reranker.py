"""重排序模型"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-Encoder 重排序"""

    def __init__(self, model: str = None):
        """
        初始化重排序模型

        Args:
            model: 模型名称（默认从 config.yaml 读取）
        """
        if model is None:
            from ..config import get_config
            config = get_config()
            model = config.rag.reranker.model
        self.model = model
        self._cross_encoder = None
        self._model_loaded = False

    def _load_model(self):
        """加载 Cross-Encoder 模型"""
        if self._model_loaded:
            return

        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self.model)
            self._model_loaded = True
            logger.info(f"Loaded Cross-Encoder model: {self.model}")
        except ImportError:
            logger.warning("sentence_transformers not available, using fallback reranker")
        except Exception as e:
            logger.warning(f"Failed to load Cross-Encoder model: {e}")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        对候选结果重排序

        Args:
            query: 查询字符串
            candidates: 候选结果列表 [{"chunk": Chunk, "score": float}, ...]
            top_k: 返回数量

        Returns:
            重排序后的结果列表
        """
        if not candidates:
            return []

        # 使用 Cross-Encoder 重排
        if self._cross_encoder is not None:
            return self._cross_encoder_rerank(query, candidates, top_k)
        else:
            return self._fallback_rerank(query, candidates, top_k)

    def _cross_encoder_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """使用 Cross-Encoder 重排序"""
        # 构建 query-document 对
        pairs = [
            (query, candidate["chunk"].content)
            for candidate in candidates
        ]

        # 批量预测相关性分数
        scores = self._cross_encoder.predict(pairs)

        # 合并原始分数和重排分数
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        # 按重排分数排序
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]

    def _fallback_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Fallback 重排序：结合原始分数和文本相似度"""
        query_words = set(query.lower().split())
        query_len = len(query_words)

        for candidate in candidates:
            chunk_words = set(candidate["chunk"].content.lower().split())
            # 计算 query 和 chunk 的词汇重叠度
            overlap = len(query_words & chunk_words) / query_len
            # 结合原始分数和重叠度
            candidate["rerank_score"] = candidate["score"] * 0.7 + overlap * 0.3

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def score(self, query: str, document: str) -> float:
        """
        计算 query 和 document 的相关性分数

        Args:
            query: 查询
            document: 文档

        Returns:
            相关性分数
        """
        if self._cross_encoder is not None:
            score = self._cross_encoder.predict([(query, document)])
            return float(score[0])
        else:
            # Fallback: 简单的词汇重叠
            query_words = set(query.lower().split())
            doc_words = set(document.lower().split())
            overlap = len(query_words & doc_words)
            return overlap / max(len(query_words), 1)

    def __repr__(self) -> str:
        return f"Reranker(model={self.model})"
