"""RRF 融合"""

import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class Fusion:
    """RRF (Reciprocal Rank Fusion) 融合多个检索结果"""

    def __init__(self, k: int = 60):
        """
        初始化融合器

        Args:
            k: RRF 公式中的常数参数
        """
        self.k = k

    def rrf_fuse(
        self,
        result_lists: List[List[Tuple[Any, float]]],
        top_k: int = 10
    ) -> List[Tuple[Any, float]]:
        """
        RRF 融合多个检索结果列表

        Args:
            result_lists: 多个检索结果列表，每个元素是 (doc_id, score) 元组
            top_k: 返回数量

        Returns:
            融合后的结果列表 [(doc_id, fused_score), ...]
        """
        if not result_lists:
            return []

        if len(result_lists) == 1:
            return result_lists[0][:top_k]

        # 构建 doc_id 到分数的映射
        doc_scores: Dict[Any, float] = {}

        for result_list in result_lists:
            for rank, (doc_id, score) in enumerate(result_list):
                # RRF 公式: 1 / (k + rank)
                rrf_score = 1.0 / (self.k + rank + 1)
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += rrf_score

        # 排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

    def weighted_fuse(
        self,
        result_lists: List[List[Tuple[Any, float]]],
        weights: List[float],
        top_k: int = 10
    ) -> List[Tuple[Any, float]]:
        """
        加权融合多个检索结果

        Args:
            result_lists: 多个检索结果列表
            weights: 对应的权重列表
            top_k: 返回数量

        Returns:
            融合后的结果列表
        """
        if not result_lists or not weights:
            return []

        if len(result_lists) != len(weights):
            raise ValueError("result_lists and weights must have the same length")

        doc_scores: Dict[Any, float] = {}

        for result_list, weight in zip(result_lists, weights):
            for doc_id, score in result_list:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += score * weight

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

    def __repr__(self) -> str:
        return f"Fusion(k={self.k})"
