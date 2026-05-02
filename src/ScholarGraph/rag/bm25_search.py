"""BM25 关键词检索"""

import logging
import math
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class BM25Search:
    """BM25 关键词检索"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化 BM25

        Args:
            k1: 词频饱和度参数
            b: 文档长度归一化参数
        """
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.avg_doc_len = 0
        self.doc_lens: List[int] = []
        self.term_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}

    def index(self, documents: List[str]) -> None:
        """
        建立索引

        Args:
            documents: 文档列表
        """
        self.documents = documents
        self.doc_lens = [len(doc.split()) for doc in documents]
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0

        # 计算词频
        self.term_freqs = [Counter(doc.lower().split()) for doc in documents]

        # 计算 IDF
        doc_count = len(documents)
        all_terms = set()
        for tf in self.term_freqs:
            all_terms.update(tf.keys())

        for term in all_terms:
            doc_with_term = sum(1 for tf in self.term_freqs if term in tf)
            self.idf[term] = math.log((doc_count - doc_with_term + 0.5) / (doc_with_term + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """
        BM25 检索

        Args:
            query: 查询字符串
            top_k: 返回数量

        Returns:
            [(doc_index, score), ...] 按分数降序
        """
        # TODO: 实现完整的 BM25 算法
        if not self.documents:
            return []

        query_terms = query.lower().split()
        scores = []

        for i, doc in enumerate(self.documents):
            score = 0.0
            doc_len = self.doc_lens[i]

            for term in query_terms:
                if term in self.term_freqs[i]:
                    tf = self.term_freqs[i][term]
                    idf = self.idf.get(term, 0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                    score += idf * numerator / denominator

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def __repr__(self) -> str:
        return f"BM25Search(k1={self.k1}, b={self.b})"
