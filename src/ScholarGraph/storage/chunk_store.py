"""Chunk 存储层（向量数据库）"""

import json
import os
import numpy as np
from typing import List, Optional, Dict, Any
import logging

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .sqlite_base import SQLiteBase
from ..models.paper import Chunk

logger = logging.getLogger(__name__)


class ChunkStore:
    """Chunk 向量存储（FAISS + SQLite）"""

    CHUNKS_TABLE = "chunks"
    INDEX_FILE = "chunk_index.faiss"

    CHUNKS_SCHEMA = {
        "id": "TEXT PRIMARY KEY",
        "paper_id": "TEXT NOT NULL",
        "chunk_type": "TEXT NOT NULL",
        "content": "TEXT NOT NULL",
        "embedding": "TEXT NOT NULL",  # JSON: list
        "metadata": "TEXT",  # JSON: dict
        "created_at": "TEXT",
    }

    def __init__(self, db_path: str = "./data/ScholarGraph.db",
                 index_path: str = "./data/chunk_index.faiss",
                 dimension: int = 384,
                 timeout: float = 30.0):
        """
        初始化 Chunk 存储

        Args:
            db_path: SQLite 数据库路径
            index_path: FAISS 索引文件路径
            dimension: 向量维度
            timeout: 连接超时时间
        """
        self.db = SQLiteBase(db_path, timeout)
        self.index_path = index_path
        self.dimension = dimension
        self._index_dir = os.path.dirname(index_path)

        self._ensure_table()
        self._load_index()

    def _ensure_table(self) -> None:
        """确保表存在"""
        self.db.create_table(self.CHUNKS_TABLE, self.CHUNKS_SCHEMA)

        # 创建索引
        try:
            self.db.execute(
                f"CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON {self.CHUNKS_TABLE} (paper_id)"
            )
            self.db.execute(
                f"CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON {self.CHUNKS_TABLE} (chunk_type)"
            )
        except Exception as e:
            logger.warning(f"Failed to create index: {e}")

    def _load_index(self) -> None:
        """加载或创建 FAISS 索引"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, using numpy fallback")
            self.index = None
            return

        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.index = self._create_index()
        else:
            self.index = self._create_index()

    def _create_index(self):
        """创建新的 FAISS 索引"""
        if not FAISS_AVAILABLE:
            return None
        # 使用内积索引（余弦相似度需要归一化）
        return faiss.IndexFlatIP(self.dimension)

    def _save_index(self) -> None:
        """保存 FAISS 索引到文件"""
        if FAISS_AVAILABLE and self.index is not None and self._index_dir:
            os.makedirs(self._index_dir, exist_ok=True)
            try:
                faiss.write_index(self.index, self.index_path)
                logger.info(f"Saved FAISS index to {self.index_path}")
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")

    def _get_embedding_matrix(self) -> np.ndarray:
        """获取所有嵌入向量的矩阵"""
        sql = f"SELECT embedding FROM {self.CHUNKS_TABLE}"
        rows = self.db.fetch_column(sql)
        if not rows:
            return np.array([])

        embeddings = []
        for row in rows:
            emb = json.loads(row) if isinstance(row, str) else row
            embeddings.append(emb)

        return np.array(embeddings).astype('float32')

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """归一化嵌入向量（用于余弦相似度）"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    def add_chunks(self, chunks: List[Chunk]) -> bool:
        """
        添加 chunks

        Args:
            chunks: Chunk 对象列表

        Returns:
            是否添加成功
        """
        if not chunks:
            return True

        try:
            # 批量插入 SQLite
            data_list = [chunk.to_dict() for chunk in chunks]
            columns = ", ".join(data_list[0].keys())
            placeholders = ", ".join(["?" for _ in data_list[0]])

            # 转换为元组列表
            params_list = [tuple(d.values()) for d in data_list]

            sql = f"INSERT OR REPLACE INTO {self.CHUNKS_TABLE} ({columns}) VALUES ({placeholders})"
            self.db.execute_many(sql, params_list)

            # 更新 FAISS 索引
            if FAISS_AVAILABLE and self.index is not None:
                embeddings = np.array([chunk.embedding for chunk in chunks]).astype('float32')
                embeddings = self._normalize_embeddings(embeddings)
                self.index.add(embeddings)
                self._save_index()

            logger.info(f"Added {len(chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            return False

    def get(self, chunk_id: str) -> Optional[Chunk]:
        """
        根据 ID 获取 chunk

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk 对象或 None
        """
        sql = f"SELECT * FROM {self.CHUNKS_TABLE} WHERE id = ?"
        row = self.db.fetch_one(sql, (chunk_id,))
        if row:
            return Chunk.from_dict(dict(row))
        return None

    def get_by_paper(self, paper_id: str) -> List[Chunk]:
        """
        获取指定论文的所有 chunks

        Args:
            paper_id: 论文 ID

        Returns:
            Chunk 对象列表
        """
        sql = f"SELECT * FROM {self.CHUNKS_TABLE} WHERE paper_id = ?"
        rows = self.db.fetch_all(sql, (paper_id,))
        return [Chunk.from_dict(dict(row)) for row in rows]

    def get_by_type(self, chunk_type: str, limit: int = 100) -> List[Chunk]:
        """
        获取指定类型的 chunks

        Args:
            chunk_type: chunk 类型
            limit: 返回数量限制

        Returns:
            Chunk 对象列表
        """
        sql = f"SELECT * FROM {self.CHUNKS_TABLE} WHERE chunk_type = ? LIMIT ?"
        rows = self.db.fetch_all(sql, (chunk_type, limit))
        return [Chunk.from_dict(dict(row)) for row in rows]

    def search(self, query_vector: List[float], top_k: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        向量检索

        Args:
            query_vector: 查询向量
            top_k: 返回数量
            filters: 过滤条件

        Returns:
            相关的 Chunk 列表
        """
        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            return self._faiss_search(query_vector, top_k, filters)
        else:
            return self._numpy_search(query_vector, top_k, filters)

    def _faiss_search(self, query_vector: List[float], top_k: int,
                       filters: Optional[Dict[str, Any]]) -> List[Chunk]:
        """使用 FAISS 进行向量检索"""
        query = np.array([query_vector]).astype('float32')
        query = self._normalize_embeddings(query)

        # 搜索
        scores, indices = self.index.search(query, top_k)

        # 获取 chunk IDs
        sql = f"SELECT id FROM {self.CHUNKS_TABLE}"
        all_ids = self.db.fetch_column(sql)

        # 构建 id 到索引的映射
        id_to_idx = {id_val: idx for idx, id_val in enumerate(all_ids)}

        # 获取结果 chunks
        result_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(all_ids):
                chunk_id = all_ids[idx]
                chunk = self.get(chunk_id)
                if chunk:
                    result_chunks.append(chunk)

        # 应用过滤器
        if filters:
            result_chunks = self._apply_filters(result_chunks, filters)

        return result_chunks[:top_k]

    def _numpy_search(self, query_vector: List[float], top_k: int,
                      filters: Optional[Dict[str, Any]]) -> List[Chunk]:
        """使用 numpy 进行向量检索（fallback）"""
        embeddings = self._get_embedding_matrix()
        if embeddings.size == 0:
            return []

        # 获取所有 chunks
        sql = f"SELECT * FROM {self.CHUNKS_TABLE}"
        all_chunks = [Chunk.from_dict(dict(row)) for row in self.db.fetch_all(sql)]

        if not all_chunks:
            return []

        # 计算余弦相似度
        query = np.array(query_vector).astype('float32')
        query = query / np.linalg.norm(query)

        similarities = np.dot(embeddings, query)

        # 排序
        top_indices = np.argsort(similarities)[::-1][:top_k]

        result_chunks = [all_chunks[i] for i in top_indices]

        # 应用过滤器
        if filters:
            result_chunks = self._apply_filters(result_chunks, filters)

        return result_chunks[:top_k]

    def _apply_filters(self, chunks: List[Chunk], filters: Dict[str, Any]) -> List[Chunk]:
        """应用过滤器"""
        result = []
        for chunk in chunks:
            match = True
            for key, value in filters.items():
                if key == "paper_id" and chunk.paper_id != value:
                    match = False
                    break
                elif key == "chunk_type" and chunk.chunk_type != value:
                    match = False
                    break
                elif key == "year" and chunk.metadata.get("year") != value:
                    match = False
                    break
                elif key == "field" and chunk.metadata.get("field") != value:
                    match = False
                    break
            if match:
                result.append(chunk)
        return result

    def delete(self, chunk_id: str) -> bool:
        """删除 chunk"""
        try:
            sql = f"DELETE FROM {self.CHUNKS_TABLE} WHERE id = ?"
            self.db.execute(sql, (chunk_id,))
            logger.info(f"Deleted chunk: {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunk: {e}")
            return False

    def delete_by_paper(self, paper_id: str) -> bool:
        """删除指定论文的所有 chunks"""
        try:
            sql = f"DELETE FROM {self.CHUNKS_TABLE} WHERE paper_id = ?"
            self.db.execute(sql, (paper_id,))
            logger.info(f"Deleted chunks for paper: {paper_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            return False

    def count(self) -> int:
        """获取 chunk 总数"""
        sql = f"SELECT COUNT(*) FROM {self.CHUNKS_TABLE}"
        return self.db.count(sql)
