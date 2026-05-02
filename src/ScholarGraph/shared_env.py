"""SharedEnvironment - 所有 Agent 和 Function 的数据共享中枢"""

from typing import Any, Optional, List, Dict
from dataclasses import dataclass, field
import logging

from .storage.paper_store import PaperStore
from .storage.chunk_store import ChunkStore
from .storage.topology_store import TopologyStore
from .models.paper import Paper, Chunk
from .models.query import QueryAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """查询上下文"""
    query: str
    query_type: str
    keywords: List[str]
    constraints: Dict[str, Any] = field(default_factory=dict)
    result: Any = None


class SharedEnvironment:
    """
    共享环境：所有 Agent/Function 的数据中枢

    提供统一的接口访问 paper_store、chunk_store 和 topology_store
    """

    def __init__(
        self,
        db_path: str = "./data/ScholarGraph.db",
        vector_index_path: str = "./data/chunk_index.faiss",
        embedding_dimension: int = 384
    ):
        """
        初始化共享环境

        Args:
            db_path: SQLite 数据库路径
            vector_index_path: FAISS 索引文件路径
            embedding_dimension: 嵌入向量维度
        """
        # 初始化存储组件
        self.paper_store = PaperStore(db_path)
        self.chunk_store = ChunkStore(
            db_path=db_path,
            index_path=vector_index_path,
            dimension=embedding_dimension
        )
        self.topology_store = TopologyStore(db_path)

        # 运行时状态
        self.current_paper: Optional[Paper] = None
        self.current_query: Optional[QueryContext] = None
        self.query_result: Optional[Dict[str, Any]] = None

        logger.info("SharedEnvironment initialized")

    # ==================== Paper 操作 ====================

    def add_paper(self, paper: Paper) -> bool:
        """
        添加论文

        Args:
            paper: Paper 对象

        Returns:
            是否添加成功
        """
        return self.paper_store.add(paper)

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """
        获取论文

        Args:
            paper_id: 论文 ID

        Returns:
            Paper 对象或 None
        """
        return self.paper_store.get(paper_id)

    def update_paper(self, paper: Paper) -> bool:
        """
        更新论文

        Args:
            paper: Paper 对象

        Returns:
            是否更新成功
        """
        # 同时更新运行时状态中的 current_paper
        if self.current_paper and self.current_paper.id == paper.id:
            # 更新现有对象的属性
            self.current_paper.__dict__.update(paper.__dict__)
            logger.debug(f"Updated current_paper in memory: {paper.id}")
        else:
            # 设置为新的 current_paper
            self.current_paper = paper
            logger.debug(f"Set current_paper: {paper.id}")

        return self.paper_store.update(paper.id, paper)

    def delete_paper(self, paper_id: str) -> bool:
        """
        删除论文

        Args:
            paper_id: 论文 ID

        Returns:
            是否删除成功
        """
        # 同时删除关联的 chunks
        self.chunk_store.delete_by_paper(paper_id)
        return self.paper_store.delete(paper_id)

    def query_papers(self, filters: Dict[str, Any], limit: int = 100) -> List[Paper]:
        """
        根据过滤条件查询论文

        Args:
            filters: 过滤条件
            limit: 返回数量限制

        Returns:
            Paper 对象列表
        """
        return self.paper_store.query(filters, limit)

    def get_all_papers(self, limit: int = 1000, offset: int = 0) -> List[Paper]:
        """获取所有论文"""
        return self.paper_store.get_all(limit, offset)

    def get_recent_papers(self, limit: int = 20) -> List[Paper]:
        """获取最近的论文"""
        return self.paper_store.get_recent(limit)

    def search_papers(self, keyword: str, limit: int = 50) -> List[Paper]:
        """搜索论文"""
        return self.paper_store.search(keyword, limit)

    def paper_exists(self, paper_id: str) -> bool:
        """检查论文是否存在"""
        return self.paper_store.exists(paper_id)

    # ==================== Chunk 操作 ====================

    def add_chunks(self, chunks: List[Chunk]) -> bool:
        """
        添加 chunks

        Args:
            chunks: Chunk 对象列表

        Returns:
            是否添加成功
        """
        return self.chunk_store.add_chunks(chunks)

    def get_chunks_by_paper(self, paper_id: str) -> List[Chunk]:
        """
        获取论文的所有 chunks

        Args:
            paper_id: 论文 ID

        Returns:
            Chunk 对象列表
        """
        return self.chunk_store.get_by_paper(paper_id)

    def search_chunks(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        向量检索 chunks

        Args:
            query_vector: 查询向量
            top_k: 返回数量
            filters: 过滤条件

        Returns:
            相关的 Chunk 列表
        """
        return self.chunk_store.search(query_vector, top_k, filters)

    def delete_chunks_by_paper(self, paper_id: str) -> bool:
        """删除论文的所有 chunks"""
        return self.chunk_store.delete_by_paper(paper_id)

    # ==================== Topology 操作 ====================

    def add_method_node(self, node) -> bool:
        """添加方法节点"""
        return self.topology_store.add_method_node(node)

    def get_method_topology(self, topic: str) -> Dict[str, Any]:
        """
        获取方法拓扑图

        Args:
            topic: 方法领域

        Returns:
            拓扑图数据
        """
        return self.topology_store.get_method_topology(topic)

    def add_pairwise_relation(self, relation) -> bool:
        """添加成对关系"""
        return self.topology_store.add_pairwise_relation(relation)

    def get_relations_between(self, paper_a_id: str, paper_b_id: str) -> List:
        """获取两篇论文之间的关系"""
        return self.topology_store.get_relations_between(paper_a_id, paper_b_id)

    # ==================== 运行时状态管理 ====================

    def set_current_paper(self, paper: Paper) -> None:
        """
        设置当前处理的论文

        Args:
            paper: Paper 对象
        """
        self.current_paper = paper
        logger.debug(f"Set current paper: {paper.id}")

    def get_current_paper(self) -> Optional[Paper]:
        """
        获取当前论文

        Returns:
            Paper 对象或 None
        """
        return self.current_paper

    def clear_current_paper(self) -> None:
        """清除当前论文"""
        self.current_paper = None

    def set_query_context(self, query: str, query_type: str,
                          keywords: List[str], constraints: Dict[str, Any] = None) -> None:
        """
        设置查询上下文

        Args:
            query: 原始查询
            query_type: 查询类型
            keywords: 关键词
            constraints: 约束条件
        """
        self.current_query = QueryContext(
            query=query,
            query_type=query_type,
            keywords=keywords,
            constraints=constraints or {}
        )
        logger.debug(f"Set query context: {query_type}")

    def get_query_context(self) -> Optional[QueryContext]:
        """获取查询上下文"""
        return self.current_query

    def clear_query_context(self) -> None:
        """清除查询上下文"""
        self.current_query = None

    def set_query_result(self, result: Dict[str, Any]) -> None:
        """
        设置查询结果

        Args:
            result: 查询结果字典
        """
        self.query_result = result

    def get_query_result(self) -> Optional[Dict[str, Any]]:
        """获取查询结果"""
        return self.query_result

    def clear_query_result(self) -> None:
        """清除查询结果"""
        self.query_result = None

    # ==================== SQL 查询 ====================

    def sql_query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        执行原始 SQL 查询（仅用于 paper_store）

        Args:
            sql: SQL 语句
            params: 参数

        Returns:
            结果列表
        """
        return self.paper_store.raw_query(sql, params)

    # ==================== 便捷方法 ====================

    def get_paper_count(self) -> int:
        """获取论文总数"""
        return self.paper_store.count()

    def get_chunk_count(self) -> int:
        """获取 chunk 总数"""
        return self.chunk_store.count()

    def reset(self) -> None:
        """重置运行时状态（保留存储数据）"""
        self.current_paper = None
        self.current_query = None
        self.query_result = None
        logger.info("SharedEnvironment reset")

    def __repr__(self) -> str:
        return (
            f"SharedEnvironment("
            f"papers={self.paper_store.count()}, "
            f"chunks={self.chunk_store.count()})"
        )
