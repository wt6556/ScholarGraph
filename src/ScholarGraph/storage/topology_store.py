"""拓扑图存储层"""

import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
import re

from .sqlite_base import SQLiteBase

logger = logging.getLogger(__name__)

def normalize_method_name(name: str) -> str:
    """标准化方法名称（仅处理括号中的缩写）"""
    if not name:
        return name
    name_clean = name.strip()

    # 已知方法的标准缩写
    KNOWN_METHODS = {"LoRA", "SoRA", "AdaLoRA", "S-LoRA", "MoELoRA", "LoRAMoE", "MOLE"}

    # 如果已是已知缩写，直接返回
    if name_clean in KNOWN_METHODS:
        return name_clean

    # 检查括号中的缩写
    match = re.search(r'\(([^)]+)\)', name_clean)
    if match:
        abbrev = match.group(1)
        if abbrev in KNOWN_METHODS:
            return abbrev

    return name_clean


@dataclass
class MethodNode:
    """方法节点"""
    id: str
    topic: str  # 方法领域，如 "LoRA"
    paper_id: str
    method_name: str
    improvement_direction: str  # efficiency, accuracy, memory, etc.
    key_innovation: str
    is_root: bool
    parent_methods: List[str]  # JSON: list of method names
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "paper_id": self.paper_id,
            "method_name": self.method_name,
            "improvement_direction": self.improvement_direction,
            "key_innovation": self.key_innovation,
            "is_root": self.is_root,
            "parent_methods": json.dumps(self.parent_methods),
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MethodNode':
        parent_methods = data.get('parent_methods', '[]')
        if isinstance(parent_methods, str):
            parent_methods = json.loads(parent_methods)
        return cls(
            id=data['id'],
            topic=data['topic'],
            paper_id=data['paper_id'],
            method_name=data['method_name'],
            improvement_direction=data.get('improvement_direction', ''),
            key_innovation=data.get('key_innovation', ''),
            is_root=bool(data.get('is_root', False)),
            parent_methods=parent_methods,
            created_at=data.get('created_at', '')
        )


@dataclass
class MethodEdge:
    """方法关系边"""
    id: str
    topic: str
    from_method: str
    to_method: str
    improvement_direction: str
    description: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "from_method": self.from_method,
            "to_method": self.to_method,
            "improvement_direction": self.improvement_direction,
            "description": self.description,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MethodEdge':
        return cls(**data)


@dataclass
class PairwiseRelation:
    """成对关系"""
    id: str
    paper_a_id: str
    paper_b_id: str
    relation_type: str  # improves, compares, similar_to, based_on
    description: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "paper_a_id": self.paper_a_id,
            "paper_b_id": self.paper_b_id,
            "relation_type": self.relation_type,
            "description": self.description,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PairwiseRelation':
        return cls(**data)


class TopologyStore:
    """拓扑图存储（SQLite）"""

    METHOD_GRAPH_TABLE = "method_graph"
    TOPOLOGY_EDGES_TABLE = "topology_edges"
    RELATIONS_TABLE = "relations"

    METHOD_GRAPH_SCHEMA = {
        "id": "TEXT PRIMARY KEY",
        "topic": "TEXT NOT NULL",
        "paper_id": "TEXT NOT NULL",
        "method_name": "TEXT NOT NULL",
        "improvement_direction": "TEXT",
        "key_innovation": "TEXT",
        "is_root": "INTEGER DEFAULT 0",
        "parent_methods": "TEXT",  # JSON
        "created_at": "TEXT",
    }

    TOPOLOGY_EDGES_SCHEMA = {
        "id": "TEXT PRIMARY KEY",
        "topic": "TEXT NOT NULL",
        "from_method": "TEXT NOT NULL",
        "to_method": "TEXT NOT NULL",
        "improvement_direction": "TEXT",
        "description": "TEXT",
        "created_at": "TEXT",
    }

    RELATIONS_SCHEMA = {
        "id": "TEXT PRIMARY KEY",
        "paper_a_id": "TEXT NOT NULL",
        "paper_b_id": "TEXT NOT NULL",
        "relation_type": "TEXT NOT NULL",
        "description": "TEXT",
        "created_at": "TEXT",
    }

    def __init__(self, db_path: str = "./data/ScholarGraph.db", timeout: float = 30.0):
        """
        初始化拓扑图存储

        Args:
            db_path: 数据库文件路径
            timeout: 连接超时时间
        """
        self.db = SQLiteBase(db_path, timeout)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """确保表存在"""
        self.db.create_table(self.METHOD_GRAPH_TABLE, self.METHOD_GRAPH_SCHEMA)
        self.db.create_table(self.TOPOLOGY_EDGES_TABLE, self.TOPOLOGY_EDGES_SCHEMA)
        self.db.create_table(self.RELATIONS_TABLE, self.RELATIONS_SCHEMA)

        # 创建索引
        try:
            self.db.execute(
                f"CREATE INDEX IF NOT EXISTS idx_method_graph_topic ON {self.METHOD_GRAPH_TABLE} (topic)"
            )
            self.db.execute(
                f"CREATE INDEX IF NOT EXISTS idx_topology_edges_topic ON {self.TOPOLOGY_EDGES_TABLE} (topic)"
            )
            self.db.execute(
                f"CREATE INDEX IF NOT EXISTS idx_relations_paper_a ON {self.RELATIONS_TABLE} (paper_a_id)"
            )
            self.db.execute(
                f"CREATE INDEX IF NOT EXISTS idx_relations_paper_b ON {self.RELATIONS_TABLE} (paper_b_id)"
            )
        except Exception as e:
            logger.warning(f"Failed to create index: {e}")

    # ==================== Method Graph 操作 ====================

    def add_method_node(self, node: MethodNode) -> bool:
        """添加方法节点"""
        try:
            data = node.to_dict()
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            sql = f"INSERT OR REPLACE INTO {self.METHOD_GRAPH_TABLE} ({columns}) VALUES ({placeholders})"
            self.db.execute(sql, tuple(data.values()))
            logger.info(f"Added method node: {node.method_name} in {node.topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to add method node: {e}")
            return False

    def get_method_nodes(self, topic: str) -> List[MethodNode]:
        """获取指定领域的所有方法节点"""
        sql = f"SELECT * FROM {self.METHOD_GRAPH_TABLE} WHERE topic = ?"
        rows = self.db.fetch_all(sql, (topic,))
        return [MethodNode.from_dict(dict(row)) for row in rows]

    def get_method_node(self, topic: str, method_name: str) -> Optional[MethodNode]:
        """获取指定方法节点"""
        sql = f"SELECT * FROM {self.METHOD_GRAPH_TABLE} WHERE topic = ? AND method_name = ?"
        row = self.db.fetch_one(sql, (topic, method_name))
        if row:
            return MethodNode.from_dict(dict(row))
        return None

    def get_root_methods(self, topic: str) -> List[MethodNode]:
        """获取指定领域的根方法（基础方法）"""
        sql = f"SELECT * FROM {self.METHOD_GRAPH_TABLE} WHERE topic = ? AND is_root = 1"
        rows = self.db.fetch_all(sql, (topic,))
        return [MethodNode.from_dict(dict(row)) for row in rows]

    # ==================== Topology Edges 操作 ====================

    def add_relation(self, edge: MethodEdge) -> bool:
        """添加方法关系边"""
        try:
            data = edge.to_dict()
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            sql = f"INSERT OR REPLACE INTO {self.TOPOLOGY_EDGES_TABLE} ({columns}) VALUES ({placeholders})"
            self.db.execute(sql, tuple(data.values()))
            logger.info(f"Added relation: {edge.from_method} -> {edge.to_method}")
            return True
        except Exception as e:
            logger.error(f"Failed to add relation: {e}")
            return False

    def get_topology_edges(self, topic: str) -> List[MethodEdge]:
        """获取指定领域的拓扑边"""
        sql = f"SELECT * FROM {self.TOPOLOGY_EDGES_TABLE} WHERE topic = ?"
        rows = self.db.fetch_all(sql, (topic,))
        return [MethodEdge.from_dict(dict(row)) for row in rows]

    def get_child_methods(self, topic: str, method_name: str) -> List[str]:
        """获取指定方法的子方法"""
        sql = f"SELECT to_method FROM {self.TOPOLOGY_EDGES_TABLE} WHERE topic = ? AND from_method = ?"
        return self.db.fetch_column(sql, (topic, method_name))

    def get_parent_methods(self, topic: str, method_name: str) -> List[str]:
        """获取指定方法的父方法"""
        sql = f"SELECT from_method FROM {self.TOPOLOGY_EDGES_TABLE} WHERE topic = ? AND to_method = ?"
        return self.db.fetch_column(sql, (topic, method_name))

    # ==================== Pairwise Relations 操作 ====================

    def add_pairwise_relation(self, relation: PairwiseRelation) -> bool:
        """添加成对关系"""
        try:
            data = relation.to_dict()
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            sql = f"INSERT OR REPLACE INTO {self.RELATIONS_TABLE} ({columns}) VALUES ({placeholders})"
            self.db.execute(sql, tuple(data.values()))
            logger.info(f"Added pairwise relation: {relation.paper_a_id} -> {relation.paper_b_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add pairwise relation: {e}")
            return False

    def get_relations_between(self, paper_a_id: str, paper_b_id: str) -> List[PairwiseRelation]:
        """获取两篇论文之间的关系"""
        sql = f"""SELECT * FROM {self.RELATIONS_TABLE}
                  WHERE (paper_a_id = ? AND paper_b_id = ?)
                     OR (paper_a_id = ? AND paper_b_id = ?)"""
        rows = self.db.fetch_all(sql, (paper_a_id, paper_b_id, paper_b_id, paper_a_id))
        return [PairwiseRelation.from_dict(dict(row)) for row in rows]

    def get_paper_relations(self, paper_id: str) -> List[PairwiseRelation]:
        """获取与指定论文相关的所有关系"""
        sql = f"SELECT * FROM {self.RELATIONS_TABLE} WHERE paper_a_id = ? OR paper_b_id = ?"
        rows = self.db.fetch_all(sql, (paper_id, paper_id))
        return [PairwiseRelation.from_dict(dict(row)) for row in rows]

    # ==================== 拓扑图构建 ====================

    def get_method_topology(self, topic: str) -> Dict[str, Any]:
        """
        获取方法拓扑图数据

        Args:
            topic: 方法领域

        Returns:
            拓扑图数据 {
                nodes: [...],
                edges: [...]
            }
        """
        nodes = self.get_method_nodes(topic)
        edges = self.get_topology_edges(topic)

        return {
            "nodes": [
                {
                    "id": node.method_name,
                    "paper_id": node.paper_id,
                    "improvement_direction": node.improvement_direction,
                    "key_innovation": node.key_innovation,
                    "is_root": node.is_root,
                    "parent_methods": node.parent_methods
                }
                for node in nodes
            ],
            "edges": [
                {
                    "from": edge.from_method,
                    "to": edge.to_method,
                    "direction": edge.improvement_direction,
                    "description": edge.description
                }
                for edge in edges
            ]
        }

    def get_all_topology(self) -> Dict[str, Any]:
        """
        获取所有拓扑数据

        Returns:
            所有领域的拓扑数据汇总
        """
        all_nodes = []
        all_edges = []
        seen_topics = set()

        # 从 method_graph 表获取所有 topic
        try:
            rows = self.db.fetch_all(f"SELECT DISTINCT topic FROM {self.METHOD_GRAPH_TABLE}")
            topics = [row['topic'] for row in rows]
        except Exception:
            topics = []

        for topic in topics:
            if topic in seen_topics:
                continue
            seen_topics.add(topic)

            topology = self.get_method_topology(topic)
            all_nodes.extend(topology.get("nodes", []))
            all_edges.extend(topology.get("edges", []))

        # 去重
        unique_nodes = {n["paper_id"]: n for n in all_nodes}
        unique_edges = []
        seen_edge_ids = set()
        for e in all_edges:
            edge_id = f"{e['from']}->{e['to']}"
            if edge_id not in seen_edge_ids:
                seen_edge_ids.add(edge_id)
                unique_edges.append(e)

        return {
            "nodes": list(unique_nodes.values()),
            "edges": unique_edges
        }

    def build_topology_from_paper(self, paper_id: str, method: str, topic: str,
                                    improvement_direction: str,
                                    parent_methods: List[str] = None,
                                    is_root: bool = False) -> bool:
        """
        从论文构建拓扑节点

        Args:
            paper_id: 论文 ID
            method: 方法名称
            topic: 方法领域
            improvement_direction: 改进方向
            parent_methods: 父方法列表
            is_root: 是否是根方法

        Returns:
            是否成功
        """
        from datetime import datetime
        import hashlib

        node_id = hashlib.md5(f"{topic}_{method}".encode()).hexdigest()[:16]
        node = MethodNode(
            id=node_id,
            topic=topic,
            paper_id=paper_id,
            method_name=method,
            improvement_direction=improvement_direction,
            key_innovation="",
            is_root=is_root,
            parent_methods=parent_methods or [],
            created_at=datetime.now().isoformat()
        )

        if not self.add_method_node(node):
            return False

        # 添加边
        if parent_methods:
            for parent in parent_methods:
                edge_id = hashlib.md5(f"{topic}_{parent}_{method}".encode()).hexdigest()[:16]
                edge = MethodEdge(
                    id=edge_id,
                    topic=topic,
                    from_method=parent,
                    to_method=method,
                    improvement_direction=improvement_direction,
                    description=f"{method} 改进了 {parent}",
                    created_at=datetime.now().isoformat()
                )
                self.add_relation(edge)

        return True

    def normalize_method_names(self) -> Dict[str, int]:
        """
        清理和统一数据库中不一致的方法命名

        Returns:
            包含更新统计的字典
        """
        stats = {"nodes_updated": 0, "edges_updated": 0, "duplicates_merged": 0}

        # 1. 收集所有需要标准化的 method_name
        sql = f"SELECT id, topic, paper_id, method_name, parent_methods FROM {self.METHOD_GRAPH_TABLE}"
        rows = self.db.fetch_all(sql)

        # 建立 method_name -> 标准名称的映射
        name_mapping = {}  # old_name -> new_standard_name
        for row in rows:
            old_name = row['method_name']
            new_name = normalize_method_name(old_name)
            if old_name != new_name:
                name_mapping[old_name] = new_name

        # 2. 更新 method_graph 表中的 method_name
        for old_name, new_name in name_mapping.items():
            # 检查有多少行需要更新
            check_sql = f"SELECT id, paper_id FROM {self.METHOD_GRAPH_TABLE} WHERE method_name = ?"
            rows_to_update = self.db.fetch_all(check_sql, (old_name,))

            for row in rows_to_update:
                old_id = row['id']
                paper_id = row['paper_id']
                # 生成基于 paper_id 的唯一 ID
                new_id = f"{paper_id[:16]}_node"
                update_sql = f"UPDATE {self.METHOD_GRAPH_TABLE} SET method_name = ?, id = ? WHERE id = ?"
                self.db.execute(update_sql, (new_name, new_id, old_id))
                stats["nodes_updated"] += 1

            # 更新 parent_methods JSON 字段
            for row in rows:
                if row['method_name'] == old_name:
                    parent_methods = row['parent_methods']
                    if isinstance(parent_methods, str):
                        try:
                            import json
                            parents = json.loads(parent_methods)
                            normalized_parents = [normalize_method_name(p) for p in parents]
                            if parents != normalized_parents:
                                update_parent_sql = f"UPDATE {self.METHOD_GRAPH_TABLE} SET parent_methods = ? WHERE id = ?"
                                self.db.execute(update_parent_sql, (json.dumps(normalized_parents), row['id']))
                        except:
                            pass

        # 3. 更新 topology_edges 表中的 from_method 和 to_method
        edges_sql = f"SELECT id, topic, from_method, to_method, improvement_direction, description, created_at FROM {self.TOPOLOGY_EDGES_TABLE}"
        edge_rows = self.db.fetch_all(edges_sql)

        for row in edge_rows:
            old_from = row['from_method']
            old_to = row['to_method']
            new_from = normalize_method_name(old_from)
            new_to = normalize_method_name(old_to)

            if old_from != new_from or old_to != new_to:
                # 删除旧记录
                delete_sql = f"DELETE FROM {self.TOPOLOGY_EDGES_TABLE} WHERE id = ?"
                self.db.execute(delete_sql, (row['id'],))

                # 插入新记录（使用基于内容的新ID）
                import hashlib
                new_edge_id = hashlib.md5(f"{row['topic']}_{new_from}_{new_to}".encode()).hexdigest()[:16]
                insert_sql = f"""
                    INSERT OR REPLACE INTO {self.TOPOLOGY_EDGES_TABLE}
                    (id, topic, from_method, to_method, improvement_direction, description, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                self.db.execute(insert_sql, (
                    new_edge_id,
                    row['topic'],
                    new_from,
                    new_to,
                    row['improvement_direction'],
                    row['description'],
                    row['created_at']
                ))
                stats["edges_updated"] += 1

        logger.info(f"Normalized method names: {stats}")
        return stats
