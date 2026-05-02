"""论文存储层"""

import json
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from .sqlite_base import SQLiteBase
from ..models.paper import Paper

logger = logging.getLogger(__name__)


class PaperStore:
    """论文存储（SQLite）"""

    TABLE_NAME = "papers"

    SCHEMA = {
        "id": "TEXT PRIMARY KEY",
        "title": "TEXT NOT NULL",
        "authors": "TEXT",  # JSON: list
        "year": "INTEGER",
        "venue": "TEXT",
        "ccf_rating": "TEXT",
        "doi": "TEXT",
        "abstract": "TEXT",
        "sections": "TEXT",  # JSON: dict
        "task": "TEXT",
        "assumption": "TEXT",
        "motivation": "TEXT",
        "method": "TEXT",
        "method_category": "TEXT",
        "core_idea": "TEXT",
        "baselines": "TEXT",  # JSON: list
        "datasets": "TEXT",  # JSON: list
        "improvements": "TEXT",  # JSON: list
        "experiment_settings": "TEXT",  # JSON: dict
        "contribution": "TEXT",
        "limitation": "TEXT",
        "paper_field": "TEXT",
        "subfield": "TEXT",
        "topic": "TEXT",  # JSON: list
        "pdf_path": "TEXT",
        "parsed_json_path": "TEXT",
        "created_at": "TEXT",
        "updated_at": "TEXT",
    }

    def __init__(self, db_path: str = "./data/ScholarGraph.db", timeout: float = 30.0):
        """
        初始化论文存储

        Args:
            db_path: 数据库文件路径
            timeout: 连接超时时间
        """
        self.db = SQLiteBase(db_path, timeout)
        self._ensure_table()

    def _ensure_table(self) -> None:
        """确保表存在"""
        self.db.create_table(self.TABLE_NAME, self.SCHEMA)

    def add(self, paper: Paper) -> bool:
        """
        添加论文

        Args:
            paper: Paper 对象

        Returns:
            是否添加成功
        """
        try:
            data = paper.to_dict()
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            sql = f"INSERT OR REPLACE INTO {self.TABLE_NAME} ({columns}) VALUES ({placeholders})"
            self.db.execute(sql, tuple(data.values()))
            logger.info(f"Added paper: {paper.id} - {paper.title[:50]}")
            return True
        except Exception as e:
            logger.error(f"Failed to add paper: {e}")
            return False

    def get(self, paper_id: str) -> Optional[Paper]:
        """
        根据 ID 获取论文

        Args:
            paper_id: 论文 ID

        Returns:
            Paper 对象或 None
        """
        sql = f"SELECT * FROM {self.TABLE_NAME} WHERE id = ?"
        row = self.db.fetch_one(sql, (paper_id,))
        if row:
            return Paper.from_dict(dict(row))
        return None

    def update(self, paper_id: str, paper: Paper) -> bool:
        """
        更新或插入论文（upsert）

        Args:
            paper_id: 论文 ID
            paper: Paper 对象

        Returns:
            是否更新成功
        """
        try:
            paper.updated_at = datetime.now()
            data = paper.to_dict()
            # 使用 INSERT OR REPLACE 实现 upsert
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?"] * len(data))
            sql = f"INSERT OR REPLACE INTO {self.TABLE_NAME} ({columns}) VALUES ({placeholders})"
            params = tuple(data.values())
            self.db.execute(sql, params)
            logger.info(f"Upserted paper: {paper_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert paper: {e}")
            return False

    def delete(self, paper_id: str) -> bool:
        """
        删除论文

        Args:
            paper_id: 论文 ID

        Returns:
            是否删除成功
        """
        try:
            sql = f"DELETE FROM {self.TABLE_NAME} WHERE id = ?"
            self.db.execute(sql, (paper_id,))
            logger.info(f"Deleted paper: {paper_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete paper: {e}")
            return False

    def query(self, filters: Dict[str, Any], limit: int = 100) -> List[Paper]:
        """
        根据过滤条件查询论文

        Args:
            filters: 过滤条件字典
            limit: 返回数量限制

        Returns:
            Paper 对象列表
        """
        conditions = []
        params = []

        for key, value in filters.items():
            if key == "year_min":
                conditions.append("year >= ?")
                params.append(value)
            elif key == "year_max":
                conditions.append("year <= ?")
                params.append(value)
            elif key == "ccf_rating":
                conditions.append("ccf_rating = ?")
                params.append(value)
            elif key == "field":
                conditions.append("paper_field = ?")
                params.append(value)
            elif key == "subfield":
                conditions.append("subfield = ?")
                params.append(value)
            elif key == "method":
                conditions.append("method LIKE ?")
                params.append(f"%{value}%")
            elif key == "task":
                conditions.append("task LIKE ?")
                params.append(f"%{value}%")
            elif key == "keyword":
                conditions.append("(title LIKE ? OR task LIKE ? OR method LIKE ?)")
                kw = f"%{value}%"
                params.extend([kw, kw, kw])
            else:
                conditions.append(f"{key} = ?")
                params.append(value)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"SELECT * FROM {self.TABLE_NAME} {where_clause} LIMIT ?"
        params.append(limit)

        rows = self.db.fetch_all(sql, tuple(params))
        return [Paper.from_dict(dict(row)) for row in rows]

    def get_all(self, limit: int = 1000, offset: int = 0) -> List[Paper]:
        """
        获取所有论文

        Args:
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            Paper 对象列表
        """
        sql = f"SELECT * FROM {self.TABLE_NAME} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        rows = self.db.fetch_all(sql, (limit, offset))
        return [Paper.from_dict(dict(row)) for row in rows]

    def count(self) -> int:
        """获取论文总数"""
        sql = f"SELECT COUNT(*) FROM {self.TABLE_NAME}"
        return self.db.count(sql)

    def exists(self, paper_id: str) -> bool:
        """检查论文是否存在"""
        sql = f"SELECT 1 FROM {self.TABLE_NAME} WHERE id = ?"
        return self.db.exists(sql, (paper_id,))

    def raw_query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        执行原始 SQL 查询

        Args:
            sql: SQL 语句
            params: 参数

        Returns:
            结果列表
        """
        return self.db.raw_query(sql, params)

    def get_by_field(self, field: str, limit: int = 100) -> List[Paper]:
        """获取指定领域的论文"""
        return self.query({"field": field}, limit)

    def get_by_year(self, year: int, limit: int = 100) -> List[Paper]:
        """获取指定年份的论文"""
        return self.query({"year": year}, limit)

    def get_by_ccf_rating(self, rating: str, limit: int = 100) -> List[Paper]:
        """获取指定 CCF 评级的论文"""
        return self.query({"ccf_rating": rating}, limit)

    def get_by_method(self, method: str, limit: int = 100) -> List[Paper]:
        """获取指定方法的论文"""
        return self.query({"method": method}, limit)

    def get_recent(self, limit: int = 20) -> List[Paper]:
        """获取最近的论文"""
        return self.get_all(limit=limit)

    def search(self, keyword: str, limit: int = 50) -> List[Paper]:
        """搜索论文"""
        return self.query({"keyword": keyword}, limit)
