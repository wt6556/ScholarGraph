"""SQLite 基础操作类"""

import sqlite3
import os
from typing import Any, List, Dict, Optional, Tuple
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class SQLiteBase:
    """SQLite 数据库基础操作类"""

    def __init__(self, db_path: str, timeout: float = 30.0):
        """
        初始化 SQLite 基础类

        Args:
            db_path: 数据库文件路径
            timeout: 连接超时时间（秒）
        """
        self.db_path = db_path
        self.timeout = timeout
        self._ensure_db_dir()

    def _ensure_db_dir(self) -> None:
        """确保数据库目录存在"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        conn.row_factory = sqlite3.Row  # 启用列名访问
        return conn

    @contextmanager
    def get_connection(self):
        """上下文管理器：自动管理数据库连接"""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        执行 SQL 语句

        Args:
            sql: SQL 语句
            params: 参数元组

        Returns:
            游标对象
        """
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            return cursor

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """
        查询单条记录

        Args:
            sql: SQL 语句
            params: 参数元组

        Returns:
            单条记录的字典形式，或 None
        """
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def fetch_all(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        查询所有记录

        Args:
            sql: SQL 语句
            params: 参数元组

        Returns:
            记录列表
        """
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def fetch_column(self, sql: str, params: tuple = ()) -> List[Any]:
        """
        查询单列的所有值

        Args:
            sql: SQL 语句（应只选择单列）
            params: 参数元组

        Returns:
            值列表
        """
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            return [row[0] for row in rows]

    def execute_many(self, sql: str, params_list: List[tuple]) -> None:
        """
        批量执行 SQL 语句

        Args:
            sql: SQL 语句
            params_list: 参数元组列表
        """
        with self.get_connection() as conn:
            conn.executemany(sql, params_list)

    def exists(self, sql: str, params: tuple = ()) -> bool:
        """
        检查记录是否存在

        Args:
            sql: SQL 语句
            params: 参数元组

        Returns:
            是否存在
        """
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchone() is not None

    def count(self, sql: str, params: tuple = ()) -> int:
        """
        统计记录数

        Args:
            sql: SQL 语句（应使用 COUNT）
            params: 参数元组

        Returns:
            记录数
        """
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            result = cursor.fetchone()
            return result[0] if result else 0

    def raw_query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        执行原始 SQL 查询（用于复杂的 SQL 查询）

        Args:
            sql: SQL 语句
            params: 参数元组

        Returns:
            结果列表
        """
        return self.fetch_all(sql, params)

    def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        """
        创建表

        Args:
            table_name: 表名
            schema: 字段字典 {字段名: 类型+约束}
        """
        columns = ", ".join([f"{name} {dtype}" for name, dtype in schema.items()])
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        self.execute(sql)

    def table_exists(self, table_name: str) -> bool:
        """
        检查表是否存在

        Args:
            table_name: 表名

        Returns:
            是否存在
        """
        sql = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        return self.exists(sql, (table_name,))

    def drop_table(self, table_name: str) -> None:
        """
        删除表

        Args:
            table_name: 表名
        """
        sql = f"DROP TABLE IF EXISTS {table_name}"
        self.execute(sql)

    def vacuum(self) -> None:
        """整理数据库文件"""
        with self.get_connection() as conn:
            conn.execute("VACUUM")

    def get_tables(self) -> List[str]:
        """
        获取所有表名

        Returns:
            表名列表
        """
        sql = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        return self.fetch_column(sql)

    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """
        获取表结构信息

        Args:
            table_name: 表名

        Returns:
            字段信息列表
        """
        sql = f"PRAGMA table_info({table_name})"
        return self.fetch_all(sql)
