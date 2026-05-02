"""Storage 模块测试"""

import sys
import os
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ScholarGraph.storage.sqlite_base import SQLiteBase
from ScholarGraph.storage.paper_store import PaperStore
from ScholarGraph.storage.chunk_store import ChunkStore
from ScholarGraph.storage.topology_store import TopologyStore, MethodNode, MethodEdge, PairwiseRelation
from ScholarGraph.models.paper import Paper


class TestSQLiteBase(unittest.TestCase):
    """SQLite 基础操作测试"""

    def setUp(self):
        """设置测试数据库"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db = SQLiteBase(self.db_path)
        print(f"\n[SETUP] Created temp database: {self.db_path}")

    def tearDown(self):
        """清理测试数据库"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up temp directory")

    def test_execute(self):
        """测试执行 SQL"""
        print("\n[TEST] test_execute")
        self.db.execute("CREATE TABLE test (id TEXT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO test (id, name) VALUES (?, ?)", ("1", "test"))
        result = self.db.fetch_one("SELECT * FROM test WHERE id = ?", ("1",))
        self.assertIsNotNone(result)
        self.assertEqual(result['name'], "test")
        print(f"  - Inserted and fetched: {result}")

    def test_fetch_one(self):
        """测试查询单条"""
        print("\n[TEST] test_fetch_one")
        self.db.execute("CREATE TABLE test (id TEXT PRIMARY KEY, value INTEGER)")
        self.db.execute("INSERT INTO test VALUES (?, ?)", ("a", 100))
        result = self.db.fetch_one("SELECT * FROM test WHERE id = ?", ("a",))
        self.assertEqual(result['value'], 100)
        print(f"  - Fetched row: {result}")

    def test_fetch_all(self):
        """测试查询所有"""
        print("\n[TEST] test_fetch_all")
        self.db.execute("CREATE TABLE test (id TEXT PRIMARY KEY, value INTEGER)")
        self.db.execute("INSERT INTO test VALUES (?, ?)", ("a", 100))
        self.db.execute("INSERT INTO test VALUES (?, ?)", ("b", 200))
        results = self.db.fetch_all("SELECT * FROM test")
        self.assertEqual(len(results), 2)
        print(f"  - Fetched {len(results)} rows")

    def test_exists(self):
        """测试记录存在检查"""
        print("\n[TEST] test_exists")
        self.db.execute("CREATE TABLE test (id TEXT PRIMARY KEY)")
        self.db.execute("INSERT INTO test VALUES (?)", ("exists",))
        self.assertTrue(self.db.exists("SELECT 1 FROM test WHERE id = ?", ("exists",)))
        self.assertFalse(self.db.exists("SELECT 1 FROM test WHERE id = ?", ("not_exists",)))
        print(f"  - exists=True check passed")

    def test_count(self):
        """测试计数"""
        print("\n[TEST] test_count")
        self.db.execute("CREATE TABLE test (id TEXT PRIMARY KEY)")
        self.db.execute("INSERT INTO test VALUES (?)", ("1",))
        self.db.execute("INSERT INTO test VALUES (?)", ("2",))
        count = self.db.count("SELECT COUNT(*) FROM test")
        self.assertEqual(count, 2)
        print(f"  - Count: {count}")

    def test_raw_query(self):
        """测试原始 SQL 查询"""
        print("\n[TEST] test_raw_query")
        self.db.execute("CREATE TABLE test (id TEXT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO test VALUES (?, ?)", ("1", "Alice"))
        results = self.db.raw_query("SELECT * FROM test WHERE name LIKE ?", ("%Alice%",))
        self.assertEqual(len(results), 1)
        print(f"  - Raw query results: {len(results)}")

    def test_create_table(self):
        """测试创建表"""
        print("\n[TEST] test_create_table")
        schema = {"id": "TEXT PRIMARY KEY", "name": "TEXT", "age": "INTEGER"}
        self.db.create_table("users", schema)
        self.assertTrue(self.db.table_exists("users"))
        print(f"  - Table 'users' created")

    def test_table_info(self):
        """测试获取表结构"""
        print("\n[TEST] test_table_info")
        schema = {"id": "TEXT PRIMARY KEY", "name": "TEXT"}
        self.db.create_table("users", schema)
        info = self.db.get_table_info("users")
        self.assertEqual(len(info), 2)
        print(f"  - Table has {len(info)} columns")


class TestPaperStore(unittest.TestCase):
    """PaperStore 测试"""

    def setUp(self):
        """设置测试数据库"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "papers.db")
        self.store = PaperStore(self.db_path)
        print(f"\n[SETUP] Created PaperStore with DB: {self.db_path}")

    def tearDown(self):
        """清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_add_paper(self):
        """测试添加论文"""
        print("\n[TEST] test_add_paper")
        paper = Paper(
            id="paper001",
            title="Test Paper",
            authors=["Alice"],
            year=2024,
            venue="ICML",
            ccf_rating="A"
        )
        result = self.store.add(paper)
        self.assertTrue(result)
        print(f"  - Added paper: {paper.title}")

    def test_get_paper(self):
        """测试获取论文"""
        print("\n[TEST] test_get_paper")
        paper = Paper(id="paper001", title="Test Paper", authors=["Alice"], year=2024)
        self.store.add(paper)
        retrieved = self.store.get("paper001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, "Test Paper")
        print(f"  - Retrieved paper: {retrieved.title}")

    def test_update_paper(self):
        """测试更新论文"""
        print("\n[TEST] test_update_paper")
        paper = Paper(id="paper001", title="Original Title", authors=["Alice"])
        self.store.add(paper)
        paper.update(title="Updated Title", year=2025)
        self.store.update("paper001", paper)
        retrieved = self.store.get("paper001")
        self.assertEqual(retrieved.title, "Updated Title")
        self.assertEqual(retrieved.year, 2025)
        print(f"  - Updated paper title: {retrieved.title}")

    def test_delete_paper(self):
        """测试删除论文"""
        print("\n[TEST] test_delete_paper")
        paper = Paper(id="paper001", title="To Delete", authors=["Alice"])
        self.store.add(paper)
        result = self.store.delete("paper001")
        self.assertTrue(result)
        retrieved = self.store.get("paper001")
        self.assertIsNone(retrieved)
        print(f"  - Deleted paper, retrieval returns None")

    def test_query_papers(self):
        """测试查询论文"""
        print("\n[TEST] test_query_papers")
        papers = [
            Paper(id="p1", title="CV Paper", authors=["A"], year=2024, paper_field="Computer Vision"),
            Paper(id="p2", title="NLP Paper", authors=["B"], year=2024, paper_field="NLP"),
            Paper(id="p3", title="ML Paper", authors=["C"], year=2023, paper_field="ML"),
        ]
        for p in papers:
            self.store.add(p)

        results = self.store.query({"field": "Computer Vision"})
        self.assertEqual(len(results), 1)
        print(f"  - Found {len(results)} paper(s) in Computer Vision")

    def test_get_all_papers(self):
        """测试获取所有论文"""
        print("\n[TEST] test_get_all_papers")
        for i in range(5):
            self.store.add(Paper(id=f"p{i}", title=f"Paper {i}", authors=["Author"]))
        all_papers = self.store.get_all()
        self.assertEqual(len(all_papers), 5)
        print(f"  - Retrieved {len(all_papers)} papers total")

    def test_paper_count(self):
        """测试论文计数"""
        print("\n[TEST] test_paper_count")
        for i in range(3):
            self.store.add(Paper(id=f"p{i}", title=f"Paper {i}", authors=["Author"]))
        count = self.store.count()
        self.assertEqual(count, 3)
        print(f"  - Paper count: {count}")

    def test_exists(self):
        """测试论文存在检查"""
        print("\n[TEST] test_exists")
        paper = Paper(id="paper001", title="Test", authors=["A"])
        self.store.add(paper)
        self.assertTrue(self.store.exists("paper001"))
        self.assertFalse(self.store.exists("nonexistent"))
        print(f"  - exists check passed")


class TestChunkStore(unittest.TestCase):
    """ChunkStore 测试"""

    def setUp(self):
        """设置测试"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "chunks.db")
        self.index_path = os.path.join(self.temp_dir, "index.faiss")
        self.store = ChunkStore(self.db_path, self.index_path, dimension=384)
        print(f"\n[SETUP] Created ChunkStore with DB: {self.db_path}")

    def tearDown(self):
        """清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_add_chunks(self):
        """测试添加 chunks"""
        print("\n[TEST] test_add_chunks")
        from ScholarGraph.models.paper import Chunk
        chunks = [
            Chunk(id="c1", paper_id="p1", chunk_type="abstract",
                  content="Abstract content", embedding=[0.1] * 384),
            Chunk(id="c2", paper_id="p1", chunk_type="method",
                  content="Method content", embedding=[0.2] * 384),
        ]
        result = self.store.add_chunks(chunks)
        self.assertTrue(result)
        self.assertEqual(self.store.count(), 2)
        print(f"  - Added {len(chunks)} chunks")

    def test_get_chunk(self):
        """测试获取 chunk"""
        print("\n[TEST] test_get_chunk")
        from ScholarGraph.models.paper import Chunk
        chunk = Chunk(id="c1", paper_id="p1", chunk_type="abstract",
                      content="Test", embedding=[0.1] * 384)
        self.store.add_chunks([chunk])
        retrieved = self.store.get("c1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "Test")
        print(f"  - Retrieved chunk: {retrieved.id}")

    def test_get_by_paper(self):
        """测试获取论文的所有 chunks"""
        print("\n[TEST] test_get_by_paper")
        from ScholarGraph.models.paper import Chunk
        chunks = [
            Chunk(id="c1", paper_id="p1", chunk_type="abstract", content="A", embedding=[0.1] * 384),
            Chunk(id="c2", paper_id="p1", chunk_type="method", content="M", embedding=[0.2] * 384),
            Chunk(id="c3", paper_id="p2", chunk_type="abstract", content="B", embedding=[0.3] * 384),
        ]
        self.store.add_chunks(chunks)
        paper_chunks = self.store.get_by_paper("p1")
        self.assertEqual(len(paper_chunks), 2)
        print(f"  - Found {len(paper_chunks)} chunks for paper p1")

    def test_search_chunks(self):
        """测试向量检索"""
        print("\n[TEST] test_search_chunks")
        from ScholarGraph.models.paper import Chunk
        chunks = [
            Chunk(id="c1", paper_id="p1", chunk_type="abstract", content="AI content", embedding=[1.0] * 384),
            Chunk(id="c2", paper_id="p2", chunk_type="abstract", content="ML content", embedding=[0.5] * 384),
        ]
        self.store.add_chunks(chunks)
        results = self.store.search([1.0] * 384, top_k=2)
        self.assertGreaterEqual(len(results), 1)
        print(f"  - Search returned {len(results)} results")


class TestTopologyStore(unittest.TestCase):
    """TopologyStore 测试"""

    def setUp(self):
        """设置测试"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "topology.db")
        self.store = TopologyStore(self.db_path)
        print(f"\n[SETUP] Created TopologyStore with DB: {self.db_path}")

    def tearDown(self):
        """清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_add_method_node(self):
        """测试添加方法节点"""
        print("\n[TEST] test_add_method_node")
        node = MethodNode(
            id="n1",
            topic="LoRA",
            paper_id="paper1",
            method_name="LoRA",
            improvement_direction="efficiency",
            key_innovation="low-rank adaptation",
            is_root=True,
            parent_methods=[],
            created_at="2024-01-01"
        )
        result = self.store.add_method_node(node)
        self.assertTrue(result)
        print(f"  - Added method node: {node.method_name}")

    def test_get_method_nodes(self):
        """测试获取方法节点"""
        print("\n[TEST] test_get_method_nodes")
        node = MethodNode(
            id="n1", topic="LoRA", paper_id="paper1", method_name="LoRA",
            improvement_direction="efficiency", key_innovation="",
            is_root=True, parent_methods=[], created_at="2024-01-01"
        )
        self.store.add_method_node(node)
        nodes = self.store.get_method_nodes("LoRA")
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].method_name, "LoRA")
        print(f"  - Found {len(nodes)} nodes for topic LoRA")

    def test_add_relation(self):
        """测试添加方法关系"""
        print("\n[TEST] test_add_relation")
        edge = MethodEdge(
            id="e1",
            topic="LoRA",
            from_method="LoRA",
            to_method="AdaLoRA",
            improvement_direction="accuracy",
            description="AdaLoRA improves LoRA",
            created_at="2024-01-01"
        )
        result = self.store.add_relation(edge)
        self.assertTrue(result)
        print(f"  - Added relation: {edge.from_method} -> {edge.to_method}")

    def test_get_topology_edges(self):
        """测试获取拓扑边"""
        print("\n[TEST] test_get_topology_edges")
        edge = MethodEdge(
            id="e1", topic="LoRA", from_method="LoRA", to_method="AdaLoRA",
            improvement_direction="accuracy", description="", created_at="2024-01-01"
        )
        self.store.add_relation(edge)
        edges = self.store.get_topology_edges("LoRA")
        self.assertEqual(len(edges), 1)
        print(f"  - Found {len(edges)} edges for LoRA")

    def test_get_method_topology(self):
        """测试获取方法拓扑图"""
        print("\n[TEST] test_get_method_topology")
        node = MethodNode(
            id="n1", topic="LoRA", paper_id="paper1", method_name="LoRA",
            improvement_direction="efficiency", key_innovation="",
            is_root=True, parent_methods=[], created_at="2024-01-01"
        )
        self.store.add_method_node(node)
        topology = self.store.get_method_topology("LoRA")
        self.assertIn("nodes", topology)
        self.assertIn("edges", topology)
        self.assertEqual(len(topology["nodes"]), 1)
        print(f"  - Topology has {len(topology['nodes'])} nodes")

    def test_add_pairwise_relation(self):
        """测试添加成对关系"""
        print("\n[TEST] test_add_pairwise_relation")
        relation = PairwiseRelation(
            id="r1",
            paper_a_id="paper1",
            paper_b_id="paper2",
            relation_type="improves",
            description="Paper2 improves Paper1",
            created_at="2024-01-01"
        )
        result = self.store.add_pairwise_relation(relation)
        self.assertTrue(result)
        print(f"  - Added pairwise relation: {relation.paper_a_id} -> {relation.paper_b_id}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Storage Tests...")
    print("=" * 60)
    unittest.main(verbosity=2)
