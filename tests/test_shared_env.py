"""SharedEnvironment 模块测试"""

import sys
import os
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ScholarGraph.shared_env import SharedEnvironment, QueryContext
from ScholarGraph.models.paper import Paper


class TestQueryContext(unittest.TestCase):
    """QueryContext 测试"""

    def test_query_context_creation(self):
        """测试 QueryContext 创建"""
        print("\n[TEST] test_query_context_creation")
        ctx = QueryContext(
            query="What is LoRA?",
            query_type="factual_qa",
            keywords=["LoRA"]
        )
        self.assertEqual(ctx.query, "What is LoRA?")
        self.assertEqual(ctx.query_type, "factual_qa")
        print(f"  - Query: {ctx.query}")
        print(f"  - Type: {ctx.query_type}")

    def test_query_context_with_constraints(self):
        """测试带约束的 QueryContext"""
        print("\n[TEST] test_query_context_with_constraints")
        ctx = QueryContext(
            query="Compare methods",
            query_type="comparison",
            keywords=["methods"],
            constraints={"year_range": [2020, 2024], "ccf_rating": "A"}
        )
        self.assertEqual(ctx.constraints["year_range"], [2020, 2024])
        self.assertEqual(ctx.constraints["ccf_rating"], "A")
        print(f"  - Constraints: {ctx.constraints}")


class TestSharedEnvironment(unittest.TestCase):
    """SharedEnvironment 测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_env.db")
        self.index_path = os.path.join(self.temp_dir, "test_index.faiss")
        self.env = SharedEnvironment(
            db_path=self.db_path,
            vector_index_path=self.index_path,
            embedding_dimension=384
        )
        print(f"\n[SETUP] Created SharedEnvironment: {self.db_path}")

    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up temp directory")

    def test_initialization(self):
        """测试初始化"""
        print("\n[TEST] test_initialization")
        self.assertIsNotNone(self.env.paper_store)
        self.assertIsNotNone(self.env.chunk_store)
        self.assertIsNotNone(self.env.topology_store)
        print(f"  - Stores initialized: paper_store, chunk_store, topology_store")

    def test_add_paper(self):
        """测试添加论文"""
        print("\n[TEST] test_add_paper")
        paper = Paper(
            id="test001",
            title="Test Paper",
            authors=["Alice"],
            year=2024,
            venue="ICML",
            ccf_rating="A"
        )
        result = self.env.add_paper(paper)
        self.assertTrue(result)
        print(f"  - Added paper: {paper.title}")

    def test_get_paper(self):
        """测试获取论文"""
        print("\n[TEST] test_get_paper")
        paper = Paper(id="test001", title="Test Paper", authors=["Alice"], year=2024)
        self.env.add_paper(paper)
        retrieved = self.env.get_paper("test001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, "Test Paper")
        print(f"  - Retrieved paper: {retrieved.title}")

    def test_update_paper(self):
        """测试更新论文"""
        print("\n[TEST] test_update_paper")
        paper = Paper(id="test001", title="Original Title", authors=["Alice"])
        self.env.add_paper(paper)
        paper.update(title="Updated Title")
        self.env.update_paper(paper)
        retrieved = self.env.get_paper("test001")
        self.assertEqual(retrieved.title, "Updated Title")
        print(f"  - Updated title: {retrieved.title}")

    def test_delete_paper(self):
        """测试删除论文"""
        print("\n[TEST] test_delete_paper")
        paper = Paper(id="test001", title="To Delete", authors=["Alice"])
        self.env.add_paper(paper)
        result = self.env.delete_paper("test001")
        self.assertTrue(result)
        retrieved = self.env.get_paper("test001")
        self.assertIsNone(retrieved)
        print(f"  - Paper deleted")

    def test_query_papers(self):
        """测试查询论文"""
        print("\n[TEST] test_query_papers")
        papers = [
            Paper(id="p1", title="CV Paper", authors=["A"], year=2024, paper_field="Computer Vision"),
            Paper(id="p2", title="NLP Paper", authors=["B"], year=2024, paper_field="NLP"),
        ]
        for p in papers:
            self.env.add_paper(p)
        results = self.env.query_papers({"field": "Computer Vision"})
        self.assertEqual(len(results), 1)
        print(f"  - Found {len(results)} paper(s) matching filter")

    def test_get_all_papers(self):
        """测试获取所有论文"""
        print("\n[TEST] test_get_all_papers")
        for i in range(3):
            self.env.add_paper(Paper(id=f"p{i}", title=f"Paper {i}", authors=["A"]))
        all_papers = self.env.get_all_papers()
        self.assertEqual(len(all_papers), 3)
        print(f"  - Retrieved {len(all_papers)} papers")

    def test_search_papers(self):
        """测试搜索论文"""
        print("\n[TEST] test_search_papers")
        papers = [
            Paper(id="p1", title="LoRA Paper", authors=["A"], year=2024),
            Paper(id="p2", title="Transformer Paper", authors=["B"], year=2024),
        ]
        for p in papers:
            self.env.add_paper(p)
        results = self.env.search_papers("LoRA")
        self.assertGreaterEqual(len(results), 1)
        print(f"  - Search found {len(results)} result(s)")

    def test_paper_exists(self):
        """测试论文存在检查"""
        print("\n[TEST] test_paper_exists")
        paper = Paper(id="test001", title="Test", authors=["A"])
        self.env.add_paper(paper)
        self.assertTrue(self.env.paper_exists("test001"))
        self.assertFalse(self.env.paper_exists("nonexistent"))
        print(f"  - exists check passed")

    def test_get_paper_count(self):
        """测试获取论文总数"""
        print("\n[TEST] test_get_paper_count")
        for i in range(5):
            self.env.add_paper(Paper(id=f"p{i}", title=f"Paper {i}", authors=["A"]))
        count = self.env.get_paper_count()
        self.assertEqual(count, 5)
        print(f"  - Paper count: {count}")

    def test_current_paper(self):
        """测试当前论文管理"""
        print("\n[TEST] test_current_paper")
        paper = Paper(id="current", title="Current Paper", authors=["A"])
        self.env.set_current_paper(paper)
        retrieved = self.env.get_current_paper()
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, "current")
        self.env.clear_current_paper()
        self.assertIsNone(self.env.get_current_paper())
        print(f"  - Current paper: {retrieved.id} (cleared)")

    def test_query_context(self):
        """测试查询上下文管理"""
        print("\n[TEST] test_query_context")
        self.env.set_query_context(
            query="What is LoRA?",
            query_type="factual_qa",
            keywords=["LoRA"],
            constraints={"field": "ML"}
        )
        ctx = self.env.get_query_context()
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.query, "What is LoRA?")
        self.env.clear_query_context()
        self.assertIsNone(self.env.get_query_context())
        print(f"  - Query context set and cleared")

    def test_query_result(self):
        """测试查询结果管理"""
        print("\n[TEST] test_query_result")
        result = {"answer": "LoRA is...", "sources": ["paper1"]}
        self.env.set_query_result(result)
        retrieved = self.env.get_query_result()
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["answer"], "LoRA is...")
        self.env.clear_query_result()
        self.assertIsNone(self.env.get_query_result())
        print(f"  - Query result set and cleared")

    def test_reset(self):
        """测试重置"""
        print("\n[TEST] test_reset")
        paper = Paper(id="test", title="Test", authors=["A"])
        self.env.set_current_paper(paper)
        self.env.set_query_context("test", "factual_qa", ["test"])
        self.env.set_query_result({"data": "test"})
        self.env.reset()
        self.assertIsNone(self.env.get_current_paper())
        self.assertIsNone(self.env.get_query_context())
        self.assertIsNone(self.env.get_query_result())
        print(f"  - Environment reset")

    def test_repr(self):
        """测试字符串表示"""
        print("\n[TEST] test_repr")
        repr_str = repr(self.env)
        self.assertIn("SharedEnvironment", repr_str)
        print(f"  - Repr: {repr_str}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running SharedEnvironment Tests...")
    print("=" * 60)
    unittest.main(verbosity=2)
