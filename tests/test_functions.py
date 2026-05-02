"""Functions 模块测试"""

import sys
import os
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ScholarGraph.functions.base import BaseFunction, FunctionResult
from ScholarGraph.functions.pdf_parser import PDFParser
from ScholarGraph.functions.understanding import Understanding
from ScholarGraph.functions.enrichment import Enrichment
from ScholarGraph.functions.classification import Classification
from ScholarGraph.functions.relation import Relation
from ScholarGraph.functions.retrieval import Retrieval
from ScholarGraph.functions.synthesis import Synthesis
from ScholarGraph.functions.embedding import Embedding
from ScholarGraph.functions.ccf_parser import CCFParser
from ScholarGraph.shared_env import SharedEnvironment
from ScholarGraph.models.paper import Paper, ParsedPaper
from ScholarGraph.models.query import QueryAnalysisResult


class TestFunctionResult(unittest.TestCase):
    """FunctionResult 测试"""

    def test_function_result_creation(self):
        """测试 FunctionResult 创建"""
        print("\n[TEST] test_function_result_creation")
        result = FunctionResult(success=True, data={"key": "value"})
        self.assertTrue(result.success)
        self.assertEqual(result.data["key"], "value")
        print(f"  - Result: success={result.success}, data={result.data}")

    def test_function_result_with_error(self):
        """测试带错误的 FunctionResult"""
        print("\n[TEST] test_function_result_with_error")
        result = FunctionResult(success=False, error="Parse failed")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Parse failed")
        print(f"  - Error: {result.error}")

    def test_function_result_with_metadata(self):
        """测试带元数据的 FunctionResult"""
        print("\n[TEST] test_function_result_with_metadata")
        result = FunctionResult(
            success=True,
            data={"processed": True},
            metadata={"duration": 0.5}
        )
        self.assertEqual(result.metadata["duration"], 0.5)
        print(f"  - Metadata: {result.metadata}")


class TestPDFParser(unittest.TestCase):
    """PDFParser 测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "pdf_test.db")
        self.index_path = os.path.join(self.temp_dir, "pdf_index.faiss")
        self.env = SharedEnvironment(
            db_path=self.db_path,
            vector_index_path=self.index_path,
            embedding_dimension=384
        )
        self.func = PDFParser()
        print(f"\n[SETUP] Created PDFParser")

    def tearDown(self):
        """清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_pdf_parser_init(self):
        """测试 PDFParser 初始化"""
        print("\n[TEST] test_pdf_parser_init")
        self.assertEqual(self.func.name, "PDFParser")
        print(f"  - Function name: {self.func.name}")

    def test_repr(self):
        """测试字符串表示"""
        print("\n[TEST] test_repr")
        repr_str = repr(self.func)
        self.assertIn("PDFParser", repr_str)
        print(f"  - Repr: {repr_str}")

    def test_validate_input(self):
        """测试输入验证"""
        print("\n[TEST] test_validate_input")
        self.assertTrue(self.func.validate_input("test.pdf"))
        print(f"  - Input validation: passed")


class TestUnderstanding(unittest.TestCase):
    """Understanding 测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "understanding_test.db")
        self.index_path = os.path.join(self.temp_dir, "understanding_index.faiss")
        self.env = SharedEnvironment(
            db_path=self.db_path,
            vector_index_path=self.index_path,
            embedding_dimension=384
        )
        self.func = Understanding()
        print(f"\n[SETUP] Created Understanding")

    def tearDown(self):
        """清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_understanding_init(self):
        """测试 Understanding 初始化"""
        print("\n[TEST] test_understanding_init")
        self.assertEqual(self.func.name, "Understanding")
        print(f"  - Function name: {self.func.name}")

    def test_execute_with_parsed_paper(self):
        """测试执行字段提取"""
        print("\n[TEST] test_execute_with_parsed_paper")
        parsed = ParsedPaper(
            title="LoRA: Low-Rank Adaptation",
            authors=["Author1", "Author2"],
            abstract="Abstract content",
            sections={"introduction": "Intro text"},
            raw_text="LoRA is a method for fine-tuning.",
            pdf_path="/path/to/paper.pdf"
        )
        result = self.func.execute(self.env, parsed)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.data)
        print(f"  - Extracted paper: {result.data.title}")


class TestEnrichment(unittest.TestCase):
    """Enrichment 测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "enrich_test.db")
        self.index_path = os.path.join(self.temp_dir, "enrich_index.faiss")
        self.env = SharedEnvironment(
            db_path=self.db_path,
            vector_index_path=self.index_path,
            embedding_dimension=384
        )
        self.func = Enrichment()
        print(f"\n[SETUP] Created Enrichment")

    def tearDown(self):
        """清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_enrichment_init(self):
        """测试 Enrichment 初始化"""
        print("\n[TEST] test_enrichment_init")
        self.assertEqual(self.func.name, "Enrichment")
        print(f"  - Function name: {self.func.name}")

    def test_execute_with_missing_fields(self):
        """测试补全缺失字段"""
        print("\n[TEST] test_execute_with_missing_fields")
        paper = Paper(id="test", title="Test Paper")
        result = self.func.execute(self.env, paper)
        self.assertTrue(result.success)
        print(f"  - Enrichment result: enriched={result.data.get('enriched', False)}")

    def test_execute_with_complete_paper(self):
        """测试完整论文"""
        print("\n[TEST] test_execute_with_complete_paper")
        paper = Paper(
            id="complete",
            title="Complete Paper",
            authors=["Author"],
            year=2024,
            venue="ICML",
            ccf_rating="A",
            task="classification",
            method="transformer",
            paper_field="ML"
        )
        result = self.func.execute(self.env, paper)
        self.assertTrue(result.success)
        print(f"  - Complete paper: enriched={result.data.get('enriched', False)}")


class TestClassification(unittest.TestCase):
    """Classification 测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "classify_test.db")
        self.index_path = os.path.join(self.temp_dir, "classify_index.faiss")
        self.env = SharedEnvironment(
            db_path=self.db_path,
            vector_index_path=self.index_path,
            embedding_dimension=384
        )
        self.func = Classification()
        print(f"\n[SETUP] Created Classification")

    def tearDown(self):
        """清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_classification_init(self):
        """测试 Classification 初始化"""
        print("\n[TEST] test_classification_init")
        self.assertEqual(self.func.name, "Classification")
        print(f"  - Function name: {self.func.name}")

    def test_execute(self):
        """测试执行分类"""
        print("\n[TEST] test_execute")
        paper = Paper(
            id="test",
            title="Test Paper",
            authors=["Author"],
            paper_field="Computer Vision",
            subfield="Image Classification",
            topic=["Deep Learning"]
        )
        result = self.func.execute(self.env, paper)
        self.assertTrue(result.success)
        self.assertIn("field", result.data)
        print(f"  - Classification result: paper_field={result.data.get('field')}")


class TestRelation(unittest.TestCase):
    """Relation 测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "relation_test.db")
        self.index_path = os.path.join(self.temp_dir, "relation_index.faiss")
        self.env = SharedEnvironment(
            db_path=self.db_path,
            vector_index_path=self.index_path,
            embedding_dimension=384
        )
        self.func = Relation()
        print(f"\n[SETUP] Created Relation")

    def tearDown(self):
        """清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_relation_init(self):
        """测试 Relation 初始化"""
        print("\n[TEST] test_relation_init")
        self.assertEqual(self.func.name, "Relation")
        print(f"  - Function name: {self.func.name}")

    def test_execute_with_paper(self):
        """测试论文关系提取"""
        print("\n[TEST] test_execute_with_paper")
        paper = Paper(id="test", title="Test Paper", authors=["Author"])
        result = self.func.execute(self.env, paper=paper)
        self.assertTrue(result.success)
        print(f"  - Relation result: topology_updated={result.data.get('topology_updated', False)}")

    def test_execute_with_query(self):
        """测试查询关系"""
        print("\n[TEST] test_execute_with_query")
        result = self.func.execute(self.env, query="哪些方法改进自 LoRA?")
        self.assertTrue(result.success)
        print(f"  - Query processed: {result.data.get('query_processed', False)}")

    def test_execute_without_input(self):
        """测试无输入情况"""
        print("\n[TEST] test_execute_without_input")
        result = self.func.execute(self.env)
        self.assertFalse(result.success)
        self.assertIn("No paper or query", result.error)
        print(f"  - Error: {result.error}")


class TestRetrieval(unittest.TestCase):
    """Retrieval 测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "retrieval_test.db")
        self.index_path = os.path.join(self.temp_dir, "retrieval_index.faiss")
        self.env = SharedEnvironment(
            db_path=self.db_path,
            vector_index_path=self.index_path,
            embedding_dimension=384
        )
        self.func = Retrieval()
        print(f"\n[SETUP] Created Retrieval")

    def tearDown(self):
        """清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_retrieval_init(self):
        """测试 Retrieval 初始化"""
        print("\n[TEST] test_retrieval_init")
        self.assertEqual(self.func.name, "Retrieval")
        print(f"  - Function name: {self.func.name}")

    def test_execute(self):
        """测试执行检索"""
        print("\n[TEST] test_execute")
        query_analysis = QueryAnalysisResult(
            query_type="factual_qa",
            original_query="What is LoRA?",
            keywords=["LoRA"],
            constraints={},
            target_function="retrieval"
        )
        result = self.func.execute(self.env, query_analysis)
        self.assertTrue(result.success)
        self.assertIsInstance(result.data, list)
        print(f"  - Retrieved {len(result.data)} papers")


class TestSynthesis(unittest.TestCase):
    """Synthesis 测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "synthesis_test.db")
        self.index_path = os.path.join(self.temp_dir, "synthesis_index.faiss")
        self.env = SharedEnvironment(
            db_path=self.db_path,
            vector_index_path=self.index_path,
            embedding_dimension=384
        )
        self.func = Synthesis()
        print(f"\n[SETUP] Created Synthesis")

    def tearDown(self):
        """清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_synthesis_init(self):
        """测试 Synthesis 初始化"""
        print("\n[TEST] test_synthesis_init")
        self.assertEqual(self.func.name, "Synthesis")
        print(f"  - Function name: {self.func.name}")

    def test_execute(self):
        """测试执行合成"""
        print("\n[TEST] test_execute")
        papers = [
            Paper(id="p1", title="Paper 1", authors=["A"], year=2024),
            Paper(id="p2", title="Paper 2", authors=["B"], year=2024),
        ]
        query_analysis = QueryAnalysisResult(
            query_type="factual_qa",
            original_query="What is LoRA?",
            keywords=["LoRA"],
            constraints={},
            target_function="retrieval"
        )
        result = self.func.execute(self.env, papers, query_analysis)
        self.assertTrue(result.success)
        self.assertEqual(result.data.papers_used, 2)
        print(f"  - Synthesized from {result.data.papers_used} papers")


class TestEmbedding(unittest.TestCase):
    """Embedding 测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "embed_test.db")
        self.index_path = os.path.join(self.temp_dir, "embed_index.faiss")
        self.env = SharedEnvironment(
            db_path=self.db_path,
            vector_index_path=self.index_path,
            embedding_dimension=384
        )
        self.func = Embedding()
        print(f"\n[SETUP] Created Embedding")

    def tearDown(self):
        """清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_embedding_init(self):
        """测试 Embedding 初始化"""
        print("\n[TEST] test_embedding_init")
        self.assertEqual(self.func.name, "Embedding")
        print(f"  - Function name: {self.func.name}")

    def test_execute(self):
        """测试执行嵌入"""
        print("\n[TEST] test_execute")
        paper = Paper(id="test", title="Test Paper", authors=["Author"])
        result = self.func.execute(self.env, paper)
        self.assertTrue(result.success)
        self.assertIsInstance(result.data, list)
        print(f"  - Generated {len(result.data)} chunks")


class TestCCFParser(unittest.TestCase):
    """CCFParser 测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "ccf_test.db")
        self.index_path = os.path.join(self.temp_dir, "ccf_index.faiss")
        self.env = SharedEnvironment(
            db_path=self.db_path,
            vector_index_path=self.index_path,
            embedding_dimension=384
        )
        self.func = CCFParser()
        print(f"\n[SETUP] Created CCFParser")

    def tearDown(self):
        """清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_ccf_parser_init(self):
        """测试 CCFParser 初始化"""
        print("\n[TEST] test_ccf_parser_init")
        self.assertEqual(self.func.name, "CCFParser")
        print(f"  - Function name: {self.func.name}")

    def test_execute(self):
        """测试执行解析"""
        print("\n[TEST] test_execute")
        result = self.func.execute(self.env, "/path/to/ccf.pdf")
        self.assertTrue(result.success)
        self.assertIn("version", result.data)
        self.assertIn("venue_aliases", result.data)
        print(f"  - CCF mappings version: {result.data.get('version')}")


class TestFunctionsImport(unittest.TestCase):
    """Functions 导入测试"""

    def test_import_all_functions(self):
        """测试导入所有 Function"""
        print("\n[TEST] test_import_all_functions")
        from ScholarGraph.functions import (
            BaseFunction, FunctionResult,
            PDFParser, Understanding, Enrichment,
            Classification, Relation, Retrieval,
            Synthesis, Embedding, CCFParser
        )
        functions = [
            PDFParser, Understanding, Enrichment,
            Classification, Relation, Retrieval,
            Synthesis, Embedding, CCFParser
        ]
        for func in functions:
            instance = func()
            self.assertIsInstance(instance, BaseFunction)
            print(f"  - {func.__name__} is BaseFunction")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Functions Tests...")
    print("=" * 60)
    unittest.main(verbosity=2)
