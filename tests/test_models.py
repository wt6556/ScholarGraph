"""Models 模块测试"""

import sys
import os
import unittest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ScholarGraph.models.paper import Paper, ParsedPaper, Chunk, generate_paper_id
from ScholarGraph.models.query import QueryAnalysisResult, SynthesizedAnswer, CritiqueResult, QueryType


class TestPaperModel(unittest.TestCase):
    """Paper 模型测试"""

    def setUp(self):
        """设置测试数据"""
        self.paper = Paper(
            id="test001",
            title="Test Paper: A Study on Machine Learning",
            authors=["Alice Wang", "Bob Chen"],
            year=2024,
            venue="ICML",
            ccf_rating="A",
            task="image classification",
            assumption="deep learning works well",
            motivation="improve accuracy",
            method="Transformer",
            method_category="architecture",
            core_idea="self-attention mechanism",
            baselines=["CNN", "RNN"],
            datasets=["ImageNet", "COCO"],
            improvements=[{"dataset": "ImageNet", "metric": "Top-1", "delta": "+2.5"}],
            experiment_settings={"batch_size": 32},
            contribution="novel approach",
            limitation="computational cost",
            paper_field="Computer Vision",
            subfield="Image Recognition",
            topic=["Deep Learning", "Transformer"]
        )

    def test_paper_creation(self):
        """测试论文创建"""
        print("\n[TEST] test_paper_creation")
        self.assertEqual(self.paper.id, "test001")
        self.assertEqual(self.paper.title, "Test Paper: A Study on Machine Learning")
        self.assertEqual(len(self.paper.authors), 2)
        self.assertEqual(self.paper.year, 2024)
        self.assertEqual(self.paper.ccf_rating, "A")
        print(f"  - Paper ID: {self.paper.id}")
        print(f"  - Title: {self.paper.title[:30]}...")
        print(f"  - Authors: {self.paper.authors}")
        print(f"  - Year: {self.paper.year}, CCF: {self.paper.ccf_rating}")

    def test_has_missing_fields(self):
        """测试缺失字段检测"""
        print("\n[TEST] test_has_missing_fields")
        # 有缺失字段
        incomplete_paper = Paper(id="incomplete", title="Incomplete Paper")
        self.assertTrue(incomplete_paper.has_missing_fields())
        print(f"  - Incomplete paper has missing: {incomplete_paper.has_missing_fields()}")

        # 无缺失字段
        self.assertFalse(self.paper.has_missing_fields())
        print(f"  - Complete paper has missing: {self.paper.has_missing_fields()}")

    def test_get_missing_fields(self):
        """测试获取缺失字段列表"""
        print("\n[TEST] test_get_missing_fields")
        incomplete_paper = Paper(id="incomplete", title="Incomplete Paper")
        missing = incomplete_paper.get_missing_fields()
        print(f"  - Missing fields: {missing}")
        self.assertIn('venue', missing)
        self.assertIn('authors', missing)

    def test_to_dict(self):
        """测试序列化"""
        print("\n[TEST] test_to_dict")
        data = self.paper.to_dict()
        print(f"  - Serialized keys: {list(data.keys())[:10]}...")
        self.assertIsInstance(data, dict)
        self.assertEqual(data['id'], "test001")
        # authors is serialized to JSON string for database storage
        self.assertIsInstance(data['authors'], str)
        # After json.loads it should be a list
        import json
        authors_loaded = json.loads(data['authors'])
        self.assertIsInstance(authors_loaded, list)

    def test_from_dict(self):
        """测试反序列化"""
        print("\n[TEST] test_from_dict")
        data = self.paper.to_dict()
        restored = Paper.from_dict(data)
        self.assertEqual(restored.id, self.paper.id)
        self.assertEqual(restored.title, self.paper.title)
        self.assertEqual(restored.year, self.paper.year)
        print(f"  - Restored paper: {restored.title[:30]}...")

    def test_update(self):
        """测试更新字段"""
        print("\n[TEST] test_update")
        self.paper.update(year=2025, venue="NeurIPS")
        self.assertEqual(self.paper.year, 2025)
        self.assertEqual(self.paper.venue, "NeurIPS")
        print(f"  - Updated year: {self.paper.year}, venue: {self.paper.venue}")

    def test_generate_paper_id(self):
        """测试生成论文 ID"""
        print("\n[TEST] test_generate_paper_id")
        id1 = generate_paper_id("Test Paper", 2024)
        id2 = generate_paper_id("Test Paper", 2024)
        id3 = generate_paper_id("Different Paper", 2024)
        print(f"  - ID1: {id1}")
        print(f"  - ID2: {id2}")
        print(f"  - ID3: {id3}")
        self.assertEqual(id1, id2)  # 相同内容应产生相同 ID
        self.assertNotEqual(id1, id3)  # 不同内容应产生不同 ID


class TestParsedPaperModel(unittest.TestCase):
    """ParsedPaper 模型测试"""

    def setUp(self):
        """设置测试数据"""
        self.parsed = ParsedPaper(
            title="Parsed Paper Title",
            authors=["Author One", "Author Two"],
            abstract="This is the abstract.",
            sections={
                "introduction": "Introduction content...",
                "method": "Method content...",
                "experiment": "Experiment content...",
                "conclusion": "Conclusion content..."
            },
            raw_text="Full raw text content...",
            pdf_path="/path/to/paper.pdf"
        )

    def test_parsed_paper_creation(self):
        """测试 ParsedPaper 创建"""
        print("\n[TEST] test_parsed_paper_creation")
        self.assertEqual(self.parsed.title, "Parsed Paper Title")
        self.assertEqual(len(self.parsed.sections), 4)
        print(f"  - Title: {self.parsed.title}")
        print(f"  - Sections: {list(self.parsed.sections.keys())}")

    def test_get_section(self):
        """测试获取章节"""
        print("\n[TEST] test_get_section")
        method_content = self.parsed.get_section("method")
        self.assertIsNotNone(method_content)
        print(f"  - Method section length: {len(method_content)} chars")

    def test_to_paper(self):
        """测试转换为 Paper"""
        print("\n[TEST] test_to_paper")
        paper = self.parsed.to_paper()
        self.assertIsInstance(paper, Paper)
        self.assertEqual(paper.title, self.parsed.title)
        self.assertEqual(paper.authors, self.parsed.authors)
        print(f"  - Converted to Paper: {paper.title[:30]}...")


class TestChunkModel(unittest.TestCase):
    """Chunk 模型测试"""

    def setUp(self):
        """设置测试数据"""
        self.chunk = Chunk(
            id="chunk001",
            paper_id="paper001",
            chunk_type="method",
            content="Method description content...",
            embedding=[0.1] * 384,
            metadata={"year": 2024, "field": "ML"}
        )

    def test_chunk_creation(self):
        """测试 Chunk 创建"""
        print("\n[TEST] test_chunk_creation")
        self.assertEqual(self.chunk.id, "chunk001")
        self.assertEqual(self.chunk.chunk_type, "method")
        self.assertEqual(len(self.chunk.embedding), 384)
        print(f"  - Chunk ID: {self.chunk.id}, Type: {self.chunk.chunk_type}")
        print(f"  - Embedding dimension: {len(self.chunk.embedding)}")

    def test_to_dict(self):
        """测试序列化"""
        print("\n[TEST] test_to_dict")
        data = self.chunk.to_dict()
        print(f"  - Serialized keys: {list(data.keys())}")
        self.assertIsInstance(data['embedding'], str)  # embedding 应被序列化为 JSON 字符串

    def test_from_dict(self):
        """测试反序列化"""
        print("\n[TEST] test_from_dict")
        data = self.chunk.to_dict()
        restored = Chunk.from_dict(data)
        self.assertEqual(restored.id, self.chunk.id)
        self.assertEqual(len(restored.embedding), 384)
        print(f"  - Restored chunk embedding length: {len(restored.embedding)}")


class TestQueryModels(unittest.TestCase):
    """查询相关模型测试"""

    def test_query_analysis_result(self):
        """测试 QueryAnalysisResult"""
        print("\n[TEST] test_query_analysis_result")
        result = QueryAnalysisResult(
            query_type="factual_qa",
            original_query="What is LoRA?",
            keywords=["LoRA"],
            constraints={"field": "NLP"},
            target_function="retrieval"
        )
        print(f"  - Query type: {result.query_type}")
        print(f"  - Keywords: {result.keywords}")
        print(f"  - Target: {result.target_function}")

    def test_synthesized_answer(self):
        """测试 SynthesizedAnswer"""
        print("\n[TEST] test_synthesized_answer")
        answer = SynthesizedAnswer(
            conclusion="LoRA is a parameter-efficient fine-tuning method.",
            summary="Brief summary",
            evidence=[{"statement": "evidence1", "source": "paper1"}],
            method_groups=[{"method": "LoRA", "papers": ["paper1"]}],
            limitations=["limited expressiveness"],
            confidence=0.85,
            papers_used=5
        )
        print(f"  - Conclusion: {answer.conclusion[:50]}...")
        print(f"  - Confidence: {answer.confidence}, Papers used: {answer.papers_used}")

    def test_critique_result(self):
        """测试 CritiqueResult"""
        print("\n[TEST] test_critique_result")
        critique = CritiqueResult(
            passed=True,
            coverage_score=0.9,
            comparison_score=0.8,
            limitation_score=0.7,
            grounding_score=0.85,
            failed_checks=[],
            needs_retrieval=False
        )
        print(f"  - Passed: {critique.passed}")
        print(f"  - Coverage: {critique.coverage_score}")
        print(f"  - Grounding: {critique.grounding_score}")


class TestQueryType(unittest.TestCase):
    """QueryType 枚举测试"""

    def test_query_type_values(self):
        """测试 QueryType 值"""
        print("\n[TEST] test_query_type_values")
        print(f"  - FACTUAL_QA: {QueryType.FACTUAL_QA.value}")
        print(f"  - COMPARISON: {QueryType.COMPARISON.value}")
        print(f"  - METHOD_TOPOLOGY: {QueryType.METHOD_TOPOLOGY.value}")
        self.assertEqual(QueryType.FACTUAL_QA.value, "factual_qa")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Models Tests...")
    print("=" * 60)
    unittest.main(verbosity=2)
