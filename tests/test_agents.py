"""Agents 模块测试"""

import sys
import os
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ScholarGraph.agents.base import BaseAgent, AgentResult
from ScholarGraph.agents.controller import Controller
from ScholarGraph.agents.query_agent import QueryAgent
from ScholarGraph.agents.critic_agent import CriticAgent
from ScholarGraph.memory.conversation_history import ConversationHistory
from ScholarGraph.memory.pipeline_state import PipelineState, PipelinePhase
from ScholarGraph.shared_env import SharedEnvironment
from ScholarGraph.models.paper import Paper
from ScholarGraph.models.query import SynthesizedAnswer


class TestAgentResult(unittest.TestCase):
    """AgentResult 测试"""

    def test_agent_result_creation(self):
        """测试 AgentResult 创建"""
        print("\n[TEST] test_agent_result_creation")
        result = AgentResult(success=True, data={"key": "value"})
        self.assertTrue(result.success)
        self.assertEqual(result.data["key"], "value")
        print(f"  - Result: success={result.success}, data={result.data}")

    def test_agent_result_with_error(self):
        """测试带错误的 AgentResult"""
        print("\n[TEST] test_agent_result_with_error")
        result = AgentResult(success=False, error="Something went wrong")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Something went wrong")
        print(f"  - Error: {result.error}")

    def test_agent_result_with_state_update(self):
        """测试带状态更新的 AgentResult"""
        print("\n[TEST] test_agent_result_with_state_update")
        result = AgentResult(
            success=True,
            data={"phase": "complete"},
            state_update={"memory": {"retries": 1}}
        )
        self.assertEqual(result.state_update["memory"]["retries"], 1)
        print(f"  - State update: {result.state_update}")


class TestBaseAgent(unittest.TestCase):
    """BaseAgent 测试"""

    def test_base_agent_init(self):
        """测试 BaseAgent 初始化"""
        print("\n[TEST] test_base_agent_init")

        class TestAgent(BaseAgent):
            def execute(self, env, *args, **kwargs):
                return AgentResult(success=True)

        agent = TestAgent("TestAgent", memory=PipelineState())
        self.assertEqual(agent.name, "TestAgent")
        self.assertIsNotNone(agent.memory)
        print(f"  - Agent name: {agent.name}, memory type: {type(agent.memory).__name__}")

    def test_base_agent_repr(self):
        """测试 BaseAgent 字符串表示"""
        print("\n[TEST] test_base_agent_repr")

        class TestAgent(BaseAgent):
            def execute(self, env, *args, **kwargs):
                return AgentResult(success=True)

        agent = TestAgent("TestAgent", memory=PipelineState())
        repr_str = repr(agent)
        self.assertIn("TestAgent", repr_str)
        print(f"  - Repr: {repr_str}")


class TestController(unittest.TestCase):
    """Controller Agent 测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "controller_test.db")
        self.index_path = os.path.join(self.temp_dir, "controller_index.faiss")
        self.env = SharedEnvironment(
            db_path=self.db_path,
            vector_index_path=self.index_path,
            embedding_dimension=384
        )
        self.controller = Controller(self.env)
        print(f"\n[SETUP] Created Controller with env: {self.db_path}")

    def tearDown(self):
        """清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_controller_init(self):
        """测试 Controller 初始化"""
        print("\n[TEST] test_controller_init")
        self.assertEqual(self.controller.name, "Controller")
        self.assertIsNotNone(self.controller.memory)
        self.assertIsNotNone(self.controller.env)
        print(f"  - Controller name: {self.controller.name}")

    def test_controller_initial_phase(self):
        """测试 Controller 初始阶段"""
        print("\n[TEST] test_controller_initial_phase")
        self.assertEqual(self.controller.memory.phase, PipelinePhase.IDLE)
        print(f"  - Initial phase: {self.controller.memory.phase.value}")

    def test_controller_unknown_action(self):
        """测试未知 action"""
        print("\n[TEST] test_controller_unknown_action")
        result = self.controller.execute(self.env, action="unknown")
        self.assertFalse(result.success)
        self.assertIn("Unknown action", result.error)
        print(f"  - Unknown action error: {result.error}")

    def test_memory_update(self):
        """测试 Memory 更新"""
        print("\n[TEST] test_memory_update")
        self.controller.memory.update_phase(PipelinePhase.PARSE)
        self.assertEqual(self.controller.memory.phase, PipelinePhase.PARSE)
        print(f"  - Updated phase: {self.controller.memory.phase.value}")


class TestQueryAgent(unittest.TestCase):
    """QueryAgent 测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "query_test.db")
        self.index_path = os.path.join(self.temp_dir, "query_index.faiss")
        self.env = SharedEnvironment(
            db_path=self.db_path,
            vector_index_path=self.index_path,
            embedding_dimension=384
        )
        self.memory = ConversationHistory(session_id="test_session")
        self.query_agent = QueryAgent(self.memory)
        print(f"\n[SETUP] Created QueryAgent with session: {self.memory.session_id}")

    def tearDown(self):
        """清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_query_agent_init(self):
        """测试 QueryAgent 初始化"""
        print("\n[TEST] test_query_agent_init")
        self.assertEqual(self.query_agent.name, "QueryAgent")
        self.assertIsNotNone(self.query_agent.memory)
        print(f"  - QueryAgent name: {self.query_agent.name}")

    def test_execute_factual_query(self):
        """测试执行事实查询"""
        print("\n[TEST] test_execute_factual_query")
        result = self.query_agent.execute(self.env, "What is LoRA?")
        self.assertTrue(result.success)
        self.assertIn("query_type", result.data)
        print(f"  - Query type: {result.data['query_type']}")

    def test_execute_comparison_query(self):
        """测试执行比较查询"""
        print("\n[TEST] test_execute_comparison_query")
        result = self.query_agent.execute(self.env, "比较 LoRA 和 AdaLoRA")
        self.assertTrue(result.success)
        self.assertEqual(result.data["query_type"], "comparison")
        print(f"  - Comparison query type: {result.data['query_type']}")

    def test_execute_method_topology_query(self):
        """测试执行方法拓扑查询"""
        print("\n[TEST] test_execute_method_topology_query")
        result = self.query_agent.execute(self.env, "哪些方法改进自 LoRA?")
        self.assertTrue(result.success)
        self.assertEqual(result.data["query_type"], "method_topology")
        print(f"  - Method topology query type: {result.data['query_type']}")

    def test_query_keywords_extraction(self):
        """测试关键词提取"""
        print("\n[TEST] test_query_keywords_extraction")
        result = self.query_agent.execute(self.env, "LoRA 和 Transformer 的比较")
        self.assertTrue(result.success)
        self.assertIn("keywords", result.data)
        print(f"  - Keywords: {result.data['keywords']}")

    def test_query_constraints_extraction(self):
        """测试约束提取"""
        print("\n[TEST] test_query_constraints_extraction")
        result = self.query_agent.execute(self.env, "LoRA 方法 2023年 CCF-A 会议")
        self.assertTrue(result.success)
        self.assertIn("constraints", result.data)
        print(f"  - Constraints: {result.data['constraints']}")

    def test_memory_records_conversation(self):
        """测试 Memory 记录对话"""
        print("\n[TEST] test_memory_records_conversation")
        self.query_agent.execute(self.env, "Hello")
        self.query_agent.execute(self.env, "What is LoRA?")
        self.assertEqual(len(self.query_agent.memory.messages), 4)  # 2 user + 2 assistant
        print(f"  - Messages recorded: {len(self.query_agent.memory.messages)}")

    def test_query_clear(self):
        """测试查询清除"""
        print("\n[TEST] test_query_clear")
        self.env.set_query_context("test", "factual_qa", ["test"])
        self.env.clear_query_context()
        self.assertIsNone(self.env.get_query_context())
        print(f"  - Query context cleared")


class TestCriticAgent(unittest.TestCase):
    """CriticAgent 测试"""

    def setUp(self):
        """设置测试环境"""
        self.memory = PipelineState()
        self.critic = CriticAgent(self.memory)
        print(f"\n[SETUP] Created CriticAgent with PipelineState")

    def test_critic_agent_init(self):
        """测试 CriticAgent 初始化"""
        print("\n[TEST] test_critic_agent_init")
        self.assertEqual(self.critic.name, "CriticAgent")
        self.assertIsNotNone(self.critic.memory)
        print(f"  - CriticAgent name: {self.critic.name}")

    def test_evaluate_good_answer(self):
        """测试评估良好答案"""
        print("\n[TEST] test_evaluate_good_answer")
        env = SharedEnvironment(db_path=":memory:", vector_index_path=":memory:", embedding_dimension=384)
        answer = SynthesizedAnswer(
            conclusion="LoRA is efficient",
            summary="Brief summary",
            evidence=[{"statement": "evidence1", "source": "paper1"}, {"statement": "evidence2", "source": "paper2"}],
            method_groups=[{"method": "LoRA", "papers": ["paper1"]}],
            limitations=["limited expressiveness"],
            confidence=0.9,
            papers_used=3
        )
        result = self.critic.execute(env, answer, "What is LoRA?")
        self.assertTrue(result.success)
        critique = result.data
        self.assertTrue(critique["passed"])
        print(f"  - Critique passed: {critique['passed']}")
        print(f"  - Coverage: {critique['coverage_score']}")

    def test_evaluate_poor_answer(self):
        """测试评估不足答案"""
        print("\n[TEST] test_evaluate_poor_answer")
        env = SharedEnvironment(db_path=":memory:", vector_index_path=":memory:", embedding_dimension=384)
        answer = SynthesizedAnswer(
            conclusion="LoRA is good",
            summary="Summary",
            evidence=[],
            method_groups=[],
            limitations=[],
            confidence=0.3,
            papers_used=1
        )
        result = self.critic.execute(env, answer, "Compare LoRA methods?")
        self.assertTrue(result.success)
        critique = result.data
        self.assertFalse(critique["passed"])
        self.assertIn("coverage_low", critique["failed_checks"])
        print(f"  - Critique passed: {critique['passed']}")
        print(f"  - Failed checks: {critique['failed_checks']}")

    def test_evaluate_increments_retry(self):
        """测试评估增加重试次数"""
        print("\n[TEST] test_evaluate_increments_retry")
        env = SharedEnvironment(db_path=":memory:", vector_index_path=":memory:", embedding_dimension=384)
        answer = SynthesizedAnswer(
            conclusion="",
            summary="",
            evidence=[],
            method_groups=[],
            limitations=[],
            confidence=0.0,
            papers_used=0
        )
        initial_retries = self.memory.retries
        self.critic.execute(env, answer, "test query")
        self.assertEqual(self.memory.retries, initial_retries + 1)
        print(f"  - Retries after failed critique: {self.memory.retries}")

    def test_memory_failed_checks_recorded(self):
        """测试 Memory 记录失败检查"""
        print("\n[TEST] test_memory_failed_checks_recorded")
        env = SharedEnvironment(db_path=":memory:", vector_index_path=":memory:", embedding_dimension=384)
        answer = SynthesizedAnswer(
            conclusion="",
            summary="",
            evidence=[],
            method_groups=[],
            limitations=[],
            confidence=0.0,
            papers_used=0
        )
        self.critic.execute(env, answer, "test")
        self.assertGreater(len(self.memory.failed_checks), 0)
        print(f"  - Failed checks in memory: {self.memory.failed_checks}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Agents Tests...")
    print("=" * 60)
    unittest.main(verbosity=2)
