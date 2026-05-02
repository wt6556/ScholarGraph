"""Memory 模块测试"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ScholarGraph.memory.conversation_history import ConversationHistory, Message
from ScholarGraph.memory.pipeline_state import PipelineState, PipelinePhase


class TestMessage(unittest.TestCase):
    """Message 测试"""

    def test_message_creation(self):
        """测试消息创建"""
        print("\n[TEST] test_message_creation")
        msg = Message(role="user", content="Hello!")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello!")
        print(f"  - Message: {msg.role}: {msg.content[:20]}...")

    def test_message_to_dict(self):
        """测试消息序列化"""
        print("\n[TEST] test_message_to_dict")
        msg = Message(role="user", content="Test")
        data = msg.to_dict()
        self.assertEqual(data['role'], "user")
        self.assertEqual(data['content'], "Test")
        print(f"  - Serialized: {data}")

    def test_message_from_dict(self):
        """测试消息反序列化"""
        print("\n[TEST] test_message_from_dict")
        data = {'role': 'assistant', 'content': 'Hi!', 'timestamp': 123456.0}
        msg = Message.from_dict(data)
        self.assertEqual(msg.role, 'assistant')
        self.assertEqual(msg.content, 'Hi!')
        print(f"  - Restored: {msg.role}: {msg.content}")


class TestConversationHistory(unittest.TestCase):
    """ConversationHistory 测试"""

    def setUp(self):
        """设置测试"""
        self.history = ConversationHistory(session_id="test_session", max_history=10)
        print(f"\n[SETUP] Created ConversationHistory: {self.history.session_id}")

    def test_append(self):
        """测试追加消息"""
        print("\n[TEST] test_append")
        self.history.append("user", "Hello!")
        self.history.append("assistant", "Hi there!")
        self.assertEqual(len(self.history.messages), 2)
        print(f"  - Messages count: {len(self.history)}")

    def test_get_context(self):
        """测试获取上下文"""
        print("\n[TEST] test_get_context")
        self.history.append("user", "What is LoRA?")
        self.history.append("assistant", "LoRA is a fine-tuning method.")
        context = self.history.get_context()
        self.assertIn("What is LoRA?", context)
        self.assertIn("LoRA is a fine-tuning method.", context)
        print(f"  - Context length: {len(context)} chars")

    def test_get_context_last_n(self):
        """测试获取最后 N 条消息"""
        print("\n[TEST] test_get_context_last_n")
        for i in range(5):
            self.history.append("user", f"Question {i}")
        context = self.history.get_context(last_n=2)
        self.assertEqual(context.count("Question"), 2)
        print(f"  - Last 2 messages retrieved")

    def test_get_history_for_llm(self):
        """测试获取适合 LLM 的格式"""
        print("\n[TEST] test_get_history_for_llm")
        self.history.append("user", "Hello")
        self.history.append("assistant", "Hi")
        history = self.history.get_history_for_llm()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['role'], 'user')
        print(f"  - LLM format: {history[0]['role']}")

    def test_clear(self):
        """测试清空历史"""
        print("\n[TEST] test_clear")
        self.history.append("user", "Test")
        self.assertEqual(len(self.history), 1)
        self.history.clear()
        self.assertEqual(len(self.history), 0)
        print(f"  - History cleared, messages: {len(self.history)}")

    def test_auto_compress(self):
        """测试自动压缩"""
        print("\n[TEST] test_auto_compress")
        history = ConversationHistory(session_id="test", max_history=5)
        for i in range(8):
            history.append("user", f"Message {i}")
        # 超过 max_history 应该触发压缩
        self.assertLessEqual(len(history.messages), 15)  # 系统消息 + 摘要 + 最近消息
        print(f"  - After {8} messages, stored: {len(history.messages)}")

    def test_summarize(self):
        """测试摘要"""
        print("\n[TEST] test_summarize")
        self.history.append("user", "Question 1")
        self.history.append("user", "Question 2")
        self.history.append("assistant", "Answer")
        summary = self.history.summarize()
        self.assertIn("对话", summary)
        self.assertIn("2", summary)  # 2 个问题
        print(f"  - Summary: {summary[:50]}...")

    def test_to_dict(self):
        """测试序列化"""
        print("\n[TEST] test_to_dict")
        self.history.append("user", "Test")
        data = self.history.to_dict()
        self.assertIn("messages", data)
        self.assertEqual(data["session_id"], "test_session")
        print(f"  - Serialized session: {data['session_id']}")

    def test_from_dict(self):
        """测试反序列化"""
        print("\n[TEST] test_from_dict")
        self.history.append("user", "Test")
        data = self.history.to_dict()
        restored = ConversationHistory.from_dict(data)
        self.assertEqual(restored.session_id, "test_session")
        self.assertEqual(len(restored.messages), 1)
        print(f"  - Restored: {restored.session_id}, messages: {len(restored.messages)}")


class TestPipelinePhase(unittest.TestCase):
    """PipelinePhase 枚举测试"""

    def test_phase_values(self):
        """测试阶段值"""
        print("\n[TEST] test_phase_values")
        print(f"  - IDLE: {PipelinePhase.IDLE.value}")
        print(f"  - PARSE: {PipelinePhase.PARSE.value}")
        print(f"  - UNDERSTAND: {PipelinePhase.UNDERSTAND.value}")
        self.assertEqual(PipelinePhase.IDLE.value, "idle")
        self.assertEqual(PipelinePhase.PARSE.value, "parse")


class TestPipelineState(unittest.TestCase):
    """PipelineState 测试"""

    def setUp(self):
        """设置测试"""
        self.state = PipelineState()
        print(f"\n[SETUP] Created PipelineState")

    def test_initial_state(self):
        """测试初始状态"""
        print("\n[TEST] test_initial_state")
        self.assertEqual(self.state.phase, PipelinePhase.IDLE)
        self.assertEqual(self.state.retries, 0)
        print(f"  - Initial phase: {self.state.phase.value}, retries: {self.state.retries}")

    def test_update_phase(self):
        """测试更新阶段"""
        print("\n[TEST] test_update_phase")
        self.state.update_phase(PipelinePhase.PARSE)
        self.assertEqual(self.state.phase, PipelinePhase.PARSE)
        self.assertEqual(self.state.retries, 0)  # 阶段变更应重置重试
        print(f"  - Updated phase: {self.state.phase.value}")

    def test_increment_retry(self):
        """测试增加重试次数"""
        print("\n[TEST] test_increment_retry")
        retries = self.state.increment_retry()
        self.assertEqual(retries, 1)
        retries = self.state.increment_retry()
        self.assertEqual(retries, 2)
        print(f"  - Retries: {self.state.retries}")

    def test_is_max_retries_exceeded(self):
        """测试是否超过最大重试"""
        print("\n[TEST] test_is_max_retries_exceeded")
        self.assertFalse(self.state.is_max_retries_exceeded())
        for _ in range(3):
            self.state.increment_retry()
        self.assertTrue(self.state.is_max_retries_exceeded())
        print(f"  - Max retries exceeded: {self.state.is_max_retries_exceeded()}")

    def test_failed_checks(self):
        """测试失败检查项"""
        print("\n[TEST] test_failed_checks")
        self.state.add_failed_check("coverage_low")
        self.state.add_failed_check("comparison_missing")
        self.assertEqual(len(self.state.failed_checks), 2)
        self.state.clear_failed_checks()
        self.assertEqual(len(self.state.failed_checks), 0)
        print(f"  - Failed checks: {len(self.state.failed_checks)} after clear")

    def test_error_handling(self):
        """测试错误处理"""
        print("\n[TEST] test_error_handling")
        self.state.set_error("Something went wrong")
        self.assertEqual(self.state.error_message, "Something went wrong")
        self.state.clear_error()
        self.assertIsNone(self.state.error_message)
        print(f"  - Error cleared")

    def test_metadata(self):
        """测试元数据"""
        print("\n[TEST] test_metadata")
        self.state.set_metadata("paper_id", "paper123")
        self.state.set_metadata("count", 5)
        self.assertEqual(self.state.get_metadata("paper_id"), "paper123")
        self.assertEqual(self.state.get_metadata("count"), 5)
        self.assertIsNone(self.state.get_metadata("nonexistent"))
        self.assertEqual(self.state.get_metadata("default", "default_value"), "default_value")
        print(f"  - Metadata paper_id: {self.state.get_metadata('paper_id')}")

    def test_reset(self):
        """测试重置"""
        print("\n[TEST] test_reset")
        self.state.update_phase(PipelinePhase.PARSE)
        self.state.increment_retry()
        self.state.add_failed_check("test")
        self.state.reset()
        self.assertEqual(self.state.phase, PipelinePhase.IDLE)
        self.assertEqual(self.state.retries, 0)
        self.assertEqual(len(self.state.failed_checks), 0)
        print(f"  - State reset to initial")

    def test_to_dict(self):
        """测试序列化"""
        print("\n[TEST] test_to_dict")
        self.state.update_phase(PipelinePhase.SYNTHESIZE)
        self.state.increment_retry()
        data = self.state.to_dict()
        self.assertEqual(data['phase'], 'synthesize')
        self.assertEqual(data['retries'], 1)
        print(f"  - Serialized phase: {data['phase']}")

    def test_from_dict(self):
        """测试反序列化"""
        print("\n[TEST] test_from_dict")
        data = {'phase': 'critic', 'retries': 2, 'failed_checks': ['test']}
        self.state.from_dict(data)
        self.assertEqual(self.state.phase, PipelinePhase.CRITIC)
        self.assertEqual(self.state.retries, 2)
        print(f"  - Restored: {self.state.phase.value}, retries: {self.state.retries}")

    def test_repr(self):
        """测试字符串表示"""
        print("\n[TEST] test_repr")
        repr_str = repr(self.state)
        self.assertIn("IDLE", repr_str)
        print(f"  - Repr: {repr_str}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Memory Tests...")
    print("=" * 60)
    unittest.main(verbosity=2)
