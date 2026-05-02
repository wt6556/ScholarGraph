"""LLM 模块测试"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ScholarGraph.llm import LLMClient, Message, ChatCompletion, LLMConfig, load_llm_config


class TestMessage(unittest.TestCase):
    """Message 测试"""

    def test_message_creation(self):
        """测试消息创建"""
        print("\n[TEST] test_message_creation")
        msg = Message(role="user", content="Hello!")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello!")
        print(f"  - Message: {msg.role}: {msg.content[:20]}...")

    def test_message_with_list_content(self):
        """测试带列表内容的消息"""
        print("\n[TEST] test_message_with_list_content")
        content = [
            {"type": "text", "text": "Hello!"},
            {"type": "text", "text": "How are you?"}
        ]
        msg = Message(role="user", content=content)
        self.assertEqual(len(msg.content), 2)
        print(f"  - List content length: {len(msg.content)}")

    def test_message_to_dict(self):
        """测试消息转字典"""
        print("\n[TEST] test_message_to_dict")
        msg = Message(role="assistant", content="I'm fine!")
        data = msg.to_dict()
        self.assertEqual(data['role'], "assistant")
        self.assertEqual(data['content'], "I'm fine!")
        print(f"  - Dict: {data}")


class TestChatCompletion(unittest.TestCase):
    """ChatCompletion 测试"""

    def test_chat_completion_creation(self):
        """测试 ChatCompletion 创建"""
        print("\n[TEST] test_chat_completion_creation")
        completion = ChatCompletion(
            content="Test response",
            model="MiniMax-M2.7",
            usage={"input_tokens": 10, "output_tokens": 20}
        )
        self.assertEqual(completion.content, "Test response")
        self.assertEqual(completion.model, "MiniMax-M2.7")
        self.assertEqual(completion.usage["input_tokens"], 10)
        print(f"  - Content: {completion.content}, Model: {completion.model}")

    def test_chat_completion_with_thinking(self):
        """测试带推理的 ChatCompletion"""
        print("\n[TEST] test_chat_completion_with_thinking")
        completion = ChatCompletion(
            content="Final answer",
            model="MiniMax-M2.7",
            thinking="Let me think..."
        )
        self.assertIsNotNone(completion.thinking)
        print(f"  - Thinking: {completion.thinking[:20]}...")


class TestLLMClient(unittest.TestCase):
    """LLMClient 测试"""

    def test_client_init(self):
        """测试客户端初始化"""
        print("\n[TEST] test_client_init")
        client = LLMClient(
            api_key="test_key",
            base_url="https://api.minimaxi.com/anthropic",
            model="MiniMax-M2.7"
        )
        self.assertEqual(client.api_key, "test_key")
        self.assertEqual(client.model, "MiniMax-M2.7")
        print(f"  - Client: {client}")

    def test_client_repr(self):
        """测试客户端字符串表示"""
        print("\n[TEST] test_client_repr")
        client = LLMClient(model="MiniMax-M2.7")
        repr_str = repr(client)
        self.assertIn("LLMClient", repr_str)
        self.assertIn("MiniMax-M2.7", repr_str)
        print(f"  - Repr: {repr_str}")

    def test_client_without_api_key(self):
        """测试不带 API Key 的初始化"""
        print("\n[TEST] test_client_without_api_key")
        client = LLMClient(
            base_url="https://api.minimaxi.com/anthropic",
            model="MiniMax-M2.7"
        )
        # API key should be empty or from env
        self.assertEqual(client.api_key, "")
        print(f"  - Client initialized without API key")

    def test_openai_client_init(self):
        """测试 OpenAI 客户端初始化"""
        print("\n[TEST] test_openai_client_init")
        client = LLMClient(
            api_key="test_key",
            base_url="https://api.minimaxi.com/v1",
            model="MiniMax-M2.7"
        )
        self.assertEqual(client.base_url, "https://api.minimaxi.com/v1")
        print(f"  - OpenAI endpoint: {client.base_url}")


class TestLLMConfig(unittest.TestCase):
    """LLMConfig 测试"""

    def test_llm_config_defaults(self):
        """测试默认配置"""
        print("\n[TEST] test_llm_config_defaults")
        config = LLMConfig()
        # 默认值来自 config/config.py 中的 LLMConfig 类
        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "gpt-4o-mini")
        print(f"  - Default config: provider={config.provider}, model={config.model}")

    def test_load_llm_config(self):
        """测试从配置文件加载"""
        print("\n[TEST] test_load_llm_config")
        config = load_llm_config()
        self.assertIsInstance(config, LLMConfig)
        self.assertIn(config.model, ["MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M2.5"])
        print(f"  - Loaded config: {config.model}")


class TestLLMImport(unittest.TestCase):
    """LLM 模块导入测试"""

    def test_import_all(self):
        """测试导入所有 LLM 模块"""
        print("\n[TEST] test_import_all")
        from ScholarGraph.llm import (
            LLMClient, Message, ChatCompletion,
            LLMConfig, load_llm_config, get_llm_client
        )
        self.assertIsNotNone(LLMClient)
        self.assertIsNotNone(Message)
        self.assertIsNotNone(ChatCompletion)
        print(f"  - All imports successful")


if __name__ == "__main__":
    print("=" * 60)
    print("Running LLM Tests...")
    print("=" * 60)
    unittest.main(verbosity=2)
