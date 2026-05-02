"""LLM 客户端 - 支持 MiniMax API (HTTP 直接调用)"""

import os
import json
import logging
import requests
from typing import List, Optional, Dict, Any, Union, Generator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """对话消息"""
    role: str  # 'user' | 'assistant' | 'system'
    content: Union[str, List[Dict[str, Any]]]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {"role": self.role, "content": self.content}


@dataclass
class ChatCompletion:
    """聊天补全结果"""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    thinking: Optional[str] = None  # MiniMax 推理过程


class LLMClient:
    """
    LLM 客户端 - 支持 MiniMax API (HTTP 直接调用)

    支持 Anthropic 兼容格式和 OpenAI 兼容格式
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.minimaxi.com/anthropic",
        model: str = "MiniMax-M2.7-highspeed",
        timeout: int = 120,
        max_retries: int = 3
    ):
        """
        初始化 LLM 客户端

        Args:
            api_key: API 密钥
            base_url: API 基础 URL
            model: 模型名称
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        thinking: bool = False,
        **kwargs
    ) -> ChatCompletion:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            system: 系统提示词
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            thinking: 是否启用推理过程 (MiniMax 特性)

        Returns:
            ChatCompletion 对象
        """
        # 转换消息格式
        msg_dicts = [msg.to_dict() for msg in messages]

        # 根据 base_url 判断 API 格式
        if "anthropic" in self.base_url.lower():
            return self._chat_anthropic(msg_dicts, system, max_tokens, temperature, thinking)
        else:
            return self._chat_openai(msg_dicts, system, max_tokens, temperature)

    def _chat_anthropic(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        thinking: bool
    ) -> ChatCompletion:
        """使用 Anthropic 兼容格式调用 HTTP API"""
        url = f"{self.base_url}/v1/messages"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

        # 构建消息内容
        chat_messages = []
        for msg in messages:
            if isinstance(msg["content"], str):
                chat_messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                # 如果是 content blocks 列表，转为字符串
                content_parts = []
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content_parts.append(block["text"])
                    elif isinstance(block, str):
                        content_parts.append(block)
                chat_messages.append({"role": msg["role"], "content": "\n".join(content_parts)})

        payload = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system:
            payload["system"] = system

        if thinking:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": 10000
            }

        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        # 解析响应
        text_content = ""
        thinking_content = None

        for block in data.get("content", []):
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "thinking":
                thinking_content = block.get("thinking", "")

        return ChatCompletion(
            content=text_content,
            model=data.get("model", self.model),
            usage=data.get("usage", {})
        )

    def _chat_openai(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> ChatCompletion:
        """使用 OpenAI 兼容格式调用 HTTP API"""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 构建消息内容
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})

        for msg in messages:
            if isinstance(msg["content"], str):
                chat_messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                content_parts = []
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content_parts.append(block["text"])
                    elif isinstance(block, str):
                        content_parts.append(block)
                chat_messages.append({"role": msg["role"], "content": "\n".join(content_parts)})

        payload = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        return ChatCompletion(
            content=data["choices"][0]["message"]["content"] or "",
            model=data.get("model", self.model),
            usage=data.get("usage", {})
        )

    def stream_chat(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        thinking: bool = False
    ) -> Generator[str, None, None]:
        """
        流式聊天请求

        Args:
            messages: 消息列表
            system: 系统提示词
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            thinking: 是否启用推理过程

        Yields:
            生成的文本片段
        """
        msg_dicts = [msg.to_dict() for msg in messages]

        if "anthropic" in self.base_url.lower():
            yield from self._stream_chat_anthropic(msg_dicts, system, max_tokens, temperature, thinking)
        else:
            yield from self._stream_chat_openai(msg_dicts, system, max_tokens, temperature)

    def _stream_chat_anthropic(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        thinking: bool
    ) -> Generator[str, None, None]:
        """Anthropic 流式聊天"""
        url = f"{self.base_url}/v1/messages"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

        chat_messages = []
        for msg in messages:
            if isinstance(msg["content"], str):
                chat_messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                content_parts = []
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content_parts.append(block["text"])
                chat_messages.append({"role": msg["role"], "content": "\n".join(content_parts)})

        payload = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        if system:
            payload["system"] = system

        if thinking:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": 10000
            }

        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith("data: "):
                    data_str = line_text[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        for block in data.get("content", []):
                            if block.get("type") == "content_block_delta":
                                if hasattr(block.get("delta", {}), "text"):
                                    yield block["delta"]["text"]
                                elif isinstance(block.get("delta", {}), dict) and block.get("delta", {}).get("type") == "thinking":
                                    pass  # skip thinking in stream
                    except json.JSONDecodeError:
                        continue

    def _stream_chat_openai(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> Generator[str, None, None]:
        """OpenAI 流式聊天"""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})

        for msg in messages:
            if isinstance(msg["content"], str):
                chat_messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                content_parts = []
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content_parts.append(block["text"])
                chat_messages.append({"role": msg["role"], "content": "\n".join(content_parts)})

        payload = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith("data: "):
                    data_str = line_text[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if data.get("choices", [{}])[0].get("delta", {}).get("content"):
                            yield data["choices"][0]["delta"]["content"]
                    except json.JSONDecodeError:
                        continue

    def __repr__(self) -> str:
        return f"LLMClient(base_url={self.base_url}, model={self.model})"