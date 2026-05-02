"""对话历史 Memory - QueryAgent 使用"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from time import time

from .base import BaseMemory


@dataclass
class Message:
    """对话消息"""
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    timestamp: float = field(default_factory=time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(**data)


class ConversationHistory(BaseMemory):
    """
    QueryAgent 用：存储多轮对话历史

    支持：
    - 追加消息
    - 获取上下文
    - 摘要压缩
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        max_history: int = 50,
        max_summary_length: int = 500
    ):
        """
        初始化对话历史

        Args:
            session_id: 会话 ID
            max_history: 最多保留的消息数
            max_summary_length: 摘要最大长度
        """
        self.session_id = session_id
        self.messages: List[Message] = []
        self.max_history = max_history
        self.max_summary_length = max_summary_length

    def append(self, role: str, content: str) -> None:
        """
        追加消息

        Args:
            role: 角色 ('user' | 'assistant' | 'system')
            content: 消息内容
        """
        self.messages.append(Message(role=role, content=content))
        # 超过上限时进行摘要压缩
        if len(self.messages) > self.max_history:
            self._compress()

    def get_context(self, last_n: Optional[int] = None) -> str:
        """
        获取对话上下文

        Args:
            last_n: 只返回最后 N 条消息，None 表示全部

        Returns:
            格式化的对话上下文字符串
        """
        msgs = self.messages[-last_n:] if last_n else self.messages
        return "\n".join([f"{m.role}: {m.content}" for m in msgs])

    def get_history_for_llm(self) -> List[Dict[str, str]]:
        """
        获取适合 LLM 的消息格式

        Returns:
            消息字典列表 [{"role": ..., "content": ...}, ...]
        """
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def summarize(self) -> str:
        """
        摘要压缩：将长历史压缩为短摘要

        Returns:
            摘要字符串
        """
        if not self.messages:
            return ""

        # 简单的摘要策略：提取关键信息
        user_msgs = [m.content for m in self.messages if m.role == 'user']
        assistant_msgs = [m.content for m in self.messages if m.role == 'assistant']

        summary = f"对话共 {len(self.messages)} 条消息"
        if user_msgs:
            summary += f"，用户询问了 {len(user_msgs)} 个问题"
        if assistant_msgs:
            summary += f"，助手回复了 {len(assistant_msgs)} 次"

        # 添加最后几个用户问题作为上下文
        recent = user_msgs[-3:] if user_msgs else []
        if recent:
            summary += f"\n最近的问题：{' | '.join(recent)}"

        # 截断
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length] + "..."

        return summary

    def _compress(self) -> None:
        """压缩历史：保留系统消息和最后几条，压缩中间部分"""
        system_msgs = [m for m in self.messages if m.role == 'system']
        recent_msgs = self.messages[-10:]  # 保留最近 10 条
        summary = self.summarize()

        # 保留摘要消息
        summary_msg = Message(
            role='system',
            content=f'[对话摘要] {summary}',
            timestamp=0
        )

        self.messages = system_msgs + [summary_msg] + recent_msgs

    def clear(self) -> None:
        """清空历史"""
        self.messages = []

    def read(self, key: str) -> Any:
        """读取值（ConversationHistory 主要通过 get_context 使用）"""
        if key == "messages":
            return self.messages
        elif key == "session_id":
            return self.session_id
        elif key == "context":
            return self.get_context()
        return None

    def update(self, key: str, value: Any) -> None:
        """更新值（ConversationHistory 主要通过 append 使用）"""
        if key == "session_id":
            self.session_id = value
        elif key == "max_history":
            self.max_history = value
        elif key == "max_summary_length":
            self.max_summary_length = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "max_history": self.max_history,
            "max_summary_length": self.max_summary_length
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationHistory':
        instance = cls(
            session_id=data.get("session_id"),
            max_history=data.get("max_history", 50),
            max_summary_length=data.get("max_summary_length", 500)
        )
        if "messages" in data:
            instance.messages = [Message.from_dict(m) for m in data["messages"]]
        return instance

    def __len__(self) -> int:
        return len(self.messages)

    def __bool__(self) -> bool:
        """Always return True since ConversationHistory object is valid even when empty"""
        return True

    def __repr__(self) -> str:
        return f"ConversationHistory(session={self.session_id}, messages={len(self.messages)})"
