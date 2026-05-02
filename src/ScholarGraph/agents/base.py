"""Agent 基类"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from ..shared_env import SharedEnvironment
from ..memory.base import BaseMemory


@dataclass
class AgentResult:
    """Agent 执行结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    state_update: Dict[str, Any] = field(default_factory=dict)  # Memory 状态更新


class BaseAgent(ABC):
    """
    Agent 基类（有 Memory）

    所有 Agent 必须继承此类并实现 execute 方法。

    特点：
    - 有 Memory，跨调用保持状态
    - 可以读取和更新 Memory
    - 可以在 env 中读写数据
    """

    def __init__(self, name: str, memory: Optional[BaseMemory] = None):
        """
        初始化 Agent

        Args:
            name: Agent 名称
            memory: Memory 对象
        """
        self.name = name
        self.memory = memory

    @abstractmethod
    def execute(self, env: SharedEnvironment, *args, **kwargs) -> AgentResult:
        """
        执行 Agent

        Args:
            env: SharedEnvironment 实例
            *args, **kwargs: Agent 特定的参数

        Returns:
            AgentResult: 执行结果
        """
        pass

    def update_memory(self, key: str, value: Any) -> None:
        """更新 Memory 中的值"""
        if self.memory and hasattr(self.memory, 'update'):
            self.memory.update(key, value)

    def read_memory(self, key: str) -> Any:
        """从 Memory 读取值"""
        if self.memory and hasattr(self.memory, 'read'):
            return self.memory.read(key)
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(memory={self.memory})"
