"""Function 基类"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from ..shared_env import SharedEnvironment


@dataclass
class FunctionResult:
    """Function 执行结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseFunction(ABC):
    """
    Function 基类（无状态）

    所有 Function 必须继承此类并实现 execute 方法。

    特点：
    - 无状态，不在内部维护任何状态
    - 所有中间结果通过 SharedEnvironment 读写
    - 执行完毕即结束，不保留任何内存
    """

    def __init__(self, name: str):
        """
        初始化 Function

        Args:
            name: Function 名称
        """
        self.name = name

    @abstractmethod
    def execute(self, env: SharedEnvironment, *args, **kwargs) -> FunctionResult:
        """
        执行 Function

        Args:
            env: SharedEnvironment 实例，用于读写数据
            *args, **kwargs: Function 特定的参数

        Returns:
            FunctionResult: 执行结果
        """
        pass

    def validate_input(self, *args, **kwargs) -> bool:
        """
        验证输入参数（可选实现）

        Returns:
            输入是否有效
        """
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
