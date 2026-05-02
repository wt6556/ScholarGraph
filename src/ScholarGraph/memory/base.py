"""Memory 基类"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseMemory(ABC):
    """Memory 基类（抽象）"""

    @abstractmethod
    def read(self, key: str) -> Any:
        """读取值"""
        pass

    @abstractmethod
    def update(self, key: str, value: Any) -> None:
        """更新值"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空 Memory"""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        pass

    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典恢复"""
        pass
