"""流程状态 Memory - Controller/CriticAgent 使用"""

from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, field

from .base import BaseMemory


class PipelinePhase(Enum):
    """流程阶段"""
    IDLE = "idle"
    PARSE = "parse"
    UNDERSTAND = "understand"
    ENRICH = "enrich"
    CLASSIFY = "classify"
    RELATE = "relate"
    EMBED = "embed"
    RETRIEVE = "retrieve"
    SYNTHESIZE = "synthesize"
    CRITIC = "critic"
    QUERY = "query"


@dataclass
class PipelineState(BaseMemory):
    """
    Controller/CriticAgent 用：存储流程状态

    维护：
    - 当前阶段
    - 重试次数
    - 失败检查项
    - 错误信息
    """

    phase: PipelinePhase = PipelinePhase.IDLE
    retries: int = 0
    max_retries: int = 3
    current_paper_id: Optional[str] = None
    failed_checks: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_phase(self, phase: PipelinePhase) -> None:
        """
        更新当前阶段

        Args:
            phase: 新阶段
        """
        self.phase = phase
        self.retries = 0  # 新阶段重置重试计数
        self.failed_checks = []  # 清空失败检查项
        self.error_message = None

    def increment_retry(self) -> int:
        """
        增加重试次数

        Returns:
            当前重试次数
        """
        self.retries += 1
        return self.retries

    def is_max_retries_exceeded(self) -> bool:
        """是否超过最大重试次数"""
        return self.retries >= self.max_retries

    def add_failed_check(self, check: str) -> None:
        """
        添加失败的检查项

        Args:
            check: 检查项名称
        """
        if check not in self.failed_checks:
            self.failed_checks.append(check)

    def clear_failed_checks(self) -> None:
        """清空失败检查项"""
        self.failed_checks = []

    def set_error(self, error: str) -> None:
        """
        设置错误信息

        Args:
            error: 错误信息
        """
        self.error_message = error

    def clear_error(self) -> None:
        """清空错误信息"""
        self.error_message = None

    def reset(self) -> None:
        """重置状态"""
        self.phase = PipelinePhase.IDLE
        self.retries = 0
        self.failed_checks = []
        self.error_message = None
        self.metadata = {}

    # BaseMemory 接口实现
    def clear(self) -> None:
        """清空 Memory"""
        self.reset()

    def read(self, key: str) -> Any:
        """读取值"""
        return getattr(self, key, None)

    def update(self, key: str, value: Any) -> None:
        """更新值"""
        if key == "phase" and isinstance(value, str):
            value = PipelinePhase(value)
        setattr(self, key, value)

    def set_metadata(self, key: str, value: Any) -> None:
        """
        设置元数据

        Args:
            key: 键
            value: 值
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        获取元数据

        Args:
            key: 键
            default: 默认值

        Returns:
            元数据值或默认值
        """
        return self.metadata.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "current_paper_id": self.current_paper_id,
            "failed_checks": self.failed_checks,
            "error_message": self.error_message,
            "metadata": self.metadata
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        从字典恢复状态

        Args:
            data: 状态字典
        """
        self.phase = PipelinePhase(data.get("phase", "idle"))
        self.retries = data.get("retries", 0)
        self.max_retries = data.get("max_retries", 3)
        self.current_paper_id = data.get("current_paper_id")
        self.failed_checks = data.get("failed_checks", [])
        self.error_message = data.get("error_message")
        self.metadata = data.get("metadata", {})

    def get_state(self) -> Dict[str, Any]:
        """获取完整状态（用于调试）"""
        return self.to_dict()

    def __repr__(self) -> str:
        return (
            f"PipelineState(phase={self.phase.value.upper()}, "
            f"retries={self.retries}/{self.max_retries}, "
            f"paper={self.current_paper_id})"
        )
