"""LLM 配置加载"""

import os
from typing import Optional

from ..config.config import Config, LLMConfig as BaseLLMConfig


def load_llm_config() -> BaseLLMConfig:
    """从配置文件加载 LLM 配置"""
    config = Config.load()
    return config.llm


# 全局 LLM 客户端实例
_llm_client: Optional["LLMClient"] = None


def get_llm_client() -> "LLMClient":
    """获取全局 LLM 客户端实例"""
    global _llm_client
    if _llm_client is None:
        from .client import LLMClient
        config = load_llm_config()
        _llm_client = LLMClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
    return _llm_client
