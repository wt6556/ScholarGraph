"""JSON 解析 / 校验工具"""

import json
import re
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def parse_json(content: str) -> Optional[Dict[str, Any]]:
    """
    解析 JSON 字符串，支持从混合文本中提取 JSON

    Args:
        content: 原始文本

    Returns:
        解析后的字典，或 None
    """
    if not content:
        return None

    # 尝试直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 对象
    json_match = re.search(r'\{[^{}]*"[^{}]*\}', content, re.DOTALL)
    if json_match:
        start = content.find('{')
        end = content.rfind('}') + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

    # 尝试提取 JSON 数组
    array_match = re.search(r'\[[^\[\]]*\]', content, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass

    logger.warning(f"Failed to parse JSON from content: {content[:200]}")
    return None


def validate_schema(data: Dict[str, Any], schema: Dict[str, type]) -> tuple:
    """
    验证数据是否符合 schema

    Args:
        data: 待验证的数据
        schema: schema 定义 {字段名: 类型}

    Returns:
        (is_valid, errors)
    """
    errors = []

    for field, expected_type in schema.items():
        if field not in data:
            errors.append(f"Missing required field: {field}")
            continue

        if not isinstance(data[field], expected_type):
            errors.append(
                f"Field '{field}' expected {expected_type.__name__}, "
                f"got {type(data[field]).__name__}"
            )

    return len(errors) == 0, errors


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    从文本中提取 JSON

    Args:
        text: 文本

    Returns:
        提取的 JSON 字典
    """
    return parse_json(text)


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    安全获取字典值

    Args:
        data: 字典
        key: 键
        default: 默认值

    Returns:
        值或默认值
    """
    try:
        return data.get(key, default)
    except (AttributeError, TypeError):
        return default
