"""文本处理工具"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    清洗文本

    Args:
        text: 原始文本

    Returns:
        清洗后的文本
    """
    if not text:
        return ""

    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)

    # 移除特殊字符（保留中文、英文、数字、常用标点）
    text = re.sub(r'[^\w\s一-鿿.,!?;:\-\'"()[\]{}]', '', text)

    return text.strip()


def split_sentences(text: str) -> List[str]:
    """
    将文本分割成句子

    Args:
        text: 文本

    Returns:
        句子列表
    """
    if not text:
        return []

    # 按常见句末标点分割
    sentences = re.split(r'[.!?。！？]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    提取关键词

    Args:
        text: 文本
        top_n: 返回数量

    Returns:
        关键词列表
    """
    if not text:
        return []

    # 简单实现：按词频提取
    words = re.findall(r'\w+', text.lower())
    word_freq = {}
    for word in words:
        if len(word) > 2:  # 忽略短词
            word_freq[word] = word_freq.get(word, 0) + 1

    # 排序
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    截断文本

    Args:
        text: 文本
        max_length: 最大长度
        suffix: 后缀

    Returns:
        截断后的文本
    """
    if not text or len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def normalize_whitespace(text: str) -> str:
    """
    规范化空白字符

    Args:
        text: 文本

    Returns:
        处理后的文本
    """
    if not text:
        return ""

    # 将所有连续空白替换为单个空格
    return ' '.join(text.split())
