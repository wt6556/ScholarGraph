"""分类体系加载模块"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import yaml
import logging

logger = logging.getLogger(__name__)


class TaxonomyNotFoundError(Exception):
    """分类体系文件未找到或无效"""
    pass


@dataclass
class Field:
    """研究领域"""
    name: str
    subfields: List[str] = field(default_factory=list)


@dataclass
class Taxonomy:
    """
    分类体系

    Attributes:
        fields: 研究领域列表
        default_field: 默认领域
        default_subfield: 默认子领域
    """
    fields: List[Field] = field(default_factory=list)
    default_field: str = "Computer Science"
    default_subfield: str = "Artificial Intelligence"

    def get_field_names(self) -> List[str]:
        """获取所有领域名称"""
        return [f.name for f in self.fields]

    def get_subfields(self, field_name: str) -> List[str]:
        """获取指定领域的所有子领域"""
        for f in self.fields:
            if f.name == field_name:
                return f.subfields
        return []

    def is_valid_field(self, field_name: str) -> bool:
        """检查领域是否有效"""
        return field_name in self.get_field_names()

    def is_valid_subfield(self, field_name: str, subfield: str) -> bool:
        """检查子领域是否有效"""
        return subfield in self.get_subfields(field_name)

    def to_llm_prompt_fields(self) -> str:
        """生成 LLM 提示用的领域列表"""
        lines = []
        for f in self.fields:
            lines.append(f"- {f.name}")
            for sf in f.subfields:
                lines.append(f"  - {sf}")
        return "\n".join(lines)

    # 常见别名映射：别名 -> field 名称
    # 注意：这些值都是有效的 field 名称
    FIELD_ALIASES = {
        "nlp": "Computer Science",
        "natural language processing": "Computer Science",
        "natural language": "Computer Science",
        "ml": "Computer Science",
        "machine learning": "Computer Science",
        "ai": "Computer Science",
        "artificial intelligence": "Computer Science",
        "cv": "Computer Science",
        "computer vision": "Computer Science",
        "cs": "Computer Science",
        "computer science": "Computer Science",
        "systems": "Systems & Security",
        "systems & security": "Systems & Security",
        "security": "Systems & Security",
        "media": "Media & Interaction",
        "media & interaction": "Media & Interaction",
        "theory": "Theory & Algorithms",
        "theory & algorithms": "Theory & Algorithms",
    }

    def normalize_field(self, field_name: str) -> str:
        """规范化领域名称"""
        if not field_name:
            return self.default_field

        # 精确匹配
        if self.is_valid_field(field_name):
            return field_name

        # 别名匹配（不区分大小写）
        normalized = field_name.strip().lower()
        if normalized in self.FIELD_ALIASES:
            canonical = self.FIELD_ALIASES[normalized]
            logger.info(f"Normalized field '{field_name}' -> '{canonical}'")
            return canonical

        # 部分匹配
        for valid_name in self.get_field_names():
            if valid_name.lower() in normalized or normalized in valid_name.lower():
                logger.info(f"Normalized field '{field_name}' -> '{valid_name}' (partial match)")
                return valid_name

        logger.warning(f"Unknown field '{field_name}', using default: {self.default_field}")
        return self.default_field

    # 子领域别名映射：field -> {alias -> subfield}
    SUBFIELD_ALIASES = {
        "Computer Science": {
            "nlp": "Natural Language Processing",
            "natural language processing": "Natural Language Processing",
            "natural language": "Natural Language Processing",
            "ml": "Machine Learning",
            "machine learning": "Machine Learning",
            "ai": "Artificial Intelligence",
            "artificial intelligence": "Artificial Intelligence",
            "cv": "Computer Vision",
            "computer vision": "Computer Vision",
            "peft": "Parameter-Efficient Fine-Tuning",
            "parameter efficient fine-tuning": "Parameter-Efficient Fine-Tuning",
            "parameter-efficient fine-tuning": "Parameter-Efficient Fine-Tuning",
            "large language model": "Large Language Model Fine-tuning",
            "llm": "Large Language Model Fine-tuning",
            "llm fine-tuning": "Large Language Model Fine-tuning",
            "fine-tuning": "Large Language Model Fine-tuning",
        },
    }

    def normalize_subfield(self, field_name: str, subfield: str) -> str:
        """规范化子领域名称"""
        if not subfield:
            return self.get_subfields(field_name)[0] if self.get_subfields(field_name) else self.default_subfield

        # 获取该 field 的子领域
        valid_subfields = self.get_subfields(field_name)
        if not valid_subfields:
            return self.default_subfield

        # 精确匹配
        if subfield in valid_subfields:
            return subfield

        # 别名匹配
        normalized = subfield.strip().lower()
        aliases = self.SUBFIELD_ALIASES.get(field_name, {})
        if normalized in aliases:
            canonical = aliases[normalized]
            if canonical in valid_subfields:
                logger.info(f"Normalized subfield '{subfield}' -> '{canonical}'")
                return canonical

        # 部分匹配
        for valid_sf in valid_subfields:
            if valid_sf.lower() in normalized or normalized in valid_sf.lower():
                logger.info(f"Normalized subfield '{subfield}' -> '{valid_sf}' (partial match)")
                return valid_sf

        logger.warning(f"Unknown subfield '{subfield}' for field '{field_name}', using default: {valid_subfields[0]}")
        return valid_subfields[0]


# 全局 taxonomy 实例
_taxonomy: Optional[Taxonomy] = None


def load_taxonomy(taxonomy_path: str = "./config/taxonomy.yaml") -> Taxonomy:
    """
    加载分类体系

    Args:
        taxonomy_path: taxonomy.yaml 文件路径

    Returns:
        Taxonomy 实例

    Raises:
        TaxonomyNotFoundError: 当分类体系文件不存在或格式无效
    """
    global _taxonomy

    if _taxonomy is not None:
        return _taxonomy

    if not os.path.exists(taxonomy_path):
        raise TaxonomyNotFoundError(
            f"分类体系文件不存在: {taxonomy_path}\n"
            "请先创建或更新分类体系: python -m ScholarGraph.cli update-taxonomy <taxonomy_file>"
        )

    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not data:
        raise TaxonomyNotFoundError(
            f"分类体系文件为空: {taxonomy_path}\n"
            "请先创建或更新分类体系: python -m ScholarGraph.cli update-taxonomy <taxonomy_file>"
        )

    fields = []
    for item in data.get('fields', []):
        name = item.get('name')
        subfields = item.get('subfields', [])
        if name:
            fields.append(Field(name=name, subfields=subfields))

    if not fields:
        raise TaxonomyNotFoundError(
            f"分类体系文件格式无效，至少需要一个 field: {taxonomy_path}\n"
            "请先创建或更新分类体系: python -m ScholarGraph.cli update-taxonomy <taxonomy_file>"
        )

    _taxonomy = Taxonomy(
        fields=fields,
        default_field=data.get('default_field', 'Computer Science'),
        default_subfield=data.get('default_subfield', 'Artificial Intelligence')
    )

    logger.info(f"Loaded taxonomy with {len(fields)} fields")
    return _taxonomy


def get_taxonomy() -> Taxonomy:
    """
    获取已加载的分类体系

    Returns:
        Taxonomy 实例

    Raises:
        TaxonomyNotFoundError: 当分类体系未加载
    """
    global _taxonomy
    if _taxonomy is None:
        return load_taxonomy()
    return _taxonomy


def reset_taxonomy() -> None:
    """重置分类体系（用于测试或重新加载）"""
    global _taxonomy
    _taxonomy = None
