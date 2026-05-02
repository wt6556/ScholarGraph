"""Classification - 论文分类"""

import logging
import json
import re
from typing import Optional, Dict, Any

from .base import BaseFunction, FunctionResult
from ..shared_env import SharedEnvironment
from ..models.paper import Paper
from ..llm import get_llm_client, Message
from ..config import load_taxonomy, TaxonomyNotFoundError

logger = logging.getLogger(__name__)

# 分类提示词模板（field/subfield 动态填充）
CLASSIFICATION_PROMPT_TEMPLATE = """你是一个学术论文分类助手。请根据论文的标题、摘要和方法判断其所属的研究领域。

可选的研究领域(field)包括：
{fields}

可选的子领域(subfield)请从上述领域中选择。

主题关键词(topic)包括领域内具体的技术或方法。

请以JSON格式返回：
{{
    "field": "研究领域（必须从上述列表中选择）",
    "subfield": "子领域",
    "topic": ["关键词1", "关键词2", ...]
}}

论文标题: {title}
摘要: {abstract}
方法: {method}
"""


class Classification(BaseFunction):
    """
    Classification: 论文 → field/subfield/topic

    输入: Paper
    输出: Paper (带分类标签)
    """

    def __init__(self):
        super().__init__("Classification")

    def execute(self, env: SharedEnvironment, paper: Paper) -> FunctionResult:
        """
        分类论文

        Args:
            env: SharedEnvironment
            paper: 论文

        Returns:
            FunctionResult.data = ClassificationResult

        Raises:
            TaxonomyNotFoundError: 当分类体系文件不存在时
        """
        try:
            # 如果已经有分类信息，直接返回
            if paper.paper_field and paper.subfield:
                logger.info(f"Paper {paper.id} already classified: {paper.paper_field}")
                return FunctionResult(
                    success=True,
                    data={
                        "field": paper.paper_field,
                        "subfield": paper.subfield,
                        "topic": paper.topic
                    }
                )

            # 使用 LLM 进行分类（会抛出 TaxonomyNotFoundError 如果 taxonomy 不存在）
            classification = self._classify_with_llm(
                title=paper.title,
                abstract=f"{paper.task or ''} {paper.method or ''}",
                method=paper.method or ""
            )

            if classification:
                # 规范化 field 和 subfield 名称
                taxonomy = load_taxonomy()
                paper.paper_field = taxonomy.normalize_field(classification.get("field", ""))
                paper.subfield = taxonomy.normalize_subfield(paper.paper_field, classification.get("subfield", ""))
                paper.topic = classification.get("topic", [])
                env.update_paper(paper)

            logger.info(f"Classified paper: {paper.id} as {paper.paper_field}/{paper.subfield}")
            return FunctionResult(
                success=True,
                data={
                    "field": paper.paper_field,
                    "subfield": paper.subfield,
                    "topic": paper.topic
                }
            )

        except TaxonomyNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to classify paper: {e}")
            return FunctionResult(success=False, error=str(e))

    def _classify_with_llm(
        self,
        title: str,
        abstract: str,
        method: str
    ) -> Optional[Dict[str, Any]]:
        """
        使用 LLM 进行分类

        Args:
            title: 论文标题
            abstract: 摘要
            method: 方法描述

        Returns:
            分类结果字典

        Raises:
            TaxonomyNotFoundError: 当分类体系文件不存在时
        """
        try:
            # 加载分类体系（如果不存在会抛出 TaxonomyNotFoundError）
            taxonomy = load_taxonomy()

            # 构建字段列表
            fields_text = taxonomy.to_llm_prompt_fields()

            client = get_llm_client()
            prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
                fields=fields_text,
                title=title,
                abstract=abstract[:2000] if abstract else "N/A",
                method=method[:500] if method else "N/A"
            )

            response = client.chat(
                messages=[Message(role="user", content=prompt)],
                max_tokens=1024,
                temperature=0.3
            )

            content = response.content.strip()

            # 提取 JSON
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse classification response: {content[:200]}")
                return None

        except TaxonomyNotFoundError:
            raise
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return None
