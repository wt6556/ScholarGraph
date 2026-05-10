"""Enrichment - 信息补全"""

import logging
from typing import Optional

from .base import BaseFunction, FunctionResult
from ..shared_env import SharedEnvironment
from ..models.paper import Paper
from ..web.crossref_client import CrossRefClient

logger = logging.getLogger(__name__)


class Enrichment(BaseFunction):
    """
    Enrichment: 补全缺失字段

    使用免费学术 API 补全论文信息：
    - CrossRef API (免费，无需 API key)
    - arXiv API (免费，用于 arXiv 论文)

    输入: Paper (有缺失)
    输出: Paper (补全后)
    """

    def __init__(self):
        super().__init__("Enrichment")

    def execute(self, env: SharedEnvironment, paper: Paper) -> FunctionResult:
        """
        补全论文信息

        Args:
            env: SharedEnvironment
            paper: 有缺失字段的论文

        Returns:
            FunctionResult.data = EnrichmentResult
        """
        crossref = CrossRefClient()  # 按需创建，符合无状态原则

        try:
            # 检查是否有缺失字段
            if not paper.has_missing_fields():
                logger.info(f"Paper {paper.id} has no missing fields")
                return FunctionResult(success=True, data={"enriched": False})

            missing = paper.get_missing_fields()
            logger.info(f"Paper {paper.id} missing fields: {missing}")

            enriched_fields = {}

            # 检查 title 是否有效（至少有空格或足够长）
            title = paper.title
            if title and len(title.strip()) > 5:
                # 尝试通过 CrossRef API 补全（结合作者和年份精确查询）
                results = crossref.query_by_title(
                    title,
                    authors=paper.authors if paper.authors else None,
                    year=paper.year,
                    limit=3
                )
                if results:
                    best_match = results[0]
                    self._apply_enrichment(paper, best_match, enriched_fields)

            # 如果有 DOI，尝试直接查询
            if paper.doi:
                result = crossref.query_by_doi(paper.doi)
                if result:
                    self._apply_enrichment(paper, result, enriched_fields)

            # 检查是否成功补全了任何字段
            if enriched_fields:
                # 更新 paper 对象
                for field, value in enriched_fields.items():
                    setattr(paper, field, value)

                # 保存到数据库
                env.update_paper(paper)

                logger.info(f"Enriched paper {paper.id} with fields: {list(enriched_fields.keys())}")
                return FunctionResult(
                    success=True,
                    data={
                        "enriched": True,
                        "fields_enriched": list(enriched_fields.keys()),
                        "remaining_missing": paper.get_missing_fields()
                    }
                )
            else:
                logger.info(f"Could not enrich paper {paper.id}")
                return FunctionResult(
                    success=True,
                    data={
                        "enriched": False,
                        "reason": "No matching papers found in academic databases",
                        "missing_fields": missing
                    }
                )

        except Exception as e:
            logger.error(f"Failed to enrich paper: {e}")
            return FunctionResult(success=False, error=str(e))

    def _apply_enrichment(self, paper: Paper, metadata: dict, enriched_fields: dict) -> None:
        """应用从 API 获取的元数据到论文对象"""
        # 补全作者
        if not paper.authors and metadata.get("authors"):
            enriched_fields["authors"] = metadata["authors"]

        # 补全年份
        if not paper.year and metadata.get("year"):
            enriched_fields["year"] = metadata["year"]

        # 补全 venue
        if not paper.venue and metadata.get("venue"):
            enriched_fields["venue"] = metadata["venue"]

        # 补全 DOI
        if not paper.doi and metadata.get("doi"):
            enriched_fields["doi"] = metadata["doi"]

        # 补全摘要
        if not paper.abstract and metadata.get("abstract"):
            # CrossRef 的摘要可能包含 XML 标签，需要清理
            abstract = metadata["abstract"]
            import re
            abstract = re.sub(r'<[^>]+>', '', abstract)
            enriched_fields["abstract"] = abstract