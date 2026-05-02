"""TaxonomyUpdater - 从不同来源获取并更新分类体系"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

import yaml

from .base import BaseFunction, FunctionResult
from ..shared_env import SharedEnvironment

logger = logging.getLogger(__name__)


class TaxonomySource(ABC):
    """分类来源抽象基类"""

    @abstractmethod
    def fetch(self) -> Dict[str, Any]:
        """
        获取分类体系

        Returns:
            包含 fields 列表的字典
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """来源名称"""
        pass


class ACSCCTaxonomySource(TaxonomySource):
    """ACM Computing Classification System (CCS) 来源

    基于 ACM 官方 Computing Classification System 整理
    来源: https://www.acm.org/publications/computing-classification-system
    """

    def name(self) -> str:
        return "ccs"

    def fetch(self) -> Dict[str, Any]:
        """
        获取 ACM CCS 分类

        基于 ACM Computing Classification System (CCS) 官方标准分类整理
        """
        logger.info("正在加载 ACM CCS 分类体系...")

        return {
            "source": "ccs",
            "source_url": "https://www.acm.org/publications/computing-classification-system",
            "fields": [
                {
                    "name": "Computing methodologies",
                    "subfields": [
                        "Artificial intelligence",
                        "Computer graphics",
                        "Machine learning",
                        "Natural language processing",
                        "Neural networks",
                        "Planning and scheduling",
                        "Robotics"
                    ]
                },
                {
                    "name": "Computer systems",
                    "subfields": [
                        "Computer architectures",
                        "Distributed computing",
                        "Operating systems",
                        "Real-time systems",
                        "Sensor networks"
                    ]
                },
                {
                    "name": "Software and its engineering",
                    "subfields": [
                        "Compilers",
                        "Development frameworks",
                        "Programming languages",
                        "Software architectures",
                        "Software testing"
                    ]
                },
                {
                    "name": "Data and information systems",
                    "subfields": [
                        "Data management systems",
                        "Data mining",
                        "Information integration",
                        "Information retrieval",
                        "Spatial-temporal systems"
                    ]
                },
                {
                    "name": "Security and privacy",
                    "subfields": [
                        "Access control",
                        "Cryptography",
                        "Database security",
                        "Network security",
                        "Software security"
                    ]
                },
                {
                    "name": "Human-centered computing",
                    "subfields": [
                        "Collaborative computing",
                        "Human computer interaction",
                        "Ubiquitous computing",
                        "Visualization"
                    ]
                }
            ],
            "updated_at": datetime.now().isoformat()
        }


class CCFTaxonomySource(TaxonomySource):
    """中国计算机学会（CCF）分类来源

    基于 CCF 推荐的计算机科学学术会议和期刊分类整理
    来源: https://ccf-cccr.ccf.org.cn/
    """

    def name(self) -> str:
        return "ccf"

    def fetch(self) -> Dict[str, Any]:
        """
        获取 CCF 分类

        基于中国计算机学会推荐的学术会议与期刊列表整理
        """
        logger.info("正在加载 CCF 分类体系...")

        return {
            "source": "ccf",
            "source_url": "https://ccf-cccr.ccf.org.cn/",
            "fields": [
                {
                    "name": "Computer Science",
                    "subfields": [
                        "Artificial Intelligence",
                        "Computer Vision",
                        "Natural Language Processing",
                        "Machine Learning",
                        "Robotics",
                        "Knowledge Graph",
                        "Speech Processing",
                        "Affective Computing"
                    ]
                },
                {
                    "name": "Systems & Security",
                    "subfields": [
                        "Computer Architecture",
                        "Operating Systems",
                        "Networks",
                        "Distributed Systems",
                        "Database Systems",
                        "Security & Privacy",
                        "Software Engineering"
                    ]
                },
                {
                    "name": "Media & Interaction",
                    "subfields": [
                        "Computer Graphics",
                        "Human-Computer Interaction",
                        "Multimedia",
                        "Virtual Reality",
                        "Augmented Reality",
                        "Visualization"
                    ]
                },
                {
                    "name": "Theory & Algorithms",
                    "subfields": [
                        "Algorithms",
                        "Complexity Theory",
                        "Cryptography",
                        "Formal Methods",
                        "Optimization"
                    ]
                },
                {
                    "name": "Interdisciplinary",
                    "subfields": [
                        "Computational Biology",
                        "Computational Chemistry",
                        "Computer Education",
                        "Economics & Computation",
                        "Quantum Computing"
                    ]
                }
            ],
            "updated_at": datetime.now().isoformat()
        }


class JournalTaxonomySource(TaxonomySource):
    """SCI 期刊分类来源

    基于 SCI (Science Citation Index) 期刊分类整理
    来源: Web of Science
    """

    def name(self) -> str:
        return "journal"

    def fetch(self) -> Dict[str, Any]:
        """
        获取 SCI 期刊分类

        基于 SCI 期刊分类整理
        """
        logger.info("正在加载 SCI 期刊分类体系...")

        return {
            "source": "journal",
            "source_url": "https://journals.clarivate.com/",
            "fields": [
                {
                    "name": "SCI - Computer Science",
                    "subfields": [
                        "Computer Science, Artificial Intelligence",
                        "Computer Science, Cybernetics",
                        "Computer Science, Hardware & Architecture",
                        "Computer Science, Information Systems",
                        "Computer Science, Interdisciplinary Applications",
                        "Computer Science, Software Engineering",
                        "Computer Science, Theory & Methods"
                    ]
                },
                {
                    "name": "SCI - Engineering",
                    "subfields": [
                        "Engineering, Aerospace",
                        "Engineering, Biomedical",
                        "Engineering, Chemical",
                        "Engineering, Civil",
                        "Engineering, Electrical & Electronic",
                        "Engineering, Industrial",
                        "Engineering, Manufacturing",
                        "Engineering, Mechanical",
                        "Engineering, Multidisciplinary"
                    ]
                },
                {
                    "name": "SCI - Mathematics",
                    "subfields": [
                        "Mathematics, Applied",
                        "Mathematics, Interdisciplinary",
                        "Mathematics, Mathematical Physics",
                        "Statistics & Probability"
                    ]
                }
            ],
            "updated_at": datetime.now().isoformat()
        }


class CustomTaxonomySource(TaxonomySource):
    """自定义分类来源（从 YAML 文件）"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def name(self) -> str:
        return "custom"

    def fetch(self) -> Dict[str, Any]:
        """从 YAML 文件加载分类"""
        logger.info(f"正在从文件加载分类体系: {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data or 'fields' not in data:
            raise ValueError(f"分类体系文件格式无效: {self.file_path}")

        data['source'] = 'custom'
        data['updated_at'] = datetime.now().isoformat()

        return data


class TaxonomyUpdater(BaseFunction):
    """
    TaxonomyUpdater: 更新分类体系

    支持从不同来源获取分类：
    - ccs: ACM Computing Classification System
    - ccf: 中国计算机学会
    - journal: SCI 期刊分类
    - custom: 自定义 YAML 文件
    """

    def __init__(self):
        super().__init__("TaxonomyUpdater")
        self._sources = {
            "ccs": ACSCCTaxonomySource(),
            "ccf": CCFTaxonomySource(),
            "journal": JournalTaxonomySource(),
        }

    def get_source(self, name: str, **kwargs) -> TaxonomySource:
        """获取指定来源"""
        if name == "custom":
            file_path = kwargs.get("file_path")
            if not file_path:
                raise ValueError("custom 来源需要提供 --file 参数")
            return CustomTaxonomySource(file_path)

        source = self._sources.get(name)
        if not source:
            raise ValueError(f"未知的分类来源: {name}，支持的来源: ccs, ccf, journal, custom")
        return source

    def execute(
        self,
        env: SharedEnvironment,
        source_name: str,
        file_path: Optional[str] = None,
        output_path: str = "./config/taxonomy.yaml"
    ) -> FunctionResult:
        """
        更新分类体系

        Args:
            env: SharedEnvironment
            source_name: 来源名称 (ccs/ccf/journal/custom)
            file_path: 自定义 YAML 文件路径（仅 custom 来源需要）
            output_path: 输出文件路径

        Returns:
            FunctionResult
        """
        try:
            # 获取来源
            source = self.get_source(source_name, file_path=file_path)

            # 备份旧文件
            import shutil
            if os.path.exists(output_path):
                backup_path = f"{output_path}.backup.{datetime.now().strftime('%Y%m%d%H%M%S')}"
                shutil.copy(output_path, backup_path)
                logger.info(f"已备份旧分类体系到: {backup_path}")

            # 获取分类数据
            taxonomy_data = source.fetch()

            # 验证必要字段
            if 'fields' not in taxonomy_data:
                return FunctionResult(
                    success=False,
                    error="分类数据缺少 'fields' 字段"
                )

            fields = taxonomy_data.get('fields', [])
            if not fields:
                return FunctionResult(
                    success=False,
                    error="'fields' 不能为空"
                )

            # 写入新分类体系
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(taxonomy_data, f, allow_unicode=True, sort_keys=False)

            logger.info(f"分类体系已更新: {output_path} (来源: {source.name()})")

            return FunctionResult(
                success=True,
                data={
                    "source": source.name(),
                    "fields_count": len(fields),
                    "output_path": output_path
                }
            )

        except Exception as e:
            logger.error(f"更新分类体系失败: {e}")
            return FunctionResult(success=False, error=str(e))