"""CCFParser - CCF 评级解析"""

import logging
import os
import re
from typing import Dict, List, Any, Optional

import yaml
import pdfplumber

from .base import BaseFunction, FunctionResult
from ..shared_env import SharedEnvironment

logger = logging.getLogger(__name__)


class CCFParser(BaseFunction):
    """
    CCFParser: CCF PDF → 评级映射表

    输入: CCF PDF 路径
    输出: CCFMappings (venue → CCF rating 映射)
    """

    def __init__(self):
        super().__init__("CCFParser")

    def execute(
        self,
        env: SharedEnvironment,
        ccf_pdf_path: str,
        output_path: str = "./config/ccf_mappings.yaml"
    ) -> FunctionResult:
        """
        解析 CCF PDF，提取会议/期刊评级信息

        Args:
            env: SharedEnvironment
            ccf_pdf_path: CCF PDF 文件路径
            output_path: 输出文件路径

        Returns:
            FunctionResult.data = CCFMappings
        """
        try:
            if not os.path.exists(ccf_pdf_path):
                logger.warning(f"CCF PDF 文件不存在: {ccf_pdf_path}，返回空映射")
                mappings = {
                    "version": "unknown",
                    "source_file": ccf_pdf_path,
                    "fields": {},
                    "venue_aliases": {},
                    "venue_ratings": {}
                }
                return FunctionResult(success=True, data=mappings)

            # 提取 PDF 文本
            venues = self._extract_venues_from_pdf(ccf_pdf_path)

            if not venues:
                return FunctionResult(
                    success=False,
                    error="未能从 PDF 中提取到会议/期刊信息"
                )

            # 生成别名映射
            venue_aliases = self._generate_aliases(venues)

            # 构建映射数据
            mappings = {
                "version": self._extract_version(ccf_pdf_path),
                "source_file": ccf_pdf_path,
                "fields": self._organize_by_field(venues),
                "venue_aliases": venue_aliases,
                "venue_ratings": {v["name"]: v["rating"] for v in venues}
            }

            # 保存到 YAML
            self._save_mappings(mappings, output_path)

            logger.info(f"Parsed CCF PDF: {ccf_pdf_path}, found {len(venues)} venues")

            return FunctionResult(success=True, data=mappings)

        except Exception as e:
            logger.error(f"Failed to parse CCF PDF: {e}")
            return FunctionResult(success=False, error=str(e))

    def _extract_venues_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """从 PDF 提取会议/期刊信息"""
        venues = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                current_section = None  # 当前领域
                current_rating = None    # 当前评级 (CCF-A, CCF-B, CCF-C)
                is_journal = True       # 当前是期刊还是会议

                for page in pdf.pages:
                    text = page.extract_text()
                    if not text:
                        continue

                    lines = text.split('\n')

                    # 首先解析上下文（section, rating, type）
                    for line in lines:
                        line = line.strip()

                        # 检测是期刊还是会议部分
                        if "推荐国际学术期刊" in line:
                            is_journal = True
                        elif "推荐国际学术会议" in line:
                            is_journal = False

                        # 检测领域标题
                        section = self._extract_section(line)
                        if section:
                            current_section = section

                        # 检测评级
                        rating = self._extract_rating(line)
                        if rating:
                            current_rating = rating

                    # 处理表格
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if not row or len(row) < 2:
                                continue

                            # 检查是否是表头
                            first_cell = str(row[0]) if row[0] else ""
                            if '序号' in first_cell or '刊物简称' in first_cell or '会议简称' in first_cell:
                                continue

                            # 提取简称（通常是第一或第二列）
                            short_name = self._extract_short_name(row, is_journal)

                            # 提取全称（通常是第三列）
                            full_name = self._extract_full_name(row, is_journal)

                            # 如果有评级和领域，添加venues
                            if current_rating and current_section:
                                if short_name:
                                    venues.append({
                                        "name": short_name,
                                        "rating": current_rating,
                                        "field": current_section
                                    })
                                if full_name and full_name != short_name:
                                    venues.append({
                                        "name": full_name,
                                        "rating": current_rating,
                                        "field": current_section
                                    })

        except Exception as e:
            logger.error(f"Error extracting from PDF: {e}")

        return venues

    def _extract_section(self, line: str) -> Optional[str]:
        """提取领域名称"""
        # 检测括号中的领域，如 "（计算机体系结构/并行与分布计算/存储系统）"
        match = re.search(r'（(.+?)）', line)
        if match:
            return match.group(1).strip()
        return None

    def _extract_rating(self, line: str) -> Optional[str]:
        """从行中提取 CCF 评级"""
        # 匹配 "一、A 类" 或 "二、B 类" 或 "三、C 类"
        # 注意：可能有空格
        match = re.search(r'[一二三]、\s*([ABC])\s*类', line)
        if match:
            rating = match.group(1)
            return f"CCF-{rating}"
        return None

    def _extract_short_name(self, row: List, is_journal: bool) -> Optional[str]:
        """从表格行提取简称"""
        if not row:
            return None

        # 期刊表格列：序号, 刊物简称, 刊物全称, 出版社, 网址
        # 会议表格列：序号, 会议简称, 会议全称, 出版社, 网址

        # 通常第二列（索引1）是简称
        if len(row) > 1 and row[1]:
            name = str(row[1]).strip().replace('\n', ' ')
            if name and len(name) > 1 and not name.startswith('http'):
                # 简称通常是全大写字母
                if re.match(r'^[A-Z]{2,}$', name):
                    return name
                # 也可能是混合大小写的简称
                if len(name) < 20 and re.match(r'^[A-Za-z0-9\-]+$', name):
                    return name

        # 有时简称在第一列（序号后面）
        if len(row) > 0:
            name = str(row[0]).strip().replace('\n', ' ')
            # 序号是纯数字，不是名称
            if re.match(r'^\d+$', name):
                return None
            if name and len(name) > 1 and len(name) < 15:
                if re.match(r'^[A-Z]{2,}$', name):
                    return name

        return None

    def _extract_full_name(self, row: List, is_journal: bool) -> Optional[str]:
        """从表格行提取全称"""
        if not row:
            return None

        # 期刊表格列：序号, 刊物简称, 刊物全称, 出版社, 网址
        # 全称通常是第三列（索引2）
        if len(row) > 2 and row[2]:
            name = str(row[2]).strip().replace('\n', ' ')
            if name and len(name) > 3 and not name.startswith('http'):
                # 全称通常包含 "Transactions", "Journal", "Conference" 等词
                if any(kw in name for kw in ['Transactions', 'Journal', 'Conference', 'Proceedings', 'Systems']):
                    return name

        return None

    def _generate_aliases(self, venues: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """生成别名映射"""
        aliases = {}

        for venue in venues:
            name = venue["name"]
            if not name:
                continue

            alias_list = []

            # 全大写版本
            alias_list.append(name.upper())

            # 如果是全称，提取可能的简称
            short = self._extract_known_short(name)
            if short:
                alias_list.append(short)

            if alias_list:
                aliases[name] = list(set(alias_list))

        return aliases

    def _extract_known_short(self, full_name: str) -> Optional[str]:
        """从全称提取已知简称"""
        # 常见的 Transactions 简称映射
        known_mappings = {
            'ACM Transactions on Computer Systems': 'TOCS',
            'ACM Transactions on Storage': 'TOS',
            'IEEE Transactions on Computers': 'TC',
            'IEEE Transactions on Parallel and Distributed Systems': 'TPDS',
            'IEEE Transactions on Computer-Aided Design': 'TCAD',
        }
        return known_mappings.get(full_name)

    def _organize_by_field(self, venues: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """按领域组织会议/期刊"""
        fields = {}

        for venue in venues:
            field = venue["field"]
            if field not in fields:
                fields[field] = []
            fields[field].append({
                "name": venue["name"],
                "rating": venue["rating"]
            })

        return fields

    def _extract_version(self, pdf_path: str) -> str:
        """从文件名提取版本信息"""
        filename = os.path.basename(pdf_path)
        # 尝试匹配年份
        year_match = re.search(r'(20\d{2})', filename)
        if year_match:
            return f"v{year_match.group(1)}"
        return "unknown"

    def _save_mappings(self, mappings: Dict[str, Any], output_path: str) -> None:
        """保存映射到 YAML 文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # 备份旧文件
        if os.path.exists(output_path):
            backup_path = f"{output_path}.backup"
            with open(output_path, 'r', encoding='utf-8') as f:
                old_content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(old_content)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(mappings, f, allow_unicode=True, sort_keys=False)