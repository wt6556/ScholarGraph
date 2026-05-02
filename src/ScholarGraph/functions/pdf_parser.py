"""PDFParser - PDF 解析"""

from typing import Optional
import logging

from .base import BaseFunction, FunctionResult
from ..shared_env import SharedEnvironment
from ..models.paper import ParsedPaper, Paper

logger = logging.getLogger(__name__)


class PDFParser(BaseFunction):
    """
    PDFParser: PDF 解析 → ParsedPaper

    输入: PDF 路径
    输出: ParsedPaper (sections, metadata)
    """

    def __init__(self):
        super().__init__("PDFParser")

    def execute(self, env: SharedEnvironment, pdf_path: str) -> FunctionResult:
        """
        解析 PDF 文件

        Args:
            env: SharedEnvironment
            pdf_path: PDF 文件路径

        Returns:
            FunctionResult.data = ParsedPaper
        """
        try:
            # TODO: 实现 GROBID 解析
            # 目前使用 pdfplumber 作为 fallback

            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                sections = {}

                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                # 简单的章节分割
                lines = text.split('\n')
                current_section = "body"
                current_content = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # 检测章节标题
                    lower_line = line.lower()
                    if any(marker in lower_line for marker in ['abstract', 'introduction', 'related work', 'method', 'experiment', 'conclusion', 'reference']):
                        if current_content:
                            sections[current_section] = "\n".join(current_content)
                        current_section = line.lower()
                        current_content = []
                    else:
                        current_content.append(line)

                if current_content:
                    sections[current_section] = "\n".join(current_content)

            # 提取基本信息（简单实现）
            title = self._extract_title(text) or "Unknown Title"
            authors = self._extract_authors(text) or []

            parsed = ParsedPaper(
                title=title,
                authors=authors,
                abstract=sections.get('abstract', ''),
                sections=sections,
                raw_text=text,
                pdf_path=pdf_path
            )

            # 注意：不创建 Paper 对象，只返回 ParsedPaper
            # Paper 对象由 Understanding 在 LLM 提取字段后创建
            logger.info(f"Parsed PDF: {title}")
            return FunctionResult(success=True, data=parsed)

        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            return FunctionResult(success=False, error=str(e))

    def _extract_title(self, text: str) -> Optional[str]:
        """提取标题（简单实现）"""
        lines = text.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                return line
        return None

    def _extract_authors(self, text: str) -> Optional[list]:
        """提取作者（简单实现）"""
        # TODO: 实现更复杂的作者提取
        return []
