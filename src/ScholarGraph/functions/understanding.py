"""Understanding - 字段提取"""

import logging
import json
import re
from typing import Optional

from .base import BaseFunction, FunctionResult
from ..shared_env import SharedEnvironment
from ..models.paper import ParsedPaper, Paper
from ..llm import get_llm_client, Message

logger = logging.getLogger(__name__)


# 字段提取提示词
EXTRACTION_PROMPT = """你是一个学术论文分析助手。请从给定论文的文本中提取以下所有信息：

## 基础信息
1. title: 论文标题（从文本中识别，必须正确，不要有拼写错误）
2. authors: 作者列表（从文本中识别，格式：["姓名1", "姓名2", ...]）
3. year: 发表年份（如果能找到）
4. abstract: 摘要（从文本中识别，完整的一段或多段摘要）

## 论文结构（请识别论文包含哪些章节，并提取每个章节的名称和内容概要）
5. sections: 论文章节结构，格式如：
   - "sections": {
       "Introduction": "介绍部分的主要内容...",
       "Method": "方法部分的核心内容...",
       "Experiment": "实验部分的设置和结果...",
       ...
     }

## 研究内容分析
6. task: 研究任务/问题（一句话描述这篇论文在解决什么问题）
7. assumption: 研究假设/前提
8. motivation: 研究动机（为什么需要这项研究）
9. method: 方法名称（请使用标准缩写，如LoRA、SoRA、AdaLoRA、S-LoRA、MoELoRA、LoRAMoE、MOLE等；如果是新方法则使用简短名称，不要使用括号内的完整描述如"Sparse Low-rank Adaptation (SoRA)"）
10. method_category: 方法类别（如：architecture, training, optimization, etc.）
11. core_idea: 核心思想（方法的关键创新点）
12. baselines: 基线方法（论文对比的方法，列表形式，使用标准缩写）
13. datasets: 使用的数据集（列表形式）
14. improvements: 性能提升（对比基线的提升，用JSON数组格式，每项包含dataset、metric、delta）

## 贡献与局限
15. contribution: 主要贡献点
16. limitation: 局限性

## 分类信息
17. field: 研究领域（如：Computer Vision, NLP, ML等）
18. subfield: 研究子领域（如：Image Classification, Object Detection等）
19. topic: 主题关键词（列表形式）

请以JSON格式返回所有字段。如果某字段无法从论文中确定，请使用null或空列表。

论文文本: {body}
"""


class Understanding(BaseFunction):
    """
    Understanding: JSON → 18+ 字段提取

    输入: ParsedPaper
    输出: Paper (task, method, dataset, ...)
    """

    def __init__(self):
        super().__init__("Understanding")

    def execute(self, env: SharedEnvironment, parsed_paper: ParsedPaper) -> FunctionResult:
        """
        提取论文字段

        Args:
            env: SharedEnvironment
            parsed_paper: 解析后的论文

        Returns:
            FunctionResult.data = Paper (含 18+ 字段)
        """
        try:
            paper = parsed_paper.to_paper()
            text = parsed_paper.get_full_text()

            # 使用 LLM 提取所有字段（包括标题、作者、摘要等）
            extracted = self._extract_with_llm(body=text)  # 发送完整文本给 LLM

            # 更新 paper 对象
            if extracted:
                # 基础字段
                paper.title = extracted.get("title") or paper.title
                paper.authors = extracted.get("authors") or []
                paper.year = extracted.get("year")
                paper.abstract = extracted.get("abstract")
                # 论文结构
                paper.sections = extracted.get("sections") or {}
                # 分析字段
                paper.task = extracted.get("task")
                paper.assumption = extracted.get("assumption")
                paper.motivation = extracted.get("motivation")
                paper.method = extracted.get("method")
                paper.method_category = extracted.get("method_category")
                paper.core_idea = extracted.get("core_idea")
                paper.baselines = extracted.get("baselines") or []
                paper.datasets = extracted.get("datasets") or []
                paper.improvements = extracted.get("improvements") or []
                paper.contribution = extracted.get("contribution")
                paper.limitation = extracted.get("limitation")
                # paper_field 和 subfield 由 Classification 函数设置（会进行规范化）
                paper.topic = extracted.get("topic") or []

                # 用正确的标题生成 ID
                from ..models.paper import generate_paper_id
                paper.id = generate_paper_id(paper.title, paper.year)

            # 更新到 env
            env.update_paper(paper)

            logger.info(f"Extracted fields for: {paper.title}")
            return FunctionResult(success=True, data=paper)

        except Exception as e:
            logger.error(f"Failed to extract fields: {e}")
            return FunctionResult(success=False, error=str(e))

    def _extract_with_llm(self, body: str) -> Optional[dict]:
        """
        使用 LLM 提取字段

        Args:
            body: 论文完整文本

        Returns:
            提取的字段字典
        """
        try:
            client = get_llm_client()
            prompt = EXTRACTION_PROMPT.replace("{body}", body if body else "N/A")

            response = client.chat(
                messages=[Message(role="user", content=prompt)],
                max_tokens=16384,
                temperature=0.3  # 低温度保证稳定性
            )

            # 解析 JSON 响应
            content = response.content.strip()

            # 尝试提取 JSON（支持 markdown 代码块和嵌套 JSON）
            # 先尝试从 markdown 代码块中提取
            json_candidates = []

            # 1. 尝试从 ```json ... ``` 中提取
            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if code_block_match:
                json_candidates.append(code_block_match.group(1))

            # 2. 尝试直接匹配 JSON 对象（处理嵌套）
            # 找到所有 { 开始的位置，从最长的可能 JSON 开始尝试
            json_start_positions = [m.start() for m in re.finditer(r'\{', content)]
            for start in json_start_positions:
                # 从每个 { 位置开始，尝试找到完整匹配的 }
                candidate = content[start:]
                # 尝试用简单的括号匹配来找到对应的结束括号
                depth = 0
                end_pos = 0
                in_string = False
                escape_next = False
                for i, c in enumerate(candidate):
                    if escape_next:
                        escape_next = False
                        continue
                    if c == '\\':
                        escape_next = True
                        continue
                    if c == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            end_pos = i + 1
                            break
                if end_pos > 0:
                    json_candidates.append(candidate[:end_pos])

            # 3. 按长度排序，优先尝试更长的（更可能是完整的 JSON）
            json_candidates.sort(key=len, reverse=True)

            for candidate in json_candidates:
                try:
                    result = json.loads(candidate)
                    return result
                except (json.JSONDecodeError, Exception) as e:
                    continue

            # 4. 如果都失败了，尝试直接解析整个 content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return None

        except Exception as e:
            logger.warning(f"LLM extraction failed: {type(e).__name__}: {e}")
            return None

    def _extract_with_rules(self, text: str, title: str) -> dict:
        """
        使用规则提取字段（备用方案）

        Args:
            text: 论文文本
            title: 论文标题

        Returns:
            提取的字段字典
        """
        result = {}

        # 简单地从文本中提取信息
        result["task"] = self._extract_task(text)
        result["method"] = self._extract_method(text, title)
        result["assumption"] = self._extract_assumption(text)
        result["motivation"] = self._extract_motivation(text)
        result["core_idea"] = self._extract_core_idea(text)
        result["baselines"] = self._extract_baselines(text)
        result["datasets"] = self._extract_datasets(text)
        result["contribution"] = self._extract_contribution(text)
        result["limitation"] = self._extract_limitation(text)

        return result

    def _extract_task(self, text: str) -> Optional[str]:
        """提取研究任务"""
        # TODO: 使用规则或 LLM
        return None

    def _extract_method(self, text: str, title: str) -> Optional[str]:
        """提取方法名称"""
        # 简单规则：从标题提取
        if '：' in title:
            return title.split('：')[0].strip()
        if ':' in title:
            return title.split(':')[0].strip()
        return None

    def _extract_assumption(self, text: str) -> Optional[str]:
        """提取假设"""
        return None

    def _extract_motivation(self, text: str) -> Optional[str]:
        """提取动机"""
        return None

    def _extract_core_idea(self, text: str) -> Optional[str]:
        """提取核心思想"""
        return None

    def _extract_baselines(self, text: str) -> list:
        """提取 baseline 方法"""
        return []

    def _extract_datasets(self, text: str) -> list:
        """提取数据集"""
        return []

    def _extract_contribution(self, text: str) -> Optional[str]:
        """提取贡献"""
        return None

    def _extract_limitation(self, text: str) -> Optional[str]:
        """提取局限性"""
        return None
