"""Relation - 关系提取"""

import logging
import json
import re
from typing import Optional, List, Dict, Any

from .base import BaseFunction, FunctionResult
from ..shared_env import SharedEnvironment
from ..models.paper import Paper
from ..storage.topology_store import MethodNode, MethodEdge
from ..llm import get_llm_client, Message

logger = logging.getLogger(__name__)

def normalize_method_name(name: str) -> str:
    """
    标准化方法名称（依赖LLM生成正确缩写，此函数仅作为保险）

    主要处理括号中的缩写，如 "Sparse Low-rank Adaptation (SoRA)" -> "SoRA"
    """
    if not name:
        return name

    name_clean = name.strip()

    # 已知方法的标准缩写
    KNOWN_METHODS = {"LoRA", "SoRA", "AdaLoRA", "S-LoRA", "MoELoRA", "LoRAMoE", "MOLE"}

    # 如果已是已知缩写，直接返回
    if name_clean in KNOWN_METHODS:
        return name_clean

    # 检查括号中的缩写
    match = re.search(r'\(([^)]+)\)', name_clean)
    if match:
        abbrev = match.group(1)
        if abbrev in KNOWN_METHODS:
            return abbrev

    return name_clean

# 关系提取提示词
RELATION_EXTRACTION_PROMPT = """你是一个学术论文关系分析助手。请分析以下论文与已有论文的关系。

新论文信息:
标题: {title}
方法: {method}
核心思想: {core_idea}
研究领域: {field}

已有论文方法列表:
{existing_methods}

请判断新论文与已有方法的关系：
- 是否有改进/继承关系？
- 是否有相似的技术路线？
- 是否有相同的应用场景？

重要提示：
- method_name（方法名称）必须使用标准缩写，如：LoRA、SoRA、AdaLoRA、S-LoRA、MoELoRA、LoRAMoE、MOLE
- from_method 和 to_method 也必须使用标准缩写，不要使用完整名称如"Sparse Low-rank Adaptation (SoRA)"
- parent_methods 数组中的方法名也必须使用标准缩写

请以JSON格式返回关系：
{{
    "relations": [
        {{
            "from_method": "已有方法(缩写)",
            "to_method": "新论文方法(缩写)",
            "improvement_direction": "accuracy/efficiency/generalization/...",
            "description": "关系描述"
        }}
    ],
    "method_name": "新论文方法名称(缩写)",
    "is_root": true/false,
    "parent_methods": ["继承自的方法(缩写)"]
}}

如果没有找到明显关系，返回空的relations数组。
"""

TOPOLOGY_QUERY_PROMPT = """你是一个学术论文拓扑分析助手。请根据用户查询确定要查询的方法领域。

用户查询: {query}

请以JSON格式返回：
{{
    "topic": "方法领域关键词",
    "intent": "查询意图 (e.g., find_improvements, compare_methods, understand_evolution)"
}}
"""


class Relation(BaseFunction):
    """
    Relation: 关系提取 + 拓扑图更新

    输入: Paper 或 Query
    输出: RelationResult / 更新 TopologyGraph
    """

    def __init__(self):
        super().__init__("Relation")

    def execute(self, env: SharedEnvironment,
                paper: Optional[Paper] = None,
                query: Optional[str] = None) -> FunctionResult:
        """
        提取关系

        Args:
            env: SharedEnvironment
            paper: 新论文（摄取阶段）
            query: 用户查询（查询阶段）

        Returns:
            FunctionResult.data = RelationResult
        """
        try:
            if paper:
                # 摄取阶段：提取新论文与已有论文的关系
                result = self._extract_relations_for_paper(env, paper)
                logger.info(f"Extracted relations for paper: {paper.id}")

            elif query:
                # 查询阶段：返回拓扑图
                topic_info = self._parse_topology_query(query)
                topology = env.get_method_topology(topic_info.get("topic", "general"))
                result = {
                    "topology": topology,
                    "query_processed": True,
                    "topic": topic_info.get("topic"),
                    "intent": topic_info.get("intent")
                }
                logger.info(f"Processed relation query: {query}")

            else:
                return FunctionResult(success=False, error="No paper or query provided")

            return FunctionResult(success=True, data=result)

        except Exception as e:
            logger.error(f"Failed to extract relations: {e}")
            return FunctionResult(success=False, error=str(e))

    def _extract_relations_for_paper(
        self,
        env: SharedEnvironment,
        paper: Paper
    ) -> Dict[str, Any]:
        """
        提取论文关系

        Args:
            env: SharedEnvironment
            paper: 新论文

        Returns:
            关系结果字典
        """
        # 获取该领域的已有方法
        field = paper.paper_field or "general"
        topology = env.get_method_topology(field)

        existing_methods = []
        if topology.get("nodes"):
            existing_methods = [node.get("id", "") for node in topology["nodes"]]

        # 使用 LLM 提取关系
        relations = self._extract_with_llm(
            title=paper.title,
            method=paper.method or "",
            core_idea=paper.core_idea or "",
            field=field,
            existing_methods=existing_methods
        )

        # 添加方法节点（标准化名称）
        method_name = normalize_method_name(relations.get("method_name", paper.method or "Unknown"))
        is_root = relations.get("is_root", len(relations.get("parent_methods", [])) == 0)
        parent_methods = [normalize_method_name(p) for p in relations.get("parent_methods", [])]

        node = MethodNode(
            id=f"{paper.id}_node",
            topic=field,
            paper_id=paper.id,
            method_name=method_name,
            improvement_direction="",
            key_innovation=paper.core_idea or "",
            is_root=is_root,
            parent_methods=parent_methods,
            created_at=paper.created_at.isoformat() if paper.created_at else ""
        )
        env.add_method_node(node)

        # 添加关系边（标准化名称）
        for rel in relations.get("relations", []):
            from_method = normalize_method_name(rel["from_method"])
            to_method = normalize_method_name(rel["to_method"])
            edge = MethodEdge(
                id=f"{from_method}_{to_method}",
                topic=field,
                from_method=from_method,
                to_method=to_method,
                improvement_direction=rel.get("improvement_direction", ""),
                description=rel.get("description", ""),
                created_at=paper.created_at.isoformat() if paper.created_at else ""
            )
            env.topology_store.add_relation(edge)

        return {
            "relations": relations.get("relations", []),
            "topology_updated": True
        }

    def _extract_with_llm(
        self,
        title: str,
        method: str,
        core_idea: str,
        field: str,
        existing_methods: List[str]
    ) -> Dict[str, Any]:
        """
        使用 LLM 提取关系

        Args:
            title: 论文标题
            method: 方法名称
            core_idea: 核心思想
            field: 研究领域
            existing_methods: 已有方法列表

        Returns:
            关系结果字典
        """
        try:
            client = get_llm_client()
            prompt = RELATION_EXTRACTION_PROMPT.format(
                title=title,
                method=method,
                core_idea=core_idea,
                field=field,
                existing_methods="\n".join([f"- {m}" for m in existing_methods[:20]]) or "无"
            )

            response = client.chat(
                messages=[Message(role="user", content=prompt)],
                max_tokens=2048,
                temperature=0.3
            )

            content = response.content.strip()

            # 尝试提取 JSON（支持嵌套）
            try:
                # 先尝试直接解析
                return json.loads(content)
            except json.JSONDecodeError:
                pass

            # 尝试从 markdown 代码块中提取
            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if code_block_match:
                try:
                    return json.loads(code_block_match.group(1))
                except json.JSONDecodeError:
                    pass

            # 尝试找到 JSON 对象的起止位置
            start = content.find('{')
            end = content.rfind('}')
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start:end+1])
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Failed to parse relation extraction response: {content[:200]}")
            return {}

        except Exception as e:
            logger.warning(f"LLM relation extraction failed: {e}")
            return {}

    def _parse_topology_query(self, query: str) -> Dict[str, str]:
        """
        解析拓扑查询

        Args:
            query: 用户查询

        Returns:
            解析结果 {topic, intent}
        """
        try:
            client = get_llm_client()
            prompt = TOPOLOGY_QUERY_PROMPT.format(query=query)

            response = client.chat(
                messages=[Message(role="user", content=prompt)],
                max_tokens=512,
                temperature=0.3
            )

            content = response.content.strip()

            # 尝试直接解析
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

            # 尝试从 markdown 代码块中提取
            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if code_block_match:
                try:
                    return json.loads(code_block_match.group(1))
                except json.JSONDecodeError:
                    pass

            # 尝试找到 JSON 起止
            start = content.find('{')
            end = content.rfind('}')
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start:end+1])
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Failed to parse topology query: {content[:200]}")
            return {"topic": "general", "intent": "explore"}

        except Exception as e:
            logger.warning(f"LLM topology query parsing failed: {e}")
            return {"topic": "general", "intent": "explore"}
