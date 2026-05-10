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
RELATION_EXTRACTION_PROMPT = """You are an academic paper relationship analysis assistant. Analyze the relationship between the new paper and existing papers.

New Paper Information:
Title: {title}
Method: {method}
Core Idea: {core_idea}
Research Field: {field}

Existing Methods:
{existing_methods}

Retrieved Context from Papers (RAG):
{rag_context}

Based on the retrieved context above, determine the relationship between the new paper and existing methods:
- Is there an improvement/inheritance relationship?
- Is there a similar technical approach?
- Is there a shared application scenario?
- Only report relationships supported by the retrieved context.

Important:
- method_name must use standard abbreviations: LoRA, SoRA, AdaLoRA, S-LoRA, MoELoRA, LoRAMoE, MOLE
- from_method and to_method must use standard abbreviations, do not use full names like "Sparse Low-rank Adaptation (SoRA)"
- Method names in parent_methods array must also use standard abbreviations

Return relationships in JSON format:
{{
    "relations": [
        {{
            "from_method": "existing method (abbreviation)",
            "to_method": "new paper method (abbreviation)",
            "improvement_direction": "accuracy/efficiency/generalization/...",
            "description": "Relationship description based on retrieved context"
        }}
    ],
    "method_name": "New paper method name (abbreviation)",
    "is_root": true/false,
    "parent_methods": ["methods inherited from (abbreviation)"]
}}

If no clear relationship is found, return an empty relations array. Only report relationships supported by the retrieved context.
"""

TOPOLOGY_QUERY_PROMPT = """You are an academic paper topology analysis assistant. Determine the method domain to query based on user input.

User Query: {query}

Return in JSON format:
{{
    "topic": "Method domain keywords",
    "intent": "Query intent (e.g., find_improvements, compare_methods, understand_evolution)"
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

    def _get_embedding_model_and_tokenizer(self):
        """获取 embedding 模型和 tokenizer"""
        from ..config import get_config
        config = get_config()
        from transformers import AutoTokenizer, AutoModel
        import torch

        model = AutoModel.from_pretrained(config.embedding.model)
        tokenizer = AutoTokenizer.from_pretrained(config.embedding.model)
        device = config.embedding.device
        pooling_strategy = getattr(config.embedding, 'pooling_strategy', 'mean')
        model.to(device)
        model.eval()

        return model, tokenizer, device, pooling_strategy

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

    def _encode_text_for_retrieval(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for retrieval."""
        import torch
        import torch.nn.functional as F

        model, tokenizer, device, pooling_strategy = self._get_embedding_model_and_tokenizer()

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        if pooling_strategy == 'last_token':
            # Last-token pooling
            attention_mask = encoded['attention_mask']
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                embeddings = outputs.last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = outputs.last_hidden_state.shape[0]
                embeddings = outputs.last_hidden_state[
                    torch.arange(batch_size, device=outputs.last_hidden_state.device),
                    sequence_lengths
                ]
        else:
            # Mean pooling
            attention_mask = encoded['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().tolist()

    def _rag_direction_a_existing_to_new(
        self,
        env: SharedEnvironment,
        paper: Paper,
        existing_methods: List[str],
        top_k: int = 20
    ) -> Dict[str, List[Any]]:
        """
        Direction A: Find existing papers that mention the new paper.

        Returns:
            Dict[paper_id, List[Chunk]] - existing papers that reference the new paper
        """
        if not existing_methods:
            return {}

        try:
            # Build query from new paper info
            query_parts = [paper.title]
            if paper.method:
                query_parts.append(paper.method)
            if paper.core_idea:
                query_parts.append(paper.core_idea[:200])
            query = " ".join(query_parts)

            # Encode and search
            query_embedding = self._encode_text_for_retrieval([query])[0]
            search_results = env.chunk_store.search_with_scores(
                query_embedding,
                top_k=top_k * 3,
                filters=None
            )

            # Group by paper_id, exclude new paper's own chunks
            paper_chunks: Dict[str, List[Any]] = {}
            for chunk, score in search_results:
                if chunk.paper_id == paper.id:
                    continue
                if chunk.paper_id not in paper_chunks:
                    paper_chunks[chunk.paper_id] = []
                paper_chunks[chunk.paper_id].append(chunk)

            # Limit to top 5 chunks per paper
            for pid in paper_chunks:
                paper_chunks[pid] = paper_chunks[pid][:5]

            logger.debug(f"Direction A: Found {len(paper_chunks)} papers referencing new paper")
            return paper_chunks

        except Exception as e:
            logger.warning(f"Direction A retrieval failed: {e}")
            return {}

    def _rag_direction_b_new_to_existing(
        self,
        env: SharedEnvironment,
        paper: Paper,
        existing_methods: List[str],
        top_k_per_paper: int = 3
    ) -> Dict[str, List[Any]]:
        """
        Direction B: Find which existing papers the new paper mentions.

        Returns:
            Dict[paper_id, List[Chunk]] - papers mentioned by the new paper
        """
        # Get new paper's chunks (already stored before Relation is called)
        new_paper_chunks = env.get_chunks_by_paper(paper.id)
        if not new_paper_chunks:
            logger.warning(f"No chunks found for new paper {paper.id}")
            return {}

        # Get existing paper titles from the same field
        field = paper.paper_field or "general"
        existing_papers = env.paper_store.get_by_field(field, limit=50) if hasattr(env, 'paper_store') else []
        existing_paper_titles: Dict[str, str] = {}
        for p in existing_papers:
            if p.id != paper.id:
                existing_paper_titles[p.id] = p.title

        # Build search terms from method names
        search_terms: Dict[str, str] = {}  # lower_term -> original_term
        for m in existing_methods:
            search_terms[m.lower()] = m
        for pid, title in existing_paper_titles.items():
            if title:
                search_terms[title.lower()] = title

        if not search_terms:
            return {}

        # Scan chunks for mentions
        paper_chunks: Dict[str, List[Any]] = {}
        for chunk in new_paper_chunks:
            chunk_lower = chunk.content.lower()
            for term_lower, term_original in search_terms.items():
                if term_lower in chunk_lower:
                    # Find which paper this term belongs to
                    if term_original in existing_methods:
                        # It's a method name - find paper_id by method
                        nodes = env.topology_store.get_method_nodes(field)
                        for node in nodes:
                            if node.method_name == term_original:
                                if node.paper_id not in paper_chunks:
                                    paper_chunks[node.paper_id] = []
                                if chunk not in paper_chunks[node.paper_id]:
                                    paper_chunks[node.paper_id].append(chunk)
                                break
                    elif term_original in existing_paper_titles.values():
                        # It's a title - find paper_id
                        for pid, title in existing_paper_titles.items():
                            if title == term_original:
                                if pid not in paper_chunks:
                                    paper_chunks[pid] = []
                                if chunk not in paper_chunks[pid]:
                                    paper_chunks[pid].append(chunk)
                                break

        # Limit chunks per paper
        for pid in paper_chunks:
            paper_chunks[pid] = paper_chunks[pid][:top_k_per_paper]

        logger.debug(f"Direction B: Found mentions of {len(paper_chunks)} papers in new paper")
        return paper_chunks

    def _build_rag_context(
        self,
        direction_a: Dict[str, List[Any]],
        direction_b: Dict[str, List[Any]],
        env: SharedEnvironment
    ) -> str:
        """
        Build structured context from bidirectional RAG results.
        """
        context_parts = []

        # Direction A: papers mentioning the new paper
        if direction_a:
            context_parts.append("=== Papers That Reference The New Paper ===")
            for paper_id, chunks in direction_a.items():
                paper = env.get_paper(paper_id)
                if paper:
                    context_parts.append(f"\nPaper: {paper.title}")
                    context_parts.append(f"Method: {paper.method or 'Unknown'}")
                else:
                    context_parts.append(f"\nPaper ID: {paper_id}")

                context_parts.append("Relevant Content:")
                for chunk in chunks[:5]:
                    content = chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content
                    context_parts.append(f"  - [{chunk.chunk_type}] {content}")

        # Direction B: papers mentioned by the new paper
        if direction_b:
            context_parts.append("\n=== Papers Referenced By The New Paper ===")
            for paper_id, chunks in direction_b.items():
                paper = env.get_paper(paper_id)
                if paper:
                    context_parts.append(f"\nPaper: {paper.title}")
                    context_parts.append(f"Method: {paper.method or 'Unknown'}")
                else:
                    context_parts.append(f"\nPaper ID: {paper_id}")

                context_parts.append("Relevant Content:")
                for chunk in chunks[:5]:
                    content = chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content
                    context_parts.append(f"  - [{chunk.chunk_type}] {content}")

        return "\n".join(context_parts) if context_parts else ""

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

        # Bidirectional RAG retrieval
        direction_a = self._rag_direction_a_existing_to_new(
            env, paper, existing_methods, top_k=20
        )
        direction_b = self._rag_direction_b_new_to_existing(
            env, paper, existing_methods, top_k_per_paper=3
        )
        rag_context = self._build_rag_context(direction_a, direction_b, env)

        logger.info(f"Direction A: {len(direction_a)} papers, Direction B: {len(direction_b)} papers")

        # 使用 LLM 提取关系
        relations = self._extract_with_llm(
            title=paper.title,
            method=paper.method or "",
            core_idea=paper.core_idea or "",
            field=field,
            existing_methods=existing_methods,
            rag_context=rag_context
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
        existing_methods: List[str],
        rag_context: str = ""
    ) -> Dict[str, Any]:
        """
        使用 LLM 提取关系

        Args:
            title: 论文标题
            method: 方法名称
            core_idea: 核心思想
            field: 研究领域
            existing_methods: 已有方法列表
            rag_context: 双向 RAG 检索的上下文

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
                existing_methods="\n".join([f"- {m}" for m in existing_methods[:20]]) or "无",
                rag_context=rag_context or "No relevant papers found."
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
