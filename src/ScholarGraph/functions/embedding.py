"""Embedding - 嵌入生成"""

import logging
from typing import List

from .base import BaseFunction, FunctionResult
from ..shared_env import SharedEnvironment
from ..models.paper import Paper, Chunk
from ..config import get_config

logger = logging.getLogger(__name__)


class Embedding(BaseFunction):
    """
    Embedding: 论文 → chunk embeddings

    输入: Paper
    输出: List[Chunk] (写入 ChunkStore)
    """

    def __init__(self):
        super().__init__("Embedding")
        self._model = None
        self._config = None

    def _get_model(self):
        """获取 embedding 模型"""
        if self._model is None:
            self._config = get_config()
            provider = self._config.embedding.provider

            if provider == "transformers":
                from transformers import AutoTokenizer, AutoModel
                import torch
                self._model = AutoModel.from_pretrained(self._config.embedding.model)
                self._tokenizer = AutoTokenizer.from_pretrained(self._config.embedding.model)
                self._device = self._config.embedding.device
                logger.info(f"Loaded transformers model: {self._config.embedding.model}")
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")

        return self._model

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """使用 transformers 生成 embeddings"""
        import torch
        import torch.nn.functional as F

        # 确保模型和 tokenizer 已加载
        self._get_model()

        # 编码文本
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # 移动到设备
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        # 前向传播
        with torch.no_grad():
            outputs = self._model(**encoded)

        # Mean pooling
        attention_mask = encoded['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().tolist()

    def execute(self, env: SharedEnvironment, paper: Paper) -> FunctionResult:
        """
        生成嵌入

        Args:
            env: SharedEnvironment
            paper: 论文

        Returns:
            FunctionResult.data = List[Chunk]
        """
        try:
            chunks: List[Chunk] = []

            # 1. 创建 chunks
            # Abstract chunk
            if paper.abstract:
                abstract_chunk = self._create_chunk(
                    paper=paper,
                    chunk_type="abstract",
                    content=paper.abstract,
                    env=env
                )
                if abstract_chunk:
                    chunks.append(abstract_chunk)

            # Method chunk
            if paper.method or paper.core_idea:
                method_content = []
                if paper.method:
                    method_content.append(f"Method: {paper.method}")
                if paper.core_idea:
                    method_content.append(f"Core Idea: {paper.core_idea}")
                if paper.method_category:
                    method_content.append(f"Category: {paper.method_category}")

                method_text = "\n".join(method_content)
                method_chunk = self._create_chunk(
                    paper=paper,
                    chunk_type="method",
                    content=method_text,
                    env=env
                )
                if method_chunk:
                    chunks.append(method_chunk)

            # Task chunk
            if paper.task:
                task_chunk = self._create_chunk(
                    paper=paper,
                    chunk_type="task",
                    content=paper.task,
                    env=env
                )
                if task_chunk:
                    chunks.append(task_chunk)

            # Experiment chunk
            if paper.datasets or paper.baselines or paper.improvements:
                exp_content = []
                if paper.datasets:
                    exp_content.append(f"Datasets: {', '.join(paper.datasets)}")
                if paper.baselines:
                    exp_content.append(f"Baselines: {', '.join(paper.baselines)}")
                if paper.improvements:
                    imp_texts = []
                    for imp in paper.improvements:
                        if isinstance(imp, dict):
                            imp_texts.append(f"{imp.get('dataset', '')}: {imp.get('metric', '')} {imp.get('delta', '')}")
                        else:
                            imp_texts.append(str(imp))
                    exp_content.append(f"Improvements: {', '.join(imp_texts)}")

                exp_text = "\n".join(exp_content)
                exp_chunk = self._create_chunk(
                    paper=paper,
                    chunk_type="experiment",
                    content=exp_text,
                    env=env
                )
                if exp_chunk:
                    chunks.append(exp_chunk)

            # Conclusion/Limitation chunk
            if paper.contribution or paper.limitation:
                concl_content = []
                if paper.contribution:
                    if isinstance(paper.contribution, list):
                        concl_content.append(f"Contributions: {', '.join(paper.contribution)}")
                    else:
                        concl_content.append(f"Contributions: {paper.contribution}")
                if paper.limitation:
                    concl_content.append(f"Limitations: {paper.limitation}")

                concl_text = "\n".join(concl_content)
                concl_chunk = self._create_chunk(
                    paper=paper,
                    chunk_type="conclusion",
                    content=concl_text,
                    env=env
                )
                if concl_chunk:
                    chunks.append(concl_chunk)

            # 2. 生成 embeddings
            if chunks:
                texts = [chunk.content for chunk in chunks]
                embeddings = self._get_embeddings(texts)

                for i, chunk in enumerate(chunks):
                    chunk.embedding = embeddings[i]

            # 3. 写入 ChunkStore
            if chunks:
                env.add_chunks(chunks)
                logger.info(f"Generated {len(chunks)} chunks for paper: {paper.id}")
            else:
                logger.info(f"No chunks generated for paper: {paper.id}")

            return FunctionResult(success=True, data=chunks)

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return FunctionResult(success=False, error=str(e))

    def _create_chunk(self, paper: Paper, chunk_type: str, content: str, env: SharedEnvironment) -> Chunk:
        """创建单个 chunk"""
        if not content or len(content.strip()) < 10:
            return None

        # 截断过长内容
        if len(content) > 2000:
            content = content[:2000]

        return Chunk.from_paper(
            paper=paper,
            chunk_type=chunk_type,
            content=content,
            embedding=[]  # 暂时留空，等生成
        )

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入"""
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        # 转换为 Python list
        return embeddings.tolist()