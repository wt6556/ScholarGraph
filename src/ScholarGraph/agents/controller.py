"""Controller - 编排器 Agent"""

import logging
from typing import Optional

from .base import BaseAgent, AgentResult
from ..shared_env import SharedEnvironment
from ..memory.pipeline_state import PipelineState, PipelinePhase
from ..models.paper import Paper, ParsedPaper
from ..functions import (
    PDFParser, Understanding, Enrichment,
    Classification, Relation, Embedding
)

logger = logging.getLogger(__name__)


class Controller(BaseAgent):
    """
    Controller: 流程编排，调度 Function

    维护 PipelineState，决定下一步调用哪个 Function
    """

    def __init__(self, env: SharedEnvironment):
        """
        初始化 Controller

        Args:
            env: SharedEnvironment 实例
        """
        super().__init__("Controller", memory=PipelineState())
        self.env = env

    def execute(self, env: SharedEnvironment,
                action: str,
                **kwargs) -> AgentResult:
        """
        执行 Controller

        Args:
            env: SharedEnvironment
            action: 'ingestion' | 'query'
            **kwargs: 额外参数

        Returns:
            AgentResult
        """
        try:
            if action == 'ingestion':
                return self.run_ingestion(**kwargs)
            elif action == 'query':
                return self.run_query(**kwargs)
            else:
                return AgentResult(success=False, error=f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"Controller execution failed: {e}")
            return AgentResult(success=False, error=str(e))

    def run_ingestion(self, pdf_path: str) -> AgentResult:
        """
        执行摄取流程

        Args:
            pdf_path: PDF 文件路径

        Returns:
            AgentResult
        """
        logger.info(f"Starting ingestion: {pdf_path}")
        self.memory.update_phase(PipelinePhase.PARSE)

        # 1. PDFParser
        pdf_parser = PDFParser()
        parse_result = pdf_parser.execute(self.env, pdf_path)
        if not parse_result.success:
            return AgentResult(success=False, error=f"PDF parsing failed: {parse_result.error}")

        # 保存 ParsedPaper 供后续使用
        parsed_paper = parse_result.data

        self.memory.update_phase(PipelinePhase.UNDERSTAND)

        # 2. Understanding (需要 ParsedPaper)
        understanding = Understanding()
        result = understanding.execute(self.env, parsed_paper)
        if not result.success:
            logger.warning(f"Understanding failed: {result.error}")

        # 从 Understanding 结果中获取更新后的 paper，并更新到 env
        if result.data:
            current_paper = result.data
            self.env.update_paper(current_paper)

        self.memory.update_phase(PipelinePhase.ENRICH)

        # 3. Enrichment (可选)
        current_paper = self.env.get_current_paper()
        if current_paper and current_paper.has_missing_fields():
            from ..functions.enrichment import Enrichment
            enrichment = Enrichment()
            enrichment.execute(self.env, current_paper)

        self.memory.update_phase(PipelinePhase.CLASSIFY)

        # 4. Classification
        classification = Classification()
        if current_paper:
            classification.execute(self.env, current_paper)

        self.memory.update_phase(PipelinePhase.RELATE)

        # 5. Relation
        relation = Relation()
        if current_paper:
            relation.execute(self.env, paper=current_paper)

        self.memory.update_phase(PipelinePhase.EMBED)

        # 6. Embedding
        embedding = Embedding()
        if current_paper:
            embedding.execute(self.env, current_paper)

        self.memory.update_phase(PipelinePhase.IDLE)
        logger.info(f"Ingestion completed: {pdf_path}")

        return AgentResult(
            success=True,
            data={"paper_id": current_paper.id if current_paper else None}
        )

    def run_query(self, query: str) -> AgentResult:
        """
        执行查询流程

        注意：查询流程主要由 QueryAgent 驱动

        Args:
            query: 用户查询

        Returns:
            AgentResult
        """
        from .query_agent import QueryAgent
        from ..memory.conversation_history import ConversationHistory

        logger.info(f"Starting query: {query}")

        # 创建 QueryAgent
        memory = ConversationHistory()
        query_agent = QueryAgent(memory)

        # 执行查询
        result = query_agent.execute(self.env, query)

        if result.success:
            # 根据结果执行相应的 Function
            target = result.data.get('target_function', 'retrieval')
            if target == 'relation':
                relation = Relation()
                relation.execute(self.env, query=query)
            elif target == 'retrieval':
                from ..functions.retrieval import Retrieval
                from ..functions.synthesis import Synthesis
                retrieval = Retrieval()
                retrieval.execute(self.env, result.data)
                synthesis = Synthesis()
                synthesis.execute(self.env, [], result.data)

        return result
