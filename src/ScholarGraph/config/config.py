"""配置管理模块"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""
    provider: str = "transformers"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    device: str = "cpu"


@dataclass
class StorageConfig:
    """存储配置"""
    sqlite_path: str = "./data/ScholarGraph.db"
    vector_index_path: str = "./data/chunk_index.faiss"
    topology_db_path: str = "./data/topology.db"


@dataclass
class RerankerConfig:
    """Reranker 配置"""
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RAGRetrievalConfig:
    """RAG 检索配置"""
    top_k: int = 20           # 初步检索返回数量（粗排）
    rerank_top_k: int = 5     # 重排后返回数量（精排）
    vector_weight: float = 0.7  # 向量检索权重
    bm25_weight: float = 0.3   # BM25 检索权重


@dataclass
class RAGConfig:
    """RAG 配置"""
    retrieval: RAGRetrievalConfig = field(default_factory=RAGRetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)


@dataclass
class Config:
    """
    配置单例

    使用方式：
        config = Config.load("./config/config.yaml")
        model = config.llm.model
    """
    _instance: Optional['Config'] = None

    # LLM
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Embedding
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # 存储
    storage: StorageConfig = field(default_factory=StorageConfig)

    # RAG
    rag: RAGConfig = field(default_factory=RAGConfig)

    # 应用
    app_name: str = "PaperAgent"
    debug: bool = False
    log_level: str = "INFO"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def load(cls, config_path: str = "./config/config.yaml") -> 'Config':
        """
        从 YAML 文件加载配置

        Args:
            config_path: 配置文件路径

        Returns:
            Config 实例
        """
        if not os.path.exists(config_path):
            return cls()

        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if data is None:
            return cls()

        instance = cls()

        # 加载 LLM 配置
        if 'llm' in data:
            llm_data = data['llm']
            api_key = llm_data.get('api_key', '')
            if api_key and api_key.startswith('${') and api_key.endswith('}'):
                env_var = api_key[2:-1]
                api_key = os.getenv(env_var, '')
            instance.llm = LLMConfig(
                provider=llm_data.get('provider', 'openai'),
                model=llm_data.get('model', 'gpt-4o-mini'),
                api_key=api_key,
                base_url=llm_data.get('base_url'),
                timeout=llm_data.get('timeout', 60),
                max_retries=llm_data.get('max_retries', 3)
            )

        # 加载 Embedding 配置
        if 'embedding' in data:
            emb_data = data['embedding']
            instance.embedding = EmbeddingConfig(
                provider=emb_data.get('provider', 'sentence-transformers'),
                model=emb_data.get('model', 'all-MiniLM-L6-v2'),
                dimension=emb_data.get('dimension', 384),
                device=emb_data.get('device', 'cpu')
            )

        # 加载存储配置
        if 'sqlite' in data:
            sqlite_data = data['sqlite']
            instance.storage = StorageConfig(
                sqlite_path=sqlite_data.get('path', './data/ScholarGraph.db'),
                vector_index_path=data.get('vector_store', {}).get('save_path', './data/chunk_index.faiss'),
                topology_db_path=sqlite_data.get('path', './data/ScholarGraph.db').replace('.db', '_topology.db')
            )

        # 加载应用配置
        if 'app' in data:
            app_data = data['app']
            instance.app_name = app_data.get('name', 'PaperAgent')
            instance.debug = app_data.get('debug', False)

        # 加载日志配置
        if 'logging' in data:
            instance.log_level = data['logging'].get('level', 'INFO')

        # 加载 RAG 配置
        if 'rag' in data:
            rag_data = data['rag']

            # 加载检索配置
            retrieval_data = rag_data.get('retrieval', {})
            retrieval_config = RAGRetrievalConfig(
                top_k=retrieval_data.get('top_k', 20),
                rerank_top_k=retrieval_data.get('rerank_top_k', 5),
                vector_weight=retrieval_data.get('vector_weight', 0.7),
                bm25_weight=retrieval_data.get('bm25_weight', 0.3)
            )

            # 加载重排配置
            reranker_data = rag_data.get('reranker', {})
            reranker_config = RerankerConfig(
                enabled=reranker_data.get('enabled', True),
                model=reranker_data.get('model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            )

            instance.rag = RAGConfig(
                retrieval=retrieval_config,
                reranker=reranker_config
            )

        return instance

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持嵌套 key，如 'llm.model'）

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值或默认值
        """
        keys = key.split('.')
        value = self
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            else:
                return default
        return value

    def __repr__(self) -> str:
        return f"Config(app={self.app_name}, llm={self.llm.provider}/{self.llm.model})"


# 全局配置实例
_default_config: Optional[Config] = None


def get_config(config_path: str = "./config/config.yaml") -> Config:
    """
    获取配置单例

    Args:
        config_path: 配置文件路径

    Returns:
        Config 实例
    """
    global _default_config
    if _default_config is None:
        _default_config = Config.load(config_path)
    return _default_config


def reset_config() -> None:
    """重置配置（用于测试）"""
    global _default_config
    _default_config = None
