"""PaperAgent 异常类型定义"""


class PaperAgentError(Exception):
    """PaperAgent 基础异常"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# 摄取流程异常
class PDFParseError(PaperAgentError):
    """PDF 解析失败"""
    pass


class UnderstandingError(PaperAgentError):
    """论文理解（字段提取）失败"""
    pass


class EnrichmentError(PaperAgentError):
    """信息补全失败"""
    pass


class ClassificationError(PaperAgentError):
    """论文分类失败"""
    pass


class RelationError(PaperAgentError):
    """关系提取失败"""
    pass


class EmbeddingError(PaperAgentError):
    """嵌入生成失败"""
    pass


# 查询流程异常
class QueryAnalysisError(PaperAgentError):
    """查询分析失败"""
    pass


class RetrievalError(PaperAgentError):
    """检索失败"""
    pass


class SynthesisError(PaperAgentError):
    """答案合成失败"""
    pass


class CritiqueError(PaperAgentError):
    """评估失败"""
    pass


# 存储异常
class StorageError(PaperAgentError):
    """存储操作失败"""
    pass


class PaperNotFoundError(StorageError):
    """论文不存在"""
    pass


class ChunkNotFoundError(StorageError):
    """Chunk 不存在"""
    pass


# 配置异常
class ConfigError(PaperAgentError):
    """配置错误"""
    pass


class APIKeyMissingError(ConfigError):
    """缺少 API Key"""
    pass


# CCF 解析异常
class CCFParseError(PaperAgentError):
    """CCF PDF 解析失败"""
    pass


class VenueNotFoundError(PaperAgentError):
    """Venue 未找到"""
    pass
