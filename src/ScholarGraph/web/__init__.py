"""Web 模块 - 外部 API 客户端"""

from .crossref_client import CrossRefClient
from .paper_with_cv import PaperWithCVClient

__all__ = [
    "CrossRefClient",
    "PaperWithCVClient",
]
