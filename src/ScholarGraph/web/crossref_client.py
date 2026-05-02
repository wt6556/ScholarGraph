"""CrossRef API 客户端"""

import logging
import time
from typing import Optional, Dict, Any, List
import requests

logger = logging.getLogger(__name__)


class CrossRefClient:
    """CrossRef API 客户端

    CrossRef API 是免费的学术论文元数据库
    文档: https://www.crossref.org/documentation/
    """

    BASE_URL = "https://api.crossref.org"

    def __init__(self, email: str = "", timeout: int = 30):
        """
        初始化 CrossRef 客户端

        Args:
            email: 邮箱（用于 User-Agent，CrossRef 推荐提供）
            timeout: 请求超时时间（秒）
        """
        self.email = email or "paper-agent@example.com"
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": f"PaperAgent/1.0 (mailto:{self.email})",
            "Accept": "application/json"
        })

    def query_by_title(self, title: str, authors: List[str] = None, year: int = None, limit: int = 5, max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        根据标题搜索论文（严格匹配：论文名+作者完全一致才返回）

        Args:
            title: 论文标题
            authors: 作者列表（用于精确匹配）
            year: 发表年份（用于精确匹配）
            limit: 返回数量
            max_retries: 最大重试次数

        Returns:
            论文列表（仅包含严格匹配的结果）
        """
        logger.info(f"[CrossRef] Searching title: {title}")

        for attempt in range(max_retries):
            try:
                url = f"{self.BASE_URL}/works"
                params = {
                    "query.title": title,
                    "rows": limit,
                    "mailto": self.email,
                }

                response = self._session.get(url, params=params, timeout=self.timeout)

                if not response.ok:
                    logger.warning(f"[CrossRef] Failed: {response.status_code} {response.reason}")
                    logger.warning(f"[CrossRef] Request URL: {response.url}")

                    if 400 <= response.status_code < 500:
                        logger.warning(f"[CrossRef] 4xx error, not retrying")
                        return []

                    wait_time = 2 ** attempt
                    logger.warning(f"[CrossRef] Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue

                data = response.json()
                items = data.get("message", {}).get("items", [])

                logger.info(f"[CrossRef] Raw items count: {len(items)}")

                results = []
                for i, item in enumerate(items):
                    parsed = self._parse_work(item)
                    logger.info(f"[CrossRef] Item {i}: title={parsed.get('title', '')[:50]}, authors={parsed.get('authors', [])}")

                    # 严格匹配：标题必须完全一致
                    if not self._title_match(parsed.get("title", ""), title):
                        logger.info(f"[CrossRef] Item {i} skipped: title doesn't match exactly")
                        continue

                    # 严格匹配：作者必须完全一致
                    if authors and not self._authors_match_strict(parsed.get("authors", []), authors):
                        logger.info(f"[CrossRef] Item {i} skipped: authors don't match exactly")
                        continue

                    results.append(parsed)
                    logger.info(f"[CrossRef] Item {i} matched!")

                logger.info(f"[CrossRef] Found {len(results)} strictly matched results")
                return results

            except requests.RequestException as e:
                logger.warning(f"[CrossRef] Request error: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                return []

        return []

    def _normalize_text(self, text: str) -> str:
        """标准化文本：转小写，移除空格和标点"""
        import re
        text = text.lower()
        text = re.sub(r'[\s\W]', '', text)
        return text

    def _title_match(self, result_title: str, query_title: str) -> bool:
        """检查标题是否完全匹配（忽略大小写和空格）"""
        if not result_title or not query_title:
            return False
        return self._normalize_text(result_title) == self._normalize_text(query_title)

    def _authors_match_strict(self, result_authors: List[str], query_authors: List[str]) -> bool:
        """检查作者是否完全匹配（作者数量和顺序都必须一致）"""
        if not result_authors or not query_authors:
            return False
        if len(result_authors) != len(query_authors):
            return False

        # 标准化后逐一比较
        result_normalized = [self._normalize_text(a) for a in result_authors]
        query_normalized = [self._normalize_text(a) for a in query_authors]

        return result_normalized == query_normalized

    def query_by_doi(self, doi: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        根据 DOI 查询论文元数据

        Args:
            doi: DOI
            max_retries: 最大重试次数

        Returns:
            论文元数据字典
        """
        for attempt in range(max_retries):
            try:
                if not doi.startswith("10."):
                    return None

                url = f"{self.BASE_URL}/works/{doi}"
                params = {"mailto": self.email}

                response = self._session.get(url, params=params, timeout=self.timeout)

                if not response.ok:
                    logger.warning(f"[CrossRef] DOI query failed: {response.status_code}")
                    if 400 <= response.status_code < 500:
                        return None

                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue

                data = response.json()
                item = data.get("message", {})
                return self._parse_work(item)

            except requests.RequestException as e:
                logger.warning(f"[CrossRef] DOI request error: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                return None

        return None

    def get_work(self, doi: str) -> Optional[Dict[str, Any]]:
        """
        获取论文详细信息

        Args:
            doi: DOI

        Returns:
            论文详情
        """
        return self.query_by_doi(doi)

    def _parse_work(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """解析 CrossRef 返回的论文数据"""
        # 提取作者
        authors = []
        for author in item.get("author", []):
            name = " ".join(filter(None, [author.get("given", ""), author.get("family", "")]))
            if name:
                authors.append(name)

        # 提取发表时间
        published = item.get("published-print") or item.get("published-online") or {}
        date_parts = published.get("date-parts", [[]])
        year = date_parts[0][0] if date_parts and date_parts[0] else None

        # 提取期刊/会议信息
        container = item.get("container-title", [])
        journal = item.get("journal", {})

        venue = container[0] if container else (journal.get("title", ""))

        return {
            "doi": item.get("DOI"),
            "title": item.get("title", [""])[0] if item.get("title") else "",
            "authors": authors,
            "year": year,
            "venue": venue,
            "volume": item.get("volume", ""),
            "issue": item.get("issue", ""),
            "pages": item.get("page", ""),
            "abstract": item.get("abstract", ""),
            "url": item.get("URL", "")
        }

    def __repr__(self) -> str:
        return f"CrossRefClient(email={self.email})"