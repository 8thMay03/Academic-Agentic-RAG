from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config.settings import settings


@dataclass
class WebSearchResult:
    sources: list[dict[str, Any]] = field(default_factory=list)
    skipped_reason: str | None = None


class WebSearchService:
    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else settings.TAVILY_API_KEY
        self._client = client

    async def search(self, query: str, max_results: int = 5) -> WebSearchResult:
        if not self._api_key:
            return WebSearchResult(skipped_reason="TAVILY_API_KEY is not configured.")

        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }

        if self._client is not None:
            response = await self._client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()
        else:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post("https://api.tavily.com/search", json=payload)
                response.raise_for_status()
                data = response.json()

        sources = data.get("results") or []
        return WebSearchResult(
            sources=[
                {
                    "title": source.get("title"),
                    "url": source.get("url"),
                    "content": source.get("content"),
                    "score": source.get("score"),
                }
                for source in sources
                if source.get("url")
            ],
        )
