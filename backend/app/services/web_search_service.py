import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config.settings import settings
from app.models.paper import Paper

ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")


@dataclass
class WebSearchResult:
    papers: list[Paper] = field(default_factory=list)
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

    async def search_papers(self, query: str, max_results: int = 5) -> WebSearchResult:
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
            papers=self._papers_from_sources(sources),
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

    def _papers_from_sources(self, sources: list[dict[str, Any]]) -> list[Paper]:
        papers: list[Paper] = []
        seen_ids: set[str] = set()

        for source in sources:
            url = str(source.get("url") or "")
            arxiv_id = self._extract_arxiv_id(url)
            if not arxiv_id or arxiv_id in seen_ids:
                continue

            seen_ids.add(arxiv_id)
            title = source.get("title") or f"arXiv paper {arxiv_id}"
            papers.append(
                Paper(
                    paper_id=arxiv_id,
                    title=str(title),
                    abstract=source.get("content"),
                    arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                )
            )

        return papers

    @staticmethod
    def _extract_arxiv_id(url: str) -> str | None:
        if "arxiv.org" not in url:
            return None
        match = ARXIV_ID_PATTERN.search(url)
        return match.group(1) if match else None
