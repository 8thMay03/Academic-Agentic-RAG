import asyncio
import importlib
from typing import Any

from app.models.paper import Paper


class SearchArxivService:
    def __init__(self, client: Any | None = None, arxiv_module: Any | None = None) -> None:
        self._client = client
        self._arxiv_module = arxiv_module

    async def search(
        self,
        query: str,
        max_results: int,
        sort_by: str = "submittedDate",
    ) -> list[Paper]:
        return await asyncio.to_thread(self._search_sync, query, max_results, sort_by)

    def _search_sync(self, query: str, max_results: int, sort_by: str) -> list[Paper]:
        arxiv = self._get_arxiv_module()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=self._resolve_sort_criterion(arxiv, sort_by),
            sort_order=arxiv.SortOrder.Descending,
        )
        client = self._client or arxiv.Client()
        return [self._to_paper(result) for result in client.results(search)]

    def _get_arxiv_module(self) -> Any:
        if self._arxiv_module is not None:
            return self._arxiv_module
        return importlib.import_module("arxiv")

    @staticmethod
    def _resolve_sort_criterion(arxiv: Any, sort_by: str) -> Any:
        sort_criteria = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
        return sort_criteria.get(sort_by, arxiv.SortCriterion.SubmittedDate)

    @staticmethod
    def _to_paper(result: Any) -> Paper:
        paper_id = result.get_short_id() if hasattr(result, "get_short_id") else result.entry_id
        authors = [author.name for author in getattr(result, "authors", [])]
        published = getattr(result, "published", None)
        arxiv_url = getattr(result, "entry_id", None)

        return Paper(
            paper_id=paper_id,
            title=result.title,
            authors=authors,
            published=published.date().isoformat() if published else None,
            abstract=getattr(result, "summary", None),
            arxiv_url=arxiv_url,
            url=arxiv_url,
            pdf_url=getattr(result, "pdf_url", None),
        )
