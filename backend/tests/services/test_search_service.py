from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.services.search_service import SearchService


class FakeSortCriterion:
    Relevance = "relevance"
    LastUpdatedDate = "lastUpdatedDate"
    SubmittedDate = "submittedDate"


class FakeSortOrder:
    Descending = "descending"


class FakeSearch:
    def __init__(self, query: str, max_results: int, sort_by: str, sort_order: str) -> None:
        self.query = query
        self.max_results = max_results
        self.sort_by = sort_by
        self.sort_order = sort_order


class FakeAuthor:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeResult:
    title = "Agentic RAG: A Survey"
    authors = [FakeAuthor("Ada Lovelace"), FakeAuthor("Alan Turing")]
    published = datetime(2026, 6, 20, tzinfo=UTC)
    summary = "A concise abstract about agentic retrieval-augmented generation."
    entry_id = "https://arxiv.org/abs/2606.12345v1"
    pdf_url = "https://arxiv.org/pdf/2606.12345v1"

    def get_short_id(self) -> str:
        return "2606.12345v1"


class FakeClient:
    def __init__(self) -> None:
        self.search = None

    def results(self, search: FakeSearch) -> list[FakeResult]:
        self.search = search
        return [FakeResult()]


@pytest.mark.asyncio
async def test_searches_arxiv_and_maps_paper_fields() -> None:
    fake_arxiv = SimpleNamespace(
        Search=FakeSearch,
        SortCriterion=FakeSortCriterion,
        SortOrder=FakeSortOrder,
    )
    fake_client = FakeClient()
    service = SearchService(client=fake_client, arxiv_module=fake_arxiv)

    papers = await service.search("Agentic RAG", max_results=5, sort_by="submittedDate")

    assert fake_client.search.query == "Agentic RAG"
    assert fake_client.search.max_results == 5
    assert fake_client.search.sort_by == "submittedDate"
    assert fake_client.search.sort_order == "descending"
    assert len(papers) == 1
    assert papers[0].title == "Agentic RAG: A Survey"
    assert papers[0].authors == ["Ada Lovelace", "Alan Turing"]
    assert papers[0].published == "2026-06-20"
    assert papers[0].abstract == "A concise abstract about agentic retrieval-augmented generation."
    assert str(papers[0].arxiv_url) == "https://arxiv.org/abs/2606.12345v1"
    assert str(papers[0].pdf_url) == "https://arxiv.org/pdf/2606.12345v1"
