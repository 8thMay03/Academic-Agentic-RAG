import pytest

from app.services.summary_service import SummaryService


MARKDOWN_SUMMARY = """## Problem
- Understand agentic RAG.

## Method
- Uses retrieval and planning.

## Key Contributions
- Combines agent control with RAG.

## Experiments
- Not specified.

## Limitations
- Not specified.

## Relevance
- Useful for research assistants."""


class FakeLLMService:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return MARKDOWN_SUMMARY


@pytest.mark.asyncio
async def test_summary_service_returns_markdown_summary() -> None:
    llm_service = FakeLLMService()
    service = SummaryService(llm_service)

    summary = await service.summarize("Agentic RAG", "This paper combines retrieval and planning.")

    assert summary == MARKDOWN_SUMMARY
    assert "Create a Markdown summary" in llm_service.prompts[0]
    assert "## Problem" in llm_service.prompts[0]
    assert "Paper title: Agentic RAG" in llm_service.prompts[0]


@pytest.mark.asyncio
async def test_summary_service_summarizes_long_text_in_chunks() -> None:
    llm_service = FakeLLMService()
    service = SummaryService(llm_service)
    long_text = ("Agentic RAG improves retrieval. " * 1200).strip()

    summary = await service.summarize("Agentic RAG", long_text)

    assert summary == MARKDOWN_SUMMARY
    assert len(llm_service.prompts) > 1
    assert "Section chunk: 1/" in llm_service.prompts[0]
    assert "Source partial summaries" in llm_service.prompts[-1]


@pytest.mark.asyncio
async def test_summary_service_rejects_empty_text() -> None:
    service = SummaryService(FakeLLMService())

    with pytest.raises(ValueError, match="text is required"):
        await service.summarize("Agentic RAG", "   ")
