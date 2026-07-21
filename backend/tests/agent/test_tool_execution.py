import asyncio

from app.agent.models import AgentLimits, ChatWorkflowRequest, ToolResult
from app.agent.tools.execution import (
    prepare_tool_input,
    run_tool_with_timeout,
    tool_limit_error,
)


class SlowToolRegistry:
    async def run(self, tool_name: str, input: dict) -> ToolResult:
        await asyncio.sleep(0.05)
        return ToolResult(tool_name=tool_name, success=True)


class FailingToolRegistry:
    async def run(self, tool_name: str, input: dict) -> ToolResult:
        raise ValueError("Embedding service is unavailable.")


def test_prepare_tool_input_injects_web_chunks_for_snippet_ingest():
    web_chunks = [{"id": "web:1"}]

    prepared = prepare_tool_input(
        "web_snippet_ingest",
        {"extra": True},
        {"web_chunks": web_chunks},
    )

    assert prepared.error is None
    assert prepared.input == {"extra": True, "web_chunks": web_chunks}


def test_prepare_tool_input_injects_chat_id_for_snippet_ingest():
    prepared = prepare_tool_input(
        "web_snippet_ingest",
        {},
        {
            "request": ChatWorkflowRequest(question="CNN là gì", chat_id="chat-1"),
            "web_chunks": [{"id": "web:1"}],
        },
    )

    assert prepared.error is None
    assert prepared.input == {"web_chunks": [{"id": "web:1"}], "chat_id": "chat-1"}


def test_prepare_tool_input_uses_first_pdf_url_for_download():
    prepared = prepare_tool_input(
        "pdf_download",
        {},
        {
            "tool_results": [
                ToolResult(
                    tool_name="arxiv_search",
                    success=True,
                    metadata={"pdf_urls": ["https://arxiv.org/pdf/2601.12345"]},
                )
            ]
        },
    )

    assert prepared.error is None
    assert prepared.input == {
        "pdf_url": "https://arxiv.org/pdf/2601.12345",
        "source_metadata": {"pdf_url": "https://arxiv.org/pdf/2601.12345"},
    }


def test_prepare_tool_input_preserves_pdf_source_metadata_for_download():
    prepared = prepare_tool_input(
        "pdf_download",
        {},
        {
            "tool_results": [
                ToolResult(
                    tool_name="arxiv_search",
                    success=True,
                    artifacts=[
                        {
                            "pdf_url": "https://arxiv.org/pdf/2601.12345",
                            "source_type": "arxiv",
                            "source_url": "https://arxiv.org/abs/2601.12345",
                            "discovered_by_query": "agentic rag",
                            "trust_level": "high",
                        }
                    ],
                )
            ]
        },
    )

    assert prepared.error is None
    assert prepared.input == {
        "pdf_url": "https://arxiv.org/pdf/2601.12345",
        "source_metadata": {
            "source_type": "arxiv",
            "source_url": "https://arxiv.org/abs/2601.12345",
            "pdf_url": "https://arxiv.org/pdf/2601.12345",
            "discovered_by_query": "agentic rag",
            "trust_level": "high",
        },
    }


def test_prepare_tool_input_reports_missing_pdf_url():
    prepared = prepare_tool_input("pdf_download", {}, {"tool_results": []})

    assert prepared.input == {}
    assert prepared.error == "No PDF URL was available from prior research results."


def test_prepare_tool_input_uses_latest_download_artifact_for_pdf_index():
    prepared = prepare_tool_input(
        "pdf_index",
        {"force": True},
        {
            "tool_results": [
                ToolResult(
                    tool_name="pdf_download",
                    success=True,
                    artifacts=[{"path": "data/pdfs/old.pdf"}],
                ),
                ToolResult(
                    tool_name="pdf_download",
                    success=True,
                    artifacts=[{"path": "data/pdfs/new.pdf"}],
                ),
            ]
        },
    )

    assert prepared.error is None
    assert prepared.input == {"force": True, "path": "data/pdfs/new.pdf", "source_metadata": {}}


def test_tool_limit_error_counts_initial_local_retrieval():
    error = tool_limit_error(
        {
            "local_chunks": [],
            "tool_results": [
                ToolResult(tool_name="local_retrieve", success=True),
                ToolResult(tool_name="local_retrieve", success=True),
            ],
        },
        "local_retrieve",
        AgentLimits(max_retrieval_rounds=3),
    )

    assert error == "Tool limit reached for local_retrieve: 3/3."


def test_tool_limit_error_stops_when_total_step_limit_is_reached():
    error = tool_limit_error(
        {
            "tool_results": [
                ToolResult(tool_name="web_search", success=True),
                ToolResult(tool_name="web_snippet_ingest", success=True),
            ],
        },
        "arxiv_search",
        AgentLimits(max_steps=2, max_arxiv_searches=2),
    )

    assert error == "Agent step limit reached: 2/2."


async def test_run_tool_with_timeout_returns_structured_failure():
    result = await run_tool_with_timeout(
        SlowToolRegistry(),
        "web_search",
        {"query": "agentic rag"},
        AgentLimits(tool_timeout_seconds=0.001),
    )

    assert result == ToolResult(
        tool_name="web_search",
        success=False,
        error="Tool timed out after 0.001s.",
    )


async def test_run_tool_with_timeout_returns_structured_exception_failure():
    result = await run_tool_with_timeout(
        FailingToolRegistry(),
        "web_snippet_ingest",
        {"web_chunks": []},
        AgentLimits(tool_timeout_seconds=1),
    )

    assert result == ToolResult(
        tool_name="web_snippet_ingest",
        success=False,
        error="Embedding service is unavailable.",
    )
