import pytest

from app.agent.models import AgentLimits, ResearchPlan, ResearchPlanStep, ToolResult
from app.agent.tools.local_retrieve_tool import LocalRetrieveTool
from app.agent.nodes.tool_executor_node import tool_executor_node
from app.agent.tools.registry import ToolRegistry
from app.agent.tools.web_search_tool import WebSearchTool
from app.agent.workflow import AgenticChatWorkflow, ChatWorkflowRequest
from tests.services.test_chat_service import (
    FakeLLMService,
    FakeRAGService,
    FakeWebSearchService,
    SUFFICIENT_LOCAL_CHUNKS,
)


FRESH_RETRIEVED_CHUNK = {
    "id": "fresh-paper:p2:c0",
    "text": "Fresh Agentic RAG systems add reflection and verification loops.",
    "metadata": {
        "paper_id": "fresh-paper",
        "title": "Fresh Agentic RAG",
        "page_number": "2",
        "chunk_id": "fresh-paper:p2:c0",
    },
    "score": 0.91,
    "retrieval_sources": ["vector"],
    "citation": {
        "paper_id": "fresh-paper",
        "title": "Fresh Agentic RAG",
        "page_number": 2,
        "chunk_id": "fresh-paper:p2:c0",
        "text": "Fresh Agentic RAG systems add reflection and verification loops.",
    },
}


class StatefulLocalRetrieveTool:
    name = "local_retrieve"

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, input: dict) -> ToolResult:
        self.calls += 1
        chunks = [] if self.calls == 1 else [FRESH_RETRIEVED_CHUNK]
        return ToolResult(tool_name=self.name, success=True, chunks=chunks)


class StaticTool:
    def __init__(self, name: str, result: ToolResult) -> None:
        self.name = name
        self._result = result
        self.calls = []

    async def run(self, input: dict) -> ToolResult:
        self.calls.append(input)
        return self._result


class FailingIfCalledToolRegistry:
    async def run(self, tool_name: str, input: dict) -> ToolResult:
        raise AssertionError(f"{tool_name} should not be called")


@pytest.mark.asyncio
async def test_agentic_rag_graph_answers_from_sufficient_local_context() -> None:
    rag = FakeRAGService(SUFFICIENT_LOCAL_CHUNKS)
    web = FakeWebSearchService()
    llm = FakeLLMService("It uses planning [paper-1:p3:c0].")
    workflow = AgenticChatWorkflow(rag, llm, web)

    result = await workflow.run(ChatWorkflowRequest("How does planning retrieve evidence?"))

    assert [event["stage"] for event in result.trace] == [
        "classify_intent",
        "query_planning",
        "query_decomposition",
        "retrieval_planning",
        "local_retrieve",
        "quality_gate",
        "draft_answer",
        "generate_answer",
        "verify_answer",
    ]
    assert result.trace[0]["intent"] == "research_qa"
    assert result.trace[1]["query_type"] == "simple_lookup"
    assert result.trace[2]["query_count"] == 1
    assert result.trace[5]["sufficient"] is True
    assert result.trace[-1]["status"] == "passed"
    assert web.calls == []
    assert result.answer == "It uses planning [1]."
    assert result.stop_reason == "answered_with_sufficient_context"


@pytest.mark.asyncio
async def test_agentic_rag_graph_routes_to_web_when_local_context_is_insufficient() -> None:
    rag = FakeRAGService([])
    web = FakeWebSearchService(
        [
            {
                "title": "Agentic RAG vs CRAG",
                "url": "https://example.com/agentic-rag-crag",
                "content": "Agentic RAG plans retrieval; CRAG corrects retrieved context.",
            }
        ]
    )
    llm = FakeLLMService("Agentic RAG plans retrieval [web:1].")
    registry = ToolRegistry(
        [
            LocalRetrieveTool(rag),
            WebSearchTool(web),
            StaticTool(
                "web_snippet_ingest",
                ToolResult(
                    tool_name="web_snippet_ingest",
                    success=True,
                    metadata={"snippets_ingested": 1},
                ),
            ),
        ]
    )
    workflow = AgenticChatWorkflow(rag, llm, web, tool_registry=registry)

    result = await workflow.run(ChatWorkflowRequest("How does Agentic RAG differ from CRAG?"))

    assert [event["stage"] for event in result.trace] == [
        "classify_intent",
        "query_planning",
        "query_decomposition",
        "retrieval_planning",
        "local_retrieve",
        "quality_gate",
        "plan",
        "execute_tool",
        "observe",
        "execute_tool",
        "observe",
        "execute_tool",
        "observe",
        "draft_answer",
        "generate_answer",
        "verify_answer",
    ]
    assert result.trace[1]["query_type"] == "comparison"
    assert result.trace[2]["query_count"] > 1
    assert result.trace[5]["reason"] == "no_local_context"
    assert result.trace[6]["step_count"] == 3
    assert result.trace[7]["tool_name"] == "local_retrieve"
    assert result.trace[8]["chunk_count"] == 0
    assert result.trace[9]["tool_name"] == "web_search"
    assert result.trace[10]["chunk_count"] == 1
    assert result.trace[11]["tool_name"] == "web_snippet_ingest"
    assert result.trace[12]["snippets_ingested"] == 1
    assert web.calls == [{"query": "How does Agentic RAG differ from CRAG?", "max_results": 5}]
    assert result.citations[0].chunk_id == "web:1"
    assert result.stop_reason == "answered_after_recovery"


@pytest.mark.asyncio
async def test_agentic_rag_graph_respects_agent_step_limit() -> None:
    rag = FakeRAGService([])
    web = FakeWebSearchService(
        [
            {
                "title": "Agentic RAG",
                "url": "https://example.com/agentic-rag",
                "content": "Agentic RAG plans retrieval before answering.",
            }
        ]
    )
    llm = FakeLLMService("Agentic RAG plans retrieval [web:1].")
    workflow = AgenticChatWorkflow(rag, llm, web)

    result = await workflow.run(
        ChatWorkflowRequest(
            "How does Agentic RAG work?",
            max_agent_steps=1,
        )
    )

    assert [event["stage"] for event in result.trace] == [
        "classify_intent",
        "query_planning",
        "query_decomposition",
        "retrieval_planning",
        "local_retrieve",
        "quality_gate",
        "plan",
        "execute_tool",
        "observe",
        "draft_answer",
    ]
    assert result.trace[6]["step_count"] == 1
    assert result.trace[7]["tool_name"] == "local_retrieve"
    assert "web_search" not in [event.get("tool_name") for event in result.trace]
    assert result.stop_reason == "no_context_available"


@pytest.mark.asyncio
async def test_agentic_rag_graph_stops_when_web_search_is_disabled() -> None:
    rag = FakeRAGService([])
    web = FakeWebSearchService(
        [
            {
                "title": "Agentic RAG",
                "url": "https://example.com/agentic-rag",
                "content": "Agentic RAG plans retrieval before answering.",
            }
        ]
    )
    workflow = AgenticChatWorkflow(rag, FakeLLMService("Should not be used."), web)

    result = await workflow.run(
        ChatWorkflowRequest(
            "How does Agentic RAG work?",
            enable_web_search=False,
        )
    )

    assert [event["stage"] for event in result.trace] == [
        "classify_intent",
        "query_planning",
        "query_decomposition",
        "retrieval_planning",
        "local_retrieve",
        "quality_gate",
        "plan",
        "execute_tool",
        "observe",
        "draft_answer",
    ]
    assert result.trace[6]["step_count"] == 1
    assert result.trace[7]["tool_name"] == "local_retrieve"
    assert result.answer == "I don't know"
    assert web.calls == []
    assert result.stop_reason == "web_search_disabled"


@pytest.mark.asyncio
async def test_agentic_rag_graph_retries_local_retrieval_before_web_search() -> None:
    local_tool = StatefulLocalRetrieveTool()
    web = FakeWebSearchService(
        [
            {
                "title": "Should not be needed",
                "url": "https://example.com/unused",
                "content": "Unused web evidence.",
            }
        ]
    )
    workflow = AgenticChatWorkflow(
        FakeRAGService([]),
        FakeLLMService("Fresh Agentic RAG systems add reflection [fresh-paper:p2:c0]."),
        web,
        tool_registry=ToolRegistry([local_tool, WebSearchTool(web)]),
    )

    result = await workflow.run(ChatWorkflowRequest("How do fresh Agentic RAG systems improve?"))

    assert local_tool.calls == 2
    assert web.calls == []
    assert [event.get("tool_name") for event in result.trace if event["stage"] == "execute_tool"] == ["local_retrieve"]
    assert result.citations[0].chunk_id == "fresh-paper:p2:c0"
    assert result.stop_reason == "answered_after_recovery"


@pytest.mark.asyncio
async def test_agentic_rag_graph_uses_web_fallback_when_auto_pdf_download_is_disabled() -> None:
    rag = FakeRAGService([])
    web = FakeWebSearchService(
        [
            {
                "title": "Latest Agentic RAG",
                "url": "https://example.com/latest-agentic-rag",
                "content": "Fresh Agentic RAG systems add reflection and verification loops.",
            }
        ]
    )
    workflow = AgenticChatWorkflow(rag, FakeLLMService("Fresh systems add reflection [web:1]."), web)

    result = await workflow.run(
        ChatWorkflowRequest(
            "What is the latest Agentic RAG approach?",
            auto_download_pdfs=False,
        )
    )

    assert [event.get("tool_name") for event in result.trace if event["stage"] == "execute_tool"] == [
        "web_search",
        "web_snippet_ingest",
    ]
    assert result.trace[0]["intent"] == "fresh_research"
    assert result.trace[6]["step_count"] == 2
    assert web.calls == [{"query": "What is the latest Agentic RAG approach?", "max_results": 5}]
    assert result.answer == "Fresh systems add reflection [1]."


@pytest.mark.asyncio
async def test_tool_executor_returns_structured_failure_when_web_search_limit_is_reached() -> None:
    state = {
        "tool_registry": FailingIfCalledToolRegistry(),
        "plan": ResearchPlan(
            goal="search",
            steps=[
                ResearchPlanStep(
                    tool_name="web_search",
                    reason="Search externally.",
                    input={"query": "agentic rag"},
                )
            ],
        ),
        "current_step_index": 0,
        "tool_results": [ToolResult(tool_name="web_search", success=True)],
        "limits": AgentLimits(max_web_searches=1),
        "trace": [],
    }

    result_state = await tool_executor_node(state)

    result = result_state["current_tool_result"]
    assert result.success is False
    assert result.tool_name == "web_search"
    assert result.error == "Tool limit reached for web_search: 1/1."
    assert result_state["tool_calls"][0].tool_name == "web_search"
    assert result_state["tool_calls"][0].input == {"query": "agentic rag"}
    assert result_state["tool_calls"][0].reason == "Search externally."
    assert result_state["tool_calls"][0].step_index == 0
    assert result_state["trace"][0]["latency_ms"] >= 0
    assert result_state["trace"][0] == {
        "stage": "execute_tool",
        "tool_name": "web_search",
        "step_index": 0,
        "success": False,
        "reason": "Tool limit reached for web_search: 1/1.",
        "latency_ms": result_state["trace"][0]["latency_ms"],
        "tool_result": {
            "tool_name": "web_search",
            "success": False,
            "error": "Tool limit reached for web_search: 1/1.",
        },
    }


@pytest.mark.asyncio
async def test_tool_executor_counts_initial_local_retrieval_against_retrieval_limit() -> None:
    state = {
        "tool_registry": FailingIfCalledToolRegistry(),
        "local_chunks": [],
        "plan": ResearchPlan(
            goal="retrieve",
            steps=[
                ResearchPlanStep(
                    tool_name="local_retrieve",
                    reason="Retrieve after ingestion.",
                    input={"question": "agentic rag"},
                )
            ],
        ),
        "current_step_index": 0,
        "tool_results": [
            ToolResult(tool_name="local_retrieve", success=True),
            ToolResult(tool_name="local_retrieve", success=True),
        ],
        "limits": AgentLimits(max_retrieval_rounds=3),
        "trace": [],
    }

    result_state = await tool_executor_node(state)

    result = result_state["current_tool_result"]
    assert result.success is False
    assert result.error == "Tool limit reached for local_retrieve: 3/3."


@pytest.mark.asyncio
async def test_tool_executor_records_prepared_tool_calls() -> None:
    tool = StaticTool(
        "web_snippet_ingest",
        ToolResult(tool_name="web_snippet_ingest", success=True, metadata={"snippets_ingested": 1}),
    )
    state = {
        "tool_registry": ToolRegistry([tool]),
        "web_chunks": [{"id": "web:1"}],
        "plan": ResearchPlan(
            goal="ingest",
            steps=[
                ResearchPlanStep(
                    tool_name="web_snippet_ingest",
                    reason="Persist web snippets.",
                    input={},
                )
            ],
        ),
        "current_step_index": 0,
        "tool_calls": [],
        "tool_results": [],
        "trace": [],
    }

    result_state = await tool_executor_node(state)

    assert tool.calls == [{"web_chunks": [{"id": "web:1"}]}]
    assert result_state["tool_calls"][0].tool_name == "web_snippet_ingest"
    assert result_state["tool_calls"][0].input == {"web_chunks": [{"id": "web:1"}]}
    assert result_state["tool_calls"][0].reason == "Persist web snippets."
    assert result_state["trace"][0]["tool_result"] == {
        "tool_name": "web_snippet_ingest",
        "success": True,
        "metadata": {"snippets_ingested": 1},
    }


@pytest.mark.asyncio
async def test_agentic_rag_graph_ingests_fresh_arxiv_pdf_then_retrieves_again() -> None:
    local_tool = StatefulLocalRetrieveTool()
    arxiv_tool = StaticTool(
        "arxiv_search",
        ToolResult(
            tool_name="arxiv_search",
            success=True,
            artifacts=[
                {
                    "paper_id": "2601.12345",
                    "title": "Fresh Agentic RAG",
                    "pdf_url": "https://arxiv.org/pdf/2601.12345",
                    "source_type": "arxiv",
                    "source_url": "https://arxiv.org/abs/2601.12345",
                    "discovered_by_query": "What is the latest Agentic RAG approach?",
                    "trust_level": "high",
                }
            ],
            metadata={
                "paper_count": 1,
                "pdf_urls": ["https://arxiv.org/pdf/2601.12345"],
            },
        ),
    )
    pdf_download_tool = StaticTool(
        "pdf_download",
        ToolResult(
            tool_name="pdf_download",
            success=True,
            artifacts=[
                {
                    "path": "data/pdfs/2601.12345.pdf",
                    "filename": "2601.12345.pdf",
                    "cached": False,
                }
            ],
            metadata={
                "path": "data/pdfs/2601.12345.pdf",
                "filename": "2601.12345.pdf",
                "cached": False,
            },
        ),
    )
    pdf_index_tool = StaticTool(
        "pdf_index",
        ToolResult(
            tool_name="pdf_index",
            success=True,
            artifacts=[
                {
                    "paper_id": "2601.12345",
                    "filename": "2601.12345.pdf",
                    "chunks_indexed": 8,
                    "cached": False,
                }
            ],
            metadata={
                "paper_id": "2601.12345",
                "filename": "2601.12345.pdf",
                "chunks_indexed": 8,
                "cached": False,
            },
        ),
    )
    registry = ToolRegistry([local_tool, arxiv_tool, pdf_download_tool, pdf_index_tool])
    workflow = AgenticChatWorkflow(
        FakeRAGService([]),
        FakeLLMService("Fresh systems add reflection [fresh-paper:p2:c0]."),
        FakeWebSearchService(),
        tool_registry=registry,
    )

    result = await workflow.run(
        ChatWorkflowRequest("What is the latest Agentic RAG approach?")
    )

    assert [event.get("tool_name") for event in result.trace if event["stage"] == "execute_tool"] == [
        "arxiv_search",
        "pdf_download",
        "pdf_index",
        "local_retrieve",
    ]
    assert result.trace[0]["intent"] == "fresh_research"
    assert result.trace[6]["step_count"] == 4
    assert result.trace[8]["paper_count"] == 1
    assert result.trace[10]["artifact_count"] == 1
    assert result.trace[12]["chunks_indexed"] == 8
    assert result.trace[14]["chunk_count"] == 1
    assert pdf_download_tool.calls == [
        {
            "pdf_url": "https://arxiv.org/pdf/2601.12345",
            "source_metadata": {
                "source_type": "arxiv",
                "source_url": "https://arxiv.org/abs/2601.12345",
                "pdf_url": "https://arxiv.org/pdf/2601.12345",
                "discovered_by_query": "What is the latest Agentic RAG approach?",
                "trust_level": "high",
            },
        }
    ]
    assert pdf_index_tool.calls == [
        {
            "path": "data/pdfs/2601.12345.pdf",
            "source_metadata": {},
        }
    ]
    assert local_tool.calls == 2
    assert result.answer == "Fresh systems add reflection [1]."
