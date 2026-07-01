import pytest

from app.services.agentic_chat_workflow import AgenticChatWorkflow, ChatWorkflowRequest
from tests.services.test_chat_service import (
    FakeLLMService,
    FakeRAGService,
    FakeWebSearchService,
    SUFFICIENT_LOCAL_CHUNKS,
)


@pytest.mark.asyncio
async def test_agentic_rag_graph_answers_from_sufficient_local_context() -> None:
    rag = FakeRAGService(SUFFICIENT_LOCAL_CHUNKS)
    web = FakeWebSearchService()
    llm = FakeLLMService("It uses planning [paper-1:p3:c0].")
    workflow = AgenticChatWorkflow(rag, llm, web)

    result = await workflow.run(ChatWorkflowRequest("How does planning retrieve evidence?"))

    assert [event["stage"] for event in result.trace] == [
        "local_retrieve",
        "quality_gate",
        "answer",
    ]
    assert result.trace[1]["sufficient"] is True
    assert web.calls == []
    assert result.answer == "It uses planning [paper-1:p3:c0]."


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
    workflow = AgenticChatWorkflow(rag, llm, web)

    result = await workflow.run(ChatWorkflowRequest("How does Agentic RAG differ from CRAG?"))

    assert [event["stage"] for event in result.trace] == [
        "local_retrieve",
        "quality_gate",
        "web_search",
        "answer",
    ]
    assert result.trace[1]["reason"] == "no_local_context"
    assert web.calls == [{"query": "How does Agentic RAG differ from CRAG?", "max_results": 5}]
    assert result.citations[0].chunk_id == "web:1"
