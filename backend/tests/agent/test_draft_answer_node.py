import pytest

from app.agent.models import ContextQuality
from app.agent.nodes.draft_answer_node import draft_answer_node
from app.agent.workflow import ChatWorkflowRequest


class FakeCitationGrounder:
    def __init__(self) -> None:
        self.calls = []

    def build_citations(self, chunks: list[dict], question: str) -> list[dict]:
        self.calls.append({"chunks": chunks, "question": question})
        return [{"chunk_id": chunk["id"]} for chunk in chunks]


class FakePromptBuilder:
    def __init__(self) -> None:
        self.calls = []

    def build(self, question: str, chunks: list[dict], chat_history=None) -> str:
        self.calls.append(
            {"question": question, "chunks": chunks, "chat_history": chat_history}
        )
        return f"Prompt for {question} with {len(chunks)} chunks."


@pytest.mark.asyncio
async def test_draft_answer_node_uses_prompt_and_citation_dependencies():
    local_chunk = {"id": "local:1", "text": "Local evidence."}
    web_chunk = {"id": "web:1", "text": "Recovered evidence."}
    citation_grounder = FakeCitationGrounder()
    prompt_builder = FakePromptBuilder()
    request = ChatWorkflowRequest("How does Agentic RAG verify?")

    state = await draft_answer_node(
        {
            "request": request,
            "quality": ContextQuality(
                sufficient=True,
                chunk_count=1,
                context_chars=15,
                reason="strong_context",
            ),
            "local_chunks": [local_chunk],
            "web_chunks": [web_chunk],
            "citation_grounder": citation_grounder,
            "prompt_builder": prompt_builder,
            "trace": [],
        }
    )

    expected_chunks = [local_chunk, web_chunk]
    assert citation_grounder.calls == [
        {"chunks": expected_chunks, "question": "How does Agentic RAG verify?"}
    ]
    assert prompt_builder.calls == [
        {
            "question": "How does Agentic RAG verify?",
            "chunks": expected_chunks,
            "chat_history": None,
        }
    ]
    assert state["prompt"] == "Prompt for How does Agentic RAG verify? with 2 chunks."
    assert state["citations"] == [{"chunk_id": "local:1"}, {"chunk_id": "web:1"}]
    assert state["trace"][-1] == {
        "stage": "draft_answer",
        "status": "ready",
        "context_count": 2,
        "citation_count": 2,
    }


@pytest.mark.asyncio
async def test_draft_answer_node_skips_prompt_when_context_is_missing():
    state = await draft_answer_node(
        {
            "request": ChatWorkflowRequest("What is unsupported?"),
            "quality": ContextQuality(
                sufficient=False,
                chunk_count=0,
                context_chars=0,
                reason="no_local_context",
            ),
            "citation_grounder": FakeCitationGrounder(),
            "prompt_builder": FakePromptBuilder(),
            "trace": [],
        }
    )

    assert state["prompt"] is None
    assert state["citations"] == []
    assert state["trace"] == [{"stage": "draft_answer", "status": "no_context"}]
