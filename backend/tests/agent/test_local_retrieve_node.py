import pytest

from app.agent.models import ToolResult
from app.agent.nodes.local_retrieve_node import local_retrieve_node
from app.agent.workflow import ChatWorkflowRequest


class FakeToolRegistry:
    def __init__(self, result: ToolResult) -> None:
        self.result = result
        self.calls = []

    async def run(self, tool_name: str, input: dict) -> ToolResult:
        self.calls.append({"tool_name": tool_name, "input": input})
        return self.result


@pytest.mark.asyncio
async def test_local_retrieve_node_uses_tool_contract():
    chunk = {
        "id": "paper-1:p1:c0",
        "text": "Agentic RAG plans before retrieving.",
        "citation": {"chunk_id": "paper-1:p1:c0"},
    }
    tool_registry = FakeToolRegistry(ToolResult(tool_name="local_retrieve", success=True, chunks=[chunk]))
    request = ChatWorkflowRequest(
        question="How does Agentic RAG retrieve?",
        paper_ids=["paper-1"],
        top_k=3,
        score_threshold=0.7,
    )

    state = await local_retrieve_node(
        {"tool_registry": tool_registry, "request": request, "trace": []}
    )

    assert tool_registry.calls == [
        {
            "tool_name": "local_retrieve",
            "input": {
                "question": "How does Agentic RAG retrieve?",
                "paper_ids": ["paper-1"],
                "top_k": 3,
                "score_threshold": 0.7,
                "chat_history": None,
            },
        }
    ]
    assert state["local_chunks"] == [chunk]
    assert state["trace"] == [
        {
            "stage": "local_retrieve",
            "chunk_count": 1,
            "paper_ids": ["paper-1"],
            "success": True,
        }
    ]
