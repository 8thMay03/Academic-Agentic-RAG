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
        chat_id="chat-1",
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
                "chat_id": "chat-1",
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
                "query_count": 1,
                "queries": ["How does Agentic RAG retrieve?"],
                "tool_result": {
                    "tool_name": "local_retrieve",
                    "success": True,
                "chunks": [
                    {
                        "id": "paper-1:p1:c0",
                        "text": "Agentic RAG plans before retrieving.",
                    }
                ],
                "chunk_count": 1,
            },
        }
    ]


@pytest.mark.asyncio
async def test_local_retrieve_node_uses_decomposed_queries():
    first_chunk = {
        "id": "gru:p1:c0",
        "text": "GRU uses reset and update gates.",
        "citation": {"chunk_id": "gru:p1:c0"},
    }
    second_chunk = {
        "id": "lstm:p1:c0",
        "text": "LSTM uses input, forget, and output gates.",
        "citation": {"chunk_id": "lstm:p1:c0"},
    }

    class MultiResultRegistry:
        def __init__(self) -> None:
            self.calls = []
            self.results = [
                ToolResult(tool_name="local_retrieve", success=True, chunks=[first_chunk]),
                ToolResult(tool_name="local_retrieve", success=True, chunks=[second_chunk]),
            ]

        async def run(self, tool_name: str, input: dict) -> ToolResult:
            self.calls.append({"tool_name": tool_name, "input": input})
            return self.results.pop(0)

    tool_registry = MultiResultRegistry()
    request = ChatWorkflowRequest(question="GRU khác gì so với LSTM")

    state = await local_retrieve_node(
        {
            "tool_registry": tool_registry,
            "request": request,
            "query_plan": type(
                "QueryPlan",
                (),
                {"search_queries": ["GRU khác gì so với LSTM", "GRU"]},
            )(),
            "trace": [],
        }
    )

    assert [call["input"]["question"] for call in tool_registry.calls] == [
        "GRU khác gì so với LSTM",
        "GRU",
    ]
    assert state["local_chunks"] == [first_chunk, second_chunk]
    assert state["trace"][0]["query_count"] == 2
