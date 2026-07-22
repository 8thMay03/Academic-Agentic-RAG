import pytest

from app.agent.models import QueryPlan, ToolResult
from app.agent.nodes.local_retrieve_node import local_retrieve_node
from app.agent.nodes.query_planning_node import (
    query_decomposition_node,
    query_planning_node,
    retrieval_planning_node,
)
from app.agent.workflow import ChatWorkflowRequest


class RecordingToolRegistry:
    def __init__(self) -> None:
        self.calls = []

    async def run(self, tool_name: str, input: dict) -> ToolResult:
        self.calls.append({"tool_name": tool_name, "input": input})
        suffix = len(self.calls)
        return ToolResult(
            tool_name=tool_name,
            success=True,
            chunks=[
                {
                    "id": f"paper:p{suffix}:c0",
                    "text": f"Evidence {suffix}.",
                    "metadata": {"chunk_id": f"paper:p{suffix}:c0"},
                    "citation": {"chunk_id": f"paper:p{suffix}:c0"},
                },
                {
                    "id": f"paper:p{suffix}:c1",
                    "text": f"Extra evidence {suffix}.",
                    "metadata": {"chunk_id": f"paper:p{suffix}:c1"},
                    "citation": {"chunk_id": f"paper:p{suffix}:c1"},
                },
            ],
        )


@pytest.mark.asyncio
async def test_retrieval_planning_adds_comparative_strategy() -> None:
    state = {
        "request": ChatWorkflowRequest(
            "How does RAG differ from CRAG?",
            top_k=5,
            score_threshold=0.7,
        )
    }

    state = await query_planning_node(state)
    state = await query_decomposition_node(state)
    state = await retrieval_planning_node(state)

    query_plan = state["query_plan"]
    assert query_plan.retrieval_mode == "comparative"
    assert query_plan.per_query_top_k == 4
    assert query_plan.score_threshold == 0.7
    assert query_plan.max_total_chunks >= 5
    assert state["trace"][-1]["retrieval_mode"] == "comparative"
    assert state["trace"][-1]["reason"] == "balanced_per_entity_local_retrieve_with_chunk_budget"


@pytest.mark.asyncio
async def test_local_retrieve_uses_planned_strategy_and_chunk_budget() -> None:
    registry = RecordingToolRegistry()
    query_plan = QueryPlan(
        original_query="Compare RAG and CRAG",
        query_type="comparison",
        search_queries=["Compare RAG and CRAG", "RAG", "CRAG"],
        reason="question_compares_multiple_items",
        retrieval_mode="comparative",
        per_query_top_k=3,
        score_threshold=0.6,
        max_total_chunks=2,
        retrieval_reason="balanced_per_entity_local_retrieve_with_chunk_budget",
    )

    state = await local_retrieve_node(
        {
            "request": ChatWorkflowRequest("Compare RAG and CRAG", top_k=5, score_threshold=0.8),
            "query_plan": query_plan,
            "tool_registry": registry,
        }
    )

    assert len(registry.calls) == 3
    assert {call["input"]["top_k"] for call in registry.calls} == {3}
    assert {call["input"]["score_threshold"] for call in registry.calls} == {0.6}
    assert len(state["local_chunks"]) == 2
    assert state["trace"][-1]["retrieval_mode"] == "comparative"
    assert state["trace"][-1]["max_total_chunks"] == 2
