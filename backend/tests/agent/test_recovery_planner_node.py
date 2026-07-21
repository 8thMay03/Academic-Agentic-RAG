import pytest

from app.agent.nodes.recovery_planner_node import recovery_planner_node
from app.agent.workflow import ChatWorkflowRequest


@pytest.mark.asyncio
async def test_recovery_planner_node_plans_web_search_from_verifier_feedback():
    prior_tool_results = [object()]
    state = await recovery_planner_node(
        {
            "request": ChatWorkflowRequest(
                question="How does Agentic RAG verify evidence?",
                top_k=3,
            ),
            "trace": [{"stage": "verify_answer", "suggested_action": "retrieve_more"}],
            "tool_results": prior_tool_results,
        }
    )

    assert state["current_step_index"] == 0
    assert state["tool_results"] == prior_tool_results
    assert state["plan"].goal == "How does Agentic RAG verify evidence?"
    assert len(state["plan"].steps) == 1
    step = state["plan"].steps[0]
    assert step.tool_name == "web_search"
    assert step.reason == "Verifier requested more evidence for unsupported claims."
    assert step.input == {
        "query": "How does Agentic RAG verify evidence?",
        "max_results": 3,
    }
    assert state["trace"][-1] == {
        "stage": "plan",
        "status": "recovery",
        "step_count": 1,
        "reason": "Verifier requested more evidence for unsupported claims.",
    }
