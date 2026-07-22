from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from app.agent.models import ChatWorkflowResult, StopReason
from app.agent.nodes.classify_intent_node import classify_intent_node
from app.agent.nodes.draft_answer_node import draft_answer_node
from app.agent.nodes.generate_answer_node import generate_answer_node
from app.agent.nodes.local_retrieve_node import local_retrieve_node
from app.agent.nodes.observer_node import observer_node
from app.agent.nodes.planner_node import planner_node
from app.agent.nodes.quality_gate_node import quality_gate_node
from app.agent.nodes.query_planning_node import (
    query_decomposition_node,
    query_planning_node,
    retrieval_planning_node,
)
from app.agent.nodes.recovery_planner_node import recovery_planner_node
from app.agent.nodes.tool_executor_node import tool_executor_node
from app.agent.nodes.verify_answer_node import verify_answer_node
from app.agent.state import AgenticRAGState

if TYPE_CHECKING:
    from app.agent.workflow import (
        AgenticChatWorkflow,
        ChatWorkflowRequest,
    )

UNKNOWN_ANSWER = "I don't know"


async def invoke_agentic_rag_graph(
    workflow: AgenticChatWorkflow,
    request: ChatWorkflowRequest,
) -> AgenticRAGState:
    graph = build_agentic_rag_graph()
    return await graph.ainvoke(workflow.initial_state(request))


async def run_verified_agentic_rag_workflow(
    workflow: AgenticChatWorkflow,
    request: ChatWorkflowRequest,
) -> ChatWorkflowResult:
    final_state = await invoke_agentic_rag_graph(workflow, request)
    answer = final_state.get("answer") or UNKNOWN_ANSWER
    citations = final_state.get("citations", []) if final_state.get("answer") else []
    stop_reason = infer_stop_reason(final_state, answer)
    return ChatWorkflowResult(
        answer=answer,
        citations=citations,
        trace=final_state.get("trace", []),
        stop_reason=stop_reason,
    )


async def stream_verified_agentic_rag_workflow(
    workflow: AgenticChatWorkflow,
    request: ChatWorkflowRequest,
):
    graph = build_agentic_rag_graph()
    seen_trace_count = 0
    final_state: AgenticRAGState = {}
    async for state in graph.astream(workflow.initial_state(request), stream_mode="values"):
        final_state = state
        trace = state.get("trace", [])
        for event in trace[seen_trace_count:]:
            yield {"type": "agent_step", "step": event}
        seen_trace_count = len(trace)

    answer = final_state.get("answer") or UNKNOWN_ANSWER
    citations = final_state.get("citations", []) if final_state.get("answer") else []
    stop_reason = infer_stop_reason(final_state, answer)
    yield {
        "type": "result",
        "result": ChatWorkflowResult(
            answer=answer,
            citations=citations,
            trace=final_state.get("trace", []),
            stop_reason=stop_reason,
        ),
    }


def build_agentic_rag_graph():
    graph = StateGraph(AgenticRAGState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("query_planning", query_planning_node)
    graph.add_node("query_decomposition", query_decomposition_node)
    graph.add_node("retrieval_planning", retrieval_planning_node)
    graph.add_node("local_retrieve", local_retrieve_node)
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("plan", planner_node)
    graph.add_node("plan_recovery", recovery_planner_node)
    graph.add_node("execute_tool", tool_executor_node)
    graph.add_node("observe", observer_node)
    graph.add_node("draft_answer", draft_answer_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("verify_answer", verify_answer_node)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "query_planning")
    graph.add_edge("query_planning", "query_decomposition")
    graph.add_edge("query_decomposition", "retrieval_planning")
    graph.add_edge("retrieval_planning", "local_retrieve")
    graph.add_edge("local_retrieve", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {
            "plan": "plan",
            "draft_answer": "draft_answer",
        },
    )
    graph.add_conditional_edges(
        "plan",
        route_after_planning,
        {
            "execute_tool": "execute_tool",
            "draft_answer": "draft_answer",
        },
    )
    graph.add_edge("execute_tool", "observe")
    graph.add_conditional_edges(
        "observe",
        route_after_observation,
        {
            "execute_tool": "execute_tool",
            "quality_gate": "quality_gate",
            "draft_answer": "draft_answer",
        },
    )
    graph.add_conditional_edges(
        "draft_answer",
        route_after_draft_answer,
        {
            "generate_answer": "generate_answer",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "generate_answer",
        route_after_generate_answer,
        {
            "verify_answer": "verify_answer",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "verify_answer",
        route_after_verify_answer,
        {
            "plan_recovery": "plan_recovery",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "plan_recovery",
        route_after_planning,
        {
            "execute_tool": "execute_tool",
            "draft_answer": "draft_answer",
        },
    )

    return graph.compile()


def route_after_quality_gate(state: AgenticRAGState) -> str:
    quality = state["quality"]
    return "draft_answer" if quality.sufficient else "plan"


def route_after_planning(state: AgenticRAGState) -> str:
    plan = state["plan"]
    return "execute_tool" if plan.steps else "draft_answer"


def route_after_observation(state: AgenticRAGState) -> str:
    plan = state["plan"]
    current_step_index = state.get("current_step_index", 0)
    if _observation_should_regrade_context(state):
        return "quality_gate"
    return "execute_tool" if current_step_index < len(plan.steps) else "draft_answer"


def route_after_draft_answer(state: AgenticRAGState) -> str:
    return "generate_answer" if state.get("prompt") else "end"


def route_after_generate_answer(state: AgenticRAGState) -> str:
    return "verify_answer" if state.get("answer") else "end"


def route_after_verify_answer(state: AgenticRAGState) -> str:
    verification = state.get("verification")
    if not verification or verification.suggested_action != "retrieve_more":
        return "end"
    return "plan_recovery" if _can_retrieve_more(state) else "end"


def infer_stop_reason(state: AgenticRAGState, answer: str | None = None) -> StopReason:
    trace = state.get("trace", [])
    verification = state.get("verification")
    if verification and _answered_unknown(answer) and verification.suggested_action in {"answer_unknown", "retrieve_more", "revise_answer"}:
        return "verification_failed_answer_unknown"

    limit_reason = _latest_limit_reason(trace)
    if limit_reason:
        return limit_reason

    if _answered_unknown(answer):
        request = state["request"]
        quality = state.get("quality")
        plan = state.get("plan")
        if quality and not quality.sufficient and not request.enable_web_search:
            return "web_search_disabled"
        if plan is not None and not plan.steps:
            return "planner_no_valid_steps"
        return "no_context_available"

    if any(event["stage"] == "execute_tool" for event in trace):
        return "answered_after_recovery"
    return "answered_with_sufficient_context"


def _can_retrieve_more(state: AgenticRAGState) -> bool:
    request = state["request"]
    if not request.enable_web_search:
        return False
    limits = state.get("limits")
    max_steps = limits.max_steps if limits else request.max_agent_steps
    max_web_searches = limits.max_web_searches if limits else 2
    trace = state.get("trace", [])
    executed_tool_count = sum(1 for event in trace if event["stage"] == "execute_tool")
    if executed_tool_count >= max_steps:
        return False
    web_search_count = sum(
        1
        for event in trace
        if event["stage"] == "execute_tool" and event.get("tool_name") == "web_search"
    )
    return web_search_count < max_web_searches


def _latest_limit_reason(trace: list[dict]) -> StopReason | None:
    for event in reversed(trace):
        if event.get("stage") != "execute_tool":
            continue
        reason = str(event.get("reason") or "")
        if reason.startswith("Agent step limit reached"):
            return "step_limit_reached"
        if reason.startswith("Tool limit reached"):
            return "tool_limit_reached"
    return None


def _answered_unknown(answer: str | None) -> bool:
    return not answer or answer.strip() == UNKNOWN_ANSWER


def _observation_has_answerable_local_evidence(state: AgenticRAGState) -> bool:
    result = state.get("current_tool_result")
    quality = state.get("quality")
    if quality and not quality.sufficient:
        return False
    return bool(
        result
        and result.success
        and result.tool_name == "local_retrieve"
        and result.chunks
    )


def _observation_should_regrade_context(state: AgenticRAGState) -> bool:
    result = state.get("current_tool_result")
    if not result or not result.success or not result.chunks:
        return False
    if result.tool_name == "local_retrieve":
        return True
    plan = state["plan"]
    current_step_index = state.get("current_step_index", 0)
    return current_step_index >= len(plan.steps)
