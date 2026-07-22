import json
import re
from typing import Any

from app.agent.models import AgentLimits, PlannerDecision, ResearchPlan, ResearchPlanStep, append_trace
from app.agent.state import AgenticRAGState
from app.config.settings import settings


async def planner_node(state: AgenticRAGState) -> AgenticRAGState:
    request = state["request"]
    quality = state["quality"]
    limits = state.get("limits") or AgentLimits(max_steps=request.max_agent_steps)
    decision, planner_source = await select_planner_decision(state, limits)
    decision = _validate_decision_tools(decision, _registered_tool_names(state))
    steps = decision.steps[: limits.max_steps]
    decision = PlannerDecision(
        goal=decision.goal,
        intent=decision.intent,
        needs_fresh_context=decision.needs_fresh_context,
        can_answer_from_local_context=decision.can_answer_from_local_context,
        selected_tools=[step.tool_name for step in steps],
        steps=steps,
        stop_condition=decision.stop_condition,
        risk_notes=decision.risk_notes,
    )
    plan = ResearchPlan(goal=request.question, steps=steps)
    trace = append_trace(
        state.get("trace", []),
        "plan",
        status="ready",
        step_count=len(plan.steps),
        reason=quality.reason,
        selected_tools=decision.selected_tools,
        stop_condition=decision.stop_condition,
        risk_notes=decision.risk_notes,
        available_tools=_registered_tool_names(state),
        planner_source=planner_source,
    )
    return {
        **state,
        "limits": limits,
        "plan": plan,
        "planner_decision": decision,
        "current_step_index": 0,
        "tool_calls": state.get("tool_calls", []),
        "tool_results": state.get("tool_results", []),
        "trace": trace,
    }


async def select_planner_decision(state: AgenticRAGState, limits: AgentLimits) -> tuple[PlannerDecision, str]:
    fallback_decision = build_planner_decision(state, limits)
    if not _should_use_llm_planner(state):
        return fallback_decision, "heuristic"

    try:
        return await build_llm_planner_decision(state, limits), "llm"
    except Exception as exc:
        risk_notes = [*fallback_decision.risk_notes, f"llm_planner_fallback:{type(exc).__name__}"]
        return (
            PlannerDecision(
                goal=fallback_decision.goal,
                intent=fallback_decision.intent,
                needs_fresh_context=fallback_decision.needs_fresh_context,
                can_answer_from_local_context=fallback_decision.can_answer_from_local_context,
                selected_tools=fallback_decision.selected_tools,
                steps=fallback_decision.steps,
                stop_condition=fallback_decision.stop_condition,
                risk_notes=risk_notes,
            ),
            "heuristic_fallback",
        )


def build_planner_decision(state: AgenticRAGState, limits: AgentLimits) -> PlannerDecision:
    request = state["request"]
    quality = state["quality"]
    intent = state.get("intent", "research_qa")
    if quality.sufficient:
        steps: list[ResearchPlanStep] = []
        stop_condition = "local_context_is_sufficient"
        risk_notes = []
    elif quality.reason == "latest_query_requires_web":
        steps = _fresh_research_steps(request)
        stop_condition = "fresh_context_retrieved_or_limits_reached"
        risk_notes = ["freshness_request_requires_external_research"]
    else:
        steps = _local_retry_steps(request, quality.reason) if quality.chunk_count == 0 else []
        if request.enable_web_search:
            steps.extend(_web_fallback_steps(request, quality.reason))
        stop_condition = "external_context_retrieved_or_no_valid_tools"
        risk_notes = [f"local_context_insufficient:{quality.reason}"]

    steps = steps[: limits.max_steps]
    return PlannerDecision(
        goal=request.question,
        intent=intent,
        needs_fresh_context=quality.reason == "latest_query_requires_web",
        can_answer_from_local_context=quality.sufficient,
        selected_tools=[step.tool_name for step in steps],
        steps=steps,
        stop_condition=stop_condition,
        risk_notes=risk_notes,
    )


async def build_llm_planner_decision(state: AgenticRAGState, limits: AgentLimits) -> PlannerDecision:
    llm_service = state["llm_service"]
    raw_response = await llm_service.complete(_planner_prompt(state, limits))
    payload = _parse_planner_json(raw_response)
    steps = [
        ResearchPlanStep(
            tool_name=str(step["tool_name"]),
            reason=str(step.get("reason") or "Planner selected this tool."),
            input=dict(step.get("input") or {}),
        )
        for step in payload.get("steps", [])
        if isinstance(step, dict) and step.get("tool_name")
    ][: limits.max_steps]
    return PlannerDecision(
        goal=str(payload.get("goal") or state["request"].question),
        intent=str(payload.get("intent") or state.get("intent", "research_qa")),
        needs_fresh_context=bool(payload.get("needs_fresh_context", False)),
        can_answer_from_local_context=bool(payload.get("can_answer_from_local_context", False)),
        selected_tools=[step.tool_name for step in steps],
        steps=steps,
        stop_condition=str(payload.get("stop_condition") or "planner_finished_or_limits_reached"),
        risk_notes=[str(note) for note in payload.get("risk_notes", []) if str(note).strip()],
    )


def _should_use_llm_planner(state: AgenticRAGState) -> bool:
    if state["quality"].sufficient:
        return False
    if not settings.ENABLE_LLM_PLANNER:
        return False
    return bool(state.get("llm_service")) and bool(_registered_tool_names(state))


def _planner_prompt(state: AgenticRAGState, limits: AgentLimits) -> str:
    request = state["request"]
    quality = state["quality"]
    return "\n".join(
        [
            "You are the planner for an Agentic RAG workflow.",
            "Choose the smallest safe sequence of tools needed before answer generation.",
            "Return only valid JSON with this schema:",
            (
                '{"goal":"string","intent":"string","needs_fresh_context":false,'
                '"can_answer_from_local_context":false,"steps":[{"tool_name":"string",'
                '"reason":"string","input":{}}],"stop_condition":"string","risk_notes":["string"]}'
            ),
            f"Question: {request.question}",
            f"Intent: {state.get('intent', 'research_qa')}",
            f"Context sufficient: {quality.sufficient}",
            f"Context reason: {quality.reason}",
            f"Max steps: {limits.max_steps}",
            f"Web search enabled: {request.enable_web_search}",
            f"Research ingest enabled: {request.enable_research_ingest}",
            f"Auto PDF download enabled: {request.auto_download_pdfs}",
            f"Available tools: {json.dumps(_available_tool_descriptions(state), ensure_ascii=False)}",
        ]
    )


def _parse_planner_json(raw_response: str) -> dict[str, Any]:
    response = raw_response.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        response = fenced_match.group(1).strip()
    payload = json.loads(response)
    if not isinstance(payload, dict):
        raise ValueError("planner response must be a JSON object")
    return payload


def _web_fallback_steps(request, reason: str) -> list[ResearchPlanStep]:
    if not request.enable_web_search:
        return []
    return [
        ResearchPlanStep(
            tool_name="web_search",
            reason=f"Local context was insufficient: {reason}",
            input={
                "query": request.question,
                "max_results": request.top_k,
            },
        ),
        ResearchPlanStep(
            tool_name="web_snippet_ingest",
            reason="Persist useful web snippets for future local retrieval.",
            input={},
        ),
    ]


def _local_retry_steps(request, reason: str) -> list[ResearchPlanStep]:
    lowered_threshold = None if request.score_threshold is None else max(0.0, request.score_threshold - 0.2)
    return [
        ResearchPlanStep(
            tool_name="local_retrieve",
            reason=f"Retry local retrieval with broader recall before external search: {reason}",
            input={
                "question": request.question,
                "chat_id": request.chat_id,
                "paper_ids": request.paper_ids,
                "top_k": max(request.top_k + 2, request.top_k * 2),
                "score_threshold": lowered_threshold,
                "chat_history": request.chat_history,
            },
        )
    ]


def _registered_tool_names(state: AgenticRAGState) -> list[str]:
    registry = state.get("tool_registry")
    if registry and hasattr(registry, "names"):
        return list(registry.names())
    return []


def _available_tool_descriptions(state: AgenticRAGState) -> list[dict]:
    registry = state.get("tool_registry")
    if registry and hasattr(registry, "descriptions"):
        return list(registry.descriptions())
    return [{"name": name} for name in _registered_tool_names(state)]


def _validate_decision_tools(decision: PlannerDecision, registered_tool_names: list[str]) -> PlannerDecision:
    if not registered_tool_names:
        return decision
    registered = set(registered_tool_names)
    valid_steps = [step for step in decision.steps if step.tool_name in registered]
    invalid_tools = [step.tool_name for step in decision.steps if step.tool_name not in registered]
    risk_notes = list(decision.risk_notes)
    if invalid_tools:
        risk_notes.append(f"planner_removed_unregistered_tools:{','.join(invalid_tools)}")
    return PlannerDecision(
        goal=decision.goal,
        intent=decision.intent,
        needs_fresh_context=decision.needs_fresh_context,
        can_answer_from_local_context=decision.can_answer_from_local_context,
        selected_tools=[step.tool_name for step in valid_steps],
        steps=valid_steps,
        stop_condition=decision.stop_condition,
        risk_notes=risk_notes,
    )


def _fresh_research_steps(request) -> list[ResearchPlanStep]:
    if not request.enable_research_ingest or not request.auto_download_pdfs:
        return _web_fallback_steps(request, "fresh research ingest is disabled")
    return [
        ResearchPlanStep(
            tool_name="arxiv_search",
            reason="The question asks for fresh research, so discover recent arXiv papers.",
            input={
                "query": request.question,
                "max_results": min(request.top_k, 3),
                "sort_by": "submittedDate",
            },
        ),
        ResearchPlanStep(
            tool_name="pdf_download",
            reason="Download the most relevant discovered PDF for full-text indexing.",
            input={},
        ),
        ResearchPlanStep(
            tool_name="pdf_index",
            reason="Index the downloaded paper into the local vector store.",
            input={},
        ),
        ResearchPlanStep(
            tool_name="local_retrieve",
            reason="Retrieve again after ingesting fresh research.",
            input={
                "question": request.question,
                "chat_id": request.chat_id,
                "paper_ids": request.paper_ids,
                "top_k": request.top_k,
                "score_threshold": request.score_threshold,
                "chat_history": request.chat_history,
            },
        ),
    ]
