from app.agent.models import AgentLimits, ResearchPlan, ResearchPlanStep, append_trace
from app.agent.state import AgenticRAGState


async def planner_node(state: AgenticRAGState) -> AgenticRAGState:
    request = state["request"]
    quality = state["quality"]
    limits = state.get("limits") or AgentLimits(max_steps=request.max_agent_steps)
    if quality.reason == "latest_query_requires_web":
        steps = _fresh_research_steps(request)
    else:
        steps = _web_fallback_steps(request, quality.reason)
    steps = steps[: limits.max_steps]
    plan = ResearchPlan(goal=request.question, steps=steps)
    trace = append_trace(
        state.get("trace", []),
        "plan",
        status="ready",
        step_count=len(plan.steps),
        reason=quality.reason,
    )
    return {
        **state,
        "limits": limits,
        "plan": plan,
        "current_step_index": 0,
        "tool_calls": [],
        "tool_results": [],
        "trace": trace,
    }


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
                "paper_ids": request.paper_ids,
                "top_k": request.top_k,
                "score_threshold": request.score_threshold,
                "chat_history": request.chat_history,
            },
        ),
    ]
