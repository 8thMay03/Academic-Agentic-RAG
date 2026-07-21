from app.agent.models import ResearchPlan, ResearchPlanStep, append_trace
from app.agent.state import AgenticRAGState


async def recovery_planner_node(state: AgenticRAGState) -> AgenticRAGState:
    request = state["request"]
    plan = ResearchPlan(
        goal=request.question,
        steps=[
            ResearchPlanStep(
                tool_name="web_search",
                reason="Verifier requested more evidence for unsupported claims.",
                input={
                    "query": request.question,
                    "max_results": request.top_k,
                },
            )
        ],
    )
    return {
        **state,
        "plan": plan,
        "current_step_index": 0,
        "trace": append_trace(
            state.get("trace", []),
            "plan",
            status="recovery",
            step_count=len(plan.steps),
            reason="Verifier requested more evidence for unsupported claims.",
        ),
    }
