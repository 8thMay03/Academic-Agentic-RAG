from app.agent.models import PlannerDecision, ResearchPlan, ResearchPlanStep, append_trace
from app.agent.state import AgenticRAGState


async def recovery_planner_node(state: AgenticRAGState) -> AgenticRAGState:
    request = state["request"]
    steps = []
    if request.enable_web_search and _tool_registered(state, "web_search"):
        steps.append(
            ResearchPlanStep(
                tool_name="web_search",
                reason="Verifier requested more evidence for unsupported claims.",
                input={
                    "query": request.question,
                    "max_results": request.top_k,
                },
            )
        )
    decision = PlannerDecision(
        goal=request.question,
        intent=state.get("intent", "research_qa"),
        needs_fresh_context=False,
        can_answer_from_local_context=False,
        selected_tools=[step.tool_name for step in steps],
        steps=steps,
        stop_condition="verification_passes_or_recovery_limit_reached",
        risk_notes=["verifier_requested_more_evidence"],
    )
    plan = ResearchPlan(goal=request.question, steps=decision.steps)
    return {
        **state,
        "plan": plan,
        "planner_decision": decision,
        "current_step_index": 0,
        "trace": append_trace(
            state.get("trace", []),
            "plan",
            status="recovery",
            step_count=len(plan.steps),
            reason="Verifier requested more evidence for unsupported claims.",
            selected_tools=decision.selected_tools,
            stop_condition=decision.stop_condition,
            risk_notes=decision.risk_notes,
        ),
    }


def _tool_registered(state: AgenticRAGState, tool_name: str) -> bool:
    registry = state.get("tool_registry")
    if not registry or not hasattr(registry, "names"):
        return True
    return tool_name in set(registry.names())
