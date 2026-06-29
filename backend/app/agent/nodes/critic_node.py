from app.agent.state import ResearchState


async def critic_node(state: ResearchState) -> ResearchState:
    issues: list[str] = []
    next_action = "end"

    papers = state.get("selected_papers") or state.get("papers", [])
    summaries = state.get("summaries", [])
    report = (state.get("report") or "").strip()

    if not papers:
        issues.append("No papers were found for the research query.")
        next_action = "search"
    elif not summaries:
        issues.append("No grounded paper summaries were produced.")
        next_action = "summarize"
    elif not state.get("comparison"):
        issues.append("No comparison was produced across the selected papers.")
        next_action = "compare"
    elif not report:
        issues.append("No final research report was produced.")
        next_action = "report"

    critique = {
        "passed": not issues,
        "issues": issues,
        "next_action": next_action,
    }

    return {
        **state,
        "critique": critique,
        "next_action": next_action,
        "reflection_iterations": state.get("reflection_iterations", 0) + 1,
    }
