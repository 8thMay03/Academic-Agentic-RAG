def route_after_search(state: dict) -> str:
    plan = state.get("plan") or {}
    selection_criteria = plan.get("selection_criteria") or {}
    min_papers = selection_criteria.get("min_papers", 1)

    if len(state.get("papers", [])) >= min_papers:
        return "select_papers"

    if state.get("search_iterations", 0) < state.get("max_search_iterations", 1):
        return "search"

    return "select_papers"


def route_after_critique(state: dict) -> str:
    critique = state.get("critique") or {}
    if critique.get("passed"):
        return "end"

    if state.get("reflection_iterations", 0) >= state.get("max_reflection_iterations", 1):
        return "end"

    next_action = critique.get("next_action")
    if next_action in {"search", "summarize", "compare", "report"}:
        return next_action

    return "end"
