def route_after_search(state: dict) -> str:
    return "summarize" if state.get("papers") else "end"

