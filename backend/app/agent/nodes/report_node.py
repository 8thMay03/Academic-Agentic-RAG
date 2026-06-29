from app.agent.state import ResearchState
from app.services.report_service import ReportService


async def report_node(state: ResearchState) -> ResearchState:
    summaries = [summary["content"] for summary in state.get("summaries", [])]
    comparison = state.get("comparison", "")
    title = f"Research Report: {state['query']}"

    if not summaries and not comparison:
        return {**state, "report": ""}

    report = await ReportService().generate_survey(title, summaries, comparison)
    return {**state, "report": report}
