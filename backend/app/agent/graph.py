from app.models.research import ResearchRequest, ResearchResponse
from langgraph.graph import END, StateGraph

from app.agent.nodes.compare_node import compare_node
from app.agent.nodes.critic_node import critic_node
from app.agent.nodes.download_node import download_node
from app.agent.nodes.embed_node import embed_node
from app.agent.nodes.parse_node import parse_node
from app.agent.nodes.planner_node import planner_node
from app.agent.nodes.report_node import report_node
from app.agent.nodes.search_node import search_node
from app.agent.nodes.select_papers_node import select_papers_node
from app.agent.nodes.summarize_node import summarize_node
from app.agent.state import ResearchState
from app.agent.workflow_router import route_after_critique, route_after_search


async def run_research_workflow(request: ResearchRequest) -> ResearchResponse:
    graph = build_research_graph()
    initial_state: ResearchState = {
        "query": request.query,
        "max_results": request.max_results,
    }
    final_state = await graph.ainvoke(initial_state)
    papers = final_state.get("selected_papers") or final_state.get("papers", [])
    return ResearchResponse(
        query=request.query,
        papers=papers,
        summary=final_state.get("summary"),
        comparison=final_state.get("comparison"),
        report=final_state.get("report"),
    )


def build_research_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("plan", planner_node)
    workflow.add_node("search", search_node)
    workflow.add_node("select_papers", select_papers_node)
    workflow.add_node("download", download_node)
    workflow.add_node("parse", parse_node)
    workflow.add_node("embed", embed_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("compare", compare_node)
    workflow.add_node("report", report_node)
    workflow.add_node("critic", critic_node)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "search")
    workflow.add_conditional_edges(
        "search",
        route_after_search,
        {
            "search": "search",
            "select_papers": "select_papers",
        },
    )
    workflow.add_edge("select_papers", "download")
    workflow.add_edge("download", "parse")
    workflow.add_edge("parse", "embed")
    workflow.add_edge("embed", "summarize")
    workflow.add_edge("summarize", "compare")
    workflow.add_edge("compare", "report")
    workflow.add_edge("report", "critic")
    workflow.add_conditional_edges(
        "critic",
        route_after_critique,
        {
            "search": "search",
            "summarize": "summarize",
            "compare": "compare",
            "report": "report",
            "end": END,
        },
    )

    return workflow.compile()
