from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from app.agent.nodes.answer_node import answer_node
from app.agent.nodes.knowledge_ingest_node import knowledge_ingest_node
from app.agent.nodes.local_retrieve_node import local_retrieve_node
from app.agent.nodes.quality_gate_node import quality_gate_node
from app.agent.nodes.web_search_node import web_search_node
from app.agent.state import AgenticRAGState

if TYPE_CHECKING:
    from app.services.agentic_chat_workflow import (
        AgenticChatWorkflow,
        ChatWorkflowRequest,
        PreparedAnswer,
    )


async def run_agentic_rag_workflow(
    workflow: AgenticChatWorkflow,
    request: ChatWorkflowRequest,
) -> PreparedAnswer:
    graph = build_agentic_rag_graph()
    final_state = await graph.ainvoke(
        {
            "workflow": workflow,
            "request": request,
            "trace": [],
        }
    )
    return workflow._prepared_answer(
        prompt=final_state.get("prompt"),
        citations=final_state.get("citations", []),
        trace=final_state.get("trace", []),
    )


def build_agentic_rag_graph():
    graph = StateGraph(AgenticRAGState)

    graph.add_node("local_retrieve", local_retrieve_node)
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("knowledge_ingest", knowledge_ingest_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("local_retrieve")
    graph.add_edge("local_retrieve", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {
            "web_search": "web_search",
            "answer": "answer",
        },
    )
    graph.add_edge("web_search", "knowledge_ingest")
    graph.add_edge("knowledge_ingest", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


def route_after_quality_gate(state: AgenticRAGState) -> str:
    quality = state["quality"]
    return "answer" if quality.sufficient else "web_search"
