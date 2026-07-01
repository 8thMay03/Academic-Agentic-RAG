from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from app.services.agentic_chat_workflow import (
        AgenticChatWorkflow,
        ChatWorkflowRequest,
        ContextQuality,
        PreparedAnswer,
    )


class AgenticRAGState(TypedDict, total=False):
    workflow: Any
    request: Any
    local_chunks: list[dict]
    web_chunks: list[dict]
    quality: Any
    citations: list[Any]
    prompt: str | None
    trace: list[dict]


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
    graph.add_edge("web_search", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


async def local_retrieve_node(state: AgenticRAGState) -> AgenticRAGState:
    workflow = state["workflow"]
    request = state["request"]
    local_chunks = await workflow._retrieve_local(request)
    trace = [
        *state.get("trace", []),
        {
            "stage": "local_retrieve",
            "chunk_count": len(local_chunks),
            "paper_ids": request.paper_ids,
        },
    ]
    return {
        **state,
        "local_chunks": local_chunks,
        "trace": trace,
    }


async def quality_gate_node(state: AgenticRAGState) -> AgenticRAGState:
    workflow = state["workflow"]
    request = state["request"]
    local_chunks = state.get("local_chunks", [])
    quality: ContextQuality = await workflow._evaluate_context(request, local_chunks)
    trace = [
        *state.get("trace", []),
        {
            "stage": "quality_gate",
            "sufficient": quality.sufficient,
            "reason": quality.reason,
            "chunk_count": quality.chunk_count,
            "context_chars": quality.context_chars,
            "top_score": quality.top_score,
            "average_score": quality.average_score,
            "source_count": quality.source_count,
            "query_coverage": quality.query_coverage,
            "self_check_used": quality.self_check_used,
            "self_check_passed": quality.self_check_passed,
        },
    ]
    return {
        **state,
        "quality": quality,
        "trace": trace,
    }


def route_after_quality_gate(state: AgenticRAGState) -> str:
    quality = state["quality"]
    return "answer" if quality.sufficient else "web_search"


async def web_search_node(state: AgenticRAGState) -> AgenticRAGState:
    workflow = state["workflow"]
    request = state["request"]
    quality = state["quality"]
    web_chunks = await workflow._search_web(request)
    trace = [
        *state.get("trace", []),
        {
            "stage": "web_search",
            "chunk_count": len(web_chunks),
            "trigger": quality.reason,
        },
    ]
    return {
        **state,
        "web_chunks": web_chunks,
        "trace": trace,
    }


async def answer_node(state: AgenticRAGState) -> AgenticRAGState:
    workflow = state["workflow"]
    request = state["request"]
    quality = state["quality"]
    local_chunks = state.get("local_chunks", [])
    web_chunks = state.get("web_chunks", [])

    chunks = local_chunks if quality.sufficient else [*local_chunks, *web_chunks]
    if not quality.sufficient and not web_chunks:
        chunks = []

    if not chunks:
        trace = [
            *state.get("trace", []),
            {"stage": "answer", "status": "no_context"},
        ]
        return {
            **state,
            "prompt": None,
            "citations": [],
            "trace": trace,
        }

    citations = workflow._citations(chunks, request.question)
    trace = [
        *state.get("trace", []),
        {
            "stage": "answer",
            "status": "ready",
            "context_count": len(chunks),
            "citation_count": len(citations),
        },
    ]
    return {
        **state,
        "prompt": workflow._build_prompt(request.question, chunks, request.chat_history),
        "citations": citations,
        "trace": trace,
    }
