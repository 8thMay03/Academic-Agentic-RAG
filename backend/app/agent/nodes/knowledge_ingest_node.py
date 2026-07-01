import logging

from app.agent.state import AgenticRAGState

logger = logging.getLogger(__name__)


async def knowledge_ingest_node(state: AgenticRAGState) -> AgenticRAGState:
    """Persist web search snippets into the local vector store for future retrieval."""
    workflow = state["workflow"]
    web_chunks = state.get("web_chunks", [])

    ingested_snippets = 0
    errors: list[str] = []

    if web_chunks:
        try:
            ingested_snippets = await workflow._ingest_web_snippets(web_chunks)
        except Exception as exc:
            logger.warning("Failed to ingest web snippets: %s", exc)
            errors.append(f"snippets: {exc}")

    trace = [
        *state.get("trace", []),
        {
            "stage": "knowledge_ingest",
            "snippets_ingested": ingested_snippets,
            "errors": errors,
        },
    ]
    return {
        **state,
        "trace": trace,
    }
