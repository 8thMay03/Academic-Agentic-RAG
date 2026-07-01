from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.agent.state import AgenticRAGState

if TYPE_CHECKING:
    from app.models.paper import Paper

logger = logging.getLogger(__name__)


async def knowledge_ingest_node(state: AgenticRAGState) -> AgenticRAGState:
    """Persist web-sourced knowledge into the local vector store.

    This node runs after ``web_search_node`` and saves two types of artefacts
    so that future queries can benefit from previously fetched web knowledge:

    1. **Web snippets** – the Tavily search result excerpts are converted into
       ``Chunk`` objects and indexed into ChromaDB.
    2. **arXiv PDFs** (best-effort) – any papers identified by their arXiv ID
       are downloaded and fully indexed via the existing PDF pipeline.
    """
    workflow = state["workflow"]
    web_chunks = state.get("web_chunks", [])
    web_papers: list[Paper] = state.get("web_papers", [])

    ingested_snippets = 0
    ingested_papers = 0
    errors: list[str] = []

    # --- 1. Index web snippets into ChromaDB ---
    if web_chunks:
        try:
            ingested_snippets = await workflow._ingest_web_snippets(web_chunks)
        except Exception as exc:
            logger.warning("Failed to ingest web snippets: %s", exc)
            errors.append(f"snippets: {exc}")

    # --- 2. Download & index arXiv PDFs (best-effort) ---
    for paper in web_papers:
        try:
            indexed = await workflow._ingest_arxiv_paper(paper)
            if indexed:
                ingested_papers += 1
        except Exception as exc:
            logger.warning("Failed to ingest paper %s: %s", paper.paper_id, exc)
            errors.append(f"paper {paper.paper_id}: {exc}")

    trace = [
        *state.get("trace", []),
        {
            "stage": "knowledge_ingest",
            "snippets_ingested": ingested_snippets,
            "papers_ingested": ingested_papers,
            "errors": errors,
        },
    ]
    return {
        **state,
        "trace": trace,
    }
