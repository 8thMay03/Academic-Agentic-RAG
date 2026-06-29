from pathlib import Path

from app.agent.state import ResearchState
from app.services.pdf_index_service import PDFIndexService


async def embed_node(state: ResearchState) -> ResearchState:
    downloaded_files = state.get("downloaded_files", [])
    indexed_paper_ids = state.get("indexed_paper_ids", [])
    errors = state.get("errors", [])
    index_service = PDFIndexService()

    for downloaded_file in downloaded_files:
        paper_id = downloaded_file["paper_id"]
        if paper_id in indexed_paper_ids:
            continue

        try:
            result = await index_service.index_pdf(Path(downloaded_file["path"]))
            indexed_paper_ids.append(result.paper_id)
        except Exception as exc:
            errors.append(
                {
                    "stage": "embed",
                    "paper_id": paper_id,
                    "path": downloaded_file["path"],
                    "error": str(exc),
                }
            )

    return {**state, "indexed_paper_ids": indexed_paper_ids, "errors": errors}
