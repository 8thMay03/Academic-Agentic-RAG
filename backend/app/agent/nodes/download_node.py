from pathlib import Path

from app.agent.state import ResearchState
from app.config.settings import settings
from app.services.pdf_service import PDFService


async def download_node(state: ResearchState) -> ResearchState:
    papers = state.get("selected_papers") or state.get("papers", [])
    destination_dir = Path(settings.DATA_DIR) / "pdfs"
    downloaded_files = state.get("downloaded_files", [])
    errors = state.get("errors", [])

    for paper in papers:
        if not paper.pdf_url:
            errors.append(
                {
                    "stage": "download",
                    "paper_id": paper.paper_id,
                    "error": "Paper has no PDF URL.",
                }
            )
            continue

        try:
            result = await PDFService().download_pdf_result(str(paper.pdf_url), destination_dir)
            downloaded_files.append(
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "path": str(result.path),
                    "cached": result.cached,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "stage": "download",
                    "paper_id": paper.paper_id,
                    "pdf_url": str(paper.pdf_url),
                    "error": str(exc),
                }
            )

    return {**state, "downloaded_files": downloaded_files, "errors": errors}
