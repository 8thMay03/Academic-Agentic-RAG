from pathlib import Path

from app.agent.state import ResearchState
from app.services.parser_service import ParserService


async def parse_node(state: ResearchState) -> ResearchState:
    downloaded_files = state.get("downloaded_files", [])
    selected_papers = state.get("selected_papers") or state.get("papers", [])
    parsed_documents = state.get("parsed_documents", [])
    errors = state.get("errors", [])

    parsed_paper_ids = {document["paper_id"] for document in parsed_documents}
    parser_service = ParserService()

    for downloaded_file in downloaded_files:
        paper_id = downloaded_file["paper_id"]
        if paper_id in parsed_paper_ids:
            continue

        try:
            text = await parser_service.parse_pdf(Path(downloaded_file["path"]))
            parsed_documents.append(
                {
                    "paper_id": paper_id,
                    "title": downloaded_file["title"],
                    "text": text,
                    "source": "pdf",
                    "path": downloaded_file["path"],
                }
            )
            parsed_paper_ids.add(paper_id)
        except Exception as exc:
            errors.append(
                {
                    "stage": "parse",
                    "paper_id": paper_id,
                    "path": downloaded_file["path"],
                    "error": str(exc),
                }
            )

    for paper in selected_papers:
        if paper.paper_id in parsed_paper_ids or not paper.abstract:
            continue
        parsed_documents.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "text": paper.abstract,
                "source": "abstract",
                "path": None,
            }
        )
        parsed_paper_ids.add(paper.paper_id)

    return {**state, "parsed_documents": parsed_documents, "errors": errors}
