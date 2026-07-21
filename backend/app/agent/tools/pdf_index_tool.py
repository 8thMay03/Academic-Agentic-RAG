from pathlib import Path

from app.agent.models import ToolResult
from app.services.pdf_index_service import PDFIndexService


class PDFIndexTool:
    name = "pdf_index"

    def __init__(self, pdf_index_service: PDFIndexService) -> None:
        self._pdf_index_service = pdf_index_service

    async def run(self, input: dict) -> ToolResult:
        force = bool(input.get("force", False))
        source_metadata = dict(input.get("source_metadata") or {})
        if input.get("path"):
            index_result = await self._pdf_index_service.index_pdf(
                Path(input["path"]),
                force=force,
                source_metadata=source_metadata,
            )
        else:
            index_result = await self._pdf_index_service.index_downloaded_pdf(
                filename=str(input["filename"]),
                force=force,
                source_metadata=source_metadata,
            )

        artifact = {
            "paper_id": index_result.paper_id,
            "filename": index_result.filename,
            "chunks_indexed": index_result.chunks_indexed,
            "cached": index_result.cached,
            **index_result.source_metadata,
        }
        return ToolResult(
            tool_name=self.name,
            success=True,
            artifacts=[artifact],
            metadata=artifact,
        )
