from pathlib import Path

from app.agent.models import ToolResult
from app.services.pdf_index_service import PDFIndexService


class PDFIndexTool:
    name = "pdf_index"
    description = "Parse, chunk, embed, and index a downloaded PDF into the local vector store."
    input_schema = {"path": "string|null", "filename": "string|null", "source_metadata": "dict", "force": "boolean"}
    when_to_use = "Use after pdf_download succeeds so local_retrieve can search the new full text."
    failure_modes = ["missing_downloaded_pdf", "pdf_parse_failed", "embedding_failure", "vector_store_write_failure"]

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
        if index_result.embedding_usage:
            artifact.update(
                {
                    "embedding_model": index_result.embedding_usage.model,
                    "embedding_input_count": index_result.embedding_usage.input_count,
                    "embedding_tokens": index_result.embedding_usage.total_tokens,
                    "embedding_estimated_cost_usd": index_result.embedding_usage.estimated_cost_usd,
                }
            )
        return ToolResult(
            tool_name=self.name,
            success=True,
            artifacts=[artifact],
            metadata=artifact,
        )
