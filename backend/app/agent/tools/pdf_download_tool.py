from pathlib import Path

from app.agent.models import ToolResult
from app.config.settings import settings
from app.services.pdf_service import PDFService


class PDFDownloadTool:
    name = "pdf_download"
    description = "Download a PDF artifact discovered by a previous search step."
    input_schema = {"pdf_url": "string", "source_metadata": "dict", "destination": "string|null"}
    when_to_use = "Use after arxiv_search or web discovery provides a trusted PDF URL."
    failure_modes = ["missing_pdf_url", "download_failed", "non_pdf_content", "tool_timeout"]

    def __init__(self, pdf_service: PDFService, default_destination_dir: str | Path | None = None) -> None:
        self._pdf_service = pdf_service
        self._default_destination_dir = Path(default_destination_dir or settings.DATA_DIR) / "pdfs"

    async def run(self, input: dict) -> ToolResult:
        pdf_url = str(input["pdf_url"])
        source_metadata = dict(input.get("source_metadata") or {})
        destination = Path(input.get("destination") or input.get("destination_dir") or self._default_destination_dir)
        result = await self._pdf_service.download_pdf_result(
            pdf_url=pdf_url,
            destination=destination,
        )
        artifact = {
            "path": result.path.as_posix(),
            "filename": result.path.name,
            "cached": result.cached,
            "pdf_url": pdf_url,
            "source_url": source_metadata.get("source_url") or pdf_url,
            "source_type": source_metadata.get("source_type", "web_pdf"),
            "discovered_by_query": source_metadata.get("discovered_by_query"),
            "trust_level": source_metadata.get("trust_level", "unknown"),
            "ingestion_status": "downloaded",
        }
        return ToolResult(
            tool_name=self.name,
            success=True,
            artifacts=[artifact],
            metadata=artifact,
        )
