from pathlib import Path

from app.agent.models import ToolResult
from app.config.settings import settings
from app.services.pdf_service import PDFService


class PDFDownloadTool:
    name = "pdf_download"

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
