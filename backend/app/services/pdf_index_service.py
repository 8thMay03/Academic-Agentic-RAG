import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.config.settings import settings
from app.parser.chunker import chunk_text_with_metadata
from app.parser.cleaner import PAGE_BREAK, clean_text
from app.parser.pdf_parser import extract_text_from_pdf
from app.vectorstore.indexing import index_chunks


@dataclass(frozen=True)
class PDFIndexResult:
    paper_id: str
    filename: str
    chunks_indexed: int
    cached: bool = False
    source_metadata: dict = field(default_factory=dict)


class PDFIndexService:
    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir or settings.DATA_DIR)
        self._pdf_dir = self._data_dir / "pdfs"
        self._manifest_path = self._data_dir / "metadata" / "pdf_index_manifest.json"

    async def index_downloaded_pdf(
        self,
        filename: str,
        force: bool = False,
        source_metadata: dict | None = None,
    ) -> PDFIndexResult:
        safe_name = Path(filename).name
        pdf_path = self._pdf_dir / safe_name
        if not safe_name.lower().endswith(".pdf") or not pdf_path.is_file():
            raise FileNotFoundError(f"Downloaded PDF not found: {filename}")

        return await self.index_pdf(pdf_path, force=force, source_metadata=source_metadata)

    async def index_all_downloaded_pdfs(self, force: bool = False) -> list[PDFIndexResult]:
        if not self._pdf_dir.exists():
            return []

        results = []
        for pdf_path in sorted(self._pdf_dir.glob("*.pdf")):
            if pdf_path.is_file():
                results.append(await self.index_pdf(pdf_path, force=force))
        return results

    async def index_pdf(
        self,
        pdf_path: Path,
        force: bool = False,
        source_metadata: dict | None = None,
    ) -> PDFIndexResult:
        paper_id = pdf_path.stem
        source_metadata = dict(source_metadata or {})
        manifest = self._read_manifest()
        signature = self._file_signature(pdf_path)
        manifest_entry = manifest.get(pdf_path.name)

        if not force and manifest_entry and self._is_current(manifest_entry, signature):
            return PDFIndexResult(
                paper_id=paper_id,
                filename=pdf_path.name,
                chunks_indexed=int(manifest_entry.get("chunks_indexed", 0)),
                cached=True,
                source_metadata=manifest_entry.get("source_metadata") or source_metadata,
            )

        raw_text = await asyncio.to_thread(extract_text_from_pdf, pdf_path)
        cleaned_pages = [clean_text(page) for page in raw_text.split(PAGE_BREAK) if page.strip()]
        cleaned_text = PAGE_BREAK.join(page for page in cleaned_pages if page)
        chunks = chunk_text_with_metadata(cleaned_text, paper_id=paper_id)
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "title": pdf_path.name,
                    "source_path": pdf_path.as_posix(),
                    **source_metadata,
                }
            )

        await index_chunks(chunks)
        manifest[pdf_path.name] = {
            **signature,
            "paper_id": paper_id,
            "chunks_indexed": len(chunks),
            "source_metadata": source_metadata,
        }
        self._write_manifest(manifest)
        return PDFIndexResult(
            paper_id=paper_id,
            filename=pdf_path.name,
            chunks_indexed=len(chunks),
            cached=False,
            source_metadata=source_metadata,
        )

    @staticmethod
    def _file_signature(pdf_path: Path) -> dict:
        stat = pdf_path.stat()
        return {
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
        }

    @staticmethod
    def _is_current(manifest_entry: dict, signature: dict) -> bool:
        return (
            manifest_entry.get("size_bytes") == signature["size_bytes"]
            and manifest_entry.get("modified_at") == signature["modified_at"]
        )

    def _read_manifest(self) -> dict:
        if not self._manifest_path.is_file():
            return {}
        return json.loads(self._manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, manifest: dict) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
