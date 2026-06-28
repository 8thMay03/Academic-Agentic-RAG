from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from app.utils.file import ensure_parent, safe_filename


class PDFDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class PDFDownloadResult:
    path: Path
    cached: bool = False


class PDFService:
    def __init__(self, client: httpx.AsyncClient | None = None, timeout: float = 60.0) -> None:
        self._client = client
        self._timeout = timeout

    async def download_pdf(self, pdf_url: str, destination: Path) -> Path:
        return (await self.download_pdf_result(pdf_url, destination)).path

    async def download_pdf_result(self, pdf_url: str, destination: Path) -> PDFDownloadResult:
        target_path = self._resolve_destination(pdf_url, destination)
        ensure_parent(target_path)
        if self._is_cached_pdf(target_path):
            return PDFDownloadResult(path=target_path, cached=True)

        temp_path = target_path.with_name(f"{target_path.name}.tmp")

        try:
            if self._client is not None:
                await self._download_with_client(self._client, pdf_url, temp_path)
            else:
                async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                    await self._download_with_client(client, pdf_url, temp_path)

            temp_path.replace(target_path)
            return PDFDownloadResult(path=target_path, cached=False)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    async def download_pdfs(self, pdf_urls: list[str], destination_dir: Path) -> list[Path]:
        return [result.path for result in await self.download_pdf_results(pdf_urls, destination_dir)]

    async def download_pdf_results(
        self,
        pdf_urls: list[str],
        destination_dir: Path,
    ) -> list[PDFDownloadResult]:
        destination_dir.mkdir(parents=True, exist_ok=True)
        downloaded_paths: list[PDFDownloadResult] = []
        for pdf_url in pdf_urls:
            downloaded_paths.append(await self.download_pdf_result(pdf_url, destination_dir))

        return downloaded_paths

    async def _download_with_client(
        self,
        client: httpx.AsyncClient,
        pdf_url: str,
        temp_path: Path,
    ) -> None:
        async with client.stream("GET", pdf_url) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise PDFDownloadError(f"Failed to download PDF: {pdf_url}") from exc

            first_chunk = b""
            with temp_path.open("wb") as file:
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    if not first_chunk:
                        first_chunk = chunk[:16]
                    file.write(chunk)

            if not first_chunk:
                raise PDFDownloadError(f"Downloaded PDF is empty: {pdf_url}")

            if not self._looks_like_pdf(response.headers.get("content-type"), first_chunk):
                raise PDFDownloadError(f"Downloaded content is not a PDF: {pdf_url}")

    @staticmethod
    def _resolve_destination(pdf_url: str, destination: Path) -> Path:
        if destination.suffix:
            return destination

        parsed_url = urlparse(pdf_url)
        filename = safe_filename(unquote(Path(parsed_url.path).name))
        if not filename:
            filename = "paper"
        if not filename.lower().endswith(".pdf"):
            filename = f"{filename}.pdf"

        return destination / filename

    @staticmethod
    def _looks_like_pdf(content_type: str | None, first_chunk: bytes) -> bool:
        normalized_content_type = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
        return normalized_content_type in {
            "application/pdf",
            "application/octet-stream",
            "binary/octet-stream",
        } or first_chunk.startswith(b"%PDF")

    @staticmethod
    def _is_cached_pdf(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False

        with path.open("rb") as file:
            return file.read(16).startswith(b"%PDF")
