from dataclasses import dataclass
import ipaddress
from pathlib import Path
import socket
from urllib.parse import unquote, urlparse, urlunparse

import httpx

from app.config.settings import settings
from app.utils.file import ensure_parent, safe_filename


class PDFDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class PDFDownloadResult:
    path: Path
    cached: bool = False


class PDFService:
    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
        max_download_bytes: int | None = None,
        allowed_domains: list[str] | None = None,
    ) -> None:
        self._client = client
        self._timeout = timeout
        self._max_download_bytes = max_download_bytes or settings.MAX_PDF_DOWNLOAD_BYTES
        self._allowed_domains = (
            [domain.lower() for domain in allowed_domains]
            if allowed_domains is not None
            else settings.pdf_download_allowed_domains
        )

    async def download_pdf(self, pdf_url: str, destination: Path) -> Path:
        return (await self.download_pdf_result(pdf_url, destination)).path

    async def download_pdf_result(self, pdf_url: str, destination: Path) -> PDFDownloadResult:
        normalized_pdf_url = self._normalize_pdf_url(pdf_url)
        self._validate_download_url(normalized_pdf_url)
        target_path = self._resolve_destination(normalized_pdf_url, destination)
        ensure_parent(target_path)
        if self._is_cached_pdf(target_path):
            return PDFDownloadResult(path=target_path, cached=True)

        temp_path = target_path.with_name(f"{target_path.name}.tmp")

        try:
            if self._client is not None:
                await self._download_with_client(self._client, normalized_pdf_url, temp_path)
            else:
                async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                    await self._download_with_client(client, normalized_pdf_url, temp_path)

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
                raise PDFDownloadError(
                    f"Failed to download PDF: {pdf_url} returned HTTP {response.status_code}"
                ) from exc
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > self._max_download_bytes:
                raise PDFDownloadError(
                    f"Downloaded PDF exceeds size limit: {pdf_url} is {content_length} bytes."
                )

            first_chunk = b""
            downloaded_bytes = 0
            with temp_path.open("wb") as file:
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > self._max_download_bytes:
                        raise PDFDownloadError(
                            f"Downloaded PDF exceeds size limit: {pdf_url} is larger than "
                            f"{self._max_download_bytes} bytes."
                        )
                    if not first_chunk:
                        first_chunk = chunk[:16]
                    file.write(chunk)

            if not first_chunk:
                raise PDFDownloadError(f"Downloaded PDF is empty: {pdf_url}")

            if not self._looks_like_pdf(response.headers.get("content-type"), first_chunk):
                content_type = response.headers.get("content-type", "unknown")
                raise PDFDownloadError(
                    f"Downloaded content is not a PDF: {pdf_url} returned {content_type}"
                )

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
    def _normalize_pdf_url(pdf_url: str) -> str:
        parsed_url = urlparse(pdf_url)
        path_parts = [part for part in parsed_url.path.split("/") if part]

        if parsed_url.netloc.endswith("arxiv.org") and len(path_parts) >= 2:
            paper_id = path_parts[1].removesuffix(".pdf")
            if path_parts[0] in {"abs", "pdf"}:
                normalized_path = f"/pdf/{paper_id}"
                return urlunparse(parsed_url._replace(path=normalized_path, query="", fragment=""))

        return pdf_url

    def _validate_download_url(self, pdf_url: str) -> None:
        parsed_url = urlparse(pdf_url)
        if parsed_url.scheme not in {"http", "https"}:
            raise PDFDownloadError(f"Unsupported PDF URL scheme: {parsed_url.scheme or 'missing'}")
        if not parsed_url.hostname:
            raise PDFDownloadError("PDF URL must include a hostname.")
        hostname = parsed_url.hostname.lower()
        if self._allowed_domains and not self._host_allowed(hostname):
            allowed = ", ".join(self._allowed_domains)
            raise PDFDownloadError(
                f"Blocked PDF URL host outside allowed domains: {hostname}. "
                f"Allowed domains: {allowed}"
            )

        try:
            addresses = socket.getaddrinfo(hostname, None)
        except socket.gaierror as exc:
            raise PDFDownloadError(f"Could not resolve PDF URL host: {hostname}") from exc

        for address in addresses:
            ip_address = ipaddress.ip_address(address[4][0])
            if (
                ip_address.is_private
                or ip_address.is_loopback
                or ip_address.is_link_local
                or ip_address.is_multicast
                or ip_address.is_reserved
                or ip_address.is_unspecified
            ):
                raise PDFDownloadError(f"Blocked unsafe PDF URL host: {hostname}")

    def _host_allowed(self, hostname: str) -> bool:
        return any(
            hostname == domain.lstrip(".")
            or (domain.startswith(".") and hostname.endswith(domain))
            or hostname.endswith(f".{domain}")
            for domain in self._allowed_domains
        )

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
