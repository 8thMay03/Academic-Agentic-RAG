import re
import unicodedata
from collections import Counter


PAGE_BREAK = "\f"
_PAGE_NUMBER_PATTERN = re.compile(r"^(?:page\s*)?\d+(?:\s*/\s*\d+)?$", re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"[ \t\r\v]+")


def clean_text(text: str) -> str:
    pages = _split_pages(_normalize_unicode(text))
    pages = [_remove_repeated_header_footer(page, pages) for page in pages]
    normalized_pages = [_normalize_page_text(page) for page in pages]
    return "\n\n".join(page for page in normalized_pages if page)


def _split_pages(text: str) -> list[str]:
    pages = text.split(PAGE_BREAK)
    return pages if len(pages) > 1 else [text]


def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return (
        text.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def _remove_repeated_header_footer(page: str, all_pages: list[str]) -> str:
    page_lines = _stripped_lines(page)
    repeated_lines = _repeated_edge_lines(all_pages)
    kept_lines = [
        line
        for line in page_lines
        if line not in repeated_lines and not _PAGE_NUMBER_PATTERN.fullmatch(line)
    ]
    return "\n".join(kept_lines)


def _repeated_edge_lines(pages: list[str], edge_size: int = 1) -> set[str]:
    if len(pages) < 2:
        return set()

    candidates: list[str] = []
    for page in pages:
        lines = _stripped_lines(page)
        page_candidates = set(lines[:edge_size] + lines[-edge_size:])
        candidates.extend(page_candidates)

    threshold = max(2, len(pages) // 2 + 1)
    return {
        line
        for line, count in Counter(candidates).items()
        if count >= threshold and not _PAGE_NUMBER_PATTERN.fullmatch(line)
    }


def _normalize_page_text(page: str) -> str:
    lines = [_WHITESPACE_PATTERN.sub(" ", line).strip() for line in page.splitlines()]
    lines = [line for line in lines if line]
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _stripped_lines(text: str) -> list[str]:
    return [_WHITESPACE_PATTERN.sub(" ", line).strip() for line in text.splitlines() if line.strip()]
