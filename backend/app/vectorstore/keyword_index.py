from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config.settings import settings
from app.vectorstore.bm25 import BM25Scorer


class PersistentKeywordIndex:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path or Path(settings.CHROMA_DIR) / "keyword_index.json")

    def upsert(self, documents: list[str], metadatas: list[dict], ids: list[str]) -> None:
        if not documents:
            return
        if len(documents) != len(metadatas) or len(documents) != len(ids):
            raise ValueError("documents, metadatas, and ids must have the same length")

        entries = {entry["id"]: entry for entry in self._read_entries()}
        for document, metadata, document_id in zip(documents, metadatas, ids, strict=True):
            entries[str(document_id)] = {
                "id": str(document_id),
                "text": document,
                "metadata": dict(metadata),
            }
        self._write_entries(list(entries.values()))

    def search(
        self,
        query: str,
        top_k: int,
        score_threshold: float | None = None,
        paper_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        entries = self._filtered_entries(paper_ids)
        if not entries:
            return []

        scorer = BM25Scorer([entry["text"] for entry in entries])
        results = []
        for document_index, score in scorer.rank(query, top_k):
            if score_threshold is not None and score < score_threshold:
                continue
            entry = entries[document_index]
            metadata = dict(entry.get("metadata") or {})
            text = str(entry.get("text") or "")
            results.append(
                {
                    "id": entry["id"],
                    "text": text,
                    "metadata": metadata,
                    "score": score,
                    "keyword_score": score,
                    "citation": self._citation(metadata, text),
                }
            )
        return results

    def _filtered_entries(self, paper_ids: list[str] | None) -> list[dict[str, Any]]:
        entries = self._read_entries()
        if not paper_ids:
            return entries
        allowed_paper_ids = set(paper_ids)
        return [
            entry
            for entry in entries
            if (entry.get("metadata") or {}).get("paper_id") in allowed_paper_ids
        ]

    def _read_entries(self) -> list[dict[str, Any]]:
        if not self._path.is_file():
            return []
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return []
        return [entry for entry in payload if isinstance(entry, dict) and entry.get("id")]

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _citation(metadata: dict, text: str) -> dict:
        page_number = metadata.get("page_number") or metadata.get("page")
        try:
            page_number = int(page_number) if page_number not in {None, ""} else None
        except (TypeError, ValueError):
            page_number = None

        return {
            "paper_id": metadata.get("paper_id", ""),
            "title": metadata.get("title", ""),
            "page_number": page_number,
            "page": page_number,
            "chunk_id": metadata.get("chunk_id", ""),
            "text": text,
        }
