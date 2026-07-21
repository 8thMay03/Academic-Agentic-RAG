import re

from app.agent.models import (
    RetrievedChunk,
    optional_float,
    retrieved_chunk_id,
    retrieved_chunk_page_number,
    retrieved_chunk_text,
    retrieved_chunk_title,
)
from app.models.citation import Citation
from app.vectorstore.bm25 import tokenize


CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
UNKNOWN_ANSWER = "I don't know"


class CitationGrounder:
    def build_citations(self, chunks: list[RetrievedChunk], question: str = "") -> list[Citation]:
        citations: list[Citation] = []
        seen_chunk_ids: set[str] = set()

        for chunk in chunks:
            citation = chunk.get("citation") or {}
            metadata = chunk.get("metadata") or {}
            chunk_id = retrieved_chunk_id(chunk)
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            page_number = retrieved_chunk_page_number(chunk)
            text = citation.get("text") or retrieved_chunk_text(chunk)

            citations.append(
                Citation(
                    paper_id=citation.get("paper_id") or metadata.get("paper_id", ""),
                    title=retrieved_chunk_title(chunk),
                    page_number=page_number,
                    page=page_number,
                    chunk_id=chunk_id,
                    url=citation.get("url") or metadata.get("url") or metadata.get("source_url") or metadata.get("pdf_url"),
                    source_type=citation.get("source_type") or metadata.get("source_type"),
                    source_url=citation.get("source_url") or metadata.get("source_url"),
                    pdf_url=citation.get("pdf_url") or metadata.get("pdf_url"),
                    trust_level=citation.get("trust_level") or metadata.get("trust_level"),
                    ingestion_status=citation.get("ingestion_status") or metadata.get("ingestion_status"),
                    text=text,
                    score=optional_float(chunk.get("score")),
                    rerank_score=optional_float(chunk.get("rerank_score")),
                    cross_encoder_score=optional_float(chunk.get("cross_encoder_score")),
                    vector_score=optional_float(chunk.get("vector_score")),
                    keyword_score=optional_float(chunk.get("keyword_score")),
                    reranker=chunk.get("reranker"),
                    retrieval_sources=list(chunk.get("retrieval_sources") or []),
                    evidence_quality=self._evidence_quality(chunk),
                    matched_terms=self._matched_terms(question, text),
                )
            )

        return citations

    def ground_answer(self, answer: str, citations: list[Citation]) -> str:
        if answer.strip() == UNKNOWN_ANSWER:
            return answer

        valid_chunk_ids = [citation.chunk_id for citation in citations if citation.chunk_id]
        if not valid_chunk_ids:
            return answer

        valid_chunk_id_set = set(valid_chunk_ids)

        def keep_valid_citations(match: re.Match[str]) -> str:
            raw_citation_ids = re.split(r"[,;\s]+", match.group(1).strip())
            supported_citation_ids = [
                citation_id
                for citation_id in raw_citation_ids
                if citation_id in valid_chunk_id_set
            ]
            if not supported_citation_ids:
                return ""
            return f"[{', '.join(supported_citation_ids)}]"

        grounded_answer = CITATION_PATTERN.sub(keep_valid_citations, answer).strip()
        if not self._answer_references_any_chunk(grounded_answer, valid_chunk_id_set):
            grounded_answer = f"{grounded_answer} [{valid_chunk_ids[0]}]"
        grounded_answer = " ".join(grounded_answer.split())
        return re.sub(r"\s+([,.!?;:])", r"\1", grounded_answer)

    def citations_referenced_by_answer(self, citations: list[Citation], answer: str) -> list[Citation]:
        if answer.strip() == UNKNOWN_ANSWER:
            return []
        referenced_chunk_ids = self._referenced_chunk_ids(answer)
        if not referenced_chunk_ids:
            return citations

        citations_by_chunk_id = {
            citation.chunk_id: citation
            for citation in citations
            if citation.chunk_id
        }
        return [
            citations_by_chunk_id[chunk_id]
            for chunk_id in referenced_chunk_ids
            if chunk_id in citations_by_chunk_id
        ]

    def _evidence_quality(self, chunk: RetrievedChunk) -> str:
        if "web" in set(chunk.get("retrieval_sources") or []):
            return "web"
        score = optional_float(chunk.get("rerank_score"))
        if score is None:
            score = optional_float(chunk.get("score"))
        if score is None:
            return "unknown"
        if score >= 0.75:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"

    @staticmethod
    def _matched_terms(question: str, text: str) -> list[str]:
        query_terms = []
        seen_terms = set()
        for term in tokenize(question):
            if term not in seen_terms:
                seen_terms.add(term)
                query_terms.append(term)

        text_terms = set(tokenize(text))
        return [term for term in query_terms if term in text_terms][:8]

    def _answer_references_any_chunk(self, answer: str, valid_chunk_ids: set[str]) -> bool:
        return any(
            chunk_id in valid_chunk_ids
            for chunk_id in self._referenced_chunk_ids(answer)
        )

    @staticmethod
    def _referenced_chunk_ids(answer: str) -> list[str]:
        chunk_ids = []
        seen_chunk_ids = set()
        for match in CITATION_PATTERN.finditer(answer):
            raw_citation_ids = re.split(r"[,;\s]+", match.group(1).strip())
            for citation_id in raw_citation_ids:
                if citation_id and citation_id not in seen_chunk_ids:
                    seen_chunk_ids.add(citation_id)
                    chunk_ids.append(citation_id)
        return chunk_ids
