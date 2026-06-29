import math
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token and token not in STOP_WORDS
    ]


class BM25Scorer:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._documents = [tokenize(document) for document in documents]
        self._k1 = k1
        self._b = b
        self._document_count = len(self._documents)
        self._document_lengths = [len(document) for document in self._documents]
        self._average_document_length = (
            sum(self._document_lengths) / self._document_count
            if self._document_count
            else 0.0
        )
        self._document_frequencies = self._document_frequencies_for(self._documents)

    def rank(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not self._documents:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored_documents = [
            (index, self._score_document(query_terms, index))
            for index in range(len(self._documents))
        ]
        scored_documents = [
            (index, score)
            for index, score in scored_documents
            if score > 0
        ]
        if not scored_documents:
            return []

        max_score = max(score for _, score in scored_documents)
        normalized_scores = [
            (index, score / max_score)
            for index, score in scored_documents
        ]
        return sorted(normalized_scores, key=lambda item: item[1], reverse=True)[:top_k]

    def _score_document(self, query_terms: list[str], document_index: int) -> float:
        document = self._documents[document_index]
        document_length = self._document_lengths[document_index]
        term_frequencies = Counter(document)
        score = 0.0

        for term in query_terms:
            term_frequency = term_frequencies.get(term, 0)
            if term_frequency == 0:
                continue

            document_frequency = self._document_frequencies.get(term, 0)
            idf = math.log(
                1 + (self._document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            length_normalizer = (
                1 - self._b
                + self._b * document_length / self._average_document_length
                if self._average_document_length
                else 1
            )
            score += idf * (
                term_frequency * (self._k1 + 1)
                / (term_frequency + self._k1 * length_normalizer)
            )

        return score

    @staticmethod
    def _document_frequencies_for(documents: list[list[str]]) -> Counter:
        document_frequencies: Counter = Counter()
        for document in documents:
            document_frequencies.update(set(document))
        return document_frequencies
