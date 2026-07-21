import re

from app.agent.citations import CITATION_PATTERN, CitationGrounder, normalize_answer_markdown
from app.agent.models import VerificationResult
from app.models.citation import Citation


UNKNOWN_ANSWER = "I don't know"


class AnswerVerifier:
    def __init__(self, citation_grounder: CitationGrounder | None = None) -> None:
        self._citation_grounder = citation_grounder or CitationGrounder()

    def verify(self, answer: str, citations: list[Citation]) -> VerificationResult:
        normalized_answer = normalize_answer_markdown(answer)
        if not normalized_answer:
            return VerificationResult(
                passed=True,
                answer=UNKNOWN_ANSWER,
                citations=[],
                issues=[],
                unsupported_claims=[],
                suggested_action="finalize",
            )

        if not citations:
            return VerificationResult(
                passed=False,
                answer=UNKNOWN_ANSWER,
                citations=[],
                issues=["answer_has_no_evidence"],
                unsupported_claims=[normalized_answer],
                suggested_action="answer_unknown",
            )

        if normalized_answer == UNKNOWN_ANSWER:
            return VerificationResult(
                passed=False,
                answer=UNKNOWN_ANSWER,
                citations=[],
                issues=["answer_unknown_despite_available_evidence"],
                unsupported_claims=[normalized_answer],
                suggested_action="retrieve_more",
            )

        grounded_answer = self._citation_grounder.ground_answer(normalized_answer, citations)
        referenced_citations = self._citation_grounder.citations_referenced_by_answer(
            citations,
            grounded_answer,
        )
        issues = []
        unsupported_claims = []
        if not self._answer_has_explicit_citation(normalized_answer):
            issues.append("answer_missing_explicit_citations")
        if grounded_answer != normalized_answer:
            issues.append("answer_citations_were_grounded")
        if not referenced_citations:
            issues.append("answer_references_no_valid_citations")
            unsupported_claims.append(normalized_answer)
            return VerificationResult(
                passed=False,
                answer=UNKNOWN_ANSWER,
                citations=[],
                issues=issues,
                unsupported_claims=unsupported_claims,
                suggested_action="retrieve_more",
            )

        if self._answer_has_explicit_citation(normalized_answer):
            filtered_answer, uncited_claims = self._remove_uncited_claims(
                grounded_answer,
                {citation.chunk_id for citation in referenced_citations if citation.chunk_id},
            )
            if uncited_claims and filtered_answer:
                grounded_answer = filtered_answer
                unsupported_claims.extend(uncited_claims)
                issues.append("answer_contains_uncited_claims")
            elif uncited_claims:
                issues.append("answer_references_no_valid_citations")
                return VerificationResult(
                    passed=False,
                    answer=UNKNOWN_ANSWER,
                    citations=[],
                    issues=issues,
                    unsupported_claims=uncited_claims,
                    suggested_action="retrieve_more",
                )

        return VerificationResult(
            passed=not issues,
            answer=grounded_answer,
            citations=referenced_citations,
            issues=issues,
            unsupported_claims=unsupported_claims,
            suggested_action="finalize" if not issues else "revise_answer",
        )

    @staticmethod
    def _answer_has_explicit_citation(answer: str) -> bool:
        return "[" in answer and "]" in answer

    @staticmethod
    def _remove_uncited_claims(answer: str, valid_chunk_ids: set[str]) -> tuple[str, list[str]]:
        if AnswerVerifier._answer_contains_markdown_blocks(answer):
            return answer, []

        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+(?!\[)", answer) if sentence.strip()]
        if len(sentences) <= 1:
            return answer, []

        supported_sentences = []
        unsupported_claims = []
        pending_uncited_sentences = []
        for sentence in sentences:
            citation_ids = AnswerVerifier._sentence_citation_ids(sentence)
            if citation_ids & valid_chunk_ids:
                supported_sentences.extend(pending_uncited_sentences)
                pending_uncited_sentences = []
                supported_sentences.append(sentence)
            else:
                pending_uncited_sentences.append(sentence)

        unsupported_claims.extend(pending_uncited_sentences)

        if not unsupported_claims:
            return answer, []
        if not supported_sentences:
            return "", unsupported_claims
        return normalize_answer_markdown(" ".join(supported_sentences)), unsupported_claims

    @staticmethod
    def _sentence_citation_ids(sentence: str) -> set[str]:
        return {
            citation_id
            for match in CITATION_PATTERN.finditer(sentence)
            for citation_id in match.group(1).replace(",", " ").replace(";", " ").split()
        }

    @staticmethod
    def _answer_contains_markdown_blocks(answer: str) -> bool:
        return bool(re.search(r"(?m)^\s{0,3}(#{1,6}\s+|[-*]\s+|\|.+\|)", answer))
