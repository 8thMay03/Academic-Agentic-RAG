import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from app.agent.citations import CITATION_PATTERN, CitationGrounder, normalize_answer_markdown
from app.agent.models import ClaimVerification, VerificationResult
from app.models.citation import Citation
from app.vectorstore.bm25 import tokenize


UNKNOWN_ANSWER = "I don't know"


class ClaimSupportJudge(Protocol):
    def assess(
        self,
        claim: str,
        evidence_text: str,
        supporting_chunk_ids: list[str],
    ) -> ClaimVerification:
        """Return whether cited evidence supports, contradicts, or is insufficient for a claim."""


class AsyncClaimSupportJudge(Protocol):
    async def assess(
        self,
        claim: str,
        evidence_text: str,
        supporting_chunk_ids: list[str],
    ) -> ClaimVerification:
        """Return async support judgment for a cited claim."""


@dataclass(frozen=True)
class VerificationDraft:
    grounded_answer: str
    referenced_citations: list[Citation]
    issues: list[str]
    unsupported_claims: list[str]


class HeuristicClaimSupportJudge:
    def assess(
        self,
        claim: str,
        evidence_text: str,
        supporting_chunk_ids: list[str],
    ) -> ClaimVerification:
        if self._claim_contradicts_text(claim, evidence_text):
            return ClaimVerification(
                claim=claim,
                status="contradicted",
                supporting_chunk_ids=supporting_chunk_ids,
                reason="claim_negation_conflicts_with_cited_evidence",
            )
        if self._claim_supported_by_text(claim, evidence_text):
            return ClaimVerification(
                claim=claim,
                status="supported",
                supporting_chunk_ids=supporting_chunk_ids,
                reason="claim_terms_overlap_cited_evidence",
            )
        return ClaimVerification(
            claim=claim,
            status="insufficient",
            supporting_chunk_ids=supporting_chunk_ids,
            reason="claim_terms_do_not_overlap_cited_evidence_enough",
        )

    @classmethod
    def _claim_contradicts_text(cls, claim: str, evidence_text: str) -> bool:
        claim_terms = AnswerVerifier._semantic_terms(CITATION_PATTERN.sub("", claim))
        evidence_terms = AnswerVerifier._semantic_terms(evidence_text)
        if not claim_terms or not evidence_terms:
            return False
        shared_terms = claim_terms & evidence_terms
        if len(shared_terms) / len(claim_terms) < 0.5:
            return False
        return cls._has_negation(claim) != cls._has_negation(evidence_text)

    @staticmethod
    def _has_negation(text: str) -> bool:
        terms = set(tokenize(text))
        return bool(terms & {"not", "no", "never", "without", "cannot", "cant", "không", "khong", "chưa", "chua"})

    @staticmethod
    def _claim_supported_by_text(claim: str, evidence_text: str) -> bool:
        claim_terms = AnswerVerifier._semantic_terms(CITATION_PATTERN.sub("", claim))
        evidence_terms = AnswerVerifier._semantic_terms(evidence_text)
        if not claim_terms:
            return True
        if not evidence_terms:
            return False
        overlap = len(claim_terms & evidence_terms) / len(claim_terms)
        return overlap >= 0.6


class LLMClaimSupportJudge:
    def __init__(
        self,
        llm_service: Any,
        fallback_judge: ClaimSupportJudge | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._fallback_judge = fallback_judge or HeuristicClaimSupportJudge()

    async def assess(
        self,
        claim: str,
        evidence_text: str,
        supporting_chunk_ids: list[str],
    ) -> ClaimVerification:
        try:
            raw_response = await self._llm_service.complete(
                self._prompt(claim, evidence_text, supporting_chunk_ids)
            )
            return self._parse_response(raw_response, claim, supporting_chunk_ids)
        except Exception:
            fallback = self._fallback_judge.assess(claim, evidence_text, supporting_chunk_ids)
            return ClaimVerification(
                claim=fallback.claim,
                status=fallback.status,
                supporting_chunk_ids=fallback.supporting_chunk_ids,
                reason=f"llm_claim_judge_fallback:{fallback.reason}",
            )

    @staticmethod
    def _prompt(claim: str, evidence_text: str, supporting_chunk_ids: list[str]) -> str:
        chunk_ids = ", ".join(supporting_chunk_ids) or "none"
        return (
            "You are a strict claim verification judge for a RAG system.\n"
            "Decide whether the cited evidence supports the claim, contradicts it, or is insufficient.\n"
            "Use only the evidence text. Do not use outside knowledge.\n"
            "Return JSON only with this schema:\n"
            "{\"status\":\"supported|contradicted|insufficient\",\"reason\":\"short reason\"}\n\n"
            f"Claim:\n{claim}\n\n"
            f"Supporting chunk ids: {chunk_ids}\n\n"
            f"Evidence:\n{evidence_text[:5000]}"
        )

    @staticmethod
    def _parse_response(
        raw_response: str,
        claim: str,
        supporting_chunk_ids: list[str],
    ) -> ClaimVerification:
        payload = LLMClaimSupportJudge._extract_json(raw_response)
        status = str(payload.get("status", "")).strip().lower()
        if status not in {"supported", "contradicted", "insufficient"}:
            raise ValueError(f"Invalid claim support status: {status}")
        reason = str(payload.get("reason") or f"llm_claim_judge_{status}").strip()
        return ClaimVerification(
            claim=claim,
            status=status,  # type: ignore[arg-type]
            supporting_chunk_ids=supporting_chunk_ids,
            reason=reason[:240],
        )

    @staticmethod
    def _extract_json(raw_response: str) -> dict:
        response = raw_response.strip()
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, flags=re.DOTALL)
            if not match:
                raise
            payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError("Claim judge response must be a JSON object.")
        return payload


class AnswerVerifier:
    def __init__(
        self,
        citation_grounder: CitationGrounder | None = None,
        claim_judge: ClaimSupportJudge | None = None,
        async_claim_judge: AsyncClaimSupportJudge | None = None,
    ) -> None:
        self._citation_grounder = citation_grounder or CitationGrounder()
        self._claim_judge = claim_judge or HeuristicClaimSupportJudge()
        self._async_claim_judge = async_claim_judge

    def verify(self, answer: str, citations: list[Citation]) -> VerificationResult:
        draft = self._prepare_verification(answer, citations)
        if isinstance(draft, VerificationResult):
            return draft

        semantic_answer, semantic_issues, semantic_claims = self._verify_cited_claims(
            draft.grounded_answer,
            draft.referenced_citations,
        )
        return self._finalize_verification(draft, semantic_answer, semantic_issues, semantic_claims)

    async def verify_async(self, answer: str, citations: list[Citation]) -> VerificationResult:
        draft = self._prepare_verification(answer, citations)
        if isinstance(draft, VerificationResult):
            return draft
        if self._async_claim_judge is None:
            return self.verify(answer, citations)

        semantic_answer, semantic_issues, semantic_claims = await self._verify_cited_claims_async(
            draft.grounded_answer,
            draft.referenced_citations,
        )
        return self._finalize_verification(draft, semantic_answer, semantic_issues, semantic_claims)

    def _prepare_verification(
        self,
        answer: str,
        citations: list[Citation],
    ) -> VerificationResult | VerificationDraft:
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

        return VerificationDraft(
            grounded_answer=grounded_answer,
            referenced_citations=referenced_citations,
            issues=issues,
            unsupported_claims=unsupported_claims,
        )

    @staticmethod
    def _finalize_verification(
        draft: VerificationDraft,
        semantic_answer: str,
        semantic_issues: list[str],
        semantic_claims: list[ClaimVerification],
    ) -> VerificationResult:
        issues = list(draft.issues)
        unsupported_claims = list(draft.unsupported_claims)
        grounded_answer = draft.grounded_answer
        if semantic_issues:
            issues.extend(semantic_issues)
            unsupported_claims.extend(
                claim.claim
                for claim in semantic_claims
                if claim.status != "supported"
            )
            grounded_answer = semantic_answer
            if not grounded_answer:
                return VerificationResult(
                    passed=False,
                    answer=UNKNOWN_ANSWER,
                    citations=[],
                    issues=issues,
                    unsupported_claims=unsupported_claims,
                    suggested_action="retrieve_more",
                    claim_verifications=semantic_claims,
                )

        return VerificationResult(
            passed=not issues,
            answer=grounded_answer,
            citations=draft.referenced_citations,
            issues=issues,
            unsupported_claims=unsupported_claims,
            suggested_action="finalize" if not issues else "revise_answer",
            claim_verifications=semantic_claims,
        )

    @staticmethod
    def _answer_has_explicit_citation(answer: str) -> bool:
        return "[" in answer and "]" in answer

    @staticmethod
    def _remove_uncited_claims(answer: str, valid_chunk_ids: set[str]) -> tuple[str, list[str]]:
        if AnswerVerifier._answer_contains_markdown_blocks(answer):
            return answer, []

        sentences = AnswerVerifier._split_answer_sentences(answer)
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

    def _verify_cited_claims(
        self,
        answer: str,
        citations: list[Citation],
    ) -> tuple[str, list[str], list[ClaimVerification]]:
        if self._answer_contains_markdown_blocks(answer):
            return answer, [], []

        citation_by_id = {
            citation.chunk_id: citation
            for citation in citations
            if citation.chunk_id
        }
        sentences = self._split_answer_sentences(answer)
        verified_sentences = []
        claim_verifications = []
        unsupported_claims = []

        for sentence in sentences:
            cited_ids = self._sentence_citation_ids(sentence)
            if not cited_ids:
                verified_sentences.append(sentence)
                continue

            cited_text = " ".join(
                citation_by_id[chunk_id].text or ""
                for chunk_id in cited_ids
                if chunk_id in citation_by_id
            )
            supporting_chunk_ids = sorted(cited_ids & set(citation_by_id))
            claim_verification = self._claim_judge.assess(
                sentence,
                cited_text,
                supporting_chunk_ids,
            )
            claim_verifications.append(claim_verification)
            if claim_verification.status == "supported":
                verified_sentences.append(sentence)
            else:
                unsupported_claims.append(sentence)

        if not unsupported_claims:
            return answer, [], claim_verifications
        return (
            normalize_answer_markdown(" ".join(verified_sentences)),
            ["answer_contains_unsupported_cited_claims"],
            claim_verifications,
        )

    async def _verify_cited_claims_async(
        self,
        answer: str,
        citations: list[Citation],
    ) -> tuple[str, list[str], list[ClaimVerification]]:
        if self._answer_contains_markdown_blocks(answer):
            return answer, [], []

        citation_by_id = {
            citation.chunk_id: citation
            for citation in citations
            if citation.chunk_id
        }
        sentences = self._split_answer_sentences(answer)
        verified_sentences = []
        claim_verifications = []
        unsupported_claims = []

        for sentence in sentences:
            cited_ids = self._sentence_citation_ids(sentence)
            if not cited_ids:
                verified_sentences.append(sentence)
                continue

            cited_text = " ".join(
                citation_by_id[chunk_id].text or ""
                for chunk_id in cited_ids
                if chunk_id in citation_by_id
            )
            supporting_chunk_ids = sorted(cited_ids & set(citation_by_id))
            claim_verification = await self._async_claim_judge.assess(  # type: ignore[union-attr]
                sentence,
                cited_text,
                supporting_chunk_ids,
            )
            claim_verifications.append(claim_verification)
            if claim_verification.status == "supported":
                verified_sentences.append(sentence)
            else:
                unsupported_claims.append(sentence)

        if not unsupported_claims:
            return answer, [], claim_verifications
        return (
            normalize_answer_markdown(" ".join(verified_sentences)),
            ["answer_contains_unsupported_cited_claims"],
            claim_verifications,
        )

    @staticmethod
    def _semantic_terms(text: str) -> set[str]:
        text = re.sub(r"\((?:p|pp)\.?\s*\d+(?:[-,]\s*\d+)*\)", " ", text, flags=re.IGNORECASE)
        normalized_terms = set()
        for term in tokenize(text):
            normalized_terms.add(
                {
                    "retrieval": "retrieve",
                    "retrieved": "retrieve",
                    "retrieving": "retrieve",
                    "decisions": "decide",
                    "decision": "decide",
                    "decides": "decide",
                    "discusses": "discuss",
                    "systems": "system",
                    "loops": "loop",
                }.get(term, term)
            )
        return normalized_terms

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

    @staticmethod
    def _split_answer_sentences(answer: str) -> list[str]:
        protected_answer = (
            answer.replace("p. ", "p<dot> ")
            .replace("pp. ", "pp<dot> ")
            .replace("e.g. ", "e<dot>g<dot> ")
            .replace("i.e. ", "i<dot>e<dot> ")
        )
        return [
            sentence.replace("<dot>", ".").strip()
            for sentence in re.split(r"(?<=[.!?])\s+(?!\[)", protected_answer)
            if sentence.strip()
        ]
