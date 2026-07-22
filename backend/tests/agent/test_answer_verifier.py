import pytest

from app.agent.evaluators.answer_verifier import AnswerVerifier, LLMClaimSupportJudge
from app.agent.models import ClaimVerification
from app.models.citation import Citation


def test_answer_verifier_returns_unknown_without_evidence() -> None:
    verifier = AnswerVerifier()

    result = verifier.verify("This answer has no evidence.", [])

    assert result.passed is False
    assert result.answer == "I don't know"
    assert result.citations == []
    assert result.issues == ["answer_has_no_evidence"]
    assert result.unsupported_claims == ["This answer has no evidence."]
    assert result.suggested_action == "answer_unknown"


def test_answer_verifier_removes_fake_citations_and_keeps_supported_evidence() -> None:
    verifier = AnswerVerifier()
    citation = Citation(
        paper_id="paper-1",
        title="Agentic RAG",
        chunk_id="paper-1:p3:c0",
        text="Agentic RAG uses planning.",
    )

    result = verifier.verify("It uses planning [fake:c0].", [citation])

    assert result.passed is False
    assert result.answer == "It uses planning. [paper-1:p3:c0]"
    assert result.citations == [citation]
    assert result.issues == ["answer_citations_were_grounded"]
    assert result.unsupported_claims == []
    assert result.suggested_action == "revise_answer"


def test_answer_verifier_revises_answer_that_omits_explicit_citations() -> None:
    verifier = AnswerVerifier()
    citation = Citation(
        paper_id="paper-1",
        title="Agentic RAG",
        chunk_id="paper-1:p3:c0",
        text="Agentic RAG uses planning.",
    )

    result = verifier.verify("It uses planning.", [citation])

    assert result.passed is False
    assert result.answer == "It uses planning. [paper-1:p3:c0]"
    assert result.citations == [citation]
    assert result.issues == [
        "answer_missing_explicit_citations",
        "answer_citations_were_grounded",
    ]
    assert result.unsupported_claims == []
    assert result.suggested_action == "revise_answer"


def test_answer_verifier_requests_more_retrieval_when_model_answers_unknown_with_evidence() -> None:
    verifier = AnswerVerifier()
    citation = Citation(
        paper_id="paper-1",
        title="GRU vs LSTM",
        chunk_id="paper-1:p3:c0",
        text="GRU and LSTM use different gating mechanisms.",
    )

    result = verifier.verify("I don't know", [citation])

    assert result.passed is False
    assert result.answer == "I don't know"
    assert result.citations == []
    assert result.issues == ["answer_unknown_despite_available_evidence"]
    assert result.unsupported_claims == ["I don't know"]
    assert result.suggested_action == "retrieve_more"


def test_answer_verifier_requests_more_retrieval_for_unusable_citation_ids() -> None:
    verifier = AnswerVerifier()
    citation = Citation(
        paper_id="paper-1",
        title="Agentic RAG",
        chunk_id=None,
        text="Agentic RAG uses planning.",
    )

    result = verifier.verify("It uses planning [fake:c0].", [citation])

    assert result.passed is False
    assert result.answer == "I don't know"
    assert result.citations == []
    assert result.issues == ["answer_references_no_valid_citations"]
    assert result.unsupported_claims == ["It uses planning [fake:c0]."]
    assert result.suggested_action == "retrieve_more"


def test_answer_verifier_accepts_supported_answer() -> None:
    verifier = AnswerVerifier()
    citation = Citation(
        paper_id="paper-1",
        title="Agentic RAG",
        chunk_id="paper-1:p3:c0",
        text="Agentic RAG uses planning.",
    )

    result = verifier.verify("It uses planning [paper-1:p3:c0].", [citation])

    assert result.passed is True
    assert result.answer == "It uses planning [paper-1:p3:c0]."
    assert result.citations == [citation]
    assert result.issues == []
    assert result.unsupported_claims == []
    assert result.suggested_action == "finalize"
    assert result.claim_verifications[0].status == "supported"
    assert result.claim_verifications[0].supporting_chunk_ids == ["paper-1:p3:c0"]


def test_answer_verifier_rejects_cited_claim_not_supported_by_cited_text() -> None:
    verifier = AnswerVerifier()
    citation = Citation(
        paper_id="paper-1",
        title="Agentic RAG",
        chunk_id="paper-1:p3:c0",
        text="Agentic RAG uses planning.",
    )

    result = verifier.verify("Agentic RAG eliminates hallucinations [paper-1:p3:c0].", [citation])

    assert result.passed is False
    assert result.answer == "I don't know"
    assert result.citations == []
    assert result.issues == ["answer_contains_unsupported_cited_claims"]
    assert result.unsupported_claims == ["Agentic RAG eliminates hallucinations [paper-1:p3:c0]."]
    assert result.suggested_action == "retrieve_more"
    assert result.claim_verifications[0].status == "insufficient"


def test_answer_verifier_rejects_cited_claim_that_contradicts_evidence() -> None:
    verifier = AnswerVerifier()
    citation = Citation(
        paper_id="paper-1",
        title="Agentic RAG",
        chunk_id="paper-1:p3:c0",
        text="Agentic RAG does not eliminate hallucinations.",
    )

    result = verifier.verify("Agentic RAG eliminates hallucinations [paper-1:p3:c0].", [citation])

    assert result.passed is False
    assert result.answer == "I don't know"
    assert result.unsupported_claims == ["Agentic RAG eliminates hallucinations [paper-1:p3:c0]."]
    assert result.suggested_action == "retrieve_more"
    assert result.claim_verifications[0].status == "contradicted"
    assert result.claim_verifications[0].reason == "claim_negation_conflicts_with_cited_evidence"


class AlwaysContradictsJudge:
    def assess(
        self,
        claim: str,
        evidence_text: str,
        supporting_chunk_ids: list[str],
    ) -> ClaimVerification:
        return ClaimVerification(
            claim=claim,
            status="contradicted",
            supporting_chunk_ids=supporting_chunk_ids,
            reason="test_llm_or_nli_judge_override",
        )


def test_answer_verifier_accepts_injected_claim_judge() -> None:
    verifier = AnswerVerifier(claim_judge=AlwaysContradictsJudge())
    citation = Citation(
        paper_id="paper-1",
        title="Agentic RAG",
        chunk_id="paper-1:p3:c0",
        text="Agentic RAG uses planning.",
    )

    result = verifier.verify("Agentic RAG uses planning [paper-1:p3:c0].", [citation])

    assert result.passed is False
    assert result.claim_verifications[0].status == "contradicted"
    assert result.claim_verifications[0].reason == "test_llm_or_nli_judge_override"


class FakeClaimJudgeLLM:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.prompts = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.asyncio
async def test_llm_claim_support_judge_parses_supported_json() -> None:
    llm = FakeClaimJudgeLLM('{"status":"supported","reason":"Evidence directly states it."}')
    judge = LLMClaimSupportJudge(llm)

    result = await judge.assess(
        "Agentic RAG uses planning [paper-1:p1:c0].",
        "Agentic RAG uses planning before answering.",
        ["paper-1:p1:c0"],
    )

    assert result.status == "supported"
    assert result.reason == "Evidence directly states it."
    assert "Return JSON only" in llm.prompts[0]


@pytest.mark.asyncio
async def test_llm_claim_support_judge_falls_back_to_heuristic_on_provider_error() -> None:
    judge = LLMClaimSupportJudge(FakeClaimJudgeLLM(RuntimeError("provider down")))

    result = await judge.assess(
        "Agentic RAG uses planning [paper-1:p1:c0].",
        "Agentic RAG uses planning before answering.",
        ["paper-1:p1:c0"],
    )

    assert result.status == "supported"
    assert result.reason == "llm_claim_judge_fallback:claim_terms_overlap_cited_evidence"


@pytest.mark.asyncio
async def test_answer_verifier_async_uses_llm_claim_judge() -> None:
    verifier = AnswerVerifier(
        async_claim_judge=LLMClaimSupportJudge(
            FakeClaimJudgeLLM('{"status":"contradicted","reason":"Evidence says the opposite."}')
        )
    )
    citation = Citation(
        paper_id="paper-1",
        title="Agentic RAG",
        chunk_id="paper-1:p3:c0",
        text="Agentic RAG uses planning.",
    )

    result = await verifier.verify_async("Agentic RAG uses planning [paper-1:p3:c0].", [citation])

    assert result.passed is False
    assert result.answer == "I don't know"
    assert result.suggested_action == "retrieve_more"
    assert result.claim_verifications[0].status == "contradicted"
    assert result.claim_verifications[0].reason == "Evidence says the opposite."


def test_answer_verifier_removes_uncited_claims_from_mixed_answer() -> None:
    verifier = AnswerVerifier()
    citation = Citation(
        paper_id="paper-1",
        title="Agentic RAG",
        chunk_id="paper-1:p3:c0",
        text="Agentic RAG uses planning.",
    )

    result = verifier.verify(
        "Agentic RAG uses planning [paper-1:p3:c0]. It eliminates hallucinations.",
        [citation],
    )

    assert result.passed is False
    assert result.answer == "Agentic RAG uses planning [paper-1:p3:c0]."
    assert result.citations == [citation]
    assert result.issues == ["answer_contains_uncited_claims"]
    assert result.unsupported_claims == ["It eliminates hallucinations."]
    assert result.suggested_action == "revise_answer"


def test_answer_verifier_keeps_paragraph_claims_when_citation_closes_the_paragraph() -> None:
    verifier = AnswerVerifier()
    citation = Citation(
        paper_id="paper-1",
        title="CNN",
        chunk_id="paper-1:p3:c0",
        text="CNN uses convolutional filters, pooling, and local receptive fields.",
    )

    result = verifier.verify(
        "CNN learns local visual patterns. It uses convolutional filters and pooling [paper-1:p3:c0].",
        [citation],
    )

    assert result.answer == "CNN learns local visual patterns. It uses convolutional filters and pooling [paper-1:p3:c0]."
    assert result.unsupported_claims == []
    assert result.suggested_action == "finalize"


def test_answer_verifier_keeps_arxiv_citation_ids_with_decimal_points() -> None:
    verifier = AnswerVerifier()
    citation = Citation(
        paper_id="2601.12345",
        title="Fresh Agentic RAG",
        chunk_id="2601.12345:p2:c0",
        text="Fresh Agentic RAG uses verification.",
    )

    result = verifier.verify(
        "Fresh Agentic RAG uses verification [2601.12345:p2:c0]. It is always correct.",
        [citation],
    )

    assert result.answer == "Fresh Agentic RAG uses verification [2601.12345:p2:c0]."
    assert result.unsupported_claims == ["It is always correct."]


def test_answer_verifier_preserves_markdown_structure() -> None:
    verifier = AnswerVerifier()
    citation = Citation(
        paper_id="paper-1",
        title="Random Forest",
        chunk_id="paper-1:p1:c0",
        text="Random Forest trains many decision trees with bootstrap samples.",
    )

    result = verifier.verify(
        (
            "Random Forest is an ensemble method [paper-1:p1:c0].\n\n"
            "## How it works\n\n"
            "- Trains many decision trees [paper-1:p1:c0].\n\n"
            "| Step | Description |\n"
            "|------|-------------|\n"
            "| 1 | Bootstrap samples |\n"
        ),
        [citation],
    )

    assert "## How it works" in result.answer
    assert "\n- Trains many decision trees [paper-1:p1:c0]." in result.answer
    assert "\n| Step | Description |\n" in result.answer
