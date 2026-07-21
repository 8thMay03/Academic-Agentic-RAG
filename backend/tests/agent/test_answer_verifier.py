from app.agent.evaluators.answer_verifier import AnswerVerifier
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
