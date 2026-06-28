from app.parser.cleaner import PAGE_BREAK, clean_text


def test_clean_text_removes_repeated_headers_footers_and_page_numbers() -> None:
    raw_text = PAGE_BREAK.join(
        [
            "\n".join(
                [
                    "Agentic RAG Survey",
                    "1",
                    "Introduction text",
                    "Confidential Draft",
                ]
            ),
            "\n".join(
                [
                    "Agentic RAG Survey",
                    "2",
                    "Methods text",
                    "Confidential Draft",
                ]
            ),
            "\n".join(
                [
                    "Agentic RAG Survey",
                    "3",
                    "Conclusion text",
                    "Confidential Draft",
                ]
            ),
        ]
    )

    cleaned = clean_text(raw_text)

    assert "Agentic RAG Survey" not in cleaned
    assert "Confidential Draft" not in cleaned
    assert "\n1\n" not in cleaned
    assert "Introduction text" in cleaned
    assert "Methods text" in cleaned
    assert "Conclusion text" in cleaned


def test_clean_text_normalizes_pdf_text_artifacts() -> None:
    raw_text = "Agentic\u00a0RAG\u2014retrieval\naug-\nmented    generation\n\n\nuses \u201cplanning\u201d."

    cleaned = clean_text(raw_text)

    assert cleaned == 'Agentic RAG-retrieval\naugmented generation\nuses "planning".'
