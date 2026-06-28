from app.services.llm_service import LLMService

SUMMARY_SECTIONS = [
    "Problem",
    "Method",
    "Key Contributions",
    "Experiments",
    "Limitations",
    "Relevance",
]


class SummaryService:
    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    async def summarize(self, title: str, text: str) -> str:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("text is required to summarize a paper.")

        chunks = self._chunk_for_summary(cleaned_text)
        if len(chunks) == 1:
            return await self.llm_service.complete(self._final_summary_prompt(title, chunks[0]))

        partial_summaries = []
        for index, chunk in enumerate(chunks, start=1):
            partial_summaries.append(
                await self.llm_service.complete(self._partial_summary_prompt(title, chunk, index, len(chunks)))
            )

        return await self.llm_service.complete(
            self._final_summary_prompt(title, "\n\n".join(partial_summaries), is_partial_summary=True)
        )

    @staticmethod
    def _chunk_for_summary(text: str, max_chars: int = 12000) -> list[str]:
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                split_at = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
                if split_at > start + max_chars // 2:
                    end = split_at + 1
            chunks.append(text[start:end].strip())
            start = end

        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _partial_summary_prompt(title: str, text: str, index: int, total: int) -> str:
        return (
            "Summarize this section of a research paper. Focus only on facts present "
            "in the provided text. Return concise Markdown bullets.\n\n"
            f"Paper title: {title}\n"
            f"Section chunk: {index}/{total}\n\n"
            f"{text}"
        )

    @staticmethod
    def _final_summary_prompt(title: str, text: str, is_partial_summary: bool = False) -> str:
        source_label = "partial summaries" if is_partial_summary else "paper text"
        sections = "\n".join(f"## {section}" for section in SUMMARY_SECTIONS)
        return (
            "Create a Markdown summary of the research paper using exactly these headings:\n\n"
            f"{sections}\n\n"
            "Rules:\n"
            "- Keep the summary concise but specific.\n"
            "- Use bullet points under each heading.\n"
            "- If a section is not supported by the provided text, write '- Not specified.'\n"
            "- Do not invent details.\n\n"
            f"Paper title: {title}\n\n"
            f"Source {source_label}:\n{text}"
        )
