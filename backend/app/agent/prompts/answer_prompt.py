from app.agent.models import (
    RetrievedChunk,
    retrieved_chunk_id,
    retrieved_chunk_page_number,
    retrieved_chunk_source_id,
    retrieved_chunk_text,
    retrieved_chunk_title,
)
from app.models.chat import ChatHistoryMessage


UNKNOWN_ANSWER = "I don't know"
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 2000


class AnswerPromptBuilder:
    def build(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        chat_history: list[ChatHistoryMessage] | None = None,
    ) -> str:
        context_blocks = []
        for index, chunk in enumerate(chunks, start=1):
            paper_id = retrieved_chunk_source_id(chunk)
            title = retrieved_chunk_title(chunk)
            page_number = retrieved_chunk_page_number(chunk) or ""
            chunk_id = retrieved_chunk_id(chunk)
            context_blocks.append(
                "\n".join(
                    [
                        f"[Context {index}]",
                        f"paper_id: {paper_id}",
                        f"title: {title}",
                        f"page_number: {page_number}",
                        f"chunk_id: {chunk_id}",
                        f"text: {retrieved_chunk_text(chunk)}",
                    ]
                )
            )

        context_text = "\n\n".join(context_blocks)
        available_chunk_ids = ", ".join(
            chunk_id
            for chunk in chunks
            if (chunk_id := self._chunk_id_for(chunk))
        )
        conversation_context = self._conversation_context(chat_history)
        conversation_section = (
            "Recent conversation:\n"
            f"{conversation_context}\n\n"
            if conversation_context
            else ""
        )
        return (
            "Answer in the same language as the user's question. If the question is Vietnamese, answer in natural Vietnamese "
            "while preserving standard technical terms and LaTeX formulas.\n"
            "Answer the question using only the retrieved local paper and web context below.\n"
            "Use the recent conversation only to resolve pronouns, ellipses, and follow-up references.\n"
            "If the context does not contain enough information to answer, respond exactly:\n"
            f"{UNKNOWN_ANSWER}\n\n"
            "Do not use outside knowledge. Do not guess.\n"
            "Write a useful, substantive answer when the context is rich: start with a direct summary, then explain "
            "the main mechanisms, distinctions, trade-offs, or steps that the context supports. Avoid one-sentence "
            "answers unless the question is trivial or the context is very sparse.\n"
            "Prefer clear structure over brevity: use short paragraphs or bullets; use a compact markdown table for "
            "comparison questions when it helps the answer scan well.\n"
            "When writing mathematical formulas, use LaTeX math delimiters. Use inline $...$ only for short symbols "
            "or tiny expressions. Put important formulas, fractions, objectives, update rules, loss functions, and "
            "attention/logistic/sigmoid equations in display math using $$...$$ on their own lines.\n"
            "For comparison questions, if separate retrieved context describes each compared item, synthesize the "
            "differences from those cited facts instead of requiring a single chunk that directly compares them.\n"
            "Every factual claim supported by retrieved context must end with one or more exact chunk_id citations "
            "in square brackets, e.g. [paper-1:p3:c0].\n"
            "Use only chunk_id values that appear in the retrieved context. Do not invent citation ids.\n\n"
            f"Available chunk_ids: {available_chunk_ids}\n\n"
            f"{conversation_section}"
            f"Question: {question}\n\n"
            "Retrieved context:\n"
            f"{context_text}"
        )

    @staticmethod
    def _conversation_context(chat_history: list[ChatHistoryMessage] | None) -> str:
        if not chat_history:
            return ""

        lines = []
        for message in chat_history[-MAX_HISTORY_MESSAGES:]:
            role = "User" if message.role == "user" else "Assistant"
            content = " ".join(message.content.split())
            if content:
                lines.append(f"{role}: {content}")

        context = "\n".join(lines)
        if len(context) <= MAX_HISTORY_CHARS:
            return context
        return context[-MAX_HISTORY_CHARS:].lstrip()

    @staticmethod
    def _chunk_id_for(chunk: RetrievedChunk) -> str:
        return retrieved_chunk_id(chunk)
