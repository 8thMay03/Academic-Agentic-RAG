from __future__ import annotations

import re

from app.agent.models import RetrievedChunk, retrieved_chunk_text


SUSPICIOUS_INSTRUCTION_PATTERNS = {
    "ignore_previous_instructions": re.compile(r"\bignore\s+(?:all\s+)?previous\s+instructions\b", re.IGNORECASE),
    "system_prompt_exfiltration": re.compile(r"\b(system|developer)\s+(prompt|message)\b", re.IGNORECASE),
    "secret_exfiltration": re.compile(r"\b(reveal|print|show|exfiltrate)\s+(?:the\s+)?(secret|api\s*key|password)\b", re.IGNORECASE),
    "role_override": re.compile(r"\b(act|behave)\s+as\s+(?:the\s+)?(system|developer|admin)\b", re.IGNORECASE),
}


def mark_suspicious_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    marked_chunks = []
    for chunk in chunks:
        reason = suspicious_instruction_reason(retrieved_chunk_text(chunk))
        if not reason:
            marked_chunks.append(chunk)
            continue

        metadata = dict(chunk.get("metadata") or {})
        citation = dict(chunk.get("citation") or {})
        metadata["security_flag"] = "suspicious_instruction"
        metadata["security_reason"] = reason
        citation["security_flag"] = "suspicious_instruction"
        citation["security_reason"] = reason
        marked_chunks.append(
            {
                **chunk,
                "metadata": metadata,
                "citation": citation,
            }
        )
    return marked_chunks


def suspicious_instruction_reason(text: str) -> str | None:
    for reason, pattern in SUSPICIOUS_INSTRUCTION_PATTERNS.items():
        if pattern.search(text):
            return reason
    return None
