from typing import Any, TypedDict


class AgenticRAGState(TypedDict, total=False):
    workflow: Any
    request: Any
    local_chunks: list[dict]
    web_chunks: list[dict]
    quality: Any
    citations: list[Any]
    prompt: str | None
    trace: list[dict]
