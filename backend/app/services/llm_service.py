from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from app.config.settings import settings


@dataclass(frozen=True)
class LLMUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class LLMService:
    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model or settings.OPENAI_CHAT_MODEL
        self._client = client
        self._api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self.last_usage: LLMUsage | None = None

    async def complete(self, prompt: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful Agentic RAG assistant. "
                        "Return concise, accurate Markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        self.last_usage = self._usage_from_response(response)
        return response.choices[0].message.content or ""

    async def stream_complete(self, prompt: str) -> AsyncIterator[str]:
        client = self._get_client()
        stream = await client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful Agentic RAG assistant. "
                        "Return concise, accurate Markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content if chunk.choices else None
            if token:
                yield token

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is required to call the LLM.")
        self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    def _usage_from_response(self, response: Any) -> LLMUsage:
        usage = getattr(response, "usage", None)
        input_tokens = int(
            getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_tokens", None)
            or 0
        )
        output_tokens = int(
            getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_tokens", None)
            or 0
        )
        total_tokens = int(getattr(usage, "total_tokens", None) or input_tokens + output_tokens)
        estimated_cost = (
            input_tokens * settings.OPENAI_CHAT_INPUT_COST_PER_1M
            + output_tokens * settings.OPENAI_CHAT_OUTPUT_COST_PER_1M
        ) / 1_000_000
        return LLMUsage(
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
        )
