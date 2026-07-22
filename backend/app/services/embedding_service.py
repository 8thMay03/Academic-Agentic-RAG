from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from app.config.settings import settings


@dataclass(frozen=True)
class EmbeddingUsage:
    model: str
    input_count: int
    prompt_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


def aggregate_embedding_usages(usages: list[EmbeddingUsage | None]) -> EmbeddingUsage | None:
    present_usages = [usage for usage in usages if usage is not None]
    if not present_usages:
        return None
    models = []
    seen_models = set()
    for usage in present_usages:
        if usage.model not in seen_models:
            seen_models.add(usage.model)
            models.append(usage.model)
    return EmbeddingUsage(
        model=", ".join(models),
        input_count=sum(usage.input_count for usage in present_usages),
        prompt_tokens=sum(usage.prompt_tokens for usage in present_usages),
        total_tokens=sum(usage.total_tokens for usage in present_usages),
        estimated_cost_usd=sum(usage.estimated_cost_usd for usage in present_usages),
    )


class EmbeddingService:
    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model or settings.OPENAI_EMBEDDING_MODEL
        self._client = client
        self._api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self.last_usage: EmbeddingUsage | None = None

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            self.last_usage = EmbeddingUsage(model=self._model, input_count=0)
            return []

        client = self._get_client()
        response = await client.embeddings.create(
            model=self._model,
            input=texts,
        )
        self.last_usage = self._usage_from_response(response, len(texts))
        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> list[float]:
        embeddings = await self.embed_texts([query])
        return embeddings[0]

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is required to create embeddings.")
        self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    def _usage_from_response(self, response: Any, input_count: int) -> EmbeddingUsage:
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0)
        total_tokens = int(getattr(usage, "total_tokens", None) or prompt_tokens)
        estimated_cost = total_tokens * settings.OPENAI_EMBEDDING_COST_PER_1M / 1_000_000
        return EmbeddingUsage(
            model=self._model,
            input_count=input_count,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
        )
