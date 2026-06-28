from typing import Any

from openai import AsyncOpenAI

from app.config.settings import settings


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

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._get_client()
        response = await client.embeddings.create(
            model=self._model,
            input=texts,
        )
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
