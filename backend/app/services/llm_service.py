from typing import Any

from openai import AsyncOpenAI

from app.config.settings import settings


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

    async def complete(self, prompt: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful research assistant. "
                        "Return concise, accurate Markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is required to call the LLM.")
        self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client
