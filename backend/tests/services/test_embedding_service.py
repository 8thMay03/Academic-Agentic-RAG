from types import SimpleNamespace

import pytest

from app.services.embedding_service import EmbeddingService


class FakeEmbeddingsClient:
    def __init__(self) -> None:
        self.model = None
        self.input = None

    async def create(self, model: str, input: list[str]):
        self.model = model
        self.input = input
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[1.0, 0.0]),
                SimpleNamespace(embedding=[0.0, 1.0]),
            ]
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsClient()


@pytest.mark.asyncio
async def test_embedding_service_embeds_texts_with_openai_client() -> None:
    client = FakeOpenAIClient()
    service = EmbeddingService(client=client, model="text-embedding-test")

    embeddings = await service.embed_texts(["hello", "world"])

    assert client.embeddings.model == "text-embedding-test"
    assert client.embeddings.input == ["hello", "world"]
    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.asyncio
async def test_embedding_service_returns_empty_list_for_empty_input() -> None:
    service = EmbeddingService(client=FakeOpenAIClient())

    assert await service.embed_texts([]) == []


@pytest.mark.asyncio
async def test_embedding_service_requires_api_key_without_injected_client() -> None:
    service = EmbeddingService(api_key="")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        await service.embed_texts(["hello"])
