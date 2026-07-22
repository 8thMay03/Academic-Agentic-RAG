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
            ],
            usage=SimpleNamespace(prompt_tokens=12, total_tokens=12),
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsClient()


@pytest.mark.asyncio
async def test_embedding_service_embeds_texts_with_openai_client(monkeypatch) -> None:
    from app.config import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "OPENAI_EMBEDDING_COST_PER_1M", 0.02)
    client = FakeOpenAIClient()
    service = EmbeddingService(client=client, model="text-embedding-test")

    embeddings = await service.embed_texts(["hello", "world"])

    assert client.embeddings.model == "text-embedding-test"
    assert client.embeddings.input == ["hello", "world"]
    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
    assert service.last_usage is not None
    assert service.last_usage.model == "text-embedding-test"
    assert service.last_usage.input_count == 2
    assert service.last_usage.total_tokens == 12
    assert service.last_usage.estimated_cost_usd == pytest.approx(12 * 0.02 / 1_000_000)


@pytest.mark.asyncio
async def test_embedding_service_returns_empty_list_for_empty_input() -> None:
    service = EmbeddingService(client=FakeOpenAIClient())

    assert await service.embed_texts([]) == []
    assert service.last_usage is not None
    assert service.last_usage.input_count == 0


@pytest.mark.asyncio
async def test_embedding_service_requires_api_key_without_injected_client() -> None:
    service = EmbeddingService(api_key="")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        await service.embed_texts(["hello"])
