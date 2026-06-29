from types import SimpleNamespace

import pytest

from app.services.llm_service import LLMService


class FakeCompletionsClient:
    def __init__(self) -> None:
        self.model = None
        self.messages = None
        self.temperature = None

    async def create(self, model: str, messages: list[dict], temperature: float, stream: bool = False):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        if stream:
            return self._stream_response()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="## Problem\n- Test summary."))]
        )

    async def _stream_response(self):
        for token in ["Hello", " world"]:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=token))])


class FakeChatClient:
    def __init__(self) -> None:
        self.completions = FakeCompletionsClient()


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = FakeChatClient()


@pytest.mark.asyncio
async def test_llm_service_completes_prompt_with_openai_client() -> None:
    client = FakeOpenAIClient()
    service = LLMService(client=client, model="gpt-test")

    response = await service.complete("Summarize this paper.")

    assert response == "## Problem\n- Test summary."
    assert client.chat.completions.model == "gpt-test"
    assert client.chat.completions.messages[-1] == {
        "role": "user",
        "content": "Summarize this paper.",
    }
    assert client.chat.completions.temperature == 0.2


@pytest.mark.asyncio
async def test_llm_service_streams_prompt_with_openai_client() -> None:
    client = FakeOpenAIClient()
    service = LLMService(client=client, model="gpt-test")

    tokens = [token async for token in service.stream_complete("Stream this.")]

    assert tokens == ["Hello", " world"]
    assert client.chat.completions.model == "gpt-test"
    assert client.chat.completions.messages[-1] == {
        "role": "user",
        "content": "Stream this.",
    }


@pytest.mark.asyncio
async def test_llm_service_requires_api_key_without_injected_client() -> None:
    service = LLMService(api_key="")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        await service.complete("Summarize this paper.")
