import httpx
import pytest

from app.services.web_search_service import WebSearchService


@pytest.mark.asyncio
async def test_web_search_service_requests_and_returns_raw_content() -> None:
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "GRU vs LSTM",
                        "url": "https://example.com/gru-lstm",
                        "content": "Short result snippet.",
                        "raw_content": "Full extracted page content about GRU and LSTM.",
                        "score": 0.91,
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = WebSearchService(api_key="test-key", client=client)
        result = await service.search("GRU vs LSTM", max_results=3)

    payload = requests[0].read().decode()

    assert '"search_depth":"advanced"' in payload
    assert '"include_raw_content":true' in payload
    assert result.sources == [
        {
            "title": "GRU vs LSTM",
            "url": "https://example.com/gru-lstm",
            "content": "Short result snippet.",
            "raw_content": "Full extracted page content about GRU and LSTM.",
            "score": 0.91,
        }
    ]
