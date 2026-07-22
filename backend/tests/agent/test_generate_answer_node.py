import pytest

from app.services.llm_service import LLMUsage
from app.agent.nodes.generate_answer_node import generate_answer_node


class FakeLLMService:
    def __init__(self, answer: str, usage: LLMUsage | None = None) -> None:
        self.answer = answer
        self.prompts = []
        self.last_usage = usage

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


@pytest.mark.asyncio
async def test_generate_answer_node_calls_llm_and_records_trace():
    llm = FakeLLMService("  Draft answer [paper:1].  ")

    state = await generate_answer_node(
        {
            "llm_service": llm,
            "prompt": "Answer from context.",
            "trace": [{"stage": "draft_answer"}],
        }
    )

    assert llm.prompts == ["Answer from context."]
    assert state["answer"] == "Draft answer [paper:1]."
    assert state["trace"][-1]["stage"] == "generate_answer"
    assert state["trace"][-1]["status"] == "completed"
    assert state["trace"][-1]["success"] is True
    assert state["trace"][-1]["answer_chars"] == 23
    assert state["trace"][-1]["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_generate_answer_node_records_llm_usage_when_available():
    llm = FakeLLMService(
        "Answer [paper:1].",
        usage=LLMUsage(
            model="gpt-test",
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
            estimated_cost_usd=0.0012,
        ),
    )

    state = await generate_answer_node(
        {
            "llm_service": llm,
            "prompt": "Answer from context.",
            "trace": [],
        }
    )

    assert state["trace"][-1]["model"] == "gpt-test"
    assert state["trace"][-1]["input_tokens"] == 120
    assert state["trace"][-1]["output_tokens"] == 30
    assert state["trace"][-1]["total_tokens"] == 150
    assert state["trace"][-1]["estimated_cost_usd"] == 0.0012


@pytest.mark.asyncio
async def test_generate_answer_node_skips_llm_when_prompt_is_missing():
    llm = FakeLLMService("Should not be used.")

    state = await generate_answer_node({"llm_service": llm, "trace": []})

    assert llm.prompts == []
    assert state["answer"] == ""
    assert state["trace"] == [
        {
            "stage": "generate_answer",
            "status": "no_prompt",
            "success": False,
            "answer_chars": 0,
        }
    ]
