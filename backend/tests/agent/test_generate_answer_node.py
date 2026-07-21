import pytest

from app.agent.nodes.generate_answer_node import generate_answer_node


class FakeLLMService:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts = []

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
    assert state["trace"][-1] == {
        "stage": "generate_answer",
        "status": "completed",
        "success": True,
        "answer_chars": 23,
    }


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
