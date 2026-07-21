from app.agent.models import append_trace
from app.agent.state import AgenticRAGState


async def generate_answer_node(state: AgenticRAGState) -> AgenticRAGState:
    prompt = state.get("prompt")
    if not prompt:
        return {
            **state,
            "answer": "",
            "trace": append_trace(
                state.get("trace", []),
                "generate_answer",
                status="no_prompt",
                success=False,
                answer_chars=0,
            ),
        }

    answer = (await state["llm_service"].complete(prompt)).strip()
    return {
        **state,
        "answer": answer,
        "trace": append_trace(
            state.get("trace", []),
            "generate_answer",
            status="completed" if answer else "empty",
            success=bool(answer),
            answer_chars=len(answer),
        ),
    }
