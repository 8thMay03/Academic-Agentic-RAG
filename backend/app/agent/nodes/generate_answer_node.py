from time import perf_counter

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

    started_at = perf_counter()
    llm_service = state["llm_service"]
    answer = (await llm_service.complete(prompt)).strip()
    latency_ms = (perf_counter() - started_at) * 1000
    usage = getattr(llm_service, "last_usage", None)
    trace_fields = {
        "status": "completed" if answer else "empty",
        "success": bool(answer),
        "answer_chars": len(answer),
        "latency_ms": latency_ms,
    }
    if usage:
        trace_fields.update(
            {
                "model": usage.model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "estimated_cost_usd": usage.estimated_cost_usd,
            }
        )
    return {
        **state,
        "answer": answer,
        "trace": append_trace(state.get("trace", []), "generate_answer", **trace_fields),
    }
