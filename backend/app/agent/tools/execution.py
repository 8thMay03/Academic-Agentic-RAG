from dataclasses import dataclass
import asyncio
from typing import Any

from app.agent.models import AgentLimits, ToolResult


@dataclass(frozen=True)
class PreparedToolInput:
    input: dict[str, Any]
    error: str | None = None


def prepare_tool_input(
    tool_name: str,
    step_input: dict[str, Any],
    state: dict[str, Any],
) -> PreparedToolInput:
    tool_input = dict(step_input)
    if tool_name == "web_snippet_ingest":
        tool_input["web_chunks"] = state.get("web_chunks", [])
    elif tool_name == "pdf_download":
        pdf_source = first_pdf_source(state.get("tool_results", []))
        if not pdf_source:
            return PreparedToolInput(
                input=tool_input,
                error="No PDF URL was available from prior research results.",
            )
        tool_input["pdf_url"] = pdf_source["pdf_url"]
        tool_input["source_metadata"] = source_metadata_from_artifact(pdf_source)
    elif tool_name == "pdf_index":
        download_artifact = latest_artifact_for_tool(state.get("tool_results", []), "pdf_download")
        if not download_artifact:
            return PreparedToolInput(
                input=tool_input,
                error="No downloaded PDF artifact was available to index.",
            )
        tool_input["path"] = download_artifact.get("path")
        tool_input["source_metadata"] = source_metadata_from_artifact(download_artifact)
    return PreparedToolInput(input=tool_input)


def tool_limit_error(
    state: dict[str, Any],
    tool_name: str,
    limits: AgentLimits,
) -> str | None:
    total_count = total_tool_count(state)
    if total_count >= limits.max_steps:
        return f"Agent step limit reached: {total_count}/{limits.max_steps}."

    prior_count = prior_tool_count(state, tool_name)
    max_count = {
        "web_search": limits.max_web_searches,
        "arxiv_search": limits.max_arxiv_searches,
        "pdf_download": limits.max_pdf_downloads,
        "local_retrieve": limits.max_retrieval_rounds,
    }.get(tool_name)
    if max_count is None or prior_count < max_count:
        return None
    return f"Tool limit reached for {tool_name}: {prior_count}/{max_count}."


async def run_tool_with_timeout(
    tool_registry: Any,
    tool_name: str,
    tool_input: dict[str, Any],
    limits: AgentLimits,
) -> ToolResult:
    try:
        return await asyncio.wait_for(
            tool_registry.run(tool_name, tool_input),
            timeout=limits.tool_timeout_seconds,
        )
    except TimeoutError:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            error=f"Tool timed out after {limits.tool_timeout_seconds:g}s.",
        )


def total_tool_count(state: dict[str, Any]) -> int:
    return len(state.get("tool_results", []))


def prior_tool_count(state: dict[str, Any], tool_name: str) -> int:
    count = sum(1 for result in state.get("tool_results", []) if result.tool_name == tool_name)
    if tool_name == "local_retrieve" and "local_chunks" in state:
        count += 1
    return count


def first_pdf_url(tool_results: list[ToolResult]) -> str | None:
    pdf_source = first_pdf_source(tool_results)
    return str(pdf_source["pdf_url"]) if pdf_source else None


def first_pdf_source(tool_results: list[ToolResult]) -> dict | None:
    for result in tool_results:
        for artifact in result.artifacts or []:
            if artifact.get("pdf_url"):
                return artifact
        metadata = result.metadata or {}
        pdf_urls = metadata.get("pdf_urls") or []
        if pdf_urls:
            source = dict(metadata)
            source["pdf_url"] = str(pdf_urls[0])
            return source
    return None


def latest_artifact_for_tool(tool_results: list[ToolResult], tool_name: str) -> dict | None:
    for result in reversed(tool_results):
        if result.tool_name == tool_name and result.artifacts:
            return result.artifacts[0]
    return None


def source_metadata_from_artifact(artifact: dict) -> dict[str, Any]:
    return {
        key: artifact[key]
        for key in (
            "source_type",
            "source_url",
            "pdf_url",
            "discovered_by_query",
            "trust_level",
            "ingestion_status",
        )
        if artifact.get(key) is not None
    }
