import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.api.security import current_tenant_id
from app.config.settings import settings
from app.models.chat import AgentRunRecord, AgentRunUsageSummary, ResearchFinding, agent_trace_payload
from app.models.citation import Citation
from app.utils.file import safe_filename


class AgentRunStore:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        root_dir = Path(base_dir or settings.DATA_DIR)
        tenant_id = current_tenant_id()
        if base_dir is None and tenant_id:
            root_dir = root_dir / "tenants" / safe_filename(tenant_id)
        self._base_dir = root_dir / "agent_runs"

    async def append_run(
        self,
        chat_id: str,
        question: str,
        answer: str,
        citations: list[Citation | dict[str, Any]],
        trace: list[dict],
        stop_reason: str | None = None,
    ) -> AgentRunRecord:
        run_id = uuid.uuid4().hex
        created_at = self._timestamp()
        normalized_citations = [
            citation if isinstance(citation, Citation) else Citation.model_validate(citation)
            for citation in citations
        ]
        normalized_trace = agent_trace_payload(trace)
        findings = self._findings_for_run(
            run_id=run_id,
            chat_id=chat_id,
            question=question,
            answer=answer,
            citations=normalized_citations,
            trace=normalized_trace,
            created_at=created_at,
        )
        record = AgentRunRecord(
            run_id=run_id,
            chat_id=chat_id,
            question=question,
            answer=answer,
            citations=normalized_citations,
            trace=normalized_trace,
            stop_reason=stop_reason,
            usage=self._usage_summary(normalized_trace),
            findings=findings,
            created_at=created_at,
        )
        path = self._run_path(chat_id, record.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(record.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return record

    async def list_runs(self, chat_id: str) -> list[AgentRunRecord]:
        chat_dir = self._chat_dir(chat_id)
        if not chat_dir.exists():
            return []
        runs = [
            AgentRunRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(chat_dir.glob("*.json"), key=lambda item: item.stat().st_mtime)
            if path.is_file()
        ]
        return runs

    async def list_findings(self, chat_id: str) -> list[ResearchFinding]:
        findings: list[ResearchFinding] = []
        for run in await self.list_runs(chat_id):
            findings.extend(run.findings)
        return findings

    def _run_path(self, chat_id: str, run_id: str) -> Path:
        return self._chat_dir(chat_id) / f"{safe_filename(run_id)}.json"

    def _chat_dir(self, chat_id: str) -> Path:
        return self._base_dir / safe_filename(chat_id)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(tz=UTC).isoformat()

    def _findings_for_run(
        self,
        run_id: str,
        chat_id: str,
        question: str,
        answer: str,
        citations: list[Citation],
        trace: list[dict],
        created_at: str,
    ) -> list[ResearchFinding]:
        summary = " ".join(answer.split()).strip()
        if not summary or summary == "I don't know":
            return []
        source_ids = self._source_ids(citations)
        citation_ids = [citation.chunk_id for citation in citations if citation.chunk_id]
        return [
            ResearchFinding(
                finding_id=f"{run_id}:f0",
                chat_id=chat_id,
                run_id=run_id,
                question=question,
                summary=summary,
                source_ids=source_ids,
                citation_ids=citation_ids,
                confidence=self._confidence(citations, trace),
                created_at=created_at,
            )
        ]

    @staticmethod
    def _source_ids(citations: list[Citation]) -> list[str]:
        source_ids = []
        seen = set()
        for citation in citations:
            source_id = citation.url or citation.paper_id or citation.title
            if source_id and source_id not in seen:
                seen.add(source_id)
                source_ids.append(source_id)
        return source_ids

    @staticmethod
    def _confidence(citations: list[Citation], trace: list[dict]) -> str:
        verification_events = [event for event in trace if event.get("stage") == "verify_answer"]
        if verification_events:
            latest_verification = verification_events[-1]
            suggested_action = latest_verification.get("suggested_action")
            status = latest_verification.get("status")
            if suggested_action in {"answer_unknown", "retrieve_more"}:
                return "low"
            if latest_verification.get("success") is False or status == "failed":
                return "low"
            if suggested_action == "revise_answer":
                return "medium"
        qualities = {citation.evidence_quality for citation in citations if citation.evidence_quality}
        if "low" in qualities:
            return "low"
        if "high" in qualities:
            return "high"
        if qualities:
            return "medium"
        return "unknown"

    @staticmethod
    def _usage_summary(trace: list[dict]) -> AgentRunUsageSummary:
        models = []
        seen_models = set()
        for event in trace:
            for model in (event.get("model"), event.get("embedding_model")):
                if model and model not in seen_models:
                    seen_models.add(model)
                    models.append(str(model))
        return AgentRunUsageSummary(
            latency_ms=sum(float(event.get("latency_ms") or 0.0) for event in trace),
            input_tokens=sum(int(event.get("input_tokens") or 0) for event in trace),
            output_tokens=sum(int(event.get("output_tokens") or 0) for event in trace),
            total_tokens=sum(int(event.get("total_tokens") or 0) for event in trace),
            embedding_tokens=sum(int(event.get("embedding_tokens") or 0) for event in trace),
            estimated_cost_usd=sum(
                float(event.get("estimated_cost_usd") or 0.0)
                + float(event.get("embedding_estimated_cost_usd") or 0.0)
                for event in trace
            ),
            tool_call_count=sum(1 for event in trace if event.get("stage") == "execute_tool"),
            models=models,
        )
