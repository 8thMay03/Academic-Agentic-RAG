import json
from datetime import UTC, datetime
from pathlib import Path

from app.config.settings import settings
from app.models.chat import ChatHistoryMessage
from app.models.citation import Citation
from app.utils.file import safe_filename


class ChatHistoryStore:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = Path(base_dir or settings.DATA_DIR) / "chat_history"

    async def get_messages(self, paper_id: str) -> list[ChatHistoryMessage]:
        path = self._history_path(paper_id)
        if not path.is_file():
            return []

        payload = json.loads(path.read_text(encoding="utf-8"))
        return [ChatHistoryMessage.model_validate(message) for message in payload.get("messages", [])]

    async def append_exchange(
        self,
        paper_id: str,
        question: str,
        answer: str,
        citations: list[Citation],
    ) -> list[ChatHistoryMessage]:
        messages = await self.get_messages(paper_id)
        created_at = datetime.now(tz=UTC).isoformat()
        messages.extend(
            [
                ChatHistoryMessage(role="user", content=question, created_at=created_at),
                ChatHistoryMessage(
                    role="assistant",
                    content=answer,
                    citations=citations,
                    created_at=created_at,
                ),
            ]
        )
        self._write_messages(paper_id, messages)
        return messages

    async def clear(self, paper_id: str) -> None:
        path = self._history_path(paper_id)
        path.unlink(missing_ok=True)

    def _write_messages(self, paper_id: str, messages: list[ChatHistoryMessage]) -> None:
        path = self._history_path(paper_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"paper_id": paper_id, "messages": [message.model_dump(mode="json") for message in messages]}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _history_path(self, paper_id: str) -> Path:
        return self._base_dir / f"{safe_filename(paper_id)}.json"
