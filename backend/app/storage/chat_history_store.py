import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.config.settings import settings
from app.models.chat import ChatHistoryMessage, ChatSession, ChatSource, ChatThread
from app.models.citation import Citation
from app.utils.file import safe_filename


class ChatHistoryStore:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = Path(base_dir or settings.DATA_DIR) / "chat_history"

    async def get_messages(self, paper_id: str) -> list[ChatHistoryMessage]:
        path = self._history_path(paper_id)
        if not path.is_file():
            return []

        return self._session_from_payload(json.loads(path.read_text(encoding="utf-8")), path).messages

    async def list_threads(self) -> list[ChatThread]:
        if not self._base_dir.exists():
            return []

        threads: list[ChatThread] = []
        for path in sorted(self._base_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            if not path.is_file():
                continue

            session = self._session_from_payload(json.loads(path.read_text(encoding="utf-8")), path)
            last_message = session.messages[-1] if session.messages else None
            first_user_message = next((message for message in session.messages if message.role == "user"), None)
            threads.append(
                ChatThread(
                    chat_id=session.chat_id,
                    title=first_user_message.content if first_user_message else session.title,
                    last_message=last_message.content if last_message else "No messages yet.",
                    updated_at=last_message.created_at if last_message else session.updated_at,
                    message_count=len(session.messages),
                    source_count=len(session.sources),
                )
            )

        return threads

    async def create_session(self, title: str | None = None) -> ChatSession:
        timestamp = self._timestamp()
        chat_id = uuid.uuid4().hex
        session = ChatSession(
            chat_id=chat_id,
            title=title or "New chat",
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._write_session(session)
        return session

    async def get_session(self, chat_id: str) -> ChatSession | None:
        path = self._history_path(chat_id)
        if not path.is_file():
            return None
        return self._session_from_payload(json.loads(path.read_text(encoding="utf-8")), path)

    async def add_source(self, chat_id: str, source: ChatSource) -> ChatSession:
        session = await self.get_session(chat_id)
        if session is None:
            timestamp = self._timestamp()
            session = ChatSession(chat_id=chat_id, title="New chat", created_at=timestamp, updated_at=timestamp)

        sources_by_id = {existing.paper_id: existing for existing in session.sources}
        sources_by_id[source.paper_id] = source
        session.sources = list(sources_by_id.values())
        session.updated_at = self._timestamp()
        if session.title == "New chat":
            session.title = source.title
        self._write_session(session)
        return session

    async def remove_source(self, chat_id: str, paper_id: str) -> ChatSession | None:
        session = await self.get_session(chat_id)
        if session is None:
            return None
        session.sources = [source for source in session.sources if source.paper_id != paper_id]
        session.updated_at = self._timestamp()
        self._write_session(session)
        return session

    async def append_exchange(
        self,
        paper_id: str,
        question: str,
        answer: str,
        citations: list[Citation],
    ) -> list[ChatHistoryMessage]:
        session = await self.get_session(paper_id)
        if session is None:
            timestamp = self._timestamp()
            session = ChatSession(chat_id=paper_id, title=paper_id, created_at=timestamp, updated_at=timestamp)

        messages = session.messages
        created_at = self._timestamp()
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
        session.messages = messages
        session.updated_at = created_at
        if session.title == "New chat":
            session.title = question
        self._write_session(session)
        return messages

    async def clear(self, paper_id: str) -> None:
        path = self._history_path(paper_id)
        if not path.is_file():
            return

        session = self._session_from_payload(json.loads(path.read_text(encoding="utf-8")), path)
        session.messages = []
        session.updated_at = self._timestamp()
        self._write_session(session)

    def _write_messages(self, paper_id: str, messages: list[ChatHistoryMessage]) -> None:
        path = self._history_path(paper_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"paper_id": paper_id, "messages": [message.model_dump(mode="json") for message in messages]}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_session(self, session: ChatSession) -> None:
        path = self._history_path(session.chat_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(session.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")

    def _history_path(self, paper_id: str) -> Path:
        return self._base_dir / f"{safe_filename(paper_id)}.json"

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(tz=UTC).isoformat()

    def _session_from_payload(self, payload: dict, path: Path) -> ChatSession:
        if "chat_id" in payload:
            return ChatSession.model_validate(payload)

        messages = [ChatHistoryMessage.model_validate(message) for message in payload.get("messages", [])]
        paper_id = payload.get("paper_id") or path.stem
        updated_at = messages[-1].created_at if messages else datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
        return ChatSession(
            chat_id=paper_id,
            title=paper_id,
            sources=[ChatSource(paper_id=paper_id, title=paper_id)],
            messages=messages,
            created_at=updated_at,
            updated_at=updated_at,
        )
