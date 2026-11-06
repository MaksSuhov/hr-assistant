from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hr_assistant.infrastructure.db.models import ChatMessageRecord, ChatSessionRecord


class ChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_session(self, session_id: str) -> ChatSessionRecord | None:
        return await self.session.get(ChatSessionRecord, session_id)

    async def add_session(self, chat: ChatSessionRecord) -> None:
        self.session.add(chat)
        await self.session.flush()

    async def add_message(self, message: ChatMessageRecord) -> None:
        self.session.add(message)
        await self.session.flush()

    async def history(self, session_id: str, limit: int = 20) -> list[ChatMessageRecord]:
        result = await self.session.scalars(
            select(ChatMessageRecord)
            .where(ChatMessageRecord.session_id == session_id)
            .order_by(ChatMessageRecord.created_at.desc())
            .limit(limit)
        )
        return list(reversed(list(result)))

