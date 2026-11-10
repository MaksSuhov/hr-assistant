from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hr_assistant.infrastructure.db.models import ResumeDocumentRecord


class ResumeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, document: ResumeDocumentRecord) -> None:
        self.session.add(document)
        await self.session.flush()

    async def get(self, resume_id: str) -> ResumeDocumentRecord | None:
        return await self.session.get(ResumeDocumentRecord, resume_id)

    async def list_for_candidate(self, candidate_id: str) -> list[ResumeDocumentRecord]:
        result = await self.session.scalars(
            select(ResumeDocumentRecord)
            .where(ResumeDocumentRecord.candidate_id == candidate_id)
            .order_by(ResumeDocumentRecord.uploaded_at.desc())
        )
        return list(result)

    async def find_by_content_hash(self, digest: str) -> ResumeDocumentRecord | None:
        return await self.session.scalar(
            select(ResumeDocumentRecord).where(ResumeDocumentRecord.content_hash == digest)
        )

    async def update_indexing_status(
        self, resume_id: str, status: str, error: str | None = None
    ) -> None:
        document = await self.get(resume_id)
        if document:
            document.indexing_status = status
            document.indexing_error = error
            await self.session.flush()

    async def reassign_candidate(self, source_id: str, target_id: str) -> int:
        documents = await self.list_for_candidate(source_id)
        for document in documents:
            document.candidate_id = target_id
        await self.session.flush()
        return len(documents)

    async def delete(self, document: ResumeDocumentRecord) -> None:
        await self.session.delete(document)
        await self.session.flush()

    async def list_all(self, limit: int = 500) -> list[ResumeDocumentRecord]:
        result = await self.session.scalars(
            select(ResumeDocumentRecord)
            .order_by(ResumeDocumentRecord.uploaded_at.desc())
            .limit(limit)
        )
        return list(result)
