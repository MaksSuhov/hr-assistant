from sqlalchemy.ext.asyncio import AsyncSession

from hr_assistant.infrastructure.db.repositories.approval_repository import ApprovalRepository
from hr_assistant.infrastructure.db.repositories.audit_repository import AuditRepository
from hr_assistant.infrastructure.db.repositories.candidate_repository import CandidateRepository
from hr_assistant.infrastructure.db.repositories.chat_repository import ChatRepository
from hr_assistant.infrastructure.db.repositories.resume_repository import ResumeRepository
from hr_assistant.infrastructure.db.repositories.vacancy_repository import VacancyRepository


class SqlAlchemyUnitOfWork:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.vacancies = VacancyRepository(session)
        self.candidates = CandidateRepository(session)
        self.resumes = ResumeRepository(session)
        self.audit = AuditRepository(session)
        self.approvals = ApprovalRepository(session)
        self.chats = ChatRepository(session)

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()

