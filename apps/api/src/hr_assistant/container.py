from hr_assistant.application.approval_service import ApprovalService
from hr_assistant.application.auth import AuthService
from hr_assistant.application.candidate_service import CandidateService
from hr_assistant.application.chat_service import ChatService
from hr_assistant.application.matching_service import MatchingService
from hr_assistant.application.search_service import SearchService
from hr_assistant.application.tool_executor import ToolExecutor
from hr_assistant.application.vacancy_service import VacancyService
from hr_assistant.core.config import Settings
from hr_assistant.infrastructure.db.database import Database
from hr_assistant.infrastructure.document.pdf_extractor import PdfTextExtractor
from hr_assistant.infrastructure.document.text_chunker import ResumeTextChunker
from hr_assistant.infrastructure.llm.openai_provider import OpenAIProvider
from hr_assistant.infrastructure.messages.smtp_sender import SmtpMessageSender
from hr_assistant.infrastructure.observability.langfuse_tracer import LangfuseTracer
from hr_assistant.infrastructure.vector.qdrant_store import QdrantResumeStore


class ServiceContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = Database(settings)
        self.auth = AuthService(self.database, settings)
        self.llm = OpenAIProvider(settings)
        self.vector_store = QdrantResumeStore(settings, self.llm)
        self.tracer = LangfuseTracer(settings)
        self.message_sender = SmtpMessageSender(settings)
        self.extractor = PdfTextExtractor()
        self.chunker = ResumeTextChunker()

        uow_provider = self.database.unit_of_work
        self.vacancies = VacancyService(uow_provider)
        self.candidates = CandidateService(
            uow_provider,
            settings,
            self.extractor,
            self.chunker,
            self.vector_store,
        )
        self.search = SearchService(uow_provider, self.vector_store)
        self.matching = MatchingService(uow_provider, self.search, self.llm, settings)
        self.approvals = ApprovalService(uow_provider, self.message_sender)
        self.tool_executor = ToolExecutor(self.candidates, self.search, self.matching)
        self.chat = ChatService(
            uow_provider,
            self.search,
            self.tool_executor,
            self.llm,
            self.tracer,
            settings,
        )

    async def close(self) -> None:
        self.tracer.flush()
        await self.llm.close()
        await self.vector_store.close()
        await self.database.close()
