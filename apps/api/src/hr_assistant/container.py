import logging
from typing import Any

from hr_assistant.application.application_service import ApplicationService
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
from hr_assistant.infrastructure.document.docx_extractor import DocxTextExtractor
from hr_assistant.infrastructure.document.extractor import CompositeTextExtractor
from hr_assistant.infrastructure.document.pdf_extractor import (
    PdfTextExtractor,
    TesseractOcrRecognizer,
)
from hr_assistant.infrastructure.document.text_chunker import ResumeTextChunker
from hr_assistant.infrastructure.llm.openai_provider import OpenAIProvider
from hr_assistant.infrastructure.messages.smtp_sender import SmtpMessageSender
from hr_assistant.infrastructure.observability.langfuse_tracer import LangfuseTracer
from hr_assistant.infrastructure.vector.qdrant_store import QdrantResumeStore

logger = logging.getLogger(__name__)


class ServiceContainer:
    def __init__(
        self,
        settings: Settings,
        *,
        llm: Any | None = None,
        vector_store: Any | None = None,
        tracer: Any | None = None,
        message_sender: Any | None = None,
        extractor: Any | None = None,
        chunker: Any | None = None,
    ) -> None:
        self.settings = settings
        self.database = Database(settings)
        self.auth = AuthService(self.database, settings)
        self.llm = llm if llm is not None else OpenAIProvider(settings)
        self.vector_store = (
            vector_store if vector_store is not None else QdrantResumeStore(settings, self.llm)
        )
        self.tracer = tracer if tracer is not None else LangfuseTracer(settings)
        self.message_sender = (
            message_sender if message_sender is not None else SmtpMessageSender(settings)
        )
        self.extractor = extractor if extractor is not None else self._build_extractor(settings)
        self.chunker = chunker if chunker is not None else ResumeTextChunker()

        uow_provider = self.database.unit_of_work
        self.vacancies = VacancyService(uow_provider)
        self.applications = ApplicationService(uow_provider)
        self.candidates = CandidateService(
            uow_provider,
            settings,
            self.extractor,
            self.chunker,
            self.vector_store,
        )
        self.search = SearchService(uow_provider, self.vector_store, self.llm, settings)
        self.matching = MatchingService(uow_provider, self.search, self.llm, settings)
        self.approvals = ApprovalService(uow_provider, self.message_sender)
        self.tool_executor = ToolExecutor(
            self.candidates, self.search, self.matching, self.applications
        )
        self.chat = ChatService(
            uow_provider,
            self.search,
            self.tool_executor,
            self.llm,
            self.tracer,
            settings,
        )

    @staticmethod
    def _build_extractor(settings: Settings) -> CompositeTextExtractor:
        ocr = TesseractOcrRecognizer(settings.ocr_languages) if settings.ocr_enabled else None
        if ocr is not None and not ocr.available:
            logger.warning(
                "OCR is enabled but tesseract is not installed; "
                "scanned PDFs will keep their selectable text only."
            )
            ocr = None
        return CompositeTextExtractor(
            pdf=PdfTextExtractor(
                ocr=ocr,
                min_text_chars=settings.ocr_min_text_chars,
                max_ocr_pages=settings.ocr_max_pages,
            ),
            docx=DocxTextExtractor(),
        )

    async def close(self) -> None:
        self.tracer.flush()
        await self.llm.close()
        await self.vector_store.close()
        await self.database.close()
