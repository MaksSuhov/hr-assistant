from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from hr_assistant.application.audit import record_event
from hr_assistant.application.errors import NotFoundError
from hr_assistant.application.ports import ResumeVectorStore, UnitOfWorkProvider
from hr_assistant.core.config import Settings
from hr_assistant.infrastructure.db.models import CandidateRecord, ResumeDocumentRecord
from hr_assistant.infrastructure.document.pdf_extractor import PdfTextExtractor
from hr_assistant.infrastructure.document.text_chunker import ResumeTextChunker
from hr_assistant.schemas.candidate import AuditEventOut, CandidateDocumentOut, CandidateOut


class CandidateService:
    def __init__(
        self,
        uow_provider: UnitOfWorkProvider,
        settings: Settings,
        extractor: PdfTextExtractor,
        chunker: ResumeTextChunker,
        vector_store: ResumeVectorStore,
    ) -> None:
        self.uow_provider = uow_provider
        self.settings = settings
        self.extractor = extractor
        self.chunker = chunker
        self.vector_store = vector_store

    async def upload_resume(
        self,
        *,
        content: bytes,
        filename: str,
        actor_id: str,
        candidate_id: str | None,
        candidate_name: str | None,
        source: str,
    ) -> CandidateOut:
        if len(content) > self.settings.max_upload_bytes:
            raise ValueError(f"PDF exceeds the {self.settings.max_upload_bytes}-byte upload limit.")
        if not filename.lower().endswith(".pdf") or not content.startswith(b"%PDF-"):
            raise ValueError("Upload a valid PDF file.")
        try:
            extracted = await asyncio.to_thread(self.extractor.extract, content)
        except Exception as error:
            raise ValueError("Could not read PDF content.") from error
        if not extracted.text:
            raise ValueError("No selectable text was found in the PDF.")

        resume_id = str(uuid.uuid4())
        resolved_candidate_id = candidate_id or str(uuid.uuid4())
        storage_path = self.settings.storage_dir / f"{resume_id}.pdf"
        await asyncio.to_thread(storage_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(storage_path.write_bytes, content)

        try:
            async with self.uow_provider() as uow:
                candidate = await uow.candidates.get(resolved_candidate_id)
                if candidate_id and not candidate:
                    raise NotFoundError(f"Candidate {candidate_id} was not found.")
                if not candidate:
                    candidate = CandidateRecord(
                        id=resolved_candidate_id,
                        full_name=(candidate_name or "").strip() or None,
                        status="new",
                    )
                    await uow.candidates.add(candidate)
                document = ResumeDocumentRecord(
                    id=resume_id,
                    candidate_id=resolved_candidate_id,
                    original_filename=Path(filename).name[:512],
                    storage_path=str(storage_path),
                    source=source.strip()[:500] or "HR upload",
                    uploaded_by=actor_id,
                    extracted_text=extracted.text,
                    indexing_status="pending",
                )
                await uow.resumes.add(document)
                await uow.commit()
        except Exception:
            await asyncio.to_thread(storage_path.unlink, missing_ok=True)
            raise

        chunks = [document.page_content for document in self.chunker.split(extracted.text)]
        indexing_status = "indexed"
        indexing_error = None
        try:
            await self.vector_store.index_resume(
                resume_id=resume_id,
                candidate_id=resolved_candidate_id,
                source_filename=Path(filename).name,
                chunks=chunks,
            )
        except Exception as error:  # the source resume remains stored and can be re-indexed later
            indexing_status = "failed"
            indexing_error = type(error).__name__

        async with self.uow_provider() as uow:
            await uow.resumes.update_indexing_status(resume_id, indexing_status)
            await record_event(
                uow,
                actor_id=actor_id,
                action="resume_uploaded",
                entity_type="candidate",
                entity_id=resolved_candidate_id,
                details={
                    "resume_id": resume_id,
                    "filename": Path(filename).name,
                    "source": source,
                    "uploaded_by": actor_id,
                    "pages": extracted.page_count,
                    "characters_extracted": len(extracted.text),
                    "text_preview": extracted.text[:400],
                    "chunks_created": len(chunks),
                    "indexing_status": indexing_status,
                    "indexing_error": indexing_error,
                },
            )
            await uow.commit()
            candidate = await uow.candidates.get(resolved_candidate_id)
            return await self._to_output(uow, candidate)

    async def get(self, candidate_id: str) -> CandidateOut:
        async with self.uow_provider() as uow:
            candidate = await uow.candidates.get(candidate_id)
            if not candidate:
                raise NotFoundError(f"Candidate {candidate_id} was not found.")
            return await self._to_output(uow, candidate)

    async def list(self, limit: int = 100) -> list[CandidateOut]:
        async with self.uow_provider() as uow:
            records = await uow.candidates.list(limit)
            return [await self._to_output(uow, record) for record in records]

    async def history(self, candidate_id: str) -> list[AuditEventOut]:
        async with self.uow_provider() as uow:
            candidate = await uow.candidates.get(candidate_id)
            if not candidate:
                raise NotFoundError(f"Candidate {candidate_id} was not found.")
            events = await uow.audit.list_for_entity("candidate", candidate_id)
            return [AuditEventOut.model_validate(event) for event in events]

    @staticmethod
    async def _to_output(uow, candidate: CandidateRecord) -> CandidateOut:
        documents = await uow.resumes.list_for_candidate(candidate.id)
        return CandidateOut(
            id=candidate.id,
            full_name=candidate.full_name,
            email=candidate.email,
            phone=candidate.phone,
            status=candidate.status,
            created_at=candidate.created_at,
            updated_at=candidate.updated_at,
            sources=[
                CandidateDocumentOut(
                    id=document.id,
                    filename=document.original_filename,
                    source=document.source,
                    uploaded_by=document.uploaded_by,
                    uploaded_at=document.uploaded_at,
                    indexing_status=document.indexing_status,
                )
                for document in documents
            ],
        )
