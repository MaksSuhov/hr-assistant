from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from hr_assistant.application.audit import record_event
from hr_assistant.application.errors import ConflictError, NotFoundError, ServiceUnavailableError
from hr_assistant.application.ports import ResumeVectorStore, UnitOfWorkProvider
from hr_assistant.core.config import Settings
from hr_assistant.domain.contacts import (
    content_hash,
    extract_email,
    extract_phone,
    normalize_email,
    normalize_phone,
)
from hr_assistant.domain.enums import ApplicationStage, later_stage
from hr_assistant.infrastructure.db.models import (
    ApplicationRecord,
    CandidateRecord,
    ResumeDocumentRecord,
)
from hr_assistant.infrastructure.document.extractor import CompositeTextExtractor
from hr_assistant.infrastructure.document.text_chunker import ResumeTextChunker
from hr_assistant.schemas.application import ApplicationOut, RetentionPurgeResult
from hr_assistant.schemas.candidate import AuditEventOut, CandidateDocumentOut, CandidateOut

logger = logging.getLogger(__name__)


class CandidateService:
    def __init__(
        self,
        uow_provider: UnitOfWorkProvider,
        settings: Settings,
        extractor: CompositeTextExtractor,
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
        email: str | None = None,
        phone: str | None = None,
        vacancy_id: str | None = None,
    ) -> CandidateOut:
        if len(content) > self.settings.max_upload_bytes:
            raise ValueError(
                f"Document exceeds the {self.settings.max_upload_bytes}-byte upload limit."
            )
        try:
            extracted = await asyncio.to_thread(self.extractor.extract, content, filename)
        except ValueError:
            raise
        except Exception as error:
            raise ValueError("Could not read the document content.") from error
        if not extracted.text:
            raise ValueError("No readable text was found in the document.")

        resume_id = str(uuid.uuid4())
        digest = content_hash(extracted.text)
        provided_email = normalize_email(email)
        provided_phone = normalize_phone(phone)
        if email and email.strip() and not provided_email:
            raise ValueError("Enter a valid email address.")
        if phone and phone.strip() and not provided_phone:
            raise ValueError("Enter a phone number with 10 to 15 digits.")
        detected_email = provided_email or extract_email(extracted.text)
        detected_phone = provided_phone or extract_phone(extracted.text)
        storage_path = self.settings.storage_dir / f"{resume_id}{Path(filename).suffix.lower()}"
        await asyncio.to_thread(storage_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(storage_path.write_bytes, content)

        candidate: CandidateRecord | None = None
        duplicate_match: str | None = None
        merged_fields: list[str] = []
        application_created = False
        try:
            async with self.uow_provider() as uow:
                if candidate_id:
                    candidate = await uow.candidates.get(candidate_id)
                    if not candidate:
                        raise NotFoundError(f"Candidate {candidate_id} was not found.")
                else:
                    candidate, duplicate_match = await self._find_duplicate_candidate(
                        uow, email=detected_email, phone=detected_phone, digest=digest
                    )
                if candidate:
                    merged_fields = self._merge_contact_fields(
                        candidate,
                        name=candidate_name,
                        email=detected_email,
                        phone=detected_phone,
                    )
                else:
                    candidate = CandidateRecord(
                        id=str(uuid.uuid4()),
                        full_name=(candidate_name or "").strip() or None,
                        email=detected_email,
                        email_normalized=detected_email,
                        phone=detected_phone,
                        phone_normalized=detected_phone,
                    )
                    await uow.candidates.add(candidate)

                existing_resume = await uow.resumes.find_by_content_hash(digest)
                if existing_resume and existing_resume.candidate_id == candidate.id:
                    raise ConflictError(
                        "This resume was already uploaded for the candidate "
                        f"({existing_resume.id}). Use POST /resumes/{existing_resume.id}/reindex "
                        "instead of uploading it again."
                    )
                if existing_resume:
                    raise ConflictError(
                        "This document is already linked to candidate "
                        f"{existing_resume.candidate_id}. Merge the duplicate candidates first."
                    )

                document = ResumeDocumentRecord(
                    id=resume_id,
                    candidate_id=candidate.id,
                    original_filename=Path(filename).name[:512],
                    storage_path=str(storage_path),
                    source=source.strip()[:500] or "HR upload",
                    uploaded_by=actor_id,
                    extracted_text=extracted.text,
                    content_hash=digest,
                    extraction_method=extracted.extraction_method,
                    indexing_status="pending",
                )
                await uow.resumes.add(document)

                application_id: str | None = None
                if vacancy_id:
                    vacancy = await uow.vacancies.get(vacancy_id)
                    if not vacancy:
                        raise NotFoundError(f"Vacancy {vacancy_id} was not found.")
                    application = await uow.applications.get_for_pair(candidate.id, vacancy_id)
                    if not application:
                        application = ApplicationRecord(
                            id=str(uuid.uuid4()),
                            candidate_id=candidate.id,
                            vacancy_id=vacancy_id,
                            stage=ApplicationStage.NEW.value,
                        )
                        await uow.applications.add(application)
                        application_created = True
                    application_id = application.id
                await uow.commit()
        except Exception:
            await asyncio.to_thread(storage_path.unlink, missing_ok=True)
            raise

        chunks = [item.page_content for item in self.chunker.split(extracted.text)]
        indexing_status = "indexed"
        indexing_error: str | None = None
        try:
            await self.vector_store.index_resume(
                resume_id=resume_id,
                candidate_id=candidate.id,
                source_filename=Path(filename).name,
                chunks=chunks,
            )
        except Exception as error:  # the source resume remains stored and can be re-indexed later
            indexing_status = "failed"
            indexing_error = type(error).__name__

        async with self.uow_provider() as uow:
            await uow.resumes.update_indexing_status(resume_id, indexing_status, indexing_error)
            await record_event(
                uow,
                actor_id=actor_id,
                action="resume_uploaded",
                entity_type="candidate",
                entity_id=candidate.id,
                details={
                    "resume_id": resume_id,
                    "vacancy_id": vacancy_id,
                    "application_id": application_id,
                    "source": source[:500],
                    "uploaded_by": actor_id,
                    "file_type": Path(filename).suffix.lower().lstrip("."),
                    "size_bytes": len(content),
                    "pages": extracted.page_count,
                    "characters_extracted": len(extracted.text),
                    "extraction_method": extracted.extraction_method,
                    "chunks_created": len(chunks),
                    "indexing_status": indexing_status,
                    "indexing_error": indexing_error,
                },
            )
            if duplicate_match:
                await record_event(
                    uow,
                    actor_id=actor_id,
                    action="candidate_deduplicated",
                    entity_type="candidate",
                    entity_id=candidate.id,
                    details={
                        "matched_by": duplicate_match,
                        "merged_fields": merged_fields,
                        "resume_id": resume_id,
                    },
                )
            if application_created and application_id:
                await record_event(
                    uow,
                    actor_id=actor_id,
                    action="application_created",
                    entity_type="candidate",
                    entity_id=candidate.id,
                    details={
                        "application_id": application_id,
                        "vacancy_id": vacancy_id,
                        "stage": ApplicationStage.NEW.value,
                    },
                )
            await uow.commit()
            stored_candidate = await uow.candidates.get(candidate.id)
            if not stored_candidate:
                raise NotFoundError(f"Candidate {candidate.id} was not found.")
            return await self._to_output(uow, stored_candidate)

    async def get(self, candidate_id: str) -> CandidateOut:
        async with self.uow_provider() as uow:
            candidate = await uow.candidates.get(candidate_id)
            if not candidate:
                raise NotFoundError(f"Candidate {candidate_id} was not found.")
            return await self._to_output(uow, candidate)

    async def list_candidates(self, limit: int = 100) -> list[CandidateOut]:
        async with self.uow_provider() as uow:
            records = await uow.candidates.list_all(limit)
            return [await self._to_output(uow, record) for record in records]

    async def history(self, candidate_id: str) -> list[AuditEventOut]:
        async with self.uow_provider() as uow:
            candidate = await uow.candidates.get(candidate_id)
            if not candidate:
                raise NotFoundError(f"Candidate {candidate_id} was not found.")
            events = await uow.audit.list_for_entity("candidate", candidate_id)
            return [AuditEventOut.model_validate(event) for event in events]

    async def reindex_resume(self, resume_id: str, actor_id: str) -> CandidateDocumentOut:
        async with self.uow_provider() as uow:
            document = await uow.resumes.get(resume_id)
            if not document:
                raise NotFoundError(f"Resume {resume_id} was not found.")
            await uow.resumes.update_indexing_status(resume_id, "pending", None)
            await uow.commit()
            candidate_id = document.candidate_id
            filename = document.original_filename
            text = document.extracted_text
            extraction_method = document.extraction_method

        chunks = [item.page_content for item in self.chunker.split(text)]
        indexing_status = "indexed"
        indexing_error: str | None = None
        try:
            # Old points may contain a different chunk count, so remove them
            # before writing the fresh vectors.
            await self.vector_store.delete_resume(resume_id)
            await self.vector_store.index_resume(
                resume_id=resume_id,
                candidate_id=candidate_id,
                source_filename=filename,
                chunks=chunks,
            )
        except Exception as error:
            indexing_status = "failed"
            indexing_error = type(error).__name__

        async with self.uow_provider() as uow:
            await uow.resumes.update_indexing_status(resume_id, indexing_status, indexing_error)
            await record_event(
                uow,
                actor_id=actor_id,
                action="resume_reindexed",
                entity_type="candidate",
                entity_id=candidate_id,
                details={
                    "resume_id": resume_id,
                    "chunks_created": len(chunks),
                    "extraction_method": extraction_method,
                    "indexing_status": indexing_status,
                    "indexing_error": indexing_error,
                },
            )
            await uow.commit()
            document = await uow.resumes.get(resume_id)
            if not document:
                raise NotFoundError(f"Resume {resume_id} was not found.")
            return self._document_output(document)

    async def delete_resume(self, resume_id: str, actor_id: str) -> None:
        async with self.uow_provider() as uow:
            document = await uow.resumes.get(resume_id)
            if not document:
                raise NotFoundError(f"Resume {resume_id} was not found.")
            candidate_id = document.candidate_id
            storage_path = Path(document.storage_path)
            source = document.source
        await self._delete_vectors(self.vector_store.delete_resume, resume_id)
        await asyncio.to_thread(storage_path.unlink, missing_ok=True)
        async with self.uow_provider() as uow:
            document = await uow.resumes.get(resume_id)
            if document:
                await uow.resumes.delete(document)
            await record_event(
                uow,
                actor_id=actor_id,
                action="resume_deleted",
                entity_type="candidate",
                entity_id=candidate_id,
                details={"resume_id": resume_id, "source": source},
            )
            await uow.commit()

    async def delete_candidate(self, candidate_id: str, actor_id: str) -> None:
        async with self.uow_provider() as uow:
            candidate = await uow.candidates.get(candidate_id)
            if not candidate:
                raise NotFoundError(f"Candidate {candidate_id} was not found.")
            documents = await uow.resumes.list_for_candidate(candidate_id)
            storage_paths = [Path(document.storage_path) for document in documents]
            application_count = len(await uow.applications.list_for_candidate(candidate_id))
        await self._delete_vectors(self.vector_store.delete_candidate, candidate_id)
        for path in storage_paths:
            await asyncio.to_thread(path.unlink, missing_ok=True)
        async with self.uow_provider() as uow:
            candidate = await uow.candidates.get(candidate_id)
            if candidate:
                await uow.candidates.delete(candidate)
            await record_event(
                uow,
                actor_id=actor_id,
                action="candidate_deleted",
                entity_type="candidate",
                entity_id=candidate_id,
                details={
                    "resumes_deleted": len(storage_paths),
                    "applications_deleted": application_count,
                },
            )
            await uow.commit()

    async def merge_candidates(
        self, *, target_id: str, source_id: str, actor_id: str
    ) -> CandidateOut:
        if target_id == source_id:
            raise ValueError("Choose two different candidates to merge.")
        async with self.uow_provider() as uow:
            target = await uow.candidates.get(target_id)
            source = await uow.candidates.get(source_id)
            if not target:
                raise NotFoundError(f"Candidate {target_id} was not found.")
            if not source:
                raise NotFoundError(f"Candidate {source_id} was not found.")

        # Move vector payloads first: if the database transaction fails the
        # source still exists, and the merge can be retried safely.
        await self._delete_vectors(self.vector_store.reassign_candidate, source_id, target_id)

        async with self.uow_provider() as uow:
            target = await uow.candidates.get(target_id)
            source = await uow.candidates.get(source_id)
            if not target or not source:
                raise NotFoundError("One of the candidates disappeared during the merge.")
            merged_fields = self._merge_profile(target, source)
            moved_resumes = await uow.resumes.reassign_candidate(source_id, target_id)
            moved_applications = await self._merge_applications(uow, source_id, target_id)
            moved_messages = await uow.approvals.reassign_outbox(source_id, target_id)
            await uow.candidates.delete(source)
            await record_event(
                uow,
                actor_id=actor_id,
                action="candidates_merged",
                entity_type="candidate",
                entity_id=target_id,
                details={
                    "source_candidate_id": source_id,
                    "resumes_moved": moved_resumes,
                    "applications_moved": moved_applications,
                    "outbox_messages_moved": moved_messages,
                    "merged_fields": merged_fields,
                },
            )
            await uow.commit()
            return await self._to_output(uow, target)

    async def purge_expired(
        self,
        *,
        days: int | None = None,
        audit_days: int | None = None,
        actor_id: str = "retention-policy",
    ) -> RetentionPurgeResult:
        """Delete candidates whose latest activity is older than the window.

        A vector-store failure leaves the candidate in place and counts it as
        failed, so a later run can finish the job. Audit events are purged
        separately, after the deletions they describe.
        """

        retention_days = self.settings.resume_retention_days if days is None else days
        audit_retention_days = (
            self.settings.audit_retention_days if audit_days is None else audit_days
        )
        if retention_days <= 0:
            return RetentionPurgeResult(
                deleted_candidates=0, deleted_resumes=0, failed_candidates=0
            )

        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        async with self.uow_provider() as uow:
            candidates = await uow.candidates.list_all(limit=10_000)
            expired: list[tuple[str, int]] = []
            for candidate in candidates:
                documents = await uow.resumes.list_for_candidate(candidate.id)
                latest = candidate.updated_at
                for document in documents:
                    if _as_utc(document.uploaded_at) > _as_utc(latest):
                        latest = document.uploaded_at
                if documents and _as_utc(latest) < cutoff:
                    expired.append((candidate.id, len(documents)))

        deleted_candidates = 0
        deleted_resumes = 0
        failed_candidates = 0
        for candidate_id, resume_count in expired:
            try:
                await self.delete_candidate(candidate_id, actor_id)
            except Exception:
                logger.exception(
                    "Could not purge candidate %s; leaving it for the next run", candidate_id
                )
                failed_candidates += 1
                continue
            deleted_candidates += 1
            deleted_resumes += resume_count

        deleted_audit_events = 0
        async with self.uow_provider() as uow:
            if audit_retention_days > 0:
                audit_cutoff = datetime.now(UTC) - timedelta(days=audit_retention_days)
                deleted_audit_events = await uow.audit.delete_older_than(audit_cutoff)
            await record_event(
                uow,
                actor_id=actor_id,
                action="retention_purge",
                entity_type="system",
                entity_id="retention",
                details={
                    "retention_days": retention_days,
                    "audit_retention_days": audit_retention_days,
                    "deleted_candidates": deleted_candidates,
                    "deleted_resumes": deleted_resumes,
                    "failed_candidates": failed_candidates,
                    "deleted_audit_events": deleted_audit_events,
                },
            )
            await uow.commit()
        return RetentionPurgeResult(
            deleted_candidates=deleted_candidates,
            deleted_resumes=deleted_resumes,
            failed_candidates=failed_candidates,
            deleted_audit_events=deleted_audit_events,
        )

    @staticmethod
    async def _delete_vectors(operation: Any, *args: str) -> None:
        try:
            await operation(*args)
        except Exception as error:
            raise ServiceUnavailableError(
                "Vector index cleanup failed; the stored records were left unchanged. Retry later."
            ) from error

    @staticmethod
    async def _find_duplicate_candidate(
        uow: Any, *, email: str | None, phone: str | None, digest: str
    ) -> tuple[CandidateRecord | None, str | None]:
        if email:
            candidate = await uow.candidates.find_by_email(email)
            if candidate:
                return candidate, "email"
        if phone:
            candidate = await uow.candidates.find_by_phone(phone)
            if candidate:
                return candidate, "phone"
        document = await uow.resumes.find_by_content_hash(digest)
        if document:
            candidate = await uow.candidates.get(document.candidate_id)
            if candidate:
                return candidate, "content_hash"
        return None, None

    @staticmethod
    def _merge_contact_fields(
        candidate: CandidateRecord,
        *,
        name: str | None,
        email: str | None,
        phone: str | None,
    ) -> list[str]:
        merged: list[str] = []
        cleaned_name = (name or "").strip()
        if cleaned_name and not candidate.full_name:
            candidate.full_name = cleaned_name[:300]
            merged.append("full_name")
        if email:
            if not candidate.email:
                candidate.email = email
                merged.append("email")
            if not candidate.email_normalized:
                candidate.email_normalized = email
        if phone:
            if not candidate.phone:
                candidate.phone = phone
                merged.append("phone")
            if not candidate.phone_normalized:
                candidate.phone_normalized = phone
        return merged

    @staticmethod
    def _merge_profile(target: CandidateRecord, source: CandidateRecord) -> list[str]:
        merged: list[str] = []
        for field in ("full_name", "email", "phone", "email_normalized", "phone_normalized"):
            if not getattr(target, field) and getattr(source, field):
                setattr(target, field, getattr(source, field))
                merged.append(field)
        return merged

    @staticmethod
    async def _merge_applications(uow: Any, source_id: str, target_id: str) -> int:
        source_applications = await uow.applications.list_for_candidate(source_id)
        moved = 0
        for application in source_applications:
            existing = await uow.applications.get_for_pair(target_id, application.vacancy_id)
            if existing:
                combined = later_stage(
                    ApplicationStage(existing.stage), ApplicationStage(application.stage)
                )
                if combined.value != existing.stage:
                    existing.stage = combined.value
                    existing.stage_changed_at = datetime.now(UTC)
                await uow.applications.delete(application)
            else:
                application.candidate_id = target_id
            moved += 1
        return moved

    @staticmethod
    async def _to_output(uow: Any, candidate: CandidateRecord) -> CandidateOut:
        documents = await uow.resumes.list_for_candidate(candidate.id)
        application_rows = await uow.applications.list_for_candidate_with_vacancy(candidate.id)
        return CandidateOut(
            id=candidate.id,
            full_name=candidate.full_name,
            email=candidate.email,
            phone=candidate.phone,
            created_at=candidate.created_at,
            updated_at=candidate.updated_at,
            sources=[CandidateService._document_output(document) for document in documents],
            applications=[
                ApplicationOut(
                    id=application.id,
                    candidate_id=application.candidate_id,
                    candidate_name=candidate.full_name,
                    vacancy_id=application.vacancy_id,
                    vacancy_title=vacancy.title,
                    stage=ApplicationStage(application.stage),
                    stage_changed_at=application.stage_changed_at,
                    created_at=application.created_at,
                    updated_at=application.updated_at,
                )
                for application, vacancy in application_rows
            ],
        )

    @staticmethod
    def _document_output(document: ResumeDocumentRecord) -> CandidateDocumentOut:
        return CandidateDocumentOut(
            id=document.id,
            filename=document.original_filename,
            source=document.source,
            uploaded_by=document.uploaded_by,
            uploaded_at=document.uploaded_at,
            indexing_status=document.indexing_status,
            indexing_error=document.indexing_error,
            extraction_method=document.extraction_method,
        )


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
