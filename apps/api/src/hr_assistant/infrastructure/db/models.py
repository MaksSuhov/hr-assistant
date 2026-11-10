from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UserRecord(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    password_hash: Mapped[str | None] = mapped_column(String(512))
    role: Mapped[str] = mapped_column(String(30), default="recruiter", nullable=False)
    oauth_provider: Mapped[str | None] = mapped_column(String(50))
    oauth_subject: Mapped[str | None] = mapped_column(String(255), unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RefreshTokenRecord(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class VacancyRecord(Base):
    __tablename__ = "vacancies"
    # `updated_at` uses a server-side `onupdate`; eager defaults fetch it in the
    # same statement. Without this, reading the record after UPDATE triggers a
    # lazy load outside the async greenlet and raises MissingGreenlet.
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    department: Mapped[str | None] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="open", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class RequirementRecord(Base):
    __tablename__ = "vacancy_requirements"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    vacancy_id: Mapped[str] = mapped_column(
        ForeignKey("vacancies.id", ondelete="CASCADE"), index=True, nullable=False
    )
    text: Mapped[str] = mapped_column(String(1000), nullable=False)
    is_must_have: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    weight: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class CandidateRecord(Base):
    __tablename__ = "candidates"
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    full_name: Mapped[str | None] = mapped_column(String(300))
    email: Mapped[str | None] = mapped_column(String(320))
    # Lower-cased email and digits-only phone used for de-duplication. They are
    # kept next to the display values so lookups do not scan every candidate.
    email_normalized: Mapped[str | None] = mapped_column(String(320), index=True)
    phone: Mapped[str | None] = mapped_column(String(80))
    phone_normalized: Mapped[str | None] = mapped_column(String(24), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ApplicationRecord(Base):
    """One candidate's progress through one vacancy.

    The pipeline stage lives here rather than on the candidate, so the same
    person can be at different stages in different vacancies.
    """

    __tablename__ = "applications"
    __table_args__ = (
        UniqueConstraint("candidate_id", "vacancy_id", name="uq_applications_candidate_vacancy"),
    )
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), index=True, nullable=False
    )
    vacancy_id: Mapped[str] = mapped_column(
        ForeignKey("vacancies.id", ondelete="CASCADE"), index=True, nullable=False
    )
    stage: Mapped[str] = mapped_column(String(20), default="new", nullable=False, index=True)
    stage_changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ResumeDocumentRecord(Base):
    __tablename__ = "resume_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), index=True, nullable=False
    )
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    source: Mapped[str] = mapped_column(String(500), default="HR upload", nullable=False)
    uploaded_by: Mapped[str] = mapped_column(String(200), nullable=False)
    extracted_text: Mapped[str] = mapped_column(Text, nullable=False)
    # SHA-256 of whitespace-normalised text. Two uploads of the same document
    # point at the same value, which makes duplicate detection cheap.
    content_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(20), default="pdf-text", nullable=False)
    indexing_status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)
    indexing_error: Mapped[str | None] = mapped_column(String(500))
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AuditEventRecord(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    actor_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ApprovalActionRecord(Base):
    __tablename__ = "approval_actions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    requested_by: Mapped[str] = mapped_column(String(200), nullable=False)
    resolved_by: Mapped[str | None] = mapped_column(String(200))
    resolution_note: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class OutboxMessageRecord(Base):
    __tablename__ = "outbox_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    approval_id: Mapped[str] = mapped_column(
        ForeignKey("approval_actions.id"), unique=True, nullable=False
    )
    candidate_id: Mapped[str] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False
    )
    recipient: Mapped[str] = mapped_column(String(320), nullable=False)
    subject: Mapped[str] = mapped_column(String(300), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="queued", nullable=False)
    last_error: Mapped[str | None] = mapped_column(String(500))
    queued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChatSessionRecord(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    actor_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ChatMessageRecord(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
