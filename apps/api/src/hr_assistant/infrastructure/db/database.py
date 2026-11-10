from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event, inspect, text
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from hr_assistant.core.config import Settings
from hr_assistant.domain.contacts import content_hash, normalize_email, normalize_phone
from hr_assistant.infrastructure.db.models import Base
from hr_assistant.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if settings.database_url.startswith("sqlite"):
            db_path = make_url(settings.database_url).database
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        if settings.database_url.startswith("sqlite"):
            event.listen(self.engine.sync_engine, "connect", _enable_sqlite_foreign_keys)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            await connection.run_sync(_upgrade_legacy_schema)

    async def close(self) -> None:
        await self.engine.dispose()

    @asynccontextmanager
    async def unit_of_work(self) -> AsyncIterator[SqlAlchemyUnitOfWork]:
        async with self.session_factory() as session:
            yield SqlAlchemyUnitOfWork(session)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
    """SQLite ignores ``ON DELETE CASCADE`` until foreign keys are enabled."""

    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _upgrade_legacy_schema(connection: Connection) -> None:
    """Bring a database created before applications/de-duplication up to date.

    The project intentionally ships without Alembic, so startup performs the
    few additive changes that ``create_all`` cannot handle. Every step is
    idempotent and safe to run on a fresh database.
    """

    inspector = inspect(connection)
    tables = set(inspector.get_table_names())
    is_sqlite = connection.dialect.name == "sqlite"

    if "candidates" in tables:
        columns = {column["name"] for column in inspector.get_columns("candidates")}
        _add_column(connection, "candidates", columns, "email_normalized", "VARCHAR(320)")
        _add_column(connection, "candidates", columns, "phone_normalized", "VARCHAR(24)")
        if "status" in columns:
            _archive_legacy_candidate_status(connection, is_sqlite)
        _create_index(connection, "candidates", "email_normalized")
        _create_index(connection, "candidates", "phone_normalized")
        _backfill_candidate_contacts(connection)

    if "resume_documents" in tables:
        columns = {column["name"] for column in inspector.get_columns("resume_documents")}
        _add_column(connection, "resume_documents", columns, "content_hash", "VARCHAR(64)")
        _add_column(
            connection,
            "resume_documents",
            columns,
            "extraction_method",
            "VARCHAR(20) NOT NULL DEFAULT 'pdf-text'",
        )
        _add_column(connection, "resume_documents", columns, "indexing_error", "VARCHAR(500)")
        _create_index(connection, "resume_documents", "content_hash")
        _backfill_resume_hashes(connection)


def _add_column(
    connection: Connection, table: str, existing: set[str], name: str, ddl_type: str
) -> None:
    if name in existing:
        return
    connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl_type}"))
    existing.add(name)


def _create_index(connection: Connection, table: str, column: str) -> None:
    connection.execute(
        text(f"CREATE INDEX IF NOT EXISTS ix_{table}_{column} ON {table} ({column})")
    )


def _archive_legacy_candidate_status(connection: Connection, is_sqlite: bool) -> None:
    """Preserve old candidate-level stages, then drop the obsolete column.

    Applications require a vacancy id, which a pre-migration candidate status
    does not carry. Rather than silently discarding that data, every non-"new"
    status is written to the audit log first. New databases never reach this.
    """

    rows = connection.execute(
        text("SELECT id, status FROM candidates WHERE status IS NOT NULL AND status != 'new'")
    ).fetchall()
    for candidate_id, status in rows:
        connection.execute(
            text(
                "INSERT INTO audit_events "
                "(id, actor_id, action, entity_type, entity_id, details) "
                "VALUES (:id, :actor, :action, :entity_type, :entity_id, :details)"
            ),
            {
                "id": str(uuid.uuid4()),
                "actor": "system-migration",
                "action": "legacy_candidate_status_archived",
                "entity_type": "candidate",
                "entity_id": candidate_id,
                "details": json.dumps({"legacy_status": status}),
            },
        )
    if not is_sqlite:
        logger.warning(
            "candidates.status is obsolete but was left in place on a non-SQLite database; "
            "drop it manually after migrating the data."
        )
        return
    try:
        connection.execute(text("ALTER TABLE candidates DROP COLUMN status"))
    except Exception:  # pragma: no cover - depends on the bundled SQLite version
        logger.exception(
            "Could not drop the obsolete candidates.status column; "
            "old rows can still be read but inserts may fail until it is removed."
        )


def _backfill_candidate_contacts(connection: Connection) -> None:
    rows = connection.execute(
        text(
            "SELECT id, email, phone, email_normalized, phone_normalized FROM candidates "
            "WHERE (email IS NOT NULL AND email_normalized IS NULL) "
            "OR (phone IS NOT NULL AND phone_normalized IS NULL)"
        )
    ).fetchall()
    for candidate_id, email, phone, email_normalized, phone_normalized in rows:
        normalized_email = email_normalized or normalize_email(
            email if isinstance(email, str) else None
        )
        normalized_phone = phone_normalized or normalize_phone(
            phone if isinstance(phone, str) else None
        )
        if normalized_email == email_normalized and normalized_phone == phone_normalized:
            continue
        connection.execute(
            text(
                "UPDATE candidates SET email_normalized = :email, phone_normalized = :phone "
                "WHERE id = :id"
            ),
            {"email": normalized_email, "phone": normalized_phone, "id": candidate_id},
        )


def _backfill_resume_hashes(connection: Connection) -> None:
    rows = connection.execute(
        text(
            "SELECT id, extracted_text, content_hash FROM resume_documents "
            "WHERE content_hash IS NULL"
        )
    ).fetchall()
    for resume_id, extracted_text, _stored_hash in rows:
        connection.execute(
            text("UPDATE resume_documents SET content_hash = :hash WHERE id = :id"),
            {"hash": content_hash(str(extracted_text or "")), "id": resume_id},
        )
