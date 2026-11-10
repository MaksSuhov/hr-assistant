import json
import sqlite3

from pydantic import SecretStr

from hr_assistant.core.config import Settings
from hr_assistant.infrastructure.db.database import Database
from hr_assistant.infrastructure.db.models import CandidateRecord


def _create_legacy_database(path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE candidates (
            id VARCHAR(36) PRIMARY KEY,
            full_name VARCHAR(300),
            email VARCHAR(320),
            phone VARCHAR(80),
            status VARCHAR(20) NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE resume_documents (
            id VARCHAR(36) PRIMARY KEY,
            candidate_id VARCHAR(36) NOT NULL,
            original_filename VARCHAR(512) NOT NULL,
            storage_path VARCHAR(1024) NOT NULL,
            source VARCHAR(500) NOT NULL,
            uploaded_by VARCHAR(200) NOT NULL,
            extracted_text TEXT NOT NULL,
            indexing_status VARCHAR(30) NOT NULL,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE audit_events (
            id VARCHAR(36) PRIMARY KEY,
            actor_id VARCHAR(200) NOT NULL,
            action VARCHAR(100) NOT NULL,
            entity_type VARCHAR(100) NOT NULL,
            entity_id VARCHAR(36) NOT NULL,
            details JSON NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    connection.execute(
        "INSERT INTO candidates (id, full_name, email, phone, status) "
        "VALUES ('c1', 'Legacy Candidate', 'Old@Example.com', '8 (999) 123-45-67', 'interview')"
    )
    connection.execute(
        "INSERT INTO resume_documents "
        "(id, candidate_id, original_filename, storage_path, source, uploaded_by, "
        "extracted_text, indexing_status) "
        "VALUES ('r1', 'c1', 'old.pdf', '/tmp/old.pdf', 'HR upload', 'hr@example.org', "
        "'Python developer', 'indexed')"
    )
    connection.commit()
    connection.close()


async def test_legacy_candidate_status_is_archived_and_schema_is_upgraded(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_legacy_database(db_path)
    settings = Settings(
        _env_file=None,
        app_env="test",
        auth_secret_key=SecretStr("test-signing-key-that-is-long-enough-32"),
        database_url=f"sqlite+aiosqlite:///{db_path}",
        storage_dir=tmp_path / "resumes",
        rate_limit_enabled=False,
    )
    database = Database(settings)

    await database.initialize()
    async with database.unit_of_work() as uow:
        candidate = await uow.candidates.get("c1")
        assert candidate is not None
        assert candidate.email_normalized == "old@example.com"
        assert candidate.phone_normalized == "79991234567"
        await uow.candidates.add(CandidateRecord(id="c2", full_name="New Candidate"))
        await uow.commit()
    await database.close()

    connection = sqlite3.connect(db_path)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(candidates)")}
    assert "status" not in columns
    assert "email_normalized" in columns
    resume_columns = {row[1] for row in connection.execute("PRAGMA table_info(resume_documents)")}
    assert "content_hash" in resume_columns
    assert "indexing_error" in resume_columns
    audit_rows = connection.execute(
        "SELECT action, details FROM audit_events WHERE action = ?",
        ("legacy_candidate_status_archived",),
    ).fetchall()
    assert audit_rows
    assert json.loads(audit_rows[0][1])["legacy_status"] == "interview"
    content_hash = connection.execute(
        "SELECT content_hash FROM resume_documents WHERE id = 'r1'"
    ).fetchone()[0]
    assert content_hash
    connection.close()
