"""Read the live candidate corpus from the application database.

The judge and the fresh-case pipeline both need more than the fragments the
retriever happened to return. This module loads the newest resumes from the
same SQLite database the API uses, de-duplicates them per candidate, and
renders a compact snapshot that fits into a single LLM prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.engine import make_url

from hr_assistant.core.config import Settings
from hr_assistant.infrastructure.db.database import Database
from hr_assistant.infrastructure.db.models import CandidateRecord, ResumeDocumentRecord

DEFAULT_CORPUS_LIMIT = 20
DEFAULT_CHARS_PER_RESUME = 1600
DEFAULT_TOTAL_CORPUS_CHARS = 60_000

_WHITESPACE = re.compile(r"[ \t\u00a0]+")


@dataclass(frozen=True)
class CorpusEntry:
    """The newest resume of one candidate."""

    candidate_id: str
    candidate_name: str | None
    resume_id: str
    source_filename: str
    text: str
    uploaded_at: datetime | None


@dataclass(frozen=True)
class Corpus:
    entries: tuple[CorpusEntry, ...]
    total_candidates: int

    def __len__(self) -> int:
        return len(self.entries)


async def load_corpus(settings: Settings, limit: int | None = DEFAULT_CORPUS_LIMIT) -> Corpus:
    """Load the freshest resumes, newest first, one resume per candidate.

    ``limit`` counts candidates rather than documents: a candidate who uploaded
    three resumes contributes one evaluation case, built from the newest file.
    """

    if limit is not None and limit < 1:
        raise ValueError("corpus limit must be at least 1")

    database_path = _sqlite_database_path(settings)
    if database_path is not None and not database_path.exists():
        # Booting the app would create an empty database; an evaluator must not
        # mistake that for "the corpus is empty".
        raise FileNotFoundError(
            f"Candidate database not found at {database_path}. Run the evaluator "
            "inside the api container, or point DATABASE_URL at the real database."
        )

    database = Database(settings)
    try:
        await database.initialize()
        async with database.session() as session:
            total_candidates = await session.scalar(
                select(func.count()).select_from(CandidateRecord)
            )
            rows = (
                await session.execute(
                    select(ResumeDocumentRecord, CandidateRecord)
                    .join(
                        CandidateRecord,
                        CandidateRecord.id == ResumeDocumentRecord.candidate_id,
                    )
                    .order_by(
                        ResumeDocumentRecord.uploaded_at.desc(),
                        ResumeDocumentRecord.id.desc(),
                    )
                )
            ).all()
    finally:
        await database.close()

    entries: list[CorpusEntry] = []
    seen: set[str] = set()
    for document, candidate in rows:
        if document.candidate_id in seen:
            continue
        seen.add(document.candidate_id)
        entries.append(
            CorpusEntry(
                candidate_id=document.candidate_id,
                candidate_name=candidate.full_name,
                resume_id=document.id,
                source_filename=document.original_filename,
                text=str(document.extracted_text or ""),
                uploaded_at=document.uploaded_at,
            )
        )
        if limit is not None and len(entries) >= limit:
            break

    return Corpus(entries=tuple(entries), total_candidates=int(total_candidates or 0))


def corpus_snapshot(
    corpus: Corpus,
    *,
    max_chars_per_resume: int = DEFAULT_CHARS_PER_RESUME,
    max_total_chars: int = DEFAULT_TOTAL_CORPUS_CHARS,
) -> dict[str, Any]:
    """Render a prompt-sized, JSON-friendly view of the whole corpus.

    The returned mapping is passed to the judge verbatim. ``truncated`` tells
    the judge that absent candidates may simply not have fitted the budget, so
    it can avoid treating them as disproven.
    """

    if max_chars_per_resume < 1:
        raise ValueError("max_chars_per_resume must be at least 1")
    if max_total_chars < 1:
        raise ValueError("max_total_chars must be at least 1")

    rendered: list[dict[str, Any]] = []
    budget = max_total_chars
    truncated = False
    for entry in corpus.entries:
        if budget <= 0:
            truncated = True
            break
        text = _normalize(entry.text)
        limit = min(max_chars_per_resume, budget)
        clipped = text[:limit]
        entry_truncated = len(text) > len(clipped)
        truncated = truncated or entry_truncated
        rendered.append(
            {
                "candidate_id": entry.candidate_id,
                "candidate_name": entry.candidate_name,
                "source_filename": entry.source_filename,
                "text": clipped,
                "truncated": entry_truncated,
            }
        )
        budget -= len(clipped)

    if len(rendered) < len(corpus.entries):
        truncated = True

    return {
        "entries": rendered,
        "included_candidates": len(rendered),
        "total_candidates": corpus.total_candidates,
        "truncated": truncated,
    }


def _normalize(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return _WHITESPACE.sub(" ", "\n".join(line for line in lines if line)).strip()


def _sqlite_database_path(settings: Settings) -> Path | None:
    if not settings.database_url.startswith("sqlite"):
        return None
    database = make_url(settings.database_url).database
    if not database or database == ":memory:":
        return None
    return Path(database)
