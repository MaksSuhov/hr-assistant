import logging
import re
from typing import Any

from hr_assistant.application.ports import ResumeVectorStore, UnitOfWorkProvider
from hr_assistant.schemas.matching import CandidateSearchHit

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self, uow_provider: UnitOfWorkProvider, vector_store: ResumeVectorStore) -> None:
        self.uow_provider = uow_provider
        self.vector_store = vector_store

    async def search_raw(
        self, query: str, limit: int = 10, candidate_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        try:
            hits = await self.vector_store.search(query, limit, candidate_ids)
            if hits:
                return hits
        except Exception:
            logger.exception("Hybrid resume search failed; falling back to local text search")
        return await self._lexical_fallback(query, limit, candidate_ids)

    async def search(self, query: str, limit: int = 10) -> list[CandidateSearchHit]:
        hits = await self.search_raw(query, limit)
        async with self.uow_provider() as uow:
            candidates = {}
            result: list[CandidateSearchHit] = []
            seen: set[tuple[str, str]] = set()
            for hit in hits:
                candidate_id = str(hit.get("candidate_id", ""))
                document_id = str(hit.get("resume_id", ""))
                key = (candidate_id, document_id)
                if not candidate_id or not document_id or key in seen:
                    continue
                seen.add(key)
                if candidate_id not in candidates:
                    candidates[candidate_id] = await uow.candidates.get(candidate_id)
                candidate = candidates[candidate_id]
                result.append(
                    CandidateSearchHit(
                        candidate_id=candidate_id,
                        candidate_name=candidate.full_name if candidate else None,
                        document_id=document_id,
                        source_filename=str(hit.get("source_filename", "")),
                        snippet=str(hit.get("text", ""))[:1200],
                        score=hit.get("score"),
                    )
                )
            return result[:limit]

    async def _lexical_fallback(
        self, query: str, limit: int, candidate_ids: list[str] | None
    ) -> list[dict[str, Any]]:
        terms = {term for term in re.findall(r"[\w+#.-]+", query.casefold()) if len(term) > 1}
        if not terms:
            return []
        async with self.uow_provider() as uow:
            documents = await uow.resumes.list_all()
            candidates = set(candidate_ids) if candidate_ids else None
            hits: list[dict[str, Any]] = []
            for document in documents:
                if candidates is not None and document.candidate_id not in candidates:
                    continue
                text = document.extracted_text
                normalized = text.casefold()
                matched = [term for term in terms if term in normalized]
                if not matched:
                    continue
                first_position = min(normalized.find(term) for term in matched)
                start = max(0, first_position - 240)
                snippet = text[start : start + 1200]
                hits.append(
                    {
                        "candidate_id": document.candidate_id,
                        "resume_id": document.id,
                        "source_filename": document.original_filename,
                        "text": snippet,
                        "score": len(matched) / len(terms),
                    }
                )
            return sorted(hits, key=lambda hit: hit["score"], reverse=True)[:limit]

