import logging
import re
from typing import Any

from hr_assistant.application.ports import ResumeReranker, ResumeVectorStore, UnitOfWorkProvider
from hr_assistant.core.config import Settings
from hr_assistant.infrastructure.llm.openai_provider import OpenAIProvider
from hr_assistant.schemas.matching import CandidateSearchHit

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(
        self,
        uow_provider: UnitOfWorkProvider,
        vector_store: ResumeVectorStore,
        llm: OpenAIProvider | None = None,
        settings: Settings | None = None,
        reranker: ResumeReranker | None = None,
    ) -> None:
        self.uow_provider = uow_provider
        self.vector_store = vector_store
        self.llm = llm
        self.settings = settings
        self.reranker = reranker

    async def search_raw(
        self,
        query: str,
        limit: int = 10,
        candidate_ids: list[str] | None = None,
        deep: bool = True,
    ) -> list[dict[str, Any]]:
        """Search resumes by text.

        ``deep=True`` adds the expensive steps: a HyDE query rewrite and local
        reranking. Batch consumers such as matching call it with ``deep=False``
        to avoid one LLM call per query and trust the hybrid RRF order instead.
        """
        settings = self.settings
        enhancements_enabled = deep and self._enhancements_enabled
        search_query = query
        if enhancements_enabled and settings is not None and settings.hyde_enabled:
            try:
                hypothetical_document = await self._hypothetical_document(query)
                search_query = f"{query}\n{hypothetical_document}"
            except Exception:
                logger.exception("HyDE query generation failed; using the original query")
        try:
            reranking_enabled = deep and self._reranking_enabled
            retrieval_limit = (
                max(limit, settings.reranking_candidate_pool)
                if reranking_enabled and settings is not None
                else limit
            )
            hits = await self.vector_store.search(search_query, retrieval_limit, candidate_ids)
            if hits:
                return hits[:limit] if not deep else await self._rerank(query, hits, limit)
        except Exception:
            logger.exception("Hybrid resume search failed; falling back to local text search")
        return await self._lexical_fallback(query, limit, candidate_ids)

    @property
    def _enhancements_enabled(self) -> bool:
        return bool(self.llm and self.llm.configured and self.settings)

    @property
    def _reranking_enabled(self) -> bool:
        return bool(
            self.reranker is not None
            and self.settings is not None
            and self.settings.reranking_enabled
        )

    async def _hypothetical_document(self, query: str) -> str:
        if self.llm is None or self.settings is None:
            return query
        response = await self.llm.require_client().chat.completions.create(
            model=self.settings.search_enhancement_model,
            temperature=0.2,
            max_tokens=220,
            extra_body=self.llm.chat_extra_body,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Составь гипотетический фрагмент резюме кандидата, который мог бы ответить "
                        "на поисковый запрос. Это поисковый текст, а не факт о реальном кандидате. "
                        "Сохрани названия технологий, должностей и навыков. Верни только "
                        "текст фрагмента, без вводных слов и объяснений."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        text = response.choices[0].message.content if response.choices else None
        return text.strip() if text and text.strip() else query

    async def _rerank(
        self, query: str, hits: list[dict[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        reranker = self.reranker
        if reranker is None or not self._reranking_enabled or len(hits) < 2:
            return hits[:limit]
        documents = [str(hit.get("text", "")) for hit in hits]
        try:
            ranked = await reranker.rerank(query, documents, limit)
        except Exception as error:
            # The reranker logs the root cause with a traceback once; repeating
            # it for every search would bury the useful lines in the log.
            logger.warning("Local reranking failed; preserving hybrid search order: %s", error)
            return hits[:limit]
        selected = [
            {**hits[index], "score": score} for index, score in ranked if 0 <= index < len(hits)
        ]
        return selected or hits[:limit]

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
