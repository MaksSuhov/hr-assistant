from types import TracebackType
from typing import Any, Protocol, Self


class UnitOfWork(Protocol):
    vacancies: Any
    candidates: Any
    applications: Any
    resumes: Any
    audit: Any
    approvals: Any
    chats: Any

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class UnitOfWorkProvider(Protocol):
    def __call__(self) -> Any: ...


class ResumeVectorStore(Protocol):
    async def index_resume(
        self,
        *,
        resume_id: str,
        candidate_id: str,
        source_filename: str,
        chunks: list[str],
    ) -> None: ...

    async def delete_resume(self, resume_id: str) -> None: ...

    async def delete_candidate(self, candidate_id: str) -> None: ...

    async def reassign_candidate(self, source_id: str, target_id: str) -> None: ...

    async def search(
        self, query: str, limit: int, candidate_ids: list[str] | None = None
    ) -> list[dict[str, Any]]: ...


class ResumeSearch(Protocol):
    async def search_raw(
        self,
        query: str,
        limit: int = 10,
        candidate_ids: list[str] | None = None,
        deep: bool = True,
    ) -> list[dict[str, Any]]: ...


class ResumeReranker(Protocol):
    """Order resume chunks by their relevance to a query.

    Implementations return ``(index, score)`` pairs for the best ``limit``
    documents, sorted by descending score. The index refers to the position in
    ``documents`` and the score is a relevance value in the 0..1 range.
    """

    async def rerank(
        self, query: str, documents: list[str], limit: int
    ) -> list[tuple[int, float]]: ...


class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def close(self) -> None: ...


class MessageSender(Protocol):
    @property
    def configured(self) -> bool: ...

    async def send(self, recipient: str, subject: str, body: str) -> None: ...
