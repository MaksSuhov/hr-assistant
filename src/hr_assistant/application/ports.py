from types import TracebackType
from typing import Any, Protocol, Self


class UnitOfWork(Protocol):
    vacancies: Any
    candidates: Any
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

    async def search(
        self, query: str, limit: int, candidate_ids: list[str] | None = None
    ) -> list[dict[str, Any]]: ...


class ResumeSearch(Protocol):
    async def search_raw(
        self, query: str, limit: int = 10, candidate_ids: list[str] | None = None
    ) -> list[dict[str, Any]]: ...


class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def close(self) -> None: ...


class MessageSender(Protocol):
    @property
    def configured(self) -> bool: ...

    async def send(self, recipient: str, subject: str, body: str) -> None: ...
