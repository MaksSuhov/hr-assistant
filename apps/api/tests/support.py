"""Deterministic fakes and fixture builders shared by the test suite."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import httpx

ResponseFactory = Callable[..., Any]

ADMIN_CREDENTIALS = {
    "email": "admin@example.org",
    "password": "correct-horse-battery-staple",
    "full_name": "Admin User",
}


def sync_csrf(client: httpx.AsyncClient) -> dict[str, str]:
    token = client.cookies["csrf_token"]
    client.headers["x-csrf-token"] = token
    return {"x-csrf-token": token}


async def register_admin(
    client: httpx.AsyncClient, payload: dict[str, str] | None = None
) -> dict[str, Any]:
    response = await client.post("/api/v1/auth/register", json=payload or ADMIN_CREDENTIALS)
    assert response.status_code == 201, response.text
    sync_csrf(client)
    return response.json()


async def upload_resume(
    client: httpx.AsyncClient,
    pdf: bytes,
    *,
    candidate_name: str | None = "Алексей Иванов",
    candidate_id: str | None = None,
    source: str = "HR upload",
    filename: str = "resume.pdf",
    endpoint: str = "/api/v1/resumes",
    authenticated: bool = True,
    email: str | None = None,
    phone: str | None = None,
    vacancy_id: str | None = None,
) -> httpx.Response:
    if authenticated:
        sync_csrf(client)
    data: dict[str, str] = {"source": source}
    if candidate_name is not None:
        data["candidate_name"] = candidate_name
    if candidate_id is not None:
        data["candidate_id"] = candidate_id
    if email is not None:
        data["email"] = email
    if phone is not None:
        data["phone"] = phone
    if vacancy_id is not None:
        data["vacancy_id"] = vacancy_id
    return await client.post(
        endpoint,
        data=data,
        files={"file": (filename, pdf, "application/pdf")},
    )


class AsyncStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> AsyncStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration as error:
            raise StopAsyncIteration from error


def text_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=None),
            )
        ],
        usage=None,
    )


def tool_call_chunk(name: str, arguments: str, call_id: str = "call-1") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id=call_id,
                            function=SimpleNamespace(name=name, arguments=arguments),
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )


def usage_chunk(prompt: int = 10, completion: int = 5) -> SimpleNamespace:
    # RouterAI ends every stream with a usage-only chunk whose `choices` is empty.
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )


def stream_response(*chunks: Any) -> ResponseFactory:
    def factory(**_kwargs: Any) -> AsyncStream:
        return AsyncStream(list(chunks))

    return factory


def json_response(payload: str | dict[str, Any]) -> ResponseFactory:
    content = payload if isinstance(payload, str) else json_dumps(payload)

    def factory(**_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return factory


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


class FakeCompletions:
    def __init__(self) -> None:
        self.responses: list[ResponseFactory] = []
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError("FakeLLM has no queued response for this call.")
        return self.responses.pop(0)(**kwargs)


class FakeLLM:
    """Duck-typed stand-in for OpenAIProvider."""

    def __init__(self, *, configured: bool = False) -> None:
        self._configured = configured
        self.completions = FakeCompletions()
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=self.completions))
        self.embedded: list[list[str]] = []
        self.closed = False

    @property
    def configured(self) -> bool:
        return self._configured

    @configured.setter
    def configured(self, value: bool) -> None:
        self._configured = value

    @property
    def chat_extra_body(self) -> dict[str, Any]:
        return {}

    def require_client(self) -> SimpleNamespace:
        if not self._configured:
            raise RuntimeError("OpenAI is not configured; set OPENAI_API_KEY to enable it.")
        return self.client

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.embedded.append(list(texts))
        return [[float(index)] * 3 for index, _ in enumerate(texts)]

    async def close(self) -> None:
        self.closed = True


class FakeReranker:
    """Duck-typed stand-in for OnnxCrossEncoderReranker.

    Scores are taken from ``scores`` by document text; documents that are not
    listed score 0.0, so a test only has to name the chunks it cares about.
    """

    def __init__(self) -> None:
        self.scores: dict[str, float] = {}
        self.calls: list[tuple[str, list[str], int]] = []
        self.fail = False

    async def rerank(self, query: str, documents: list[str], limit: int) -> list[tuple[int, float]]:
        self.calls.append((query, list(documents), limit))
        if self.fail:
            raise RuntimeError("reranker is unavailable")
        scored = [(index, self.scores.get(text, 0.0)) for index, text in enumerate(documents)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]


class FakeVectorStore:
    def __init__(self) -> None:
        self.points: list[dict[str, Any]] = []
        self.queries: list[tuple[str, int, list[str] | None]] = []
        self.deleted_resumes: list[str] = []
        self.deleted_candidates: list[str] = []
        self.reassignments: list[tuple[str, str]] = []
        self.fail_index = False
        self.fail_search = False
        self.fail_delete = False
        self.closed = False

    async def ensure_collection(self) -> None:
        return None

    async def index_resume(
        self,
        *,
        resume_id: str,
        candidate_id: str,
        source_filename: str,
        chunks: list[str],
    ) -> None:
        if self.fail_index:
            raise RuntimeError("vector index is unavailable")
        for index, text in enumerate(chunks):
            self.points.append(
                {
                    "candidate_id": candidate_id,
                    "resume_id": resume_id,
                    "source_filename": source_filename,
                    "chunk_index": index,
                    "text": text,
                }
            )

    async def delete_resume(self, resume_id: str) -> None:
        if self.fail_delete:
            raise RuntimeError("vector store is unavailable")
        self.deleted_resumes.append(resume_id)
        self.points = [point for point in self.points if point["resume_id"] != resume_id]

    async def delete_candidate(self, candidate_id: str) -> None:
        if self.fail_delete:
            raise RuntimeError("vector store is unavailable")
        self.deleted_candidates.append(candidate_id)
        self.points = [point for point in self.points if point["candidate_id"] != candidate_id]

    async def reassign_candidate(self, source_id: str, target_id: str) -> None:
        if self.fail_delete:
            raise RuntimeError("vector store is unavailable")
        self.reassignments.append((source_id, target_id))
        for point in self.points:
            if point["candidate_id"] == source_id:
                point["candidate_id"] = target_id

    async def search(
        self,
        query: str,
        limit: int,
        candidate_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if self.fail_search:
            raise RuntimeError("vector search is unavailable")
        self.queries.append((query, limit, candidate_ids))
        terms = {term for term in query.casefold().split() if term}
        scored: list[tuple[float, dict[str, Any]]] = []
        for point in self.points:
            if candidate_ids is not None and point["candidate_id"] not in candidate_ids:
                continue
            text = str(point["text"]).casefold()
            score = sum(1 for term in terms if term in text) / len(terms) if terms else 0.0
            if score > 0:
                scored.append((score, {**point, "score": score}))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in scored[:limit]]

    async def close(self) -> None:
        self.closed = True


class _Span:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class FakeTracer:
    """No-op tracer that keeps spans so tests can inspect what was recorded."""

    enabled = False

    def __init__(self) -> None:
        self.spans: list[_Span] = []
        self.flushed = False

    @contextmanager
    def trace(self, _name: str, _input_data: Any = None, **_kwargs: Any) -> Iterator[_Span]:
        span = _Span()
        self.spans.append(span)
        yield span

    @contextmanager
    def observation(self, _name: str, _input_data: Any = None, **_kwargs: Any) -> Iterator[_Span]:
        span = _Span()
        self.spans.append(span)
        yield span

    @contextmanager
    def generation(self, _name: str, **_kwargs: Any) -> Iterator[_Span]:
        span = _Span()
        self.spans.append(span)
        yield span

    def flush(self) -> None:
        self.flushed = True


class FakeSender:
    def __init__(self, *, configured: bool = False) -> None:
        self._configured = configured
        self.sent: list[tuple[str, str, str]] = []

    @property
    def configured(self) -> bool:
        return self._configured

    @configured.setter
    def configured(self, value: bool) -> None:
        self._configured = value

    async def send(self, recipient: str, subject: str, body: str) -> None:
        if not self._configured:
            raise RuntimeError("SMTP is not configured.")
        self.sent.append((recipient, subject, body))


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content_stream(text: str) -> bytes:
    lines = text.splitlines() or [""]
    parts = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
    for line in lines:
        parts.append(f"({_escape_pdf_text(line)}) Tj")
        parts.append("T*")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1")


def build_pdf(
    text: str = "Python developer with FastAPI experience", *, pages: list[str] | None = None
) -> bytes:
    """Build a minimal valid PDF with a selectable text layer.

    pypdf extracts text written with a standard Helvetica font from this layout,
    so upload tests do not depend on binary fixtures.
    """
    page_texts = pages if pages is not None else [text]
    kids = " ".join(f"{4 + 2 * index} 0 R" for index in range(len(page_texts)))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>".encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for index, page_text in enumerate(page_texts):
        stream = _content_stream(page_text)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * index} 0 R >>"
            ).encode("latin-1")
        )
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode("latin-1")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )

    buffer = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for object_id, body in enumerate(objects, 1):
        offsets.append(len(buffer))
        buffer += f"{object_id} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"
    xref_position = len(buffer)
    buffer += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    buffer += b"0000000000 65535 f \n"
    for offset in offsets:
        buffer += f"{offset:010d} 00000 n \n".encode("latin-1")
    buffer += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_position}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(buffer)


def build_docx(
    text: str = "Python developer with FastAPI", *, table: list[list[str]] | None = None
) -> bytes:
    """Build a DOCX with paragraphs and an optional table."""

    from io import BytesIO

    from docx import Document

    document = Document()
    for line in text.splitlines():
        document.add_paragraph(line)
    if table:
        doc_table = document.add_table(rows=len(table), cols=len(table[0]))
        for row_index, row in enumerate(table):
            for column_index, value in enumerate(row):
                doc_table.cell(row_index, column_index).text = value
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def sse_events(body: str) -> list[tuple[str, str]]:
    """Parse a buffered SSE response into (event, data) pairs."""
    events: list[tuple[str, str]] = []
    event_name = "message"
    data_lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("event:"):
            event_name = line.partition(":")[2].strip()
        elif line.startswith("data:"):
            data_lines.append(line.partition(":")[2].strip())
        elif not line.strip() and data_lines:
            events.append((event_name, "\n".join(data_lines)))
            event_name = "message"
            data_lines = []
    if data_lines:
        events.append((event_name, "\n".join(data_lines)))
    return events
