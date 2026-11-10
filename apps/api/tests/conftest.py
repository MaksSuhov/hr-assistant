from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from hr_assistant.app import create_app
from hr_assistant.container import ServiceContainer
from hr_assistant.core.config import Settings
from tests.support import (
    FakeLLM,
    FakeReranker,
    FakeSender,
    FakeTracer,
    FakeVectorStore,
    build_pdf,
    register_admin,
    sync_csrf,
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        app_name="HR Assistant Test",
        app_env="test",
        auth_secret_key=SecretStr("test-signing-key-that-is-long-enough-32"),
        auth_cookie_secure=False,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'hr_assistant.db'}",
        storage_dir=tmp_path / "resumes",
        openai_api_key=None,
        openai_base_url=None,
        qdrant_url="http://127.0.0.1:1",
        hyde_enabled=False,
        reranking_enabled=False,
        chat_reasoning_enabled=False,
        rate_limit_enabled=False,
        log_level="CRITICAL",
        max_upload_bytes=1024 * 1024,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def fake_tracer() -> FakeTracer:
    return FakeTracer()


@pytest.fixture
def fake_sender() -> FakeSender:
    return FakeSender()


@pytest.fixture
async def container(
    settings: Settings,
    fake_llm: FakeLLM,
    fake_vector_store: FakeVectorStore,
    fake_reranker: FakeReranker,
    fake_tracer: FakeTracer,
    fake_sender: FakeSender,
) -> AsyncIterator[ServiceContainer]:
    service_container = ServiceContainer(
        settings,
        llm=fake_llm,
        vector_store=fake_vector_store,
        tracer=fake_tracer,
        message_sender=fake_sender,
        reranker=fake_reranker,
    )
    await service_container.database.initialize()
    try:
        yield service_container
    finally:
        await service_container.close()


@pytest.fixture
async def app(settings: Settings, container: ServiceContainer) -> AsyncIterator[FastAPI]:
    yield create_app(settings, container=container)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        yield http


@pytest.fixture
async def auth_client(client: httpx.AsyncClient) -> httpx.AsyncClient:
    await register_admin(client)
    sync_csrf(client)
    return client


@pytest.fixture
def make_pdf() -> Callable[..., bytes]:
    return build_pdf
