from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from hr_assistant.core.config import Settings
from hr_assistant.infrastructure.db.models import Base
from hr_assistant.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork


class Database:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if settings.database_url.startswith("sqlite"):
            db_path = make_url(settings.database_url).database
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    @asynccontextmanager
    async def unit_of_work(self) -> AsyncIterator["SqlAlchemyUnitOfWork"]:
        async with self.session_factory() as session:
            yield SqlAlchemyUnitOfWork(session)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session
