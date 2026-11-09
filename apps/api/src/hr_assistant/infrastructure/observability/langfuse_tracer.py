from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from hr_assistant.core.config import Settings


class LangfuseTracer:
    """Langfuse client wrapper; every method becomes a no-op without credentials."""

    def __init__(self, settings: Settings) -> None:
        self._client = None
        public_key = settings.langfuse_public_key
        secret_key = settings.langfuse_secret_key
        if settings.langfuse_configured and public_key is not None and secret_key is not None:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=public_key.get_secret_value(),
                secret_key=secret_key.get_secret_value(),
                base_url=settings.langfuse_base_url,
            )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def trace(
        self,
        name: str,
        input_data: Any = None,
        *,
        as_type: Any = "agent",
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Iterator[Any | None]:
        """Open the root observation and attach trace-level attributes to it."""
        if self._client is None:
            yield None
            return
        from langfuse import propagate_attributes

        with self._client.start_as_current_observation(
            as_type=as_type, name=name, input=input_data
        ) as observation:
            with propagate_attributes(
                session_id=session_id, user_id=user_id, trace_name=name, tags=tags
            ):
                yield observation

    @contextmanager
    def observation(
        self,
        name: str,
        input_data: Any = None,
        *,
        as_type: Any = "span",
    ) -> Iterator[Any | None]:
        if self._client is None:
            yield None
            return
        with self._client.start_as_current_observation(
            as_type=as_type, name=name, input=input_data
        ) as observation:
            yield observation

    @contextmanager
    def generation(
        self,
        name: str,
        *,
        model: str,
        input_data: Any = None,
    ) -> Iterator[Any | None]:
        if self._client is None:
            yield None
            return
        with self._client.start_as_current_observation(
            as_type="generation", name=name, model=model, input=input_data
        ) as observation:
            yield observation

    def flush(self) -> None:
        if self._client:
            self._client.shutdown()
