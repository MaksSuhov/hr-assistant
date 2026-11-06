from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from hr_assistant.core.config import Settings


class LangfuseTracer:
    def __init__(self, settings: Settings) -> None:
        self._client = None
        if settings.langfuse_configured:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=settings.langfuse_public_key.get_secret_value(),
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                base_url=settings.langfuse_base_url,
            )

    @contextmanager
    def observation(self, name: str, input_data: dict[str, Any]) -> Iterator[Any | None]:
        if self._client is None:
            yield None
            return
        with self._client.start_as_current_observation(
            as_type="agent", name=name, input=input_data
        ) as observation:
            yield observation

    def flush(self) -> None:
        if self._client:
            self._client.shutdown()
