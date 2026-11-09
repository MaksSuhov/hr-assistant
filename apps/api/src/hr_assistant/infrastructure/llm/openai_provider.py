from typing import Any

from openai import AsyncOpenAI

from hr_assistant.core.config import Settings


class OpenAIProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: AsyncOpenAI | None = None
        if settings.openai_configured and settings.openai_api_key is not None:
            self.client = AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                base_url=settings.openai_base_url,
            )

    @property
    def configured(self) -> bool:
        return self.client is not None

    def require_client(self) -> AsyncOpenAI:
        if self.client is None:
            raise RuntimeError("OpenAI is not configured; set OPENAI_API_KEY to enable it.")
        return self.client

    @property
    def chat_extra_body(self) -> dict[str, Any]:
        """Provider-specific chat parameters the OpenAI SDK passes through verbatim.

        Reasoning models such as qwen3.7 spend seconds emitting a hidden
        reasoning block before the first visible token. Gateways expose that
        switch with the OpenRouter-style `reasoning` object, which is not a
        standard OpenAI field, so it travels in `extra_body`.
        """
        if self.settings.chat_reasoning_enabled:
            return {}
        return {"reasoning": {"enabled": False}}

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.client:
            raise RuntimeError("OpenAI is not configured; set OPENAI_API_KEY to enable embeddings.")
        response = await self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=texts,
            dimensions=self.settings.openai_embedding_dimensions,
        )
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    async def close(self) -> None:
        if self.client:
            await self.client.close()
