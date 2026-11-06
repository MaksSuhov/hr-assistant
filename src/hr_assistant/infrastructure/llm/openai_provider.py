from openai import AsyncOpenAI

from hr_assistant.core.config import Settings


class OpenAIProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: AsyncOpenAI | None = None
        if settings.openai_configured:
            self.client = AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                base_url=settings.openai_base_url,
            )

    @property
    def configured(self) -> bool:
        return self.client is not None

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

