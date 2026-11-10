import asyncio
import logging
import math
import threading
import time
from typing import Any

from hr_assistant.core.config import Settings

logger = logging.getLogger(__name__)

# Cross-encoder models cap at 512 tokens and fastembed already truncates to the
# model's context. Cutting long chunks down first keeps tokenization cheap.
MAX_DOCUMENT_CHARS = 2000
# A missing onnxruntime install or an unsupported model name will not fix itself,
# so a failed load is cached instead of being retried on every search. A
# transient download error is retried after this cooldown.
LOAD_RETRY_AFTER_SECONDS = 300.0


def relevance_from_logit(value: float) -> float:
    """Map a cross-encoder logit to a 0..1 relevance score."""

    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    factor = math.exp(value)
    return factor / (1.0 + factor)


class OnnxCrossEncoderReranker:
    """Local cross-encoder reranker backed by fastembed's ONNX runtime.

    The model is loaded on the first search so application startup does not wait
    for a ~1 GB download, and every inference call runs in a worker thread because
    ONNX Runtime is synchronous and CPU bound.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._encoder: Any | None = None
        self._load_error: BaseException | None = None
        self._failed_at = 0.0
        # Guards the one-time model download and load against concurrent searches.
        self._lock = threading.Lock()

    async def rerank(self, query: str, documents: list[str], limit: int) -> list[tuple[int, float]]:
        if not documents or limit <= 0:
            return []
        prepared = [document[:MAX_DOCUMENT_CHARS] for document in documents]
        scores = await asyncio.to_thread(self._score, query, prepared)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [(index, relevance_from_logit(score)) for index, score in ranked[:limit]]

    def _score(self, query: str, documents: list[str]) -> list[float]:
        encoder = self._get_encoder()
        return [
            float(score)
            for score in encoder.rerank(
                query, documents, batch_size=self.settings.reranking_batch_size
            )
        ]

    def _get_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        with self._lock:
            if self._encoder is not None:
                return self._encoder
            self._raise_if_recently_failed()
            try:
                from fastembed.common.types import Device
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                logger.info(
                    "Loading local reranker model %s; the first run downloads it.",
                    self.settings.reranking_model,
                )
                self._encoder = TextCrossEncoder(
                    model_name=self.settings.reranking_model,
                    cache_dir=(
                        str(self.settings.reranking_cache_dir)
                        if self.settings.reranking_cache_dir
                        else None
                    ),
                    cuda=Device(self.settings.reranking_device),
                )
            except Exception as error:
                self._load_error = error
                self._failed_at = time.monotonic()
                logger.exception(
                    "Local reranker model %s could not be loaded.",
                    self.settings.reranking_model,
                )
                raise
            return self._encoder

    def _raise_if_recently_failed(self) -> None:
        if self._load_error is None:
            return
        if time.monotonic() - self._failed_at < LOAD_RETRY_AFTER_SECONDS:
            raise RuntimeError(
                f"The local reranker model {self.settings.reranking_model} is unavailable."
            ) from self._load_error
        self._load_error = None
