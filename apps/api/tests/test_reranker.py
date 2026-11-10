import pytest

from hr_assistant.infrastructure.vector.reranker import (
    MAX_DOCUMENT_CHARS,
    OnnxCrossEncoderReranker,
    relevance_from_logit,
)


class _StubEncoder:
    """Stands in for TextCrossEncoder so tests never download the model."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, documents: list[str], batch_size: int) -> list[float]:
        self.calls.append((query, list(documents), batch_size))
        return list(self.scores)


async def test_rerank_orders_by_logit_and_maps_scores_to_the_unit_interval(settings):
    reranker = OnnxCrossEncoderReranker(settings)
    reranker._encoder = _StubEncoder([-2.0, 3.0, 0.0])  # type: ignore[attr-defined]

    ranked = await reranker.rerank("python", ["a", "b", "c"], limit=2)

    assert [index for index, _ in ranked] == [1, 2]
    assert ranked[0][1] > 0.9
    assert ranked[1][1] == pytest.approx(0.5)


def test_relevance_from_logit_stays_in_the_unit_interval():
    assert relevance_from_logit(0.0) == 0.5
    assert 0.0 < relevance_from_logit(-10.0) < 0.5
    assert 0.5 < relevance_from_logit(10.0) < 1.0
    assert relevance_from_logit(1000.0) == 1.0
    assert relevance_from_logit(-1000.0) == 0.0


async def test_rerank_truncates_long_documents_before_scoring(settings):
    reranker = OnnxCrossEncoderReranker(settings)
    encoder = _StubEncoder([1.0])
    reranker._encoder = encoder  # type: ignore[attr-defined]

    await reranker.rerank("python", ["x" * (MAX_DOCUMENT_CHARS + 500)], limit=1)

    assert len(encoder.calls[0][1][0]) == MAX_DOCUMENT_CHARS
    assert encoder.calls[0][2] == settings.reranking_batch_size


async def test_rerank_returns_nothing_for_empty_input(settings):
    reranker = OnnxCrossEncoderReranker(settings)

    assert await reranker.rerank("python", [], limit=5) == []
    assert await reranker.rerank("python", ["doc"], limit=0) == []


async def test_failed_model_load_is_not_retried_on_every_search(settings, monkeypatch):
    attempts: list[dict[str, object]] = []

    class BrokenTextCrossEncoder:
        def __init__(self, **kwargs: object) -> None:
            attempts.append(kwargs)
            raise RuntimeError("the model could not be downloaded")

    monkeypatch.setattr("fastembed.rerank.cross_encoder.TextCrossEncoder", BrokenTextCrossEncoder)
    reranker = OnnxCrossEncoderReranker(settings)

    with pytest.raises(RuntimeError):
        await reranker.rerank("python", ["doc"], limit=1)
    with pytest.raises(RuntimeError):
        await reranker.rerank("python", ["doc"], limit=1)

    assert len(attempts) == 1
    assert attempts[0]["model_name"] == settings.reranking_model
