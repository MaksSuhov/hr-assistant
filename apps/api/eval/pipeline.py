"""The fresh-case evaluation pipeline.

One invocation does the whole loop: read the newest resumes from the database,
turn each into a question with a reference answer, ask the assistant through
its normal chat path, and score the result with a judge that can see the entire
candidate corpus instead of only the retrieved fragments.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from eval.cases import EvalCase, build_cases
from eval.corpus import (
    DEFAULT_CHARS_PER_RESUME,
    DEFAULT_CORPUS_LIMIT,
    DEFAULT_TOTAL_CORPUS_CHARS,
    corpus_snapshot,
    load_corpus,
)
from eval.judge import evaluate_judge
from hr_assistant.core.config import Settings

DEFAULT_CONCURRENCY = 4


class ChatStream(Protocol):
    @property
    def configured(self) -> bool: ...

    def stream_chat(
        self,
        message: str,
        conversation_id: str | None,
        actor_id: str,
        include_eval_metrics: bool = False,
    ) -> Any: ...


async def collect_answers(
    chat: ChatStream,
    cases: list[EvalCase] | tuple[EvalCase, ...],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    actor_id: str = "eval-runner",
) -> list[dict[str, Any]]:
    """Ask the assistant every case and capture the answer plus its contexts."""

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(concurrency)

    async def ask(case: EvalCase) -> dict[str, Any]:
        answer_parts: list[str] = []
        contexts: list[dict[str, Any]] = []
        timing: dict[str, Any] | None = None
        error: str | None = None
        async with semaphore:
            try:
                async for event, payload in chat.stream_chat(
                    case.question,
                    None,
                    actor_id,
                    include_eval_metrics=True,
                ):
                    if event == "contexts":
                        contexts = list(payload.get("contexts", []))
                    elif event == "token":
                        answer_parts.append(str(payload.get("text", "")))
                    elif event == "eval_metrics":
                        timing = dict(payload)
            except Exception as failure:  # one failed case must not kill the run
                error = f"{type(failure).__name__}: {failure}"[:300]
        return {
            **case.as_record(),
            "answer": "".join(answer_parts),
            "contexts": contexts,
            "timing": timing,
            "error": error,
        }

    return list(await asyncio.gather(*(ask(case) for case in cases)))


async def run_pipeline(
    settings: Settings,
    *,
    limit: int = DEFAULT_CORPUS_LIMIT,
    generator: str = "llm",
    case_model: str | None = None,
    judge_model: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_chars_per_resume: int = DEFAULT_CHARS_PER_RESUME,
    max_total_chars: int = DEFAULT_TOTAL_CORPUS_CHARS,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run fresh cases end to end and return one JSON-ready report."""

    if limit < 1:
        raise ValueError("limit must be at least 1")

    # Cases come from the freshest candidates, but the judge sees the whole
    # database: that is what lets it notice a missed candidate or a claim that
    # contradicts a resume the retriever never returned.
    corpus = await load_corpus(settings, limit=None)
    if not corpus.entries:
        raise ValueError(
            "The candidate database has no resumes. Upload resumes first, "
            "or pass a dataset to `python -m eval judge`."
        )
    fresh_entries = corpus.entries[:limit]

    cases = await build_cases(
        fresh_entries,
        settings,
        generator=generator,
        model=case_model,
        concurrency=concurrency,
    )
    snapshot = corpus_snapshot(
        corpus,
        max_chars_per_resume=max_chars_per_resume,
        max_total_chars=max_total_chars,
    )
    report: dict[str, Any] = {
        "group": "pipeline",
        "mode": "fresh_cases",
        "generator": generator,
        "cases": len(cases),
        "corpus": {
            "loaded_candidates": len(corpus.entries),
            "cases_from_freshest": len(fresh_entries),
            "included_candidates": snapshot["included_candidates"],
            "total_candidates": snapshot["total_candidates"],
            "truncated": snapshot["truncated"],
        },
        "answer_model": settings.openai_chat_model,
        "judge_model": judge_model or settings.eval_judge_model or settings.openai_chat_model,
    }
    if dry_run:
        report["mode"] = "dry_run"
        report["questions"] = [case.as_record() for case in cases]
        return report

    from hr_assistant.container import ServiceContainer

    container = ServiceContainer(settings)
    try:
        await container.database.initialize()
        if not container.chat.configured:
            raise RuntimeError(
                "Set OPENAI_API_KEY to run the fresh-case pipeline: it must call the "
                "assistant to produce answers. Use --dry-run to only build the cases."
            )
        answered = await collect_answers(
            container.chat,
            cases,
            concurrency=concurrency,
        )
    finally:
        await container.close()

    judged = await evaluate_judge(
        answered,
        settings,
        report["judge_model"],
        concurrency,
        corpus=snapshot,
    )
    report["metrics"] = judged["metrics"]
    report["per_case"] = _merge(answered, judged["per_sample"])
    report["judge"] = judged
    return report


def _merge(answered: list[dict[str, Any]], scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores_by_id = {str(item.get("id")): item for item in scored}
    merged: list[dict[str, Any]] = []
    for record in answered:
        judged = scores_by_id.get(str(record.get("id")), {})
        merged.append(
            {
                "id": record.get("id"),
                "candidate_id": record.get("candidate_id"),
                "candidate_name": record.get("candidate_name"),
                "question_type": record.get("question_type"),
                "question": record.get("question"),
                "reference_answer": record.get("reference_answer"),
                "evidence": record.get("evidence"),
                "generator": record.get("generator"),
                "generator_error": record.get("generator_error"),
                "answer": record.get("answer"),
                "retrieved_contexts": len(record.get("contexts", [])),
                "timing": record.get("timing"),
                "error": record.get("error"),
                "scores": {key: value for key, value in judged.items() if key not in {"id"}},
            }
        )
    return merged
