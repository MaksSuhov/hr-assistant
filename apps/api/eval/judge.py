from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hr_assistant.core.config import Settings

JUDGE_SYSTEM_PROMPT = """You are a careful evaluator of a Russian-language HR RAG assistant.
Assess only the supplied question, answer, retrieved contexts, and optional reference answer.
Return one JSON object with numeric scores from 0 to 1 for faithfulness, answer_relevancy,
and context_relevancy. faithfulness measures whether answer claims are supported by contexts;
answer_relevancy measures whether the answer addresses the question; context_relevancy is the
mean relevance of retrieved contexts to the question. If no context was retrieved, faithfulness
and context_relevancy are 0. If a reference answer is supplied, also score context_precision as
the fraction of retrieved contexts useful for answering it, and context_recall as the fraction
of its key facts supported by the contexts. If there is no reference answer, return null for both.
Do not reward plausibility when the supplied context does not support a claim. Return no extra keys
or explanatory text."""

JUDGE_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_relevancy",
    "context_precision",
    "context_recall",
)


def _as_contexts(record: dict[str, Any]) -> list[dict[str, str]]:
    contexts = record.get("contexts", [])
    if not isinstance(contexts, list):
        raise ValueError("contexts must be a list")
    result = []
    for index, context in enumerate(contexts):
        if isinstance(context, dict):
            result.append(
                {
                    "id": str(context.get("id", index + 1)),
                    "text": str(context.get("text", "")),
                }
            )
        else:
            result.append({"id": str(index + 1), "text": str(context)})
    return result


def _validate_scores(payload: Any, reference_available: bool) -> dict[str, float | None]:
    if not isinstance(payload, dict):
        raise ValueError("Judge response must be a JSON object")
    scores: dict[str, float | None] = {}
    for metric in JUDGE_METRICS:
        value = payload.get(metric)
        if metric in {"context_precision", "context_recall"} and not reference_available:
            scores[metric] = None
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"Judge response is missing numeric score '{metric}'")
        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Judge score '{metric}' must be between 0 and 1")
        scores[metric] = score
    return scores


async def evaluate_judge(
    records: list[dict[str, Any]],
    settings: Settings,
    model: str,
    concurrency: int = 4,
) -> dict[str, Any]:
    try:
        from openai import AsyncOpenAI, OpenAIError
    except ImportError as error:
        raise RuntimeError(
            "The judge group requires project dependencies. Run `python -m pip install -e .` "
            "in the active Python environment."
        ) from error

    if not settings.openai_configured:
        raise RuntimeError("Set OPENAI_API_KEY to run judge evaluation.")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    client = AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def score_record(index: int, record: dict[str, Any]) -> dict[str, Any]:
        question = record.get("question", record.get("query"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Record {index + 1} needs a non-empty question or query")
        answer = record.get("answer")
        if not isinstance(answer, str):
            raise ValueError(f"Record {index + 1} needs an answer string")
        contexts = _as_contexts(record)
        reference = record.get("reference_answer")
        if reference is not None and not isinstance(reference, str):
            raise ValueError(f"Record {index + 1}: reference_answer must be a string")
        input_data = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "reference_answer": reference,
        }
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
            except OpenAIError as error:
                raise RuntimeError(f"Judge API request failed: {error}") from error
        content = response.choices[0].message.content
        if not content:
            raise ValueError(f"Judge returned an empty response for sample {index + 1}")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError(f"Judge returned invalid JSON for sample {index + 1}") from error
        scores = _validate_scores(parsed, reference_available=reference is not None)
        return {"id": str(record.get("id", f"sample-{index + 1}")), **scores}

    try:
        per_sample = await asyncio.gather(
            *(score_record(index, record) for index, record in enumerate(records))
        )
    finally:
        await client.close()

    totals: dict[str, list[float]] = defaultdict(list)
    for result in per_sample:
        for metric in JUDGE_METRICS:
            value = result[metric]
            if value is not None:
                totals[metric].append(value)
    return {
        "group": "judge",
        "model": model,
        "samples": len(per_sample),
        "metrics": {
            metric: sum(totals[metric]) / len(totals[metric]) if totals[metric] else None
            for metric in JUDGE_METRICS
        },
        "per_sample": per_sample,
    }
