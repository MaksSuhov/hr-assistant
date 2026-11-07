import asyncio
import json
import math
import time
from typing import Any


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


async def evaluate_performance(
    records: list[dict[str, Any]],
    base_url: str,
    actor_id: str = "eval-runner",
    concurrency: int = 1,
) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as error:
        raise RuntimeError(
            "Performance evaluation requires project dependencies. Run `python -m pip install -e .` "
            "in the active Python environment."
        ) from error

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(concurrency)
    endpoint = f"{base_url.rstrip('/')}/api/v1/chat/stream"
    timeout = httpx.Timeout(300.0, connect=15.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async def measure(index: int, record: dict[str, Any]) -> dict[str, Any]:
            message = record.get("question", record.get("query", record.get("message")))
            if not isinstance(message, str) or not message.strip():
                raise ValueError(f"Record {index + 1} needs a non-empty question, query, or message")
            payload: dict[str, Any] = {"message": message}
            conversation_id = record.get("conversation_id")
            if conversation_id:
                payload["conversation_id"] = str(conversation_id)
            headers = {"X-Actor-Id": actor_id, "X-Eval-Metrics": "true"}
            async with semaphore:
                started = time.perf_counter()
                async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", errors="replace")
                        raise RuntimeError(
                            f"Chat endpoint returned HTTP {response.status_code}: {body[:500]}"
                        )
                    event_name = "message"
                    server_metrics = None
                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            event_name = line.partition(":")[2].strip()
                        elif line.startswith("data:"):
                            data = line.partition(":")[2].strip()
                            if event_name == "eval_metrics":
                                server_metrics = json.loads(data)
                            elif event_name == "error":
                                error_data = json.loads(data)
                                raise RuntimeError(
                                    f"Chat stream failed for sample {index + 1}: "
                                    f"{error_data.get('message', 'unknown error')}"
                                )
                            event_name = "message"
                client_latency_ms = (time.perf_counter() - started) * 1000
            if not isinstance(server_metrics, dict):
                raise RuntimeError(
                    "The server did not emit eval_metrics. Ensure the project version with "
                    "X-Eval-Metrics support is running."
                )
            return {
                "id": str(record.get("id", f"sample-{index + 1}")),
                **server_metrics,
                "client_latency_ms": round(client_latency_ms, 3),
            }

        per_sample = await asyncio.gather(
            *(measure(index, record) for index, record in enumerate(records))
        )

    metrics: dict[str, dict[str, float | None]] = {}
    for key in ("ttft_ms", "ttfrt_ms", "latency_ms", "client_latency_ms"):
        values = [float(row[key]) for row in per_sample if row.get(key) is not None]
        metrics[key] = {
            "mean": sum(values) / len(values) if values else None,
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return {
        "group": "performance",
        "samples": len(per_sample),
        "concurrency": concurrency,
        "metrics": metrics,
        "per_sample": per_sample,
    }
