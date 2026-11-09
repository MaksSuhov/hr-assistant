import asyncio
from typing import Any


async def fetch_live_retrieval(
    records: list[dict[str, Any]],
    base_url: str,
    limit: int | None = None,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    try:
        import httpx
    except ImportError as error:
        raise RuntimeError(
            "Live retrieval requires project dependencies. Run `python -m pip install -e .` "
            "in the active Python environment."
        ) from error

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(concurrency)
    endpoint = f"{base_url.rstrip('/')}/api/v1/candidates/search"
    timeout = httpx.Timeout(120.0, connect=15.0)

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def search(index: int, record: dict[str, Any]) -> dict[str, Any]:
            query = record.get("query", record.get("question"))
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"Record {index + 1} needs a non-empty query or question")
            relevant = record.get("relevant_candidate_ids", record.get("expected_candidate_ids"))
            if not isinstance(relevant, list):
                raise ValueError(f"Record {index + 1} needs relevant_candidate_ids")
            top_k = limit if limit is not None else int(record.get("limit", 10))
            if not 1 <= top_k <= 50:
                raise ValueError(f"Record {index + 1}: retrieval limit must be between 1 and 50")

            async with semaphore:
                response = await client.post(
                    endpoint,
                    json={"query": query, "limit": top_k},
                )
            if response.is_error:
                raise RuntimeError(
                    f"Candidate search returned HTTP {response.status_code}: {response.text[:500]}"
                )
            hits = response.json()
            if not isinstance(hits, list):
                raise ValueError("Candidate search must return a list of hits")
            candidate_ids = []
            seen = set()
            for hit in hits:
                if not isinstance(hit, dict) or hit.get("candidate_id") is None:
                    continue
                candidate_id = str(hit["candidate_id"])
                if candidate_id not in seen:
                    seen.add(candidate_id)
                    candidate_ids.append(candidate_id)
            return {
                **record,
                "relevant_candidate_ids": relevant,
                "retrieved_candidate_ids": candidate_ids,
            }

        return await asyncio.gather(
            *(search(index, record) for index, record in enumerate(records))
        )
