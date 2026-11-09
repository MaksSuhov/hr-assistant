from collections.abc import Iterable
from typing import Any


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def evaluate_retrieval(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    hit_rates: list[float] = []
    reciprocal_ranks: list[float] = []
    average_precisions: list[float] = []
    true_positives = false_positives = false_negatives = 0
    candidate_mode = any("retrieved_candidate_ids" in record for record in records) or any(
        ("relevant_candidate_ids" in record or "expected_candidate_ids" in record)
        and "relevant_context_ids" not in record
        and "expected_context_ids" not in record
        for record in records
    )
    precision_name = "precision" if candidate_mode else "context_precision"
    recall_name = "recall" if candidate_mode else "context_recall"

    for index, record in enumerate(records):
        if candidate_mode:
            relevant_raw = record.get(
                "relevant_candidate_ids", record.get("expected_candidate_ids")
            )
            retrieved_raw = record.get("retrieved_candidate_ids")
        else:
            relevant_raw = record.get("relevant_context_ids", record.get("expected_context_ids"))
            retrieved_raw = record.get("retrieved_context_ids")
        if retrieved_raw is None:
            if candidate_mode:
                continue
            contexts = record.get("contexts", [])
            if not isinstance(contexts, list):
                raise ValueError(f"Record {index + 1}: contexts must be a list")
            retrieved_raw = [
                context.get("id")
                for context in contexts
                if isinstance(context, dict) and context.get("id") is not None
            ]
            retrieved_raw.extend(context for context in contexts if not isinstance(context, dict))
        if relevant_raw is None or not isinstance(relevant_raw, list):
            continue
        if not isinstance(retrieved_raw, list):
            raise ValueError(
                f"{record.get('id', f'sample-{index + 1}')}: retrieved ids must be a list"
            )

        relevant = set(_unique(relevant_raw))
        ranked = _unique(retrieved_raw)
        retrieved = set(ranked)
        tp = len(relevant & retrieved)
        fp = len(retrieved - relevant)
        fn = len(relevant - retrieved)
        precision = tp / len(retrieved) if retrieved else 0.0
        recall = tp / len(relevant) if relevant else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        hit_rate = float(tp > 0)
        first_relevant_rank = next(
            (rank for rank, context_id in enumerate(ranked, 1) if context_id in relevant), None
        )
        reciprocal_rank = 1 / first_relevant_rank if first_relevant_rank else 0.0
        precision_at_relevant = [
            sum(context_id in relevant for context_id in ranked[:rank]) / rank
            for rank, context_id in enumerate(ranked, 1)
            if context_id in relevant
        ]
        average_precision = sum(precision_at_relevant) / len(relevant) if relevant else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        hit_rates.append(hit_rate)
        reciprocal_ranks.append(reciprocal_rank)
        average_precisions.append(average_precision)
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        per_sample.append(
            {
                "id": str(record.get("id", f"sample-{index + 1}")),
                precision_name: precision,
                recall_name: recall,
                "f1": f1,
                "hit_rate": hit_rate,
                "reciprocal_rank": reciprocal_rank,
                "average_precision": average_precision,
            }
        )

    if not per_sample:
        raise ValueError(
            "No retrieval records found. Add relevant/retrieved candidate IDs or "
            "relevant/retrieved context IDs to the dataset."
        )

    micro_precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    micro_recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "group": "retrieval",
        "samples": len(per_sample),
        "metrics": {
            precision_name: _mean(precisions),
            recall_name: _mean(recalls),
            "f1_macro_over_queries": _mean(f1s),
            "f1_micro": micro_f1,
            "hit_rate": _mean(hit_rates),
            "mrr": _mean(reciprocal_ranks),
            "map": _mean(average_precisions),
        },
        "per_sample": per_sample,
    }
