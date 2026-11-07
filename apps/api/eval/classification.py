from collections import defaultdict
from typing import Any


def evaluate_classification(records: list[dict[str, Any]]) -> dict[str, Any]:
    confusion: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    sample_count = 0
    for index, record in enumerate(records):
        expected_raw = record.get("expected_labels", record.get("expected_candidate_ids"))
        predicted_raw = record.get("predicted_labels", record.get("predicted_candidate_ids"))
        if expected_raw is None or predicted_raw is None:
            continue
        if not isinstance(expected_raw, list) or not isinstance(predicted_raw, list):
            raise ValueError(f"Record {index + 1}: expected and predicted labels must be lists")
        expected = {str(label) for label in expected_raw}
        predicted = {str(label) for label in predicted_raw}
        for label in expected | predicted:
            tp, fp, fn = confusion[label]
            confusion[label] = [
                tp + int(label in expected and label in predicted),
                fp + int(label not in expected and label in predicted),
                fn + int(label in expected and label not in predicted),
            ]
        sample_count += 1

    if not sample_count:
        raise ValueError(
            "No classification records found. Add expected_labels and predicted_labels "
            "(or expected_candidate_ids and predicted_candidate_ids)."
        )

    total_tp = sum(counts[0] for counts in confusion.values())
    total_fp = sum(counts[1] for counts in confusion.values())
    total_fn = sum(counts[2] for counts in confusion.values())
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    class_f1s: list[float] = []
    supports: list[int] = []
    for tp, fp, fn in confusion.values():
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        class_f1s.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        supports.append(tp + fn)
    support_total = sum(supports)
    macro_f1 = sum(class_f1s) / len(class_f1s) if class_f1s else 0.0
    weighted_f1 = (
        sum(f1 * support for f1, support in zip(class_f1s, supports, strict=True))
        / support_total
        if support_total
        else 0.0
    )
    return {
        "group": "classification",
        "samples": sample_count,
        "labels": len(confusion),
        "metrics": {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
        },
    }
