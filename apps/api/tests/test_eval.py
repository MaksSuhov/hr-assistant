import json

import pytest

from eval.classification import evaluate_classification
from eval.io import load_records, write_report
from eval.judge import _as_contexts, _validate_scores
from eval.performance import _percentile
from eval.retrieval import evaluate_retrieval


def test_load_records_reads_jsonl(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")

    assert load_records(path) == [{"id": "a"}, {"id": "b"}]


def test_load_records_reads_json_arrays_and_single_objects(tmp_path):
    array_path = tmp_path / "records.json"
    array_path.write_text(json.dumps([{"id": "a"}]), encoding="utf-8")
    object_path = tmp_path / "record.json"
    object_path.write_text(json.dumps({"id": "b"}), encoding="utf-8")

    assert load_records(array_path) == [{"id": "a"}]
    assert load_records(object_path) == [{"id": "b"}]


def test_load_records_rejects_broken_datasets(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_records(missing)

    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_records(broken)

    scalar = tmp_path / "scalar.json"
    scalar.write_text("42", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_records(scalar)


def test_write_report_prints_and_persists(tmp_path, capsys):
    destination = tmp_path / "reports" / "report.json"

    write_report({"metric": 1}, destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == {"metric": 1}
    assert '"metric": 1' in capsys.readouterr().out


def test_retrieval_metrics_for_candidates():
    report = evaluate_retrieval(
        [
            {
                "relevant_candidate_ids": ["a", "b"],
                "retrieved_candidate_ids": ["a", "c"],
            }
        ]
    )

    metrics = report["metrics"]
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1_macro_over_queries"] == pytest.approx(0.5)
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["map"] == pytest.approx(0.5)


def test_retrieval_counts_plain_context_strings_as_own_ids():
    report = evaluate_retrieval(
        [
            {
                "relevant_context_ids": ["ctx-1"],
                "contexts": [{"id": "ctx-1", "text": "a"}, "free text"],
            }
        ]
    )

    assert report["metrics"]["context_precision"] == 0.5
    assert report["metrics"]["context_recall"] == 1.0


def test_retrieval_requires_data():
    with pytest.raises(ValueError, match="No retrieval records"):
        evaluate_retrieval([{"id": "empty"}])


def test_classification_metrics():
    report = evaluate_classification([{"expected_labels": ["a"], "predicted_labels": ["a", "b"]}])

    metrics = report["metrics"]
    assert metrics["micro_precision"] == pytest.approx(0.5)
    assert metrics["micro_recall"] == 1.0
    assert metrics["micro_f1"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(0.5)
    assert metrics["weighted_f1"] == pytest.approx(1.0)


def test_classification_requires_data():
    with pytest.raises(ValueError, match="No classification records"):
        evaluate_classification([{"id": "empty"}])


def test_judge_contexts_are_normalized():
    assert _as_contexts({"contexts": [{"id": "x", "text": "one"}, "two"]}) == [
        {"id": "x", "text": "one"},
        {"id": "2", "text": "two"},
    ]
    with pytest.raises(ValueError, match="contexts must be a list"):
        _as_contexts({"contexts": "not-a-list"})


def test_judge_scores_are_validated():
    scores = _validate_scores(
        {
            "faithfulness": 0.5,
            "answer_relevancy": 1,
            "context_relevancy": 0,
            "context_precision": None,
            "context_recall": None,
        },
        reference_available=False,
    )
    assert scores["context_precision"] is None

    with pytest.raises(ValueError, match="missing numeric score"):
        _validate_scores({"faithfulness": 0.5}, reference_available=False)
    with pytest.raises(ValueError, match="between 0 and 1"):
        _validate_scores(
            {
                "faithfulness": 2,
                "answer_relevancy": 1,
                "context_relevancy": 1,
                "context_precision": None,
                "context_recall": None,
            },
            reference_available=False,
        )
    with pytest.raises(ValueError, match="missing numeric score"):
        _validate_scores(
            {
                "faithfulness": True,
                "answer_relevancy": 1,
                "context_relevancy": 1,
                "context_precision": None,
                "context_recall": None,
            },
            reference_available=False,
        )


def test_percentile_handles_empty_and_bounds():
    assert _percentile([], 0.5) is None
    assert _percentile([1, 2, 3, 4], 0.5) == 2
    assert _percentile([1, 2, 3, 4], 0.95) == 4
