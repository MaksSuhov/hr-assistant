import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from eval.cases import build_template_case
from eval.classification import evaluate_classification
from eval.corpus import Corpus, CorpusEntry, corpus_snapshot, load_corpus
from eval.io import load_records, write_report
from eval.judge import _as_contexts, _validate_scores
from eval.performance import _percentile
from eval.pipeline import collect_answers
from eval.retrieval import evaluate_retrieval
from hr_assistant.core.config import Settings
from hr_assistant.infrastructure.db.database import Database
from hr_assistant.infrastructure.db.models import CandidateRecord, ResumeDocumentRecord


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
    with pytest.raises(ValueError, match="must be a number"):
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


def _eval_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        auth_secret_key=SecretStr("test-signing-key-that-is-long-enough-32"),
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'eval.db'}",
        openai_api_key=None,
        qdrant_url="http://127.0.0.1:1",
        rate_limit_enabled=False,
        log_level="CRITICAL",
    )


async def _seed_corpus(
    settings: Settings,
    candidates: list[tuple[str, str | None, list[tuple[str, str, datetime]]]],
) -> None:
    database = Database(settings)
    await database.initialize()
    try:
        async with database.session() as session:
            for candidate_id, name, _resumes in candidates:
                session.add(CandidateRecord(id=candidate_id, full_name=name))
            await session.flush()
            for candidate_id, _name, resumes in candidates:
                for resume_id, text, uploaded_at in resumes:
                    session.add(
                        ResumeDocumentRecord(
                            id=resume_id,
                            candidate_id=candidate_id,
                            original_filename=f"{resume_id}.pdf",
                            storage_path=f"/tmp/{resume_id}.pdf",
                            uploaded_by="test",
                            extracted_text=text,
                            content_hash=resume_id.ljust(64, "0"),
                            uploaded_at=uploaded_at,
                        )
                    )
            await session.commit()
    finally:
        await database.close()


async def test_load_corpus_returns_freshest_resume_per_candidate(tmp_path):
    settings = _eval_settings(tmp_path)
    await _seed_corpus(
        settings,
        [
            (
                "c-old",
                "Old Candidate",
                [("r-old-1", "old text", datetime(2026, 1, 1))],
            ),
            (
                "c-new",
                "New Candidate",
                [
                    ("r-new-1", "older resume", datetime(2026, 1, 2)),
                    ("r-new-2", "newest resume", datetime(2026, 1, 3)),
                ],
            ),
        ],
    )

    corpus = await load_corpus(settings, limit=None)

    assert [entry.candidate_id for entry in corpus.entries] == ["c-new", "c-old"]
    assert corpus.entries[0].resume_id == "r-new-2"
    assert corpus.entries[0].candidate_name == "New Candidate"
    assert corpus.total_candidates == 2


async def test_load_corpus_honours_limit_and_rejects_zero(tmp_path):
    settings = _eval_settings(tmp_path)
    await _seed_corpus(
        settings,
        [
            ("c1", "One", [("r1", "one", datetime(2026, 1, 1))]),
            ("c2", "Two", [("r2", "two", datetime(2026, 1, 2))]),
        ],
    )

    corpus = await load_corpus(settings, limit=1)

    assert [entry.candidate_id for entry in corpus.entries] == ["c2"]
    with pytest.raises(ValueError, match="at least 1"):
        await load_corpus(settings, limit=0)


def _entry(candidate_id: str, text: str) -> CorpusEntry:
    return CorpusEntry(
        candidate_id=candidate_id,
        candidate_name=f"Name {candidate_id}",
        resume_id=f"{candidate_id}-resume",
        source_filename=f"{candidate_id}.pdf",
        text=text,
        uploaded_at=None,
    )


def test_corpus_snapshot_truncates_and_reports_loss():
    corpus = Corpus(
        entries=(_entry("c1", "abcdefghij"), _entry("c2", "klmnopqrst")),
        total_candidates=5,
    )

    snapshot = corpus_snapshot(corpus, max_chars_per_resume=5, max_total_chars=8)

    assert [item["text"] for item in snapshot["entries"]] == ["abcde", "klm"]
    assert snapshot["included_candidates"] == 2
    assert snapshot["total_candidates"] == 5
    assert snapshot["truncated"] is True

    clipped = corpus_snapshot(corpus, max_chars_per_resume=5, max_total_chars=4)
    assert [item["text"] for item in clipped["entries"]] == ["abcd"]
    assert clipped["included_candidates"] == 1
    assert clipped["truncated"] is True


def test_template_case_extracts_resume_facts():
    entry = _entry(
        "c1",
        "Backend-разработчик (Python)\n6 лет опыта. PostgreSQL, Docker, FastAPI, Python.",
    )

    case = build_template_case(entry, 0)

    assert case.candidate_id == "c1"
    assert case.generator == "template"
    assert case.question
    assert "6" in case.reference_answer
    assert "Python" in case.reference_answer
    assert case.question_type in {"fact", "skill"}


def test_judge_scores_accept_corpus_metrics():
    payload = {
        "faithfulness": 0.8,
        "answer_relevancy": 0.9,
        "context_relevancy": 0.7,
        "context_precision": None,
        "context_recall": None,
        "corpus_faithfulness": 0.6,
        "corpus_completeness": None,
    }

    scores = _validate_scores(payload, reference_available=False, corpus_available=True)
    assert scores["corpus_faithfulness"] == pytest.approx(0.6)
    assert scores["corpus_completeness"] is None

    without_corpus = _validate_scores(payload, reference_available=False, corpus_available=False)
    assert without_corpus["corpus_faithfulness"] is None
    assert without_corpus["corpus_completeness"] is None

    with pytest.raises(ValueError, match="between 0 and 1"):
        _validate_scores(
            {**payload, "corpus_completeness": 1.5},
            reference_available=False,
            corpus_available=True,
        )


class _FakeChat:
    configured = True

    async def stream_chat(
        self,
        message: str,
        conversation_id: str | None,
        actor_id: str,
        include_eval_metrics: bool = False,
    ) -> Any:
        assert include_eval_metrics
        yield (
            "contexts",
            {"contexts": [{"id": "r1", "candidate_id": "c1", "text": "resume fragment"}]},
        )
        yield "token", {"text": "Ответ "}
        yield "token", {"text": "ассистента"}
        yield "eval_metrics", {"ttft_ms": 12.5}


class _FailingChat:
    configured = True

    async def stream_chat(
        self,
        message: str,
        conversation_id: str | None,
        actor_id: str,
        include_eval_metrics: bool = False,
    ) -> Any:
        raise RuntimeError("chat exploded")
        yield  # pragma: no cover - makes this an async generator


async def test_collect_answers_captures_answer_contexts_and_timing():
    case = build_template_case(_entry("c1", "6 лет опыта Python"), 0)

    records = await collect_answers(_FakeChat(), [case])

    assert records[0]["answer"] == "Ответ ассистента"
    assert records[0]["contexts"] == [{"id": "r1", "candidate_id": "c1", "text": "resume fragment"}]
    assert records[0]["timing"] == {"ttft_ms": 12.5}
    assert records[0]["error"] is None


async def test_collect_answers_isolates_a_failing_case():
    case = build_template_case(_entry("c1", "6 лет опыта Python"), 0)

    records = await collect_answers(_FailingChat(), [case])

    assert records[0]["answer"] == ""
    assert records[0]["contexts"] == []
    assert "chat exploded" in (records[0]["error"] or "")
