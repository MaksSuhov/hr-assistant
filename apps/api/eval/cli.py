import argparse
import asyncio
import sys
from typing import Any

from eval.io import load_records, write_report


def _dataset_argument(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        "dataset",
        nargs=None if required else "?",
        help="Path to a JSON or JSONL evaluation dataset",
    )
    parser.add_argument("--output", help="Optional path for a JSON report")


def _pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many of the freshest candidates to evaluate (default: 20)",
    )
    parser.add_argument("--model", help="Judge model; defaults to EVAL_JUDGE_MODEL")
    parser.add_argument("--case-model", help="Model that writes fresh questions")
    parser.add_argument("--generator", choices=["llm", "template"], default="llm")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--chars-per-resume", type=int, default=1600)
    parser.add_argument("--max-corpus-chars", type=int, default=60_000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build the fresh cases; do not call the assistant or the judge",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval",
        description="Run one evaluation group at a time for the HR assistant.",
    )
    commands = parser.add_subparsers(dest="group", required=True)

    judge = commands.add_parser(
        "judge",
        help="LLM-as-a-judge; without DATASET it scores fresh cases from the database",
    )
    _dataset_argument(judge, required=False)
    judge.add_argument("--model", help="Judge model; defaults to OPENAI_CHAT_MODEL")
    judge.add_argument("--concurrency", type=int, default=4)
    judge.add_argument(
        "--corpus",
        choices=["auto", "full", "none"],
        default="auto",
        help=(
            "Give the judge the whole candidate database as evidence. "
            "'auto' skips it when the database is not reachable."
        ),
    )
    judge.add_argument(
        "--corpus-limit",
        type=int,
        default=0,
        help="Max candidates loaded for the judge; 0 means the whole database",
    )
    judge.add_argument("--chars-per-resume", type=int, default=1600)
    judge.add_argument("--max-corpus-chars", type=int, default=60_000)
    judge.add_argument("--limit", type=int, default=20, help="Fresh cases when DATASET is omitted")
    judge.add_argument("--generator", choices=["llm", "template"], default="llm")
    judge.add_argument("--case-model", help="Model that writes fresh questions")
    judge.add_argument("--dry-run", action="store_true")

    pipeline = commands.add_parser(
        "pipeline",
        help="Fresh cases from the database: questions, answers, and judge scores",
    )
    _pipeline_arguments(pipeline)
    pipeline.add_argument("--output", help="Optional path for a JSON report")

    retrieval = commands.add_parser("retrieval", help="Retrieval precision, recall, and ranking")
    _dataset_argument(retrieval)
    retrieval.add_argument("--base-url", help="Run live search against this API URL")
    retrieval.add_argument("--limit", type=int, help="Override the search result limit")
    retrieval.add_argument("--concurrency", type=int, default=4)

    classification = commands.add_parser(
        "classification", help="Micro, macro, and weighted F1 for label predictions"
    )
    _dataset_argument(classification)

    performance = commands.add_parser(
        "performance", help="Measure TTFT, TTFRT, and latency over the SSE chat API"
    )
    _dataset_argument(performance)
    performance.add_argument("--base-url", default="http://localhost:8000")
    performance.add_argument("--actor-id", default="eval-runner")
    performance.add_argument("--concurrency", type=int, default=1)
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.group in {"judge", "pipeline"}:
        return _run_judge(args)

    records = load_records(args.dataset)
    if args.group == "retrieval":
        from eval.retrieval import evaluate_retrieval

        if args.base_url:
            from eval.live_retrieval import fetch_live_retrieval

            records = asyncio.run(
                fetch_live_retrieval(records, args.base_url, args.limit, args.concurrency)
            )
            report = evaluate_retrieval(records)
            report["run_mode"] = "live_api"
            return report
        return evaluate_retrieval(records)
    if args.group == "classification":
        from eval.classification import evaluate_classification

        return evaluate_classification(records)
    if args.group == "performance":
        from eval.performance import evaluate_performance

        return asyncio.run(
            evaluate_performance(records, args.base_url, args.actor_id, args.concurrency)
        )
    raise ValueError(f"Unsupported evaluation group: {args.group}")


def _run_judge(args: argparse.Namespace) -> dict[str, Any]:
    from hr_assistant.core.config import get_settings

    settings = get_settings()
    judge_model = args.model or settings.eval_judge_model or settings.openai_chat_model

    dataset = getattr(args, "dataset", None)
    if args.group == "pipeline" or not dataset:
        from eval.pipeline import run_pipeline

        return asyncio.run(
            run_pipeline(
                settings,
                limit=args.limit,
                generator=args.generator,
                case_model=args.case_model,
                judge_model=judge_model,
                concurrency=args.concurrency,
                max_chars_per_resume=args.chars_per_resume,
                max_total_chars=args.max_corpus_chars,
                dry_run=args.dry_run,
            )
        )

    from eval.judge import evaluate_judge

    records = load_records(dataset)
    corpus = _judge_corpus(settings, args)
    return asyncio.run(
        evaluate_judge(records, settings, judge_model, args.concurrency, corpus=corpus)
    )


def _judge_corpus(settings: Any, args: argparse.Namespace) -> dict[str, Any] | None:
    """Load the corpus for offline judging, degrading gracefully in auto mode."""

    if args.corpus == "none":
        return None

    from eval.corpus import corpus_snapshot, load_corpus

    try:
        corpus = asyncio.run(load_corpus(settings, limit=args.corpus_limit or None))
    except Exception as error:
        if args.corpus == "full":
            raise RuntimeError(
                f"Could not read the candidate corpus from the database: {error}"
            ) from error
        print(
            f"warning: judge corpus unavailable ({error}); scoring without it",
            file=sys.stderr,
        )
        return None

    if not corpus.entries:
        if args.corpus == "full":
            raise RuntimeError("The candidate database has no resumes to give the judge.")
        print(
            "warning: the candidate database has no resumes; scoring without corpus access",
            file=sys.stderr,
        )
        return None
    return corpus_snapshot(
        corpus,
        max_chars_per_resume=args.chars_per_resume,
        max_total_chars=args.max_corpus_chars,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = _run(args)
        write_report(report, args.output)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
