import argparse
import asyncio
from typing import Any

from eval.io import load_records, write_report


def _dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", help="Path to a JSON or JSONL evaluation dataset")
    parser.add_argument("--output", help="Optional path for a JSON report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval",
        description="Run one evaluation group at a time for the HR assistant.",
    )
    commands = parser.add_subparsers(dest="group", required=True)

    judge = commands.add_parser("judge", help="LLM-as-a-judge: answer and context quality")
    _dataset_argument(judge)
    judge.add_argument("--model", help="Judge model; defaults to OPENAI_CHAT_MODEL")
    judge.add_argument("--concurrency", type=int, default=4)

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
    if args.group == "judge":
        from eval.judge import evaluate_judge
        from hr_assistant.core.config import get_settings

        settings = get_settings()
        model = args.model or settings.eval_judge_model or settings.openai_chat_model
        return asyncio.run(evaluate_judge(records, settings, model, args.concurrency))
    if args.group == "performance":
        from eval.performance import evaluate_performance

        return asyncio.run(
            evaluate_performance(records, args.base_url, args.actor_id, args.concurrency)
        )
    raise ValueError(f"Unsupported evaluation group: {args.group}")


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
