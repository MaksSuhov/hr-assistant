# Eval groups

The evaluator runs one group per command so you can measure a single aspect without paying for or waiting on checks that are not needed.

| Command | Aspect | Metrics | External calls |
|---|---|---|---|
| `python -m eval pipeline [--limit N]` | Fresh cases from the database: questions, real answers, judge scores | judge metrics plus per-case timing | SQLite + Qdrant + chat API |
| `python -m eval judge [DATASET]` | Generated answer and context quality; without `DATASET` it runs the pipeline | faithfulness, answer relevancy, context relevancy, context precision/recall with a reference, corpus faithfulness/completeness with database access | OpenAI-compatible chat API |
| `python -m eval retrieval DATASET` | Ranked retrieval results | precision/recall, hit-rate, F1, MRR, MAP | None; optional live API |
| `python -m eval classification DATASET` | Candidate or label prediction | micro-F1, macro-F1, weighted-F1 | None |
| `python -m eval performance DATASET` | Streaming chat response time | TTFT, TTFRT, server latency, client latency; mean/p50/p95/min/max | Running project API |

Use the same JSON or JSONL dataset and include only the fields needed by the group being run. `data/example.jsonl` is one synthetic smoke-test record, not a golden set for the real resume corpus. Reports are printed as JSON; pass `--output path.json` to save a copy.

## The fresh-case pipeline

`python -m eval pipeline` is the end-to-end loop that runs when you do not have a saved dataset:

1. It reads the newest resumes from the application database, one resume per candidate, newest first. The freshest `--limit` (default 20) become cases; the rest of the database is still loaded as evidence for the judge.
2. For each case resume it writes a natural recruiter question and a reference answer. The default `--generator llm` uses the chat model (`--case-model` overrides it); `--generator template` is an offline fallback with less variety.
3. It asks the assistant through its normal chat path (`ChatService.stream_chat`) and captures the answer, the exact retrieved contexts, and timing.
4. It scores every case with the judge, which receives a snapshot of the **whole** candidate corpus in addition to the retrieved contexts.

The split matters: fresh cases keep the evaluation current, while full-corpus access lets the judge catch an answer that contradicts a resume the retriever never returned, or a list question that silently omits a qualifying candidate.

Run it where the database lives — inside the API container:

```bash
docker compose exec -T api python -m eval pipeline --limit 20 --dry-run
docker compose exec -T api python -m eval pipeline --limit 20 --output /tmp/report.json
```

`--dry-run` builds and prints the cases without calling the assistant or the judge. `--chars-per-resume` and `--max-corpus-chars` control how much of the corpus reaches the judge prompt. The report contains `corpus`, `metrics`, `per_case` (question, reference answer, actual answer, context count, timing, scores, rationale), and the full nested `judge` report.

`python -m eval judge` with no dataset is a shortcut for the same pipeline. With a dataset it scores saved runs instead.

## Data flow and metrics

Each JSONL line is an independent evaluation case. The evaluator reads the fields required by the selected group, runs only that group, then prints aggregate and, where applicable, per-case JSON scores.

- **Judge:** provide `question` or `query`, the model's actual `answer`, the actual retrieved `contexts`, and optionally the reference `reference_answer`. The configured OpenAI-compatible model scores each answer/context pair from 0 to 1. Aggregate scores are arithmetic means. `context_precision` and `context_recall` are null when no reference answer is provided. With corpus access (`--corpus auto` by default, `full` to require it, `none` to disable), the judge also scores `corpus_faithfulness` — whether the answer agrees with the resumes in the database — and `corpus_completeness` — whether a list/comparison question accounts for every qualifying candidate in the corpus. These catch what retrieved-only judging cannot: contradictions with the database and silently missed candidates. The judge is a scoring function, not a case source; the pipeline supplies the cases.
- **Retrieval:** provide reference relevant IDs and either retrieved IDs or use live mode. Offline mode compares `relevant_context_ids` with the ranked `retrieved_context_ids` and reports context precision/recall. Live mode calls the app's `/api/v1/candidates/search`, deduplicates returned candidate IDs in rank order, and compares them with `relevant_candidate_ids`. Hit-rate is the share of queries with any relevant result; MRR rewards the rank of the first relevant result; MAP averages precision at each relevant rank. F1 is also reported, including micro aggregation and mean per-query F1.
- **Classification:** provide `expected_labels` and `predicted_labels` per case. Micro-F1 pools label decisions; macro-F1 averages class F1 equally; weighted-F1 weights each class by its support.
- **Performance:** provide `question`, `query`, or `message`. The runner calls the streaming chat API and asks for timing metadata with `X-Eval-Metrics: true`. TTFT is request start to the first model output chunk, TTFRT is request start to the first visible answer text token, and latency ends after the answer is saved. Results include mean, p50, p95, min, and max, plus client-observed latency.

## Golden set

There is exactly one golden set: `data/golden_set.jsonl`. It contains 20 synthetic Russian HR questions with reference answers, supporting fixture snippets, and relevant demo IDs. It is a **frozen regression fixture**, not a snapshot of the live index: the `demo-candidate-*` IDs are invented because the file has to be reproducible without a database.

Do not create ID-mapped copies of it for a local corpus. The fresh-case pipeline reads the current candidate UUIDs from the database, so there is nothing to remap:

```bash
docker compose exec -T api python -m eval pipeline --limit 20
```

Use the fixture file for offline judging of saved runs and for retrieval tests that do not depend on the live database. Its ID-based retrieval scores are only meaningful against a corpus that actually uses those IDs; for the live index, use the pipeline.

The golden file stores `reference_answer`, `gold_contexts`, `relevant_context_ids`, and `relevant_candidate_ids`; it intentionally does not contain a generated `answer` or predicted IDs. For Judge evaluation, add actual model outputs and actual retrieved contexts from the runs being scored.

To evaluate the running candidate index with a prepared dataset (not the pipeline), add `--base-url http://localhost:8000` to the retrieval command. This mode reads `query` or `question` and `relevant_candidate_ids`, calls `/api/v1/candidates/search`, then computes the ranked retrieval metrics. Without `--base-url`, retrieval uses predictions already stored in the dataset. The endpoint requires an authenticated session with the `access_token` cookie and the `csrf_token`/`X-CSRF-Token` pair; the live runner does not send them yet, so it returns 401. The in-process pipeline avoids this entirely.

The judge sends question, answer, contexts, optional reference answer, and — when corpus access is enabled — the corpus snapshot to the configured model.

MRR is mean reciprocal rank. MMR (maximal marginal relevance) is a reranking strategy, rather than a quality metric, so the retrieval report uses standard ranking measures MRR and MAP. Offline retrieval uses `relevant_context_ids` / `retrieved_context_ids`; live retrieval uses `relevant_candidate_ids` and the API's candidate IDs.