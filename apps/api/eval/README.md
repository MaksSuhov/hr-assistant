# Eval groups

The evaluator runs one group per command so you can measure a single aspect without paying for or waiting on checks that are not needed.

| Command | Aspect | Metrics | External calls |
|---|---|---|---|
| `python -m eval judge DATASET` | Generated answer and retrieved context quality | faithfulness, answer relevancy, context relevancy, context precision/recall when a reference answer is present | OpenAI-compatible chat API |
| `python -m eval retrieval DATASET` | Ranked retrieval results | precision/recall, hit-rate, F1, MRR, MAP | None; optional live API |
| `python -m eval classification DATASET` | Candidate or label prediction | micro-F1, macro-F1, weighted-F1 | None |
| `python -m eval performance DATASET` | Streaming chat response time | TTFT, TTFRT, server latency, client latency; mean/p50/p95/min/max | Running project API |

Use the same JSON or JSONL dataset and include only the fields needed by the group being run. `data/example.jsonl` is one synthetic smoke-test record, not a golden set for the real resume corpus. Reports are printed as JSON; pass `--output path.json` to save a copy.

## Data flow and metrics

Each JSONL line is an independent evaluation case. The evaluator reads the fields required by the selected group, runs only that group, then prints aggregate and, where applicable, per-case JSON scores.

- **Judge:** provide `question` or `query`, the model's actual `answer`, the actual retrieved `contexts`, and optionally the reference `reference_answer`. The configured OpenAI-compatible model scores each answer/context pair from 0 to 1. Aggregate scores are arithmetic means. `context_precision` and `context_recall` are omitted from scoring when no reference answer is provided. This group evaluates saved runs; it does not call the HR chat API to generate an answer.
- **Retrieval:** provide reference relevant IDs and either retrieved IDs or use live mode. Offline mode compares `relevant_context_ids` with the ranked `retrieved_context_ids` and reports context precision/recall. Live mode calls the app's `/api/v1/candidates/search`, deduplicates returned candidate IDs in rank order, and compares them with `relevant_candidate_ids`. Hit-rate is the share of queries with any relevant result; MRR rewards the rank of the first relevant result; MAP averages precision at each relevant rank. F1 is also reported, including micro aggregation and mean per-query F1.
- **Classification:** provide `expected_labels` and `predicted_labels` per case. Micro-F1 pools label decisions; macro-F1 averages class F1 equally; weighted-F1 weights each class by its support.
- **Performance:** provide `question`, `query`, or `message`. The runner calls the streaming chat API and asks for timing metadata with `X-Eval-Metrics: true`. TTFT is request start to the first model output chunk, TTFRT is request start to the first visible answer text token, and latency ends after the answer is saved. Results include mean, p50, p95, min, and max, plus client-observed latency.

## Golden set

`data/golden_set.jsonl` contains 20 synthetic Russian HR questions, reference answers, supporting fixture snippets, and relevant demo candidate/context IDs. It covers direct fact lookup, multi-constraint matching, comparisons, and negative evidence. The `demo-candidate-*` records are invented fixtures because no labeled resume corpus was present in the project. This is a starter gold set, not ground truth for the live index. Replace/map its IDs and facts to anonymized real resumes before using retrieval scores to judge production search quality.

The golden file stores `reference_answer`, `gold_contexts`, `relevant_context_ids`, and `relevant_candidate_ids`; it intentionally does not contain a generated `answer` or predicted IDs. For Judge evaluation, add actual model outputs and actual retrieved contexts from the runs being scored. For retrieval live mode, the gold candidate IDs must match IDs in the app's index.

To evaluate the running candidate index, add `--base-url http://localhost:8000` to the retrieval command. This mode reads `query` or `question` and `relevant_candidate_ids`, calls `/api/v1/candidates/search`, then computes the ranked retrieval metrics. Without `--base-url`, retrieval uses predictions already stored in the dataset.

The judge sends question, answer, contexts, and optional reference answer to the configured model.

MRR is mean reciprocal rank. MMR (maximal marginal relevance) is a reranking strategy, rather than a quality metric, so the retrieval report uses standard ranking measures MRR and MAP. Offline retrieval uses `relevant_context_ids` / `retrieved_context_ids`; live retrieval uses `relevant_candidate_ids` and the API's candidate IDs.
