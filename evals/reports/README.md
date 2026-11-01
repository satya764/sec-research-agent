# Eval reports

Raw records for the runs in the main README's Results table. Oct 31 2026, Apple, Microsoft and NVIDIA 10-Ks,
37 verified questions from `evals/golden.jsonl`. Retrieval ran as plain hybrid search (no Cohere key, so no reranker),
even where an older file name says `hybrid_rerank`.

| File | Model | Questions |
| --- | --- | --- |
| `claude-haiku-5-5-full.json` | Claude Haiku 5.5 | all 37 |
| `claude-haiku-5-5-narrative.json` | Claude Haiku 5.5 | 3 narrative, graded against reference answers |
| `claude-sonnet-5-5-full.json` | Claude Sonnet 5.5 | all 37 |
| `claude-sonnet-5-5-narrative.json` | Claude Sonnet 5.5 | 3 narrative, graded against reference answers |
| `qwen3-8b-local-full.json` | Qwen3 8B (Ollama, 16K context) | all 37 |
| `qwen3-8b-local-narrative.json` | Qwen3 8B | 3 narrative, graded against reference answers |

The full runs started before the narrative reference answers were written, so their narrative rows are scored on
citations only; the narrative files score them for correctness. Claude runs use Claude Haiku as the judge.
The local runs use Qwen3 itself as the judge, so everything stays free and offline.

Each record has the answer, the cited source ids, every tool call, cost, latency and the grades.
