# Text2Cypher — public exploration bundle

Self-contained snapshot for sharing: a LoRA fine-tuning notebook, the synthetic dataset it trains on, and evaluation results from one completed run. No product-specific context — just an experiment in schema-conditioned natural language → Cypher generation.

## What's inside

| Path | Description |
|------|-------------|
| `code/runpod/simple_training.ipynb` | End-to-end training + eval notebook (Unsloth, Qwen3-4B-Instruct, LoRA) |
| `dataset/synthetic_cypher_dataset.parquet` | ~10.7k rows / 184 schemas / 36 domains — schema, question, gold Cypher |
| `evals/eval_results_*.csv` | Per-row gold vs predicted Cypher (validation + heldout splits) |
| `evals/semantic_rescore_*.csv` | Structural semantic re-score of those predictions |
| `evals/SUMMARY.md` | Headline accuracy numbers and split notes |
| `scripts/semantic_rescore.py` | Re-run semantic scoring on any `eval_results_*.csv` |
| `test_benchmark/report/` | Aggregate results from a second, execution-based benchmark run (see below) |
| `logs/` | Empty by default — training runs write timestamped logs here |
| `models/outputs/` | Empty by default — checkpoints and adapters land here after training |

## Quick start (RunPod or any GPU box)

1. Clone or copy this `public/` folder.
2. Open `code/runpod/simple_training.ipynb` and set `os.chdir(...)` to your workspace root (the parent of `public/`).
3. Run top-to-bottom on a GPU with [Unsloth](https://github.com/unslothai/unsloth) preinstalled (RunPod Unsloth template works).

Paths are relative to the repo/workspace root (`public/dataset`, `public/logs`, etc.).

## Analyzing the existing eval run

No model weights are included (too large for git). The **eval CSVs are from a completed training run** and are ready to analyze as-is:

```bash
cd public
python scripts/semantic_rescore.py evals/eval_results_heldout.csv evals/eval_results_validation.csv
```

See `evals/SUMMARY.md` for headline numbers. Key takeaway: **heldout semantic accuracy ~58%** vs **~7% exact match** — most "failures" are cosmetic wording differences, not wrong graph logic.

## Second benchmark: execution-based, against [neo4j/text2cypher-2024v1](https://huggingface.co/datasets/neo4j/text2cypher-2024v1)

Separately from the synthetic-dataset eval above, this model was also scored against a
curated subset of the public **`neo4j/text2cypher-2024v1`** test split — schema + question
in, generated Cypher executed against a real Neo4j database, correctness judged by
comparing the actual result set to gold (not string similarity).

| done/total | overall pass | execution pass | semantic-fallback pass | avg gen (s) |
|---|---|---|---|---|
| 1346/1346 | 558/1346 (41.5%) | 364/901 | 194/445 | 5.46 |

`execution pass` = rows scored by directly executing predicted vs. gold Cypher and
comparing result sets. `semantic-fallback pass` = rows with no live database attached,
scored by comparing query structure (labels, relationship types, WHERE conditions,
aggregations) instead.

The row-level question/gold-query data for this benchmark isn't included here — it's a
curated/audited selection built on top of the public test split, not something I generated
myself, so I'm only sharing the aggregate numbers rather than republishing someone else's
curation work. The full report (methodology notes) is in `test_benchmark/report/`.

## Dataset columns

`schema`, `question`, `cypher`, `schema_id`, `domain`, `hop_count`, `aggregation_present`, `subquery_present`, `varlen_present`, `where_predicate_count`, `complexity`

Train/val/heldout splits are **schema-grouped** (80/10/10 by `schema_id`) inside the notebook — not pre-baked into separate files.

## What is intentionally not included

- Model weights / LoRA adapters (~GBs) — train locally or ask separately
- Raw generation batch JSONs and QA ledgers (internal pipeline artifacts)
- Hugging Face tokens or any credentials
- Row-level data for the `neo4j/text2cypher-2024v1` benchmark (see above) — the public
  dataset itself is linked instead of republishing the curated subset used to score it
