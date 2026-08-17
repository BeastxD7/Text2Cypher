# Evaluation summary

Headline numbers from the completed training run (Qwen3-4B-Instruct + LoRA r=16, schema-grouped 80/10/10 split).


## Validation split (1307 rows)

| Metric | Accuracy |
|--------|----------|
| Exact match (strict string) | 6.8% |
| Semantic match | 57.2% |
| Core logic match | 76.4% |

**By complexity (semantic match):**

| Complexity | Semantic | Core logic | n |
|------------|----------|------------|---|
| easy | 69.2% | 91.2% | 399 |
| medium | 57.1% | 74.4% | 515 |
| complex | 45.0% | 64.1% | 393 |


## Heldout split (745 rows)

| Metric | Accuracy |
|--------|----------|
| Exact match (strict string) | 6.8% |
| Semantic match | 58.1% |
| Core logic match | 74.1% |

**By complexity (semantic match):**

| Complexity | Semantic | Core logic | n |
|------------|----------|------------|---|
| easy | 73.7% | 92.5% | 293 |
| medium | 51.0% | 64.9% | 194 |
| complex | 45.7% | 60.1% | 258 |


## Notes for analysis

- **Heldout** is the real generalization signal (schemas never used for training or checkpoint selection).
- **Validation** was used for `eval_loss` / early stopping — treat its generation accuracy as mildly optimistic.
- Exact match is a lower bound; most failures are cosmetic (variable names, column aliases, extra ORDER BY).
- Re-run `python scripts/semantic_rescore.py evals/eval_results_<split>.csv` from `public/` to regenerate semantic scores.
- Full per-row gold/predicted pairs are in `evals/eval_results_*.csv`.
