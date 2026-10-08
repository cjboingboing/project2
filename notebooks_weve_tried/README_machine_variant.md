# Pure learned weights — no floors

One of three weightings of the same ranking system. The other two are `notebooks/` (the shipped pipeline, now on the hybrid weights).

## What differs from `notebooks/`

**Three lines, and nothing else:**

1. `WEIGHTS_HYBRID` in `merchant_ranking_machine.ipynb` — the weights below.
2. `PROCESSED` / `PROCESSED_PATH` in every notebook — points at `data/variants/machine/processed/`
   so the three weightings never overwrite each other's results.
3. `SCRIPTS` in `segmentation.ipynb` — loads `top10_per_segment.py` from this folder's own
   `scripts/`, which carries the same weights.

No logic was touched. Every other line is byte-identical to `notebooks/`, which can be checked with
`diff`. The notebooks here are the work of the whole group; only the numbers our model produced have
moved.

## The weights

| Feature | Weight | Where it comes from |
| --- | --- | --- |
| `estimated_bnpl_revenue` | 0.724 | mean of the three fold fits |
| `active_month_ratio` | 0.218 | mean of the three fold fits |
| `customer_spread` | 0.058 | mean of the three fold fits |
| everything else | 0.00 | the fitted model sets them to zero |

Weights are the mean of the non-negative fits from folds B, C and FINAL in
`notebooks/rolling_validation.ipynb`, with terms below 0.01 dropped and the rest renormalised to
sum to 1. The mean is used rather than any single fold's fit because the three agreed closely —
revenue 0.738 / 0.702 / 0.713 — so averaging removes a little noise without hiding disagreement.

**This variant drops the ABS features entirely.** It scores highest of the three, and it is the one
that conflicts with the brief's requirement that external data inform the ranking. It is kept so the
trade-off can be shown rather than argued.

## Where the results land

`data/variants/machine/processed/` holds this variant's outputs. Shared inputs in that folder
(`transactions_clean`, `merchant_features`, the fraud tables, the behaviour segments, and so on) are
symlinks back to `data/tables/processed/`, so nothing is duplicated on disk.

Files this variant writes:

- `merchant_ranking_machine.parquet`, `top100_machine.csv`, `weights_machine.csv`, `machine_holdout_results.csv`
- `merchant_ranking_machine_fraud_adjusted.parquet`, `top100_machine_fraud_adjusted.csv`, `merchant_risk`
- `top10_per_segment.csv`

## How to run it

In order, from the project root:

```
notebooks_weve_tried/merchant_ranking_machine.ipynb
notebooks_weve_tried/fraud_penalty.ipynb
notebooks_weve_tried/fraud_imputation_check.ipynb
notebooks_weve_tried/segmentation.ipynb
python notebooks_weve_tried/scripts/top10_per_segment.py
notebooks_weve_tried/final_findings.ipynb
```

Or double-click `jalankan_3_varian.command`, which runs both variants end to end and prints the
comparison.
