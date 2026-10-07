# Hybrid weights — learned shape, with floors

One of three weightings of the same ranking system. The other two are `notebooks/` (the hand-chosen
weights the group has been using) and the sibling variant folder.

## What differs from `notebooks/`

**Three lines, and nothing else:**

1. `WEIGHTS_V2` in `merchant_ranking_v2.ipynb` — the weights below.
2. `PROCESSED` / `PROCESSED_PATH` in every notebook — points at `data/variants/hybrid/processed/`
   so the three weightings never overwrite each other's results.
3. `SCRIPTS` in `segmentation.ipynb` — loads `top10_per_segment.py` from this folder's own
   `scripts/`, which carries the same weights.

No logic was touched. Every other line is byte-identical to `notebooks/`, which can be checked with
`diff`. The notebooks here are the work of the whole group; only the numbers our model produced have
moved.

## The weights

| Feature | Weight | Where it comes from |
| --- | --- | --- |
| `estimated_bnpl_revenue` | 0.70 | the fitted model chose ~0.72 across folds |
| `active_month_ratio` | 0.18 | the fitted model chose ~0.22 — the correction the hand weights missed |
| `customer_spread` | 0.05 | the fitted model chose ~0.06 |
| `customer_income` (ABS) | 0.03 | **floor** — the model sets it to 0; the brief requires external data in the ranking |
| `customer_irsad` (ABS) | 0.02 | **floor**, same reason |
| `log_value_growth` | 0.02 | **floor** — token weight on a forward-looking term |
| everything else | 0.00 | the model sets these to 0 in every fold |

The floors cost **1.44 capture-rate points** against the unconstrained fit, measured across folds
B, C and FINAL in `notebooks/rolling_validation.ipynb`. That is the number to quote when asked what
keeping the ABS features costs.

## Where the results land

`data/variants/hybrid/processed/` holds this variant's outputs. Shared inputs in that folder
(`transactions_clean`, `merchant_features`, the fraud tables, the behaviour segments, and so on) are
symlinks back to `data/tables/processed/`, so nothing is duplicated on disk.

Files this variant writes:

- `merchant_ranking_v2.parquet`, `top100_v2.csv`, `weights_v2.csv`, `v2_holdout_results.csv`
- `merchant_ranking_v2_fraud_adjusted.parquet`, `top100_v2_fraud_adjusted.csv`, `merchant_risk`
- `top10_per_segment.csv`

## How to run it

In order, from the project root:

```
notebooks_hybrid/merchant_ranking_v2.ipynb
notebooks_hybrid/fraud_penalty.ipynb
notebooks_hybrid/fraud_imputation_check.ipynb
notebooks_hybrid/segmentation.ipynb
python notebooks_hybrid/scripts/top10_per_segment.py
notebooks_hybrid/final_findings.ipynb
```

Or double-click `jalankan_3_varian.command`, which runs both variants end to end and prints the
comparison.
