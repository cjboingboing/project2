# MAST30034 Project 2 — BNPL

## Done

**Merged**
- Unioned the 3 transaction snapshots (`transactions_20210228_...` / `..._20210828_...` / `..._20220228_...`) into one `transactions_clean` table, tagged with source order.
- Left-joined merchant and consumer fraud probability tables onto transactions (on `merchant_abn`/`order_datetime` and `user_id`/`order_datetime`); unmatched rows are left null for later imputation.

**Cleaned**
- `tbl_merchants.parquet`: parsed the free-text `tags` column into `keyword_phrase`, `revenue_code`, `take_rate`; malformed ABNs (wrong length) nulled out.
- Merchant keywords: lemmatized (WordNet) and auto-corrected rare/misspelled words to the nearest frequent word (Levenshtein distance ≤ 3), filler words dropped.
- Transactions: deduplicated (confirmed no duplicate `order_id`s).
- No outlier detection has been done yet — deliberately left out of `process.py`.

All cleaned outputs are written to `data/tables/processed/` (`merchants_clean`, `transactions_clean`, `consumer_fraud_probability`, `merchant_fraud_probability`, `consumers`) as Parquet.

## Left to do

- **Imputation** of null fraud probabilities on unmatched transactions — `notebooks/imputation.ipynb` is stubbed out, not yet implemented (likely a model-based approach).
- **Outlier detection** on transactions/merchants.
- **External data — ABS**: not yet pulled in. `tbl_consumer.csv` has `postcode`/`state`, so the plan is to link in ABS SEIFA (socio-economic index) and/or Census data by postcode → SA2 concordance for consumer-level demographic/socio-economic features.
- EDA and feature engineering on the merged dataset.
- Modelling.

## Running

```bash
python scripts/process.py
# or, on a real Spark cluster:
spark-submit scripts/process.py
```

Raw data goes in `data/tables/` (gitignored — not committed, re-downloadable).
