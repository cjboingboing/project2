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

**External data — ABS**
- `download_external.py`: 2021 Census by postal area → `abs_census_poa` (population, median age, income, rent, mortgage). Zero medians set to null.
- `join_external.py`: joined onto consumers by postcode → `consumers_abs` (one row per `user_id`). Postcodes padded to 4 digits.

**ABS findings**
- 499,999 consumers; 416,818 (83.4%) matched an ABS postal area.
- 83,181 (16.6%) unmatched, left null with `abs_matched = False`:
  - 65,023 PO box / mail-only postcodes (no residents, so no Census data)
  - 18,009 postcodes with no ABS postal area (mostly CBD, e.g. `2001`, `3001`)
  - 149 with postcode `9999`
- 7,909 NT/ACT postcodes were missing the leading zero (`862` → `0862`), fixed by padding.
- Some matched areas still have null medians (rent and mortgage most affected).

## Left to do

- **Imputation** of null fraud probabilities on unmatched transactions — `notebooks/imputation.ipynb` is stubbed out, not yet implemented (likely a model-based approach).
- **Outlier detection** on transactions/merchants.
- **Unmatched ABS consumers**: impute (e.g. state medians), map PO boxes to a location, or keep null.
- **Join `consumers_abs` onto transactions** on `user_id`.
- EDA and feature engineering on the merged dataset.
- Modelling.

## Running

```bash
python scripts/download_external.py
python scripts/join_external.py
python scripts/process.py
# or, on a real Spark cluster:
spark-submit scripts/process.py
```

Raw data goes in `data/tables/` (gitignored — not committed, re-downloadable).
