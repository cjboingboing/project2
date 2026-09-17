# MAST30034 Project 2 — BNPL

## Done

**Merged**
- Unioned the 3 transaction snapshots (`transactions_20210228_...` / `..._20210828_...` / `..._20220228_...`) into one `transactions_clean` table, tagged with source order.
- Left-joined merchant and consumer fraud probability tables onto transactions (on `merchant_abn`/`order_datetime` and `user_id`/`order_datetime`); unmatched rows are left null for later imputation.

**Cleaned**
- `tbl_merchants.parquet`: parsed the free-text `tags` column into `keyword_phrase`, `revenue_code`, `take_rate`; malformed ABNs (wrong length) nulled out.
- Merchant keywords: lemmatized (WordNet) and auto-corrected rare/misspelled words to the nearest frequent word (Levenshtein distance ≤ 3), filler words dropped.
- Transactions: deduplicated on (`order_datetime`, `user_id`, `dollar_value`) — `order_id` itself is always unique, so it can't catch the same purchase recorded twice.
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

**External data — ABS at SA2 level**
- `download_external_sa2.py`: builds the SA2 side of the external data.
  - `abs_census_sa2`: 2021 Census G01/G02 by SA2, plus all four SEIFA 2021 indexes
    (IRSD, IRSAD, IER, IEO — score and decile). 2,472 SA2s.
  - `abs_poa_sa2_correspondence` / `abs_poa_sa2_primary`: postcode → SA2, built by
    overlaying the Mesh Block allocation files for postal areas and SA2s and weighting
    each overlap by its 2021 Census population. `primary` gives one row per postcode
    (its dominant SA2, plus `sa2_share` and `n_sa2s` as confidence).
  - `SA2_2021_AUST_simplified.geojson`: SA2 boundaries, simplified to ~100m, for maps.
- `join_external_sa2.py`: places each consumer in their postcode's primary SA2 and
  attaches the Census and SEIFA columns → `consumers_sa2`.

**SA2 findings** (`notebooks/abs_sa2.ipynb`)
- 2,644 postcodes → 5,904 postcode × SA2 pairs; 1 SA2 in the correspondence has no
  Census row (a non-residential area the Census does not publish).
- Postcodes do not nest inside SA2s: only 45.1% sit wholly inside one. 65.4% have a
  primary SA2 covering 90%+ of their population, 25.1% cover 50–90%, and 9.5% cover
  under half — mostly large rural postcodes. Worst case is 2611 (ACT), spread over 21
  SA2s with its primary holding 13.3%. Melbourne's 3000 splits five ways.
- Hence `sa2_share` is kept on every row: aggregations to merchant level should weight
  by it rather than treating every consumer's SA2 as equally certain.
- Only 3 postcodes have no residents to weight by (PO box / commercial); they fall back
  to dwelling, then mesh block counts, recorded in `weight_basis`.
- SEIFA is missing for 119 SA2s (too few people to index) and medians for up to 111.

## Left to do

- **Imputation** of null fraud probabilities on unmatched transactions — `notebooks/imputation.ipynb` is stubbed out, not yet implemented (likely a model-based approach).
- **Outlier detection** on transactions/merchants.
- **Unmatched ABS consumers**: impute (e.g. state medians), map PO boxes to a location, or keep null.
- **Join `consumers_abs` / `consumers_sa2` onto transactions** on `user_id`.
- EDA and feature engineering on the merged dataset.
- Modelling.
- **Ranking system**: proposal in `docs/ranking_system_brainstorm.md` — needs the group
  to settle the open questions at the end of it (rank on firm revenue or merchant GMV;
  drop fraud or price it in).

## Running

```bash
python scripts/download_external.py      # ABS by postcode
python scripts/download_external_sa2.py  # ABS by SA2 + postcode->SA2 correspondence
python scripts/join_external.py
python scripts/join_external_sa2.py
python scripts/process.py
# or, on a real Spark cluster:
spark-submit scripts/process.py
```

`download_external_sa2.py` pulls ~160MB from the ABS and converts three large
workbooks, so the first run takes a few minutes; everything after that is cached in
`data/tables/external/` and re-runs in seconds.

Raw data goes in `data/tables/` (gitignored — not committed, re-downloadable).
