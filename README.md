# MAST30034 Project 2 — BNPL

Rank ~4,400 merchants and recommend the top 100 overall, plus the top 10 in each segment.

Start with **`notebooks/final_findings.ipynb`** — the summary for the presentation. Everything below is how it was built.

## Pipeline

| Step | Script / notebook | Output |
|---|---|---|
| Clean | `process_cleaning.py` | `transactions_clean`, `merchants_clean`, `consumers`, fraud tables |
| External — SA2 | `download_external_sa2.py` → `join_external_sa2.py` | `abs_census_sa2`, `abs_poa_sa2_primary` → `consumers_sa2` |
| EDA + features | `01_data_quality_eda.ipynb` | `merchant_features` (24 columns) |
| Segments | `segment_behaviour.py` | `merchant_behaviour_segments` (4 segments) |
| Product check | `segment_merchants.py` | `merchant_segments` (supporting only, see Findings) |
| Ranking | `merchant_ranking_hybrid.ipynb`, `validation_timesplit.ipynb`, `learned_weights.ipynb`, `rolling_validation.ipynb` | `merchant_ranking_hybrid.parquet`, `top100_hybrid.csv` |
| Fraud | `fraud_models.ipynb`, `fraud_imputation_check.ipynb`, `fraud_penalty.ipynb` | consumer tier classifier, imputed scores, merchant risk tiers and fraud penalty |
| Top 10 per segment | `top10_per_segment.py` | `top10_per_segment.csv` |
| Slide figures | `make_plots.py` | PNGs under `plots/` |

The ranking weights live in two places and must match: `WEIGHTS_HYBRID` in
`notebooks/merchant_ranking_hybrid.ipynb` and the copy in `scripts/top10_per_segment.py`.

`notebooks_weve_tried/` holds the alternatives that were built, measured and rejected — the v1
ranking and the whole pipeline under the pure learned weights. The notebooks that produced and
tested the shipped weights (`learned_weights.ipynb`, `rolling_validation.ipynb`) are in
`notebooks/`, since they justify what ships rather than compete with it. Its README explains what each one was and why it is not the shipped
version.

## Findings

**Data** (`01_data_quality_eda.ipynb`)
- 14,195,505 transactions, one row per `order_id`. 4,422 ABNs trade but only 4,026 have metadata — the other 396 (580,830 txns, ~$202.65M) are excluded from anything needing category or take rate.
- Outliers kept, not removed: the largest is ~$105,194 at an antique merchant already averaging ~$20,250.

**ABS** (`abs_sa2.ipynb`)
- 83.4% of consumers match an area; the same 16.6% miss under both postcode and SA2, because the SA2 bridge is built from postal areas. Mostly PO box postcodes (65,023), then postcodes with no ABS area (18,009).
- 7,909 NT/ACT postcodes lose their leading zero in `tbl_consumer.csv` and need padding (`862` → `0862`).
- Only 45.1% of postcodes sit inside one SA2, so `sa2_share` travels with every row as a confidence weight. SEIFA is missing for 119 SA2s.

**Segments** (`segmentation.ipynb`) — by behaviour, on the 6 agreed dimensions; k=4 chosen by silhouette (0.589)

| Segment | Merchants | Character |
|---|---|---|
| Established | 1,616 (40.1%) | wide customer base, steady |
| Rising | 969 (24.1%) | mid-size, growing |
| Fading | 888 (22.1%) | mid-size, shrinking |
| Intermittent | 553 (13.7%) | trade some months only, erratic |

- Established holds **87.9% of the $91.2M** BNPL revenue. The best merchant outside it ranks 780th overall, which is exactly why the top 10 is taken per segment.
- Behaviour is not product: every segment holds 13–26% of every category. That justifies segmenting this way rather than by what merchants sell.
- Percentile-rank the dimensions before clustering — raw values put 76% of merchants in one cluster.
- `value_cv` is named "stability" but measures **volatility** (high = erratic).
- Merchants are scored within their own segment using the v2 weights, so ranks are comparable to peers rather than to the whole market.

**Ranking** (`merchant_ranking_hybrid.ipynb`, `validation_timesplit.ipynb`)
- Train Feb 2021 – Feb 2022, test Mar – Aug 2022. v1 → v2: Spearman 0.740 → 0.905, capture 74.3% → 85.4%.
- **The shipped weights are now the hybrid set** (`notebooks/learned_weights.ipynb`,
  `notebooks/rolling_validation.ipynb`): revenue 0.70, `active_month_ratio` 0.18, `customer_spread` 0.05,
  ABS 0.05, growth 0.02. Three independent fits across walk-forward folds all put
  `active_month_ratio` near 0.22 where the hand weights had given it 0.02 — that single row is the
  correction. On the September holdout: 91.1% capture, 80 of the realised top 100, against 87.8%
  and 74 for the hand weights.
- Dropping the ABS features entirely (the pure learned fit) reaches 91.5% and the same 80 correct
  picks, so keeping them costs **0.4 points and no accuracy**. That variant is kept in
  `notebooks_weve_tried/` rather than shipped, because the brief requires external data to inform
  the ranking.
- **Past revenue alone still wins**: 99.9% capture, 99 of the realised top 100. v2 costs ~$1.54M over six months, in exchange for expressing fragility and customer risk. Unsettled.
- ABS features barely predict revenue (+0.029 income, +0.022 IRSAD) → 5% weight, kept as a risk lens.

**Fraud** (`fraud_models.ipynb`, `fraud_imputation_check.ipynb`, `fraud_penalty.ipynb`)
- Supplied fraud scores cover ~1% of transactions, so a consumer classifier predicts P(score ≥ 30 / 50 / 70) from transaction behaviour, and the gaps are imputed from a spend-band lookup.
- Merchants get risk tiers and an expected fraud loss, which feeds a penalty into the v2 ranking.

## Left to do

- **Decide: the weighted score or the naive baseline** — past revenue alone still captures ~100% in every window tested. The weighted score is kept for fraud, segments and explainability; that trade-off needs stating in the report, not resolving in code.
- **The 396 merchants with no metadata**: decide what to report.
- **Sign off segment names** (Established / Rising / Fading / Intermittent, set in `NAMES` in `scripts/top10_per_segment.py`).
- Run every notebook end to end before submission — a few still have cells without saved output.
- Slides (due Sun 11 Oct).

## Running

```bash
pip install -r requirements.txt           # PySpark also needs Java 17
python scripts/download_external_sa2.py   # ABS by SA2 (~160MB first run, then cached)
python scripts/join_external_sa2.py
python scripts/process_cleaning.py        # cleaned tables (~14M transactions)
# then notebooks/01_data_quality_eda.ipynb   -> merchant_features
python scripts/segment_behaviour.py       # 4 segments from the 6 dimensions
python scripts/segment_merchants.py       # product categories (supporting check)
# then notebooks/merchant_ranking_hybrid.ipynb   -> merchant_ranking_hybrid.parquet
# then the three fraud notebooks             -> risk tiers and the fraud penalty
python scripts/top10_per_segment.py       # top 10 per segment
python scripts/make_plots.py              # slide figures into plots/
```

Raw data goes in `data/tables/` (gitignored — not committed, re-downloadable).
