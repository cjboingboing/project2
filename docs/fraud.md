# Fraud diagnostics and adjustment

The authoritative values are calculated in the reproduced-results section in `README.md`, the executed
notebooks and `data/tables/processed/results_summary.json`. Historical hardcoded
metrics have been removed to avoid mixing runs and reporting periods.

`fraud_models.ipynb` trains three consumer classifiers on supplied score thresholds
30/50/70, with disjoint training/test users, and merchant classifiers evaluated by
leave-one-merchant-out predictions on the scored merchant-days. Input rows are
sorted before splitting/training, and random seeds and thread counts are fixed.
It reads the canonical `consumers_sa2` enrichment and exports models/evaluation
frames under `data/tables/models/`. These are retrospective diagnostic models;
features can describe the whole scoring window, rather than what was available
before each individual purchase. Results are not prospective fraud validation.

`fraud_penalty.ipynb` uses supplied probabilities, not XGBoost predictions:

1. A score below MIN_SCORE contributes zero; the default is 30.
2. Consumer probability is allocated by each transaction's share of user-day spend.
3. Combined probability is `1 - (1 - consumer_probability) * (1 - merchant_probability)`.
4. Expected transaction loss is probability × purchase amount × severity.
5. Each merchant's scoring-window loss rate is applied to its **March 2021–September
   2022 production GMV**, then subtracted from estimated BNPL revenue.
6. The same rank-space revenue term is recomputed. All other score terms are retained.

Default severity is 1.0. `hybrid_fraud_sensitivity.csv` records every tested setting;
use its actual minimum/maximum overlap when reporting robustness. Both the full
adjusted Parquet and Top 100 CSV include `fraud_review`. Review flags and inferred
risk tiers do not enter expected loss or the ranking score.

Scores cover February 2021–February 2022 and are extremely sparse. Missing means
unknown, even though this observed-score scenario assigns zero to missing scores.
Calibration is unknown. Expected loss is neither a calibrated prediction nor a
mathematically established lower bound. The separate imputation experiment must
not be described as the adopted ranking input. Alternative loss totals used for
ranking are also restricted to the production window; raw full-dataset diagnostic
plots are explicitly labelled separately.

The user-day allocation figure illustrates the arithmetic before thresholding.
A score of 18.1 contributes no loss under the adopted threshold of 30.
