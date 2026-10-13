# MAST30034 Assignment 2: merchant onboarding analysis

The shipped ranking uses fixed **Hybrid** weights: estimated BNPL revenue 70%,
active-month ratio 18%, customer spread 5%, ABS household income 3%, IRSAD 2%,
and growth 2%. The fixed definitions are included directly in the ranking notebooks
and existing segment scripts.

## Data periods and interpretation

- The supplied cleaned dataset covers **28 February 2021–26 October 2022**.
- Every production ranking feature, including customer profiles, uses the **19
  complete months from 1 March 2021–30 September 2022**. Partial February and
  October are retained in the cleaned data, but excluded from production features.
- September evaluation builds predictors from **1 March 2021–31 August 2022** and
  measures realised revenue in September. Customer membership, concentration,
  income and IRSAD are rebuilt from the history window, not copied from production.
- September is also inspected by learned-weight and rolling comparisons. Report
  it as a history-only evaluation, **not an independent, never-inspected holdout**.
- Fixed Hybrid/Machine configurations are not overwritten by learned fits.
  The past-revenue benchmark must be included when discussing model performance.

## Environment

The tested local environment uses **Python 3.10 and Java 21**. Use a virtual
environment and install the direct dependencies in `requirements.txt`:

```bash
python3.10 -m venv .proj2_bnpl
source .proj2_bnpl/bin/activate
python -m pip install -r requirements.txt
java -version
```

Select this Python environment as the kernel when running notebooks. For Spark,
set both worker and driver to the selected interpreter before starting Python or
Jupyter:

```bash
export PYSPARK_PYTHON="$(command -v python)"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
```

Each script/notebook sets Spark to UTC. Transaction dates are also stored as
calendar dates, so switching local timezones cannot remove the first evaluation
day. Cleaning can download NLTK WordNet and omw-1.4 on its first run. The ABS
script reuses cached files.

## Inputs

Supply the gitignored raw data under `data/tables/`:

```text
transactions_20210228_20210827_snapshot/
transactions_20210828_20220227_snapshot/
transactions_20220228_20220828_snapshot/
tbl_merchants.parquet
consumer_user_details.parquet
tbl_consumer.csv
consumer_fraud_probability.csv
merchant_fraud_probability.csv
```

Raw data is only read. `data/tables/external/` contains the ABS download cache.
`consumers_sa2/` is the canonical enriched consumer table; no `consumers_abs` alias
or untracked manual file renaming is required.

## Reproduce the project

Run the existing scripts and then run all cells in the notebooks in the order
below. These steps **overwrite derived data, models, figures and notebook outputs**.
Use the same Python environment throughout and save each successfully executed
notebook. Raw input files are only read. No Git operations are required.

The complete main order is:

1. `scripts/process_cleaning.py`
2. `scripts/download_external_sa2.py`
3. `scripts/join_external_sa2.py`
4. `notebooks/01_data_quality_eda.ipynb`
5. `notebooks/abs_sa2.ipynb`
6. `scripts/segment_behaviour.py`
7. `scripts/segment_merchants.py`
8. `notebooks/validation_timesplit.ipynb`
9. `notebooks/learned_weights.ipynb`
10. `notebooks/rolling_validation.ipynb`
11. `notebooks/merchant_ranking_hybrid.ipynb`
12. `notebooks/fraud_models.ipynb`
13. `notebooks/fraud_imputation_check.ipynb`
14. `notebooks/fraud_penalty.ipynb`
15. `scripts/top10_per_segment.py`
16. `notebooks/segmentation.ipynb`
17. `scripts/make_plots.py`
18. `notebooks/final_findings.ipynb`
19. `scripts/compare_variants.py`


Cleaning and ABS preparation are independent. Calendar-window feature functions
and ranking arithmetic are included directly in the relevant notebooks. The
existing `compare_variants.py` script checks cross-file consistency, independently
rebuilds production customer profiles, refreshes legacy v2 filenames and writes
the calculated results section at the end of this README.

Archived notebooks are historical experiments, not prerequisites. Their outputs
are not authoritative current results.

## Authoritative outputs

All main outputs live under `data/tables/processed/`:

| Output | Producer / meaning |
|---|---|
| `merchant_features/` | EDA, production-window commercial features |
| `merchant_behaviour_segments/` | Behavioural clustering |
| `merchant_segments/` | Separate product-category check |
| `merchant_ranking_hybrid_features.parquet` | Production features including customer context |
| `merchant_ranking_hybrid.parquet`, `top100_hybrid.csv`, `top150_hybrid.csv` | Commercial Hybrid ranking |
| `weights_hybrid.csv` | Fixed shipped weights |
| `hybrid_holdout_results.csv` | September evaluation rebuilt from pre-September features |
| `top10_per_segment.csv` | Same weights, percentiles recomputed within each segment |
| `merchant_ranking_hybrid_fraud_adjusted.parquet`, `top100_hybrid_fraud_adjusted.csv` | Separate fraud-adjusted ranking; both include review flags |
| `hybrid_fraud_sensitivity.csv` | Actual threshold/severity sweep |
| `learned_vs_hand_results.csv`, `rolling_validation_results.csv` | Exploratory comparisons |
| `results_summary.json` | Calculated results after consistency checks pass |

The reproduced-results section below is generated from current outputs. Use it and
`notebooks/final_findings.ipynb` for reporting; do not copy historical notebook
numbers into the presentation. Legacy `*_v2` outputs are byte-identical compatibility
copies refreshed by the audit. `merchant_ranking_baseline.parquet` and
`top100_baseline.csv` are refreshed from the same production snapshot using v1 scores.

The presentation `Ass2_ADS_Group 13.pdf` is outside this repository and is not
modified by this pipeline. Recheck its claims against the latest generated results.

## Fraud and model limitations

XGBoost diagnoses **supplied fraud-score tiers**, not independently observed fraud.
Its predictions do not supply ranking loss probabilities. The adopted penalty uses
supplied scores at or above **30**, severity **1.0**, user-day spend allocation and
combined consumer/merchant probabilities. The scoring-window loss rate is applied
to production-window GMV before recomputing the revenue score term.

Missing scores are unknown risk, although the implemented observed-score scenario
assigns them zero loss. Score calibration is unknown, so neither calibrated fraud
forecasts nor a guaranteed loss lower bound are claimed. Spend-band imputation is
sensitivity analysis only. Fraud risk tiers remain review flags rather than exclusions.

## Optional Machine comparison

Prepare the shared inputs and regenerate the Machine notebooks using the existing
tooling, then run these steps:

```text
python notebooks_weve_tried/scripts/build_variants.py
notebooks_weve_tried/merchant_ranking_machine.ipynb
notebooks_weve_tried/fraud_penalty_machine.ipynb
notebooks_weve_tried/fraud_imputation_check_machine.ipynb
python notebooks_weve_tried/scripts/top10_per_segment.py
notebooks_weve_tried/segmentation_machine.ipynb
notebooks_weve_tried/final_findings_machine.ipynb
python scripts/compare_variants.py
```

Outputs are isolated under `data/variants/machine/processed/`. The fixed comparison
uses 72.4% revenue, 21.8% activity and 5.8% spread, with the same feature windows and
scoring rules as Hybrid. Regeneration clears variant notebook outputs; rerun and
save them afterward.

<!-- reproduced-results:start -->
# Reproduced results

Generated by `scripts/compare_variants.py`; do not edit numerical values manually.

All production features use 2021-03-01 to 2022-09-30, in UTC.
September predictors are rebuilt through August; September is reused in exploratory comparisons.

Clean transactions: 14,195,505; modelling-window transactions: 13,440,766; rankable merchants: 4,026.
Top 100 estimated revenue: $28,726,631.15 (31.4833% of ranked revenue).

| Rank | Merchant | Estimated BNPL revenue |
|---|---|---:|
| 1 | Dignissim Maecenas Foundation | $590,673.67 |
| 2 | Leo In Consulting | $581,136.25 |
| 3 | Arcu Sed Eu Incorporated | $432,454.15 |
| 4 | Mauris Non Institute | $544,064.20 |
| 5 | Lacus Consulting | $555,429.64 |

| September picker | Spearman | Capture rate | Top 100 hits |
|---|---:|---:|---:|
| v1 (original score) | 0.739729 | 64.5215% | 46 |
| hybrid (frozen before holdout) | 0.907500 | 91.1149% | 80 |
| baseline: past revenue | 0.939527 | 99.9618% | 99 |
| perfect foresight | 1.000000 | 100.0000% | 100 |

| Segment | Within-segment rank | Merchant | Revenue | Global rank |
|---|---:|---|---:|---:|
| Intermittent | 1 | Non Leo Foundation | $2,814.83 | 2758 |
| Intermittent | 2 | Ullamcorper Magna Sed Incorporated | $11,028.83 | 2642 |
| Intermittent | 3 | Litora Torquent Corporation | $8,457.62 | 2734 |
| Established | 1 | Dignissim Maecenas Foundation | $590,673.67 | 1 |
| Established | 2 | Odio Phasellus Institute | $536,062.14 | 9 |
| Established | 3 | Arcu Sed Eu Incorporated | $432,454.15 | 3 |
| Fading | 1 | Ipsum LLP | $46,451.75 | 922 |
| Fading | 2 | Nascetur Ridiculus Corp. | $14,120.18 | 1233 |
| Fading | 3 | Sem Ut Incorporated | $11,712.21 | 1408 |
| Rising | 1 | Morbi Non Corp. | $29,663.51 | 884 |
| Rising | 2 | Ligula Donec Limited | $37,835.94 | 823 |
| Rising | 3 | Nunc Sit Incorporated | $12,652.72 | 1175 |

| Fraud example | Commercial rank | Adjusted rank | Revenue | Expected loss | Loss/revenue |
|---|---:|---:|---:|---:|---:|
| Semper Auctor PC | 89 | 96 | $355,968.33 | $73,911.72 | 20.7636% |
| Id Erat Etiam Consulting | 99 | 98 | $94,996.76 | $1.90 | 0.0020% |
| Malesuada Vel Associates | 102 | 103 | $184,241.74 | $18.78 | 0.0102% |

Sensitivity settings retain 97–100 commercial Top 100 members.

| Diagnostic threshold | Consumer single-day average precision | Merchant out-of-fold average precision |
|---|---:|---:|
| ≥30 | 0.970661 | 0.975881 |
| ≥50 | 0.916194 | 0.910838 |
| ≥70 | 0.762138 | 0.580106 |

Fraud-adjusted shortlist overlap: 99/100; expected loss: $266,358.59.
Leaving commercial Top 100: [{'name': 'Rutrum Magna Cras LLC', 'rank_overall': 100}]; entering: [{'name': 'Semper Cursus Integer Limited', 'rank_adjusted': 100}].

Authoritative files live in `data/tables/processed/`: `top100_hybrid.csv`, `top10_per_segment.csv`,
`top100_hybrid_fraud_adjusted.csv`, `hybrid_holdout_results.csv` and `results_summary.json`.
Legacy v2 filenames are byte-identical compatibility copies, refreshed after this audit passes.

The supplied presentation PDF is external to the repository and has not been edited. Its values must be compared with this new feature snapshot.
<!-- reproduced-results:end -->
