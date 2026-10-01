# Fraud: classifiers, merchant risk and the ranking penalty

**Short version:** don't impute the fraud scores. Use them as flags, and turn them into an expected dollar loss in the ranking. With the default settings, the penalty keeps 99 of v2's top 100 and mostly reorders the middle and tail of the ranking.

## Why not imputation

- **The scores are a pre-screened sample.** Consumer scores exist for 34,765 user-days, about 0.25% of transactions, and none is below 8.29%. Merchant scores exist for 114 merchant-days across 61 merchants, and none is below 18%.
- **Imputing would extrapolate.** A model fit only on flagged rows gives every ordinary transaction a 10–15% score.
- **The regression wasn't good enough.** Log-target OLS, random forest and XGBoost reached an RMSE of about 7, which is R² ≈ 0.5. That isn't precise enough to use a prediction as a probability.
- **The scores measure size.** Both scores follow how large a transaction or day is **relative to that user's or merchant's usual**. On days with a single transaction, the consumer score has a 0.91 rank correlation with log transaction value.

## Linking scores to transactions

Scores are recorded per user per day, and `order_datetime` has no time of day. A score links to one exact transaction only when that user made a single transaction that day, which is 27% of scored days. On shared days, the consumer classifier picks out one transaction: the day's top transaction has a median score of 0.94, against 0.06 for the next.

Side effect: the "earliest transaction that day" rule in `imputation.ipynb` is effectively arbitrary.

## Notebooks

| Notebook | What it does | Output |
|---|---|---|
| `fraud_classifier.ipynb` | Per-transaction consumer tier classifier. Three models for score ≥30, ≥50 and ≥70; split by user; each user-day weighted to count once | none |
| `merchant_risk.ipynb` | Merchant-level summary features and a classifier for "is this merchant ever scored?"; gives each merchant a risk tier | `merchant_risk` |
| `merchant_fraud_penalty.ipynb` | Merchant tier classifier on the 114 scored days; expected fraud loss; fraud-adjusted v2 ranking | `merchant_ranking_v2_fraud_adjusted.parquet`, `top100_v2_fraud_adjusted.csv` |

Run order: `join_external_sa2.py` → `01_data_quality_eda.ipynb` → `merchant_ranking_v2.ipynb` → `merchant_risk.ipynb` → `merchant_fraud_penalty.ipynb`.

## Results

PR-AUC measures how well a model ranks positives first. A random ranking scores the prevalence, shown in brackets.

**Consumer tier classifier.** Tested on held-out users; "exact" means days with a single transaction.

| Cutoff | Exact days | All days |
|---|---|---|
| ≥30 | 0.97 (0.07) | 0.96 (0.06) |
| ≥50 | 0.89 (0.02) | 0.88 (0.02) |
| ≥70 | 0.93 (0.003) | 0.70 (0.003) |

≥70 has only 20 test days, so its numbers are noisy.

**Merchant tier classifier.** 114 scored days, with each merchant held out in turn:

| Cutoff | Scored days | PR-AUC |
|---|---|---|
| ≥30 | 73 | 0.96 (0.64) |
| ≥50 | 27 | 0.91–0.93 (0.24) |
| ≥70 | 7 | too few to trust |

At ≥70, a single transaction is the merchant's whole day of revenue.

**Merchant "ever scored?" classifier:**

| Features | PR-AUC |
|---|---|
| Spread and variance features alone | 0.15 (0.014) |
| + size | 0.76 |

Size dominates. Scored merchants are big-ticket merchants, so the `suspect` flag is partly a size flag. Don't use it as a penalty.

**Merchant risk tiers:**

| Tier | Rule | Merchants |
|---|---|---|
| high | highest score ≥70 | 7 |
| flagged | 50–70 | 18 |
| watch | 30–50 | 21 |
| scored_low | all scores <30 | 15 |
| suspect | never scored, but looks like scored merchants | 13 |

## The penalty

For each transaction:

- **P(fraud)** = 1 − (1 − p_merchant)(1 − p_consumer).
- **Expected loss** = severity × P(fraud) × value. A fraudulent BNPL transaction costs the principal, not just the fee.
- **Applied to the ranking:**
  - fraud-adjusted revenue = `estimated_bnpl_revenue` − loss rate × `total_value`
  - it replaces the 70% revenue term in the v2 score; every other term is unchanged.

The scores' calibration is unknown, so quote the sweep, not a single number:

| Scores counted | Loss severity | Spearman vs v2 | Top-100 overlap | Loss > revenue |
|---|---|---|---|---|
| all | 1.0 | 0.853 | 97 | 553 |
| all | 0.25 | 0.961 | 99 | 214 |
| **≥30 (default)** | **1.0** | **0.975** | **99** | **171** |
| ≥50 | 1.0 | 0.981 | 99 | 103 |

- **Consumer scores make up more than 95% of expected loss.** They follow transaction size, so counting the 8–15% "scored low" days hits big-ticket merchants hardest.
- **The fraud-adjusted top 100** has 5 `watch`, 9 `scored_low` and 1 `suspect` merchant. These are review flags, not exclusions.

## Limitations

- **The expected loss is a lower bound.** Unscored transactions count as 0, and scores cover only Feb 2021 – Feb 2022.
- **The merchant penalty can't be validated.** The test window has no fraud scores.
- **Look-ahead features.** Features use the whole period, including days after a given score. That's fine for description, but not for a live system.
- **Unmatched merchants.** 13 of the 61 scored merchants have no row in `merchants_clean`, so they can't be ranked.
