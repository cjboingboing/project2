# MAST30034 Assignment 2: Merchant Onboarding Analysis

This project develops a data-driven ranking system to identify the **Top 100 merchants out of 4,026 rankable merchants** for a Buy Now Pay Later (BNPL) provider.

The objective is to identify merchants that offer strong commercial value while considering transaction activity, customer behaviour, growth, demographic context, and potential fraud risks.

The final framework uses a **Hybrid ranking model** with fixed weights:

| Feature | Weight |
|---|---:|
| Estimated BNPL revenue | 70% |
| Active-month ratio | 18% |
| Customer spread | 5% |
| ABS household income | 3% |
| IRSAD | 2% |
| Merchant growth | 2% |
| **Total** | **100%** |

The project also includes merchant segmentation, historical ranking evaluation, fraud-risk analysis, and a final onboarding recommendation.

## Project Timeline

The project was organised into six development sprints, followed by the final presentation in Week 11.

| Period | Sprint | Tasks and Deliverables |
|---|---|---|
| Week 6 | Sprint 1 | Explored the supplied datasets, began exploratory analysis, established the repository structure, and identified relevant external ABS datasets. |
| Week 7 | Sprint 2 | Developed the data-cleaning pipeline, investigated missing values and outliers, performed data quality checks, and integrated ABS geographic information. |
| Week 8 | Sprint 3 | Improved the ETL pipeline, prepared cleaned transaction and customer datasets, and investigated the supplied fraud records. |
| Week 9 | Sprint 4 | Engineered merchant features, developed the initial ranking methodology, and evaluated early ranking performance. |
| Mid-semester break | Sprint 5 | Refined the Hybrid ranking, investigated model weights, developed merchant segmentation, and produced within-segment rankings. |
| Week 10 | Sprint 6 | Finalised the Top 100 shortlist, performed model comparisons and fraud sensitivity analysis, checked results, and prepared the final presentation. |
| **Week 11** | **Final Presentation** | Presented the methodology, data preparation, Hybrid ranking, merchant segmentation, fraud analysis, model evaluation, limitations, and recommendations. |

## Generative AI Usage Declaration

Generative AI tools were used to assist with code development, debugging and troubleshooting.

AI-assisted suggestions were subject to review and testing, with the group retaining responsibility for the final implementation, methodology, interpretation, and results.

No project datasets or confidential information were intentionally shared with external AI tools. AI assistance was limited to development-related tasks, such as reviewing code structures and resolving errors.

## Data Periods and Interpretation

The cleaned transaction dataset covers **28 February 2021 to 26 October 2022**.

For the production merchant ranking, the intended modelling period is:

**1 March 2021 – 30 September 2022 (19 complete months).**

Partial months in February 2021 and October 2022 are excluded from production ranking features.

For the September evaluation:

- Historical predictors use transactions from 1 March 2021 to 31 August 2022.
- September 2022 revenue is used to evaluate the selected merchants.
- The evaluation measures how well a ranking based on historical information captures future revenue.
- The past-revenue baseline is included as a benchmark.

September was also inspected during exploratory model comparisons, so this evaluation should not be described as a completely independent, untouched holdout.

### Timezone Handling

All date processing should use UTC consistently.

During data review, an unspecified timezone in the original cleaning execution initially caused **26,571 transactions dated 1 September 2022** to be excluded from the evaluation.

A subsequent UTC-controlled execution resolved this issue and successfully completed the main pipeline.

The historical ranking and validation results reported below are based on the successful UTC-controlled execution.

## Environment

The tested execution environment used:

- Python 3.10
- Java 21
- PySpark
- Jupyter Notebook
- Dependencies listed in `requirements.txt`

Create the Python environment:

```bash
python3.10 -m venv .proj2_bnpl
source .proj2_bnpl/bin/activate
python -m pip install -r requirements.txt
java -version
```

Set the Spark driver and worker interpreters before running Python or Jupyter:

```bash
export PYSPARK_PYTHON="$(command -v python)"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
```

Run all cleaning, preprocessing, feature engineering, and evaluation steps with consistent UTC settings.

Exact dependency versions may affect some model diagnostics, particularly XGBoost results.

## Input Data

The raw datasets are stored under `data/tables/`:

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

These inputs contain transaction records, merchant information, consumer information, and supplied fraud scores.

The project also incorporates external Australian Bureau of Statistics (ABS) data for demographic and socioeconomic context.

Raw data files should remain unchanged. Processed datasets and generated results are stored separately.

## Data Cleaning and Preparation

The cleaning pipeline prepares transaction, merchant, and consumer data for analysis and ranking.

The main tasks include:

1. Combining transaction snapshots.
2. Investigating missing values and duplicate records.
3. Checking invalid values and transaction dates.
4. Joining transaction records with merchant and consumer information.
5. Integrating postcode-level and SA2-level ABS information.
6. Preparing merchant-level features for ranking.
7. Retaining supplied fraud information for separate risk analysis.

### Cleaned Dataset Summary

| Metric | Value |
|---|---:|
| Clean transactions | 14,195,505 |
| Transactions in the 19-month modelling window | 13,440,766 |
| Merchants with transactions | 4,422 |
| Merchants without take-rate information | 396 |
| **Rankable merchants** | **4,026** |

Merchants without take-rate information cannot have their estimated BNPL revenue calculated reliably and are excluded from the commercial ranking.

## External ABS Data

The project integrates ABS Census and Socio-Economic Indexes for Areas (SEIFA) information to provide additional customer-area context.

Consumer postcode information is linked to Statistical Area Level 2 (SA2) regions using a population-weighted postcode-to-Mesh-Block-to-SA2 mapping approach.

The data reports:

- 2,644 postcodes
- 5,904 postcode-to-SA2 pairs
- 83.4% of consumers assigned to an SA2
- 45.1% of postcodes associated with only one SA2

The Hybrid ranking allocates a combined **5% weight to ABS household income and IRSAD**.

These variables provide geographic and socioeconomic context, but have weak predictive relationships with merchant revenue in this dataset.

ABS values describe geographic areas rather than the actual income or socioeconomic status of individual customers.

## Merchant Ranking Methodology

### Estimated BNPL Revenue

Merchant value is measured using estimated BNPL revenue rather than transaction volume or sales alone.

For merchant \(m\):

**Estimated BNPL Revenue (m) = GMV (m) × Take Rate (m)**

For example, a merchant generating $1,000,000 in sales with a 5% take rate produces an estimated $50,000 in BNPL revenue.

### Hybrid Ranking

The Hybrid model combines six features:

| Feature | Description |
|---|---|
| Estimated revenue | Commercial value generated through transaction volume and take rate |
| Active-month ratio | Consistency of merchant activity across the modelling window |
| Customer spread | Diversity of the merchant's customer base |
| ABS household income | Area-level household income associated with customers |
| IRSAD | Area-level socioeconomic advantage and disadvantage |
| Growth | Changes in merchant activity or revenue over time |

Feature scores are normalised and combined using the fixed Hybrid weights.

The resulting score determines each merchant's overall commercial rank.

### Model Development

The initial ranking was tested using later transaction periods to evaluate its ability to identify valuable merchants.

The presentation reports an initial development test with:

- 59.8% revenue capture.
- 42 merchants overlapping with the realised Top 100.

Following further evaluation and adjustments, the final Hybrid model achieved higher September performance.

The initial development test and the September evaluation are separate comparisons and should not be interpreted as identical experiments.

## Model Evaluation

### September 2022 Evaluation

The ranking is constructed using historical information available before September 2022, then compared with realised September revenue.

The main evaluation metrics are:

**Spearman rank correlation:** Measures how closely the predicted merchant ranking agrees with the ranking based on realised revenue.

**Revenue capture rate:** Measures the proportion of realised September BNPL revenue captured by the selected 100 merchants, relative to the revenue earned by the perfect-foresight Top 100.

**Top 100 hits:** Counts how many selected merchants are also in the realised September Top 100.

### Verified September Results

The following Hybrid and baseline metrics were confirmed in the execution.

| Ranking approach | Spearman | Revenue capture | Top 100 hits |
|---|---:|---:|---:|
| Original v1 | 0.739729 | 64.5215% | 46 |
| **Hybrid** | **0.907500** | **91.1149%** | **80** |
| Past-revenue baseline | 0.939527 | 99.9618% | 99 |
| Perfect foresight | 1.000000 | 100.0000% | 100 |

The final Hybrid model significantly improves on the original v1 ranking.

However, the **past-revenue baseline performs better than Hybrid** in the September evaluation.

The review also reported that the past-revenue baseline outperformed Hybrid across the evaluated rolling folds.

This suggests that recent historical revenue is a particularly strong predictor of short-term merchant revenue in this dataset.

The Hybrid model is retained as a broader merchant assessment framework that includes activity consistency, growth, customer behaviour, and demographic context rather than optimising only for short-term revenue capture.

### Machine Comparison

The presentation also includes an alternative Machine ranking using weights of:

| Feature | Weight |
|---|---:|
| Estimated revenue | 72.4% |
| Active-month ratio | 21.8% |
| Customer spread | 5.8% |

The data reports approximately 91.5% revenue capture and 0.914 Spearman correlation for Machine.

These Machine results are presentation-reported findings. The earlier execution did not run the optional Machine branch, so its metrics were not independently reproduced in that review.

## Final Top 100 Merchant Ranking

### Verified Main Results

The following values come from the execution.

| Metric | Result |
|---|---:|
| Rankable merchants | **4,026** |
| Selected merchants | **100** |
| **Top 100 estimated BNPL revenue** | **$28,816,080.63** |
| **Share of total ranked revenue** | **31.5813%** |
| Hybrid September Spearman | 0.907500 |
| Hybrid September capture rate | 91.1149% |
| Hybrid Top 100 hits | 80 |

The execution reproduced the saved overall merchant ranks and Top 100 selections.

### Top Five Merchants

The following ranking and merchant-level figures are reported in the presentation. The review separately confirms the first-ranked merchant and its revenue.

| Rank | Merchant | Estimated BNPL revenue |
|---|---|---:|
| 1 | Dignissim Maecenas Foundation | $590,673.67 |
| 2 | Leo In Consulting | $581,136.25 |
| 3 | Arcu Sed Eu Incorporated | $432,454.15 |
| 4 | Mauris Non Institute | $544,064.20 |
| 5 | Lacus Consulting | $555,429.64 |

The ranking is based on the combined Hybrid score rather than revenue alone. Consequently, merchants with lower revenue may rank above merchants with higher revenue.

The commercial Top 100 shortlist is intended to support onboarding decisions, not automatically determine merchant eligibility.

## Merchant Segmentation

Merchant segmentation identifies different patterns of trading behaviour that may not be visible from the overall ranking alone.

The project tested alternative clustering structures and selected four behavioural segments.

### Segment Results

The following segment sizes were reproduced in execution.

| Segment | Merchants | Description |
|---|---:|---|
| Established | 1,616 | Large, consistently active merchants |
| Rising | 969 | Merchants with stronger recent growth |
| Fading | 888 | Merchants with declining or relatively flat activity |
| Intermittent | 553 | Merchants with less consistent trading activity |
| **Total** | **4,026** | |

The presentation reports that Established merchants account for approximately 87.9% of ranked revenue.

This concentration explains why the overall Top 100 is dominated by Established merchants.

### Within-Segment Ranking

Each segment is ranked separately using the same fixed Hybrid weights, with percentile scores recalculated within the segment.

This allows merchants to be compared with other merchants exhibiting similar behaviour.

For example, the presentation identifies **Morbi Non Corp.** as:

- Rank 1 in the Rising segment.
- Rank 884 in the overall ranking.
- Approximately $29.7k in estimated BNPL revenue.

Within-segment rankings help identify smaller or growing merchants that are unlikely to appear in the overall Top 100.

### Segment Reproducibility

The code successfully reproduced the segment counts, but the fresh Top 10 shortlists did not completely match the saved segment outputs.

The reported overlaps across the four segment lists were:

- 10/10
- 6/10
- 7/10
- 7/10

Therefore, the segment counts are reproduced, while some individual within-segment shortlist memberships depend on the execution snapshot.

The generated `top10_per_segment.csv` should be used when reporting results from a specific run.

## Fraud Risk Analysis

Fraud risk is analysed separately from the commercial merchant ranking.

The supplied fraud data has substantial missing coverage, with approximately **99% of transactions lacking a fraud score**.

Missing fraud information is treated as unknown risk rather than evidence that a transaction or merchant is safe.

The presentation also identifies a change in supplied fraud-score behaviour around 28 August 2021.

### XGBoost Fraud Diagnostics

XGBoost is used to examine patterns associated with the supplied fraud scores.

It is **not used to generate the final merchant ranking**.

The diagnostic models investigate score thresholds of:

- 30
- 50
- 70

The models predict whether observations exceed the corresponding supplied-score thresholds, not whether independently verified fraud occurred.

### Verified XGBoost Results

The following average-precision results come from execution.

| Threshold | Consumer single-transaction days | Merchant out-of-fold |
|---|---:|---:|
| ≥30 | 0.969776 | 0.973982 |
| ≥50 | 0.917566 | 0.907900 |
| ≥70 | 0.801061 | 0.591373 |

These metrics indicate how well the diagnostic models identify the supplied high-score categories.

They do not establish calibrated real-world fraud probabilities.

### Fraud Penalty Method

The fraud-adjusted ranking uses a scenario-based expected-loss estimate derived from the supplied fraud scores.

The adopted scenario uses:

- Minimum score threshold of 30.
- Loss severity of 1.0.
- User-day spend allocation.
- Combined consumer and merchant fraud-score information.

The estimated loss affects the revenue component before the Hybrid score is recalculated.

Where supplied scores are missing, the implemented observed-score scenario assigns zero estimated loss. This is a modelling assumption and should not be interpreted as zero actual fraud risk.

Fraud-score imputation was investigated as sensitivity analysis and was not used as the final ranking input.

## Fraud-Adjusted Ranking Results

### Results

| Metric | Result |
|---|---:|
| Merchants affected by fraud adjustment | 1,117 |
| Merchants with reduced ranking scores | 245 |
| Original Top 100 retained | **99/100** |
| Estimated fraud loss | **$266,370.55** |
| Net estimated revenue | **$28,460,465.11** |

The fraud adjustment changes only one merchant in the Top 100 shortlist.

According to the presentation:

- Rutrum Magna Cras LLC leaves the commercial Top 100.
- Semper Cursus Integer Limited enters at adjusted rank 100.

The presentation also reports that 97–100 merchants remain in the Top 100 across the tested fraud threshold and severity settings.

### Example: Semper Auctor PC

The presentation identifies Semper Auctor PC as a merchant requiring additional review.

| Metric | Value |
|---|---:|
| Commercial rank | 89 |
| Fraud-adjusted rank | 96 |
| Estimated BNPL revenue | $355,968.33 |
| Estimated fraud loss | $73,911.72 |
| Loss-to-revenue ratio | 20.8% |

This illustrates how a commercially attractive merchant may still warrant closer risk investigation.

Fraud-adjusted rankings and flags are used as decision-support information, not automatic onboarding exclusions.

## Reproducing the Project

Run the following scripts and notebooks in sequence, using the same Python environment and UTC settings.

The documented main pipeline contains 19 steps:

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

All 19 main steps completed successfully in the execution.

### Main Outputs

The key files are stored under `data/tables/processed/`:

| Output | Description |
|---|---|
| `merchant_features/` | Merchant commercial features |
| `merchant_behaviour_segments/` | Behavioural segmentation |
| `merchant_segments/` | Product-category segmentation |
| `merchant_ranking_hybrid_features.parquet` | Hybrid ranking input features |
| `merchant_ranking_hybrid.parquet` | Overall merchant ranking |
| `top100_hybrid.csv` | Top 100 commercial merchants |
| `top150_hybrid.csv` | Top 150 commercial merchants |
| `weights_hybrid.csv` | Fixed Hybrid weights |
| `hybrid_holdout_results.csv` | September historical evaluation |
| `top10_per_segment.csv` | Within-segment merchant rankings |
| `merchant_ranking_hybrid_fraud_adjusted.parquet` | Fraud-adjusted ranking |
| `top100_hybrid_fraud_adjusted.csv` | Fraud-adjusted Top 100 |
| `hybrid_fraud_sensitivity.csv` | Fraud sensitivity analysis |
| `learned_vs_hand_results.csv` | Weight comparison results |
| `rolling_validation_results.csv` | Historical rolling evaluation |
| `results_summary.json` | Consolidated calculated results |

Generated outputs should be refreshed together to avoid mixing results from different execution snapshots.

## Limitations

The project has several limitations.

**1. Historical revenue remains a stronger short-term predictor.**

The past-revenue benchmark outperforms the final Hybrid ranking in September and the evaluated rolling folds. This limits claims that the Hybrid model is the best approach for predicting short-term revenue.

**2. Fraud scores are incomplete.**

Approximately 99% of transactions lack supplied fraud scores. The risk adjustment therefore cannot provide a complete estimate of real fraud exposure.

**3. Fraud-score calibration is unknown.**

The supplied scores are not independently verified fraud outcomes. XGBoost diagnostics cannot establish the true probability of fraud.

**4. ABS data is area-level information.**

Household income and IRSAD describe geographic areas rather than individual consumers and provide relatively weak revenue-prediction signals.

**5. Historical evaluation is limited.**

The September evaluation covers one month, and that period was also inspected during exploratory modelling. It should not be interpreted as a fully independent final test.

**6. Merchant segments simplify behaviour.**

Four clusters provide useful summaries but cannot capture every merchant's unique commercial circumstances.

**7. Execution differences affect some outputs.**

The review identified differences in saved segment shortlists and XGBoost diagnostic metrics. Results should be associated with the execution environment and output snapshot used to generate them.

## Final Recommendation

The recommended onboarding process is:

1. Use the Hybrid Top 100 as the main commercial shortlist.
2. Review Top 10 merchants within each behavioural segment to identify additional opportunities.
3. Examine supplied fraud scores, sensitivity results, and data-quality flags before onboarding.
4. Compare ranking results against the historical-revenue baseline.
5. Apply human review before final onboarding decisions.

The Hybrid ranking provides a structured commercial assessment, while segmentation and fraud analysis provide additional context for selecting merchants.

**Final onboarding decisions should remain subject to human assessment rather than relying entirely on automated scores.**

## Conclusion

This project develops a merchant-ranking framework that reduces 4,026 eligible merchants to a shortlist of 100 onboarding candidates.

The execution confirms that the Top 100 generates approximately **$28.82 million in estimated BNPL revenue**, representing **31.5813% of total ranked revenue**.

The Hybrid ranking achieves **91.1149% September revenue capture**, with **80 merchants overlapping the realised Top 100**.

Although the past-revenue baseline performs better in short-term revenue prediction, the Hybrid system combines commercial value with additional merchant characteristics that may be useful for broader onboarding assessment.

Merchant segmentation identifies opportunities outside the overall Top 100, while fraud analysis supports additional review of merchants with potential risk exposure.

The final system is designed to support a transparent, reproducible, and evidence-based merchant selection process.

---

## Verified Execution Results

These values are the primary numerical results reported in this README.

| Result | Verified execution |
|---|---:|
| Rankable merchants | 4,026 |
| Top 100 estimated BNPL revenue | $28,816,080.63 |
| Share of ranked revenue | 31.5813% |
| Hybrid September Spearman | 0.907500 |
| Hybrid September capture rate | 91.1149% |
| Hybrid Top 100 hits | 80 |
| Past-revenue baseline capture rate | 99.9618% |
| Past-revenue baseline hits | 99 |
| Fraud-adjusted Top 100 overlap | 99/100 |
| Fraud-adjusted estimated loss | $266,370.55 |
| Fraud-adjusted net estimated revenue | $28,460,465.11 |

### Verified Segment Counts

| Segment | Merchants |
|---|---:|
| Established | 1,616 |
| Rising | 969 |
| Fading | 888 |
| Intermittent | 553 |

### Verified Fraud Diagnostic Metrics

| Threshold | Consumer AP | Merchant AP |
|---|---:|---:|
| ≥30 | 0.969776 | 0.973982 |
| ≥50 | 0.917566 | 0.907900 |
| ≥70 | 0.801061 | 0.591373 |
