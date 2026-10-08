# Running the Project

## 1. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

PySpark also requires **Java 17**.

Check that Java is installed with:

```bash
java -version
```

---

## 2. Add the Raw Data

Place the supplied raw datasets under:

```text
data/tables/
```

The raw data is gitignored and should **not** be committed to the repository.

The transaction data is stored across the supplied transaction folders, together with files such as:

```text
data/tables/
├── transactions_20210228_20210827/
├── transactions_20210828_20220227/
├── transactions_20220228_20220828/
├── consumer_fraud_probability.csv
├── merchant_fraud_probability.csv
├── consumer_user_details.parquet
├── tbl_consumer.csv
└── ...
```

---

## 3. Download and Join ABS External Data

Download the ABS SA2 data:

```bash
python scripts/download_external_sa2.py
```

Then create the postcode / SA2 correspondence and join the external data:

```bash
python scripts/join_external_sa2.py
```

Main outputs are written under:

```text
data/tables/processed/
```

including:

```text
abs_census_sa2/
abs_poa_sa2_correspondence/
abs_poa_sa2_primary/
consumers_sa2/
```

The first ABS download is approximately **160 MB**. Subsequent runs use the cached files.

---

## 4. Clean the Raw Data

Run:

```bash
python scripts/process_cleaning.py
```

This produces the main cleaned datasets:

```text
data/tables/processed/
├── transactions_clean/
├── merchants_clean/
├── consumers/
├── consumer_fraud_probability/
├── merchant_fraud_probability/
└── duplicate_order_ids/
```

The cleaned transaction table contains approximately **14.2 million transactions**.

---

## 5. Run Data Quality and Feature Engineering

Open and run all cells in:

```text
notebooks/01_data_quality_eda.ipynb
```

This performs the main data-quality checks and creates merchant-level features.

Main output:

```text
data/tables/processed/merchant_features/
```

---

## 6. Run the ABS Analysis

Open and run:

```text
notebooks/abs_sa2.ipynb
```

This checks the ABS / SA2 enrichment, postcode matching and socioeconomic variables used in the ranking.

---

## 7. Build the Behavioural Segments

Run:

```bash
python scripts/segment_behaviour.py
```

The supporting segmentation analysis can also be reviewed in:

```text
notebooks/segmentation.ipynb
```

Main output:

```text
data/tables/processed/merchant_behaviour_segments/
```

The final four behavioural segments are:

- `Established`
- `Rising`
- `Fading`
- `Intermittent`

---

## 8. Run the Product-Category Check

Run:

```bash
python scripts/segment_merchants.py
```

Main output:

```text
data/tables/processed/merchant_segments/
```

This is a supporting check used to confirm that the behavioural segments are not simply reproducing merchant product categories.

---

## 9. Build the Final Merchant Ranking

Open and run:

```text
notebooks/merchant_ranking_v2.ipynb
```

This applies the final Hybrid merchant-ranking weights.

The final Hybrid weights are:

| Feature | Weight |
|---|---:|
| Estimated BNPL revenue | 70% |
| Active-month ratio | 18% |
| Customer spread | 5% |
| ABS household income | 3% |
| ABS IRSAD | 2% |
| Growth | 2% |

Main outputs include:

```text
data/tables/processed/
├── merchant_ranking_v2.parquet
├── merchant_ranking_v2_features.parquet
├── top100_v2.csv
├── top150_v2.csv
└── weights_v2.csv
```

---

## 10. Run Time-Based Validation

Open and run:

```text
notebooks/validation_timesplit.ipynb
```

This evaluates the ranking using later data rather than the data used to construct the ranking.

The final September holdout results are saved to:

```text
data/tables/processed/v2_holdout_results.csv
```

The final comparison includes:

- v1
- hand-weighted v2
- Hybrid
- pure learned / Machine
- past-revenue baseline

---

## 11. Review the Learned-Weight Experiments

The experiments used to derive the machine-learned and Hybrid weights are stored under:

```text
notebooks_weve_tried/
```

The most relevant notebooks are:

```text
notebooks_weve_tried/learned_weights.ipynb
notebooks_weve_tried/rolling_validation.ipynb
notebooks_weve_tried/merchant_ranking_v2_machine.ipynb
```

These are kept for comparison and reproducibility.

They are **not the final shipped ranking pipeline**.

The final shipped ranking uses the Hybrid weights in:

```text
notebooks/merchant_ranking_v2.ipynb
```

---

## 12. Run the Fraud Diagnostics

Open and run:

```text
notebooks/fraud_models.ipynb
```

This uses XGBoost to understand which behaviours are associated with the supplied fraud scores.

The XGBoost models are **diagnostic only**.

Their predictions do not enter the final merchant ranking.

---

## 13. Run the Fraud Sensitivity Analysis

Open and run:

```text
notebooks/fraud_imputation_check.ipynb
```

This tests:

- missing-score behaviour
- spend-band imputation
- fraud-score thresholds such as `30`, `50` and `70`
- the fraud-scoring regime change around 28 August 2021

Imputed fraud scores are used for **sensitivity analysis only**.

They are not used as the final ranking input.

---

## 14. Apply the Fraud Penalty

Open and run:

```text
notebooks/fraud_penalty.ipynb
```

The fraud penalty uses the supplied fraud scores to calculate:

```text
expected fraud loss
= fraud chance × purchase amount
```

and then:

```text
fraud-adjusted BNPL revenue
= estimated BNPL revenue − expected fraud loss
```

The Hybrid ranking is then recomputed using fraud-adjusted revenue.

Main outputs include:

```text
data/tables/processed/
├── merchant_ranking_v2_fraud_adjusted.parquet
└── top100_v2_fraud_adjusted.csv
```

---

## 15. Generate the Top 10 Merchants per Segment

Run:

```bash
python scripts/top10_per_segment.py
```

Main output:

```text
data/tables/processed/top10_per_segment.csv
```

This applies the same final Hybrid weights within each behavioural segment.

---

## 16. Generate Presentation Figures

Run:

```bash
python scripts/make_plots.py
```

Figures are written under:

```text
plots/
├── eda/
├── fraud/
├── models/
├── ranking/
└── segments/
```

---

## 17. Review the Final Findings

Finally, open and run:

```text
notebooks/final_findings.ipynb
```

This notebook brings together:

- cleaned-data findings
- ABS enrichment
- behavioural segmentation
- ranking performance
- September holdout results
- fraud diagnostics
- fraud-adjusted ranking
- final merchant recommendations

This is the main notebook to use when reviewing the project or preparing the presentation.

---

## Recommended Run Order

For a complete run from raw data to final outputs:

```text
1.  pip install -r requirements.txt

2.  python scripts/download_external_sa2.py
3.  python scripts/join_external_sa2.py
4.  python scripts/process_cleaning.py

5.  notebooks/01_data_quality_eda.ipynb
6.  notebooks/abs_sa2.ipynb

7.  python scripts/segment_behaviour.py
8.  python scripts/segment_merchants.py

9.  notebooks/merchant_ranking_v2.ipynb
10. notebooks/validation_timesplit.ipynb

11. notebooks/fraud_models.ipynb
12. notebooks/fraud_imputation_check.ipynb
13. notebooks/fraud_penalty.ipynb

14. python scripts/top10_per_segment.py
15. python scripts/make_plots.py

16. notebooks/final_findings.ipynb
```

---

## Main Final Outputs

After the full pipeline has run, the key outputs should be available under:

```text
data/tables/processed/
```

including:

```text
merchant_behaviour_segments/
merchant_features/
merchant_ranking_v2.parquet
merchant_ranking_v2_fraud_adjusted.parquet

top10_per_segment.csv
top100_v2.csv
top100_v2_fraud_adjusted.csv
top150_v2.csv

v2_holdout_results.csv
weights_v2.csv
```