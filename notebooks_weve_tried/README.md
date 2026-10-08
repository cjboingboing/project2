# Notebooks we've tried

Everything in this folder was built, run and measured, and then **not** shipped. It is kept
because the reasoning behind the shipped ranking is only legible next to the things it was
chosen over.

The pipeline that produces the submitted results is `notebooks/`.

## What is in here

| Notebook | What it was | Why it is not the shipped version |
| --- | --- | --- |
| `merchant_ranking_baseline.ipynb` | the first ranking (v1) — ten features, weights chosen by hand | Tested on a time split and it lost badly: 74.3% capture against 99.9% for simply ranking by past revenue. Two faults were found: shrinkage applied in dollar space pushed thin-evidence merchants *up*, and revenue carried only 30% of the weight. Both are fixed in the shipped v2. |
| `*_machine.ipynb` (5 notebooks) | the whole pipeline under the **pure learned weights** | Scores highest of the three (91.5% capture against the hybrid's 91.1%) but sets both ABS features to zero, which conflicts with the brief's requirement that external data inform the ranking. Kept so the trade-off can be shown with output rather than argued. |
| `scripts/top10_per_segment.py` | the segment ranking under those same pure weights | Reads from `data/variants/machine/processed`. |
| `scripts/build_variants.py` | archived tooling | How the variant folders were generated from `notebooks/` in the first place. |

## The decision these notebooks produced

| Weights | Capture rate (Sept) | Precision@100 | Uses ABS |
| --- | --- | --- | --- |
| v1, by hand | 64.5% | 46 | yes |
| v2, by hand | 87.8% | 74 | yes |
| **hybrid — shipped** | **91.1%** | **80** | **yes** |
| pure learned | 91.5% | 80 | no |
| baseline: past revenue | 100% | 99 | n/a |

Keeping the ABS features costs **0.4 points** of capture rate and no correct picks at all, which is
why the hybrid is the one in `notebooks/`.

The row that should be read twice is the last one: ranking merchants by past revenue alone beats
everything we built. That finding is in the write-up, not buried here.

## Running the machine variant

It writes to its own folder (`data/variants/machine/processed`) and touches nothing the shipped
pipeline produces:

```
notebooks_weve_tried/merchant_ranking_machine.ipynb
notebooks_weve_tried/fraud_penalty_machine.ipynb
notebooks_weve_tried/fraud_imputation_check_machine.ipynb
notebooks_weve_tried/segmentation_machine.ipynb
python notebooks_weve_tried/scripts/top10_per_segment.py
notebooks_weve_tried/final_findings_machine.ipynb
```

`merchant_ranking_baseline.ipynb` reads the shared tables and writes nothing, so it can be run on
its own.

The two notebooks that produced the shipped weights — `learned_weights.ipynb` (where they were
fitted) and `rolling_validation.ipynb` (where they were tested across four windows) — now live in
`notebooks/`, because they are the justification for the shipped model rather than an alternative
to it.
