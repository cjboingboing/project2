# Historical and optional comparisons

The main pipeline and authoritative results are described in `../README.md`.

- `merchant_ranking_baseline.ipynb` retains the historical v1 method. It reads
  production-window features. The main output audit also refreshes v1 exports
  from the canonical production snapshot.
- The five `*_machine.ipynb` notebooks and their segment helper implement a
  **fixed** comparator: 72.4% revenue, 21.8% active months, 5.8% customer spread.
  They use the current main feature/scoring logic and write only variant outputs.
- `scripts/build_variants.py` regenerates these optional sources from main notebooks.
  It uses the repository location, not a machine-specific absolute path.

Follow the manual Machine run order in the repository README. The existing
`notebooks_weve_tried/scripts/build_variants.py` prepares shared inputs and generates
variant notebooks; execute those notebooks and then compare the results.
The diagnostic models are shared; ranking, risk and shortlist outputs are isolated
under `data/variants/machine/processed/`.

Frozen Machine weights are a historical comparator, not a claim that rerunning
rolling fits must yield the same exact coefficients. The current rolling notebook
uses the same calendar definitions as the main ranking. September is inspected
across experiments, so it is not an independent model-selection holdout.

The plain past-revenue benchmark is included in every evaluation. Do not claim a
learned or Hybrid approach is the best revenue predictor without comparing it.
