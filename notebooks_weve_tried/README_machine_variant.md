# Fixed Machine comparison

See `README.md` in this directory and the repository README for execution and
output paths. The fixed weights are included directly in the variant notebooks and segment script:
revenue 72.4%, active-month ratio 21.8%, customer spread 5.8%; all other weights zero.
They are retained for comparison, not automatically refitted or overwritten.

All ranking features and calendar windows match the main Hybrid pipeline.
`python notebooks_weve_tried/scripts/build_variants.py` prepares shared input
links and generates variant sources. Execute them in the repository README order. No nonexistent launcher or manual aliases are required.

The old coefficient-averaging derivation belongs to the historical experiment.
Current fitted coefficients are exported separately as `rolling_learned_weights.csv`.
