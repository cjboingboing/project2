# Historical proposal — not the implemented ranking

This brainstorm is retained as design history. Its proposed forecasting method and expected benchmark comparisons are not current findings. See README.md and README.md for the implemented pipeline and reproduced results.

# Ranking system — brainstorm

Sam, for the Week 8 checkpoint. This is a proposal to argue about, not a decision.
Everything below is costed against fields we actually have, and the ABS work
(`download_external_sa2.py`, `join_external_sa2.py`) is what supplies the consumer
side of it.

## 1. What "best merchant" should mean

The firm earns `take_rate` on every dollar a merchant transacts, and wears the loss
when a transaction turns out to be fraud. It can onboard 100 merchants a year, so
the question is not "who is biggest today" but **which 100 merchants will return the
most, net of fraud, over the next year, with the least chance of that estimate being
wrong.**

So the score we are really after is:

```
score = expected BNPL revenue over the next 12 months
      − expected fraud loss
      × a discount for how uncertain that estimate is
```

Ranking on raw historical revenue (the obvious baseline) fails this on three counts:
it rewards merchants that are already past their peak, ignores that a 6.5% take rate
earns four times what a 1.6% one does on the same basket, and puts a merchant with
three weeks of history next to one with eighteen months as if we knew them equally well.
We should still build that baseline, and beat it.

## 2. Components

### 2.1 Scale — what they do now

- `gmv_annualised` — total `dollar_value` over the window, scaled to a year by the
  number of days the merchant is actually observed (not the window length, or new
  merchants are punished twice).
- `bnpl_revenue = gmv_annualised × take_rate` — this is the number the firm cares
  about, and it costs us nothing to use dollars instead of a z-score.
- `n_customers`, `n_transactions`, `mean_basket` — kept as diagnostics and for the
  "interesting merchants" section, not stacked into the score (they are ~collinear
  with GMV and would triple-count scale).

### 2.2 Growth — where they are heading

- Fit `log(monthly GMV) ~ month` per merchant, keep the slope and its standard error.
  Log because the spread across merchants is orders of magnitude; the slope is then a
  monthly growth rate, which is interpretable in the presentation.
- Simple forecast: use the fitted trend to project the next 12 months rather than
  assuming flat. Needs a floor — a merchant with two months of data can produce an
  absurd extrapolation, hence the shrinkage in 2.5.
- Sanity check against the snapshot boundaries: the three transaction files are
  consecutive windows, so any "growth" that lines up exactly with a file boundary is
  an artefact of the data, not the merchant.

### 2.3 Fraud — what it costs us

We have daily fraud probabilities for both merchants and consumers, and they are
sparse (most transactions have no row, which is why `process.py` leaves them null).

- `expected_fraud_loss = Σ dollar_value × P(fraud)` over the merchant's transactions,
  once the nulls are imputed. Subtract it from revenue in dollars — far easier to
  defend than a fraud "penalty weight" pulled out of the air.
- `fraud_rate = expected_fraud_loss / gmv` as the comparable version.
- Worth separating **merchant-driven** fraud from **customer-driven** fraud: a
  merchant whose customers are individually risky (high consumer-side probabilities)
  is a different proposition from one whose own transactions look manufactured. The
  first is a customer-base problem we can price; the second is a reason to decline.
- The imputation this depends on is James/Calum's open item — our score is only as
  good as that model, so state the dependency in the notebook.

### 2.4 Customer base — who is buying, and can they pay the instalments

This is where the ABS data earns its place, and it is the only part of the ranking
that uses anything external.

Each consumer is placed in an SA2 (`consumers_sa2`), so per merchant we can build:

- `median_customer_income` — population-weighted median household income of the SA2s
  the merchant's customers live in. "Pay in 5 instalments" is a credit product; a
  customer base that can absorb a missed instalment is worth more than one that cannot.
- `mean_customer_seifa_irsad` — the ABS advantage/disadvantage index, which is a
  cleaner single summary of socio-economic position than income alone and is already
  decile-scored, so it needs no normalising.
- `geographic_spread` — number of distinct SA2s, and the HHI of customers across them.
  A merchant concentrated in one SA2 is exposed to that one area; a national footprint
  is more durable.
- `customer_concentration` — HHI of GMV across `user_id`. A merchant whose top five
  customers are 60% of revenue is fragile no matter how big the total is, and this is
  also a fraud smell.
- `repeat_rate` — share of customers with more than one order. Repeat purchase is the
  single best predictor that next year's revenue looks like this year's.

Caveat to write down: `sa2_share` says how much of a consumer's postcode actually sits
in the SA2 we assigned them. For 65% of postcodes it is 0.9 or better, but 1 in 10 sits
below 0.5, where the income figure is a blur across several areas. Weighting each
consumer by `sa2_share` when aggregating to the merchant is the honest way to handle it.

### 2.5 Uncertainty — how much we trust any of this

Two merchants with the same estimated revenue are not equally good bets if one has
eighteen months of history and the other has three weeks.

- Shrink every per-merchant estimate towards its segment mean by
  `n / (n + k)`, with `n` the merchant's transaction count and `k` tuned so that a
  merchant needs a few hundred transactions before it is taken at face value.
  This is the standard empirical-Bayes fix and it handles the spec's "new merchant
  with little information" case without a special rule.
- `revenue_volatility` — coefficient of variation of monthly GMV. Penalise it lightly;
  some of it is seasonality, not risk.
- Report `days_observed` next to the rank so a reader can see which entries are thin.

## 3. Putting it together

Two candidates, and I would build both because the comparison is itself a finding:

**A. Dollar model (preferred).** Everything in one unit:

```
expected_value = forecast_gmv × take_rate × (1 − fraud_rate) × shrinkage
                 × quality_multiplier(repeat_rate, customer income, concentration)
```

Defensible line by line, no arbitrary weights except inside the multiplier, and the
output is a dollar figure a non-technical audience understands immediately.

**B. Weighted rank score (sanity check).** Percentile-rank each feature within segment,
weighted sum. Crude, but immune to the long tail and to one bad forecast, so if A and B
disagree violently on a merchant that merchant is worth looking at by hand.

Ranks should be formed **within segment**, then merged for the overall top 100, or
groceries will swamp everything on volume alone.

## 4. Segments (Sprint 5, but it constrains the above)

The spec wants three to five. The cleaned `keywords_clean` column is the natural basis.
My suggestion, from skimming the tag vocabulary:

1. Everyday retail (groceries, florists, stationery, gifts)
2. Technology and electronics (computers, telecom, cameras)
3. Home and furniture (furniture, garden, appliances)
4. Health, beauty and leisure (cosmetics, jewellery, hobby, music)
5. Services and other (repair, motor vehicle, digital services)

Do it as a manual taxonomy over the top ~100 cleaned keywords rather than k-means on
TF-IDF — the clusters will need hand-fixing anyway, and a taxonomy we chose is easier
to justify in the presentation than one a clustering handed us. Cross-check the result
against `revenue_code` and `take_rate`: if a segment's take rates are all over the
place, the segment is probably two segments.

## 5. Validating it

We need to show the ranking is predictive rather than descriptive:

- **Time split.** Build the ranking on Feb 2021 – Feb 2022 only. Score merchants on
  what they actually did Mar – Aug 2022. Report Spearman correlation between predicted
  rank and realised revenue, and the overlap of the predicted top 100 with the realised
  top 100.
- **Beat the baseline.** The baseline is "rank by past revenue". If we cannot beat it,
  the extra machinery is not earning its keep and we should say so.
- **Stability.** Perturb the weights ±20% and count how many of the top 100 change.
  A ranking that reshuffles under small weight changes is not a ranking.
- **Eyeball the top 10 and bottom 10** of each segment. If any of them look obviously
  wrong, the feature causing it usually shows up immediately.

## 6. The "interesting merchants" the spec asks for

Falls out of the features above, and is the part an audience remembers:

- Small base, large baskets: low `n_customers`, top-decile `mean_basket`.
- Big but risky: top-decile GMV, top-decile `fraud_rate`.
- Thin but promising: high growth slope, low `days_observed` — the merchants the
  shrinkage deliberately holds back, worth naming as "watch list, revisit next year".
- Concentration risk: top-decile customer HHI.

## 7. Open questions for the group

1. Do we rank on the **firm's revenue** (take rate × GMV) or on **merchant GMV**?
   I think revenue, but it changes the top 100 a lot and we should decide as a group
   rather than discover we disagree in Week 10.
2. Fraud: drop flagged transactions, or keep them with an `is_fraud` flag and price
   them in? I lean towards keeping and pricing, since removing them hides exactly the
   merchants we are trying to spot.
3. How much does the ranking depend on the fraud imputation? If it moves the top 100
   materially, that is worth a slide of its own.
4. Do we forecast (2.2) or just annualise? Forecasting is a Sprint 4 deliverable in the
   spec, so probably yes, but a bad forecast is worse than none.

## 8. What I can take on next

The consumer-side features in 2.4 — they run off `consumers_sa2`, which is my work,
and none of them depend on the imputation landing first.
