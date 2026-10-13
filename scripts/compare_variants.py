"""Compare the shipped ranking against the alternatives we tried.

The shipped pipeline (notebooks/) now runs on the hybrid weights and writes to
data/tables/processed. The pure-learned-weight variant lives in notebooks_weve_tried/
and writes to data/variants/machine/processed.

Run after both have been executed:

    python scripts/compare_variants.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    "shipped (hybrid)": (PROJECT_ROOT / "data" / "tables" / "processed", "hybrid"),
    "machine weights":  (PROJECT_ROOT / "data" / "variants" / "machine" / "processed", "machine"),
}


def load(folder: Path, name: str):
    path = folder / name
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    tops, rows = {}, []
    for label, (folder, tag) in VARIANTS.items():
        top = load(folder, f"top100_{tag}.csv")
        if top is None:
            print(f"[not generated] {label}: {folder}/top100_{tag}.csv is missing")
            continue
        tops[label] = set(top["merchant_abn"])
        rows.append({
            "variant": label,
            "merchants in top 100": len(top),
            "BNPL revenue in top 100 ($M)": round(top["estimated_bnpl_revenue"].sum() / 1e6, 2),
            "median transactions": int(top["transaction_count"].median()),
            "fraud-adjusted list": "yes" if load(folder, f"top100_{tag}_fraud_adjusted.csv") is not None else "missing",
            "top 10 per segment": "yes" if load(folder, "top10_per_segment.csv") is not None else "missing",
        })

    if not rows:
        print("No variant results are available")
        return

    print("\n=== RESULTS ===")
    print(pd.DataFrame(rows).set_index("variant").to_string())

    if len(tops) == 2:
        a, b = list(tops)
        shared = len(tops[a] & tops[b])
        print(f"\n=== TOP 100 OVERLAP ===")
        print(f"{a} vs {b}: {shared}/100 shared merchants")
        print(f"only in '{a}': {len(tops[a] - tops[b])}")
        print(f"only in '{b}': {len(tops[b] - tops[a])}")



import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "tables" / "processed"
FEATURE_START = "2021-03-01"
FEATURE_END = "2022-09-30"
K = 300
WEIGHTS_HYBRID = {
    "estimated_bnpl_revenue": .70, "log_value_growth": .02,
    "transactions_per_customer": 0., "repeat_rate": 0., "stability": 0.,
    "active_month_ratio": .18, "customer_spread": .05,
    "customer_income": .03, "customer_irsad": .02,
}
WEIGHTS_MACHINE = {**dict.fromkeys(WEIGHTS_HYBRID, 0.),
                   "estimated_bnpl_revenue": .724,
                   "active_month_ratio": .218, "customer_spread": .058}
WEIGHTS_V1 = {
    "estimated_bnpl_revenue": .30, "log_value_growth": .20,
    "transactions_per_customer": .10, "stability": .10,
    "unique_customers": .05, "active_month_ratio": .05, "repeat_rate": .05,
    "customer_income": .05, "customer_irsad": .05, "customer_spread": .05,
}
SHRINK_IN_RANK_SPACE = {"estimated_bnpl_revenue", "log_value_growth"}


def derive(df):
    d = df.copy()
    d["stability"] = -d["value_cv"]
    d["customer_spread"] = 1 - d["customer_hhi"]
    return d


def design(df, features, k=K):
    d = derive(df)
    ranks = pd.DataFrame({c: d[c].rank(pct=True) * 100 for c in features}).fillna(50.)
    sh = d["transaction_count"] / (d["transaction_count"] + k)
    for c in SHRINK_IN_RANK_SPACE & set(features):
        ranks[c] = sh * ranks[c] + (1 - sh) * 50.
    return ranks


def score_hybrid(df, weights=WEIGHTS_HYBRID, k=K, shrink=SHRINK_IN_RANK_SPACE):
    # Keep the original callable interface used by both ranking notebooks.
    if shrink != SHRINK_IN_RANK_SPACE:
        raise ValueError("Use the shared revenue/growth rank-space shrinkage definition")
    ranks = design(df, weights, k)
    return sum(ranks[c] * w for c, w in weights.items()), ranks


def score_v1(df, weights=WEIGHTS_V1, k=K):
    d = derive(df)
    sh = d["transaction_count"] / (d["transaction_count"] + k)
    for c in SHRINK_IN_RANK_SPACE:
        d[c] = sh * d[c] + (1 - sh) * d[c].mean()
    ranks = pd.DataFrame({c: d[c].rank(pct=True) * 100 for c in weights}).fillna(50.)
    return sum(ranks[c] * w for c, w in weights.items()), ranks


def revenue_term(values, n, k=K):
    sh = n / (n + k)
    return sh * values.rank(pct=True) * 100 + (1 - sh) * 50.



def audit_outputs():
    p = PROCESSED
    rank = pd.read_parquet(p / "merchant_ranking_hybrid.parquet")
    fraud = pd.read_parquet(p / "merchant_ranking_hybrid_fraud_adjusted.parquet")
    segments = pd.read_parquet(p / "merchant_behaviour_segments")
    shortlist = pd.read_csv(p / "top100_hybrid.csv")
    segment_top = pd.read_csv(p / "top10_per_segment.csv")
    assert len(rank) == rank.merchant_abn.nunique() == 4026
    assert sorted(rank.rank_overall) == list(range(1, len(rank) + 1))
    assert set(rank.merchant_abn) == set(fraud.merchant_abn) == set(segments.merchant_abn)
    assert shortlist.merchant_abn.tolist() == rank.nsmallest(100, "rank_overall").merchant_abn.tolist()
    assert segment_top.groupby("segment_name").size().eq(10).all()
    joined = segment_top.merge(rank, on="merchant_abn", suffixes=("_segment", "_global"), validate="many_to_one")
    assert joined.rank_overall_segment.eq(joined.rank_overall_global).all()
    assert np.allclose(joined.estimated_bnpl_revenue_segment, joined.estimated_bnpl_revenue_global)
    assert np.allclose(score_hybrid(rank)[0], rank.score, atol=1e-10)
    assert np.allclose(fraud.fraud_adjusted_revenue,
                       fraud.estimated_bnpl_revenue - fraud.expected_fraud_loss)
    assert np.allclose(fraud.score_adjusted,
                       fraud.score - .7 * revenue_term(fraud.estimated_bnpl_revenue, fraud.transaction_count)
                       + .7 * revenue_term(fraud.fraud_adjusted_revenue, fraud.transaction_count))
    adjusted_csv = pd.read_csv(p / "top100_hybrid_fraud_adjusted.csv")
    assert adjusted_csv.merchant_abn.tolist() == fraud.nsmallest(100, "rank_adjusted").merchant_abn.tolist()
    assert "fraud_review" in fraud
    machine_path = PROJECT_ROOT / "data" / "variants" / "machine" / "processed"
    if (machine_path / "merchant_ranking_machine.parquet").exists():
        machine = pd.read_parquet(machine_path / "merchant_ranking_machine.parquet")
        assert np.allclose(score_hybrid(machine, weights=WEIGHTS_MACHINE)[0], machine.score)
        comparison = rank.merge(machine, on="merchant_abn", suffixes=("_hybrid", "_machine"), validate="one_to_one")
        for field in ["transaction_count", "estimated_bnpl_revenue", "customer_hhi", "customer_income", "customer_irsad"]:
            assert np.allclose(comparison[field + "_hybrid"], comparison[field + "_machine"], equal_nan=True)
        machine_top = pd.read_csv(machine_path / "top100_machine.csv")
        assert machine_top.merchant_abn.tolist() == machine.nsmallest(100, "rank_overall").merchant_abn.tolist()
        machine_fraud = pd.read_parquet(machine_path / "merchant_ranking_machine_fraud_adjusted.parquet")
        assert np.allclose(machine_fraud.score_adjusted,
                           machine_fraud.score - .724 * revenue_term(machine_fraud.estimated_bnpl_revenue, machine_fraud.transaction_count)
                           + .724 * revenue_term(machine_fraud.fraud_adjusted_revenue, machine_fraud.transaction_count))
    # Independently rebuild customer concentration/repeat behaviour from only
    # production transactions, catching accidental use of February/October.
    data = ds.dataset(p / "transactions_clean", format="parquet")
    date = ds.field("order_datetime")
    tx = data.to_table(columns=["merchant_abn", "user_id", "dollar_value"],
                       filter=(date >= pd.Timestamp(FEATURE_START)) &
                              (date <= pd.Timestamp(FEATURE_END))).to_pandas()
    pc = tx.groupby(["merchant_abn", "user_id"]).dollar_value.agg(["sum", "size"])
    totals = pc["sum"].groupby(level=0).sum()
    hhi = pc["sum"].pow(2).groupby(level=0).sum() / totals.pow(2)
    repeat = pc["size"].gt(1).groupby(level=0).mean()
    indexed = rank.set_index("merchant_abn")
    assert np.allclose(hhi.reindex(indexed.index), indexed.customer_hhi, rtol=1e-9, atol=1e-12)
    assert np.allclose(repeat.reindex(indexed.index), indexed.repeat_rate)
    consumers = pd.read_parquet(p / "consumers_sa2", columns=[
        "user_id", "sa2_share", "abs_median_household_income_weekly", "seifa_irsad_decile"])
    profile = pc.reset_index()[["merchant_abn", "user_id"]].merge(consumers, on="user_id", how="left", validate="many_to_one")
    for source, target in [("abs_median_household_income_weekly", "customer_income"),
                           ("seifa_irsad_decile", "customer_irsad")]:
        weights = profile.sa2_share.fillna(1.).where(profile[source].notna(), 0.)
        numerator = (profile[source].fillna(0.) * weights).groupby(profile.merchant_abn).sum()
        denominator = weights.groupby(profile.merchant_abn).sum()
        expected = (numerator / denominator).reindex(indexed.index)
        assert np.allclose(expected, indexed[target], rtol=1e-9, atol=1e-8, equal_nan=True)
    counts = tx.groupby("merchant_abn").size().reindex(indexed.index)
    assert counts.eq(indexed.transaction_count).all()
    # Baseline exports are derived from the same production feature snapshot.
    baseline = rank.copy()
    baseline["score"] = baseline["score_v1"]
    baseline["rank_overall"] = baseline["rank_v1"]
    baseline = baseline.sort_values("rank_overall")
    baseline.to_parquet(p / "merchant_ranking_baseline.parquet", index=False)
    baseline.head(100).to_csv(p / "top100_baseline.csv", index=False)
    aliases = {"merchant_ranking_v2.parquet": "merchant_ranking_hybrid.parquet",
               "merchant_ranking_v2_features.parquet": "merchant_ranking_hybrid_features.parquet",
               "merchant_ranking_features.parquet": "merchant_ranking_hybrid_features.parquet",
               "merchant_ranking_v2_fraud_adjusted.parquet": "merchant_ranking_hybrid_fraud_adjusted.parquet",
               "top100_v2.csv": "top100_hybrid.csv", "top150_v2.csv": "top150_hybrid.csv",
               "top100_v2_fraud_adjusted.csv": "top100_hybrid_fraud_adjusted.csv",
               "weights_v2.csv": "weights_hybrid.csv", "v2_holdout_results.csv": "hybrid_holdout_results.csv"}
    for old, current in aliases.items():
        shutil.copyfile(p / current, p / old)
        assert hashlib.sha256((p / old).read_bytes()).digest() == hashlib.sha256((p / current).read_bytes()).digest()
    holdout = pd.read_csv(p / "hybrid_holdout_results.csv")
    sweep = pd.read_csv(p / "hybrid_fraud_sensitivity.csv")
    top, adjusted = rank.nsmallest(100, "rank_overall"), fraud.nsmallest(100, "rank_adjusted")
    metrics = []
    model_dir = PROJECT_ROOT / "data" / "tables" / "models"
    consumer = pd.read_parquet(model_dir / "consumer_fraud_test.parquet")
    days = pd.read_parquet(model_dir / "merchant_fraud_days.parquet")
    for cutoff in [30, 50, 70]:
        model = XGBClassifier()
        model.load_model(model_dir / f"consumer_fraud_ge{cutoff}.json")
        predicted = model.predict_proba(consumer[model.get_booster().feature_names])[:, 1]
        single = consumer.day_txn_count.eq(1)
        metrics.append({"cutoff": cutoff, "single_transaction_days": int(single.sum()),
                        "single_day_positives": int(consumer.loc[single, "fraud_probability"].ge(cutoff).sum()),
                        "consumer_average_precision": float(average_precision_score(
                            consumer.loc[single, "fraud_probability"].ge(cutoff), predicted[single])),
                        "merchant_oof_average_precision": float(average_precision_score(
                            days.fraud_probability.ge(cutoff), days[f"p_ge_{cutoff}_oof"]))})
    results = {"production_period": [FEATURE_START, FEATURE_END], "timezone": "Etc/UTC",
               "holdout_note": "History-only predictors; September reused in exploratory comparisons",
               "clean_transactions": data.count_rows(), "production_transactions": len(tx),
               "rankable_transactions": int(rank.transaction_count.sum()), "rankable_merchants": len(rank),
               "weights": WEIGHTS_HYBRID, "top100_revenue": float(top.estimated_bnpl_revenue.sum()),
               "top100_revenue_share": float(top.estimated_bnpl_revenue.sum()/rank.estimated_bnpl_revenue.sum()),
               "top5": top.head(5)[["merchant_abn", "name", "rank_overall", "estimated_bnpl_revenue"]].to_dict("records"),
               "segment_top3": segment_top.groupby("segment_name", sort=False).head(3)[
                   ["segment_name", "rank_in_segment", "name", "rank_overall", "estimated_bnpl_revenue"]].to_dict("records"),
               "fraud_examples": fraud[fraud.name.isin(["Semper Auctor PC", "Malesuada Vel Associates", "Id Erat Etiam Consulting"])][
                   ["name", "rank_overall", "rank_adjusted", "estimated_bnpl_revenue", "expected_fraud_loss", "loss_to_revenue"]].to_dict("records"),
               "segments": segments.groupby("segment_id").size().to_dict(),
               "holdout": holdout.to_dict("records"), "fraud_sensitivity": sweep.to_dict("records"),
               "fraud_affected": int(fraud.expected_fraud_loss.gt(0).sum()),
               "fraud_lower_scores": int((fraud.score_adjusted < fraud.score - 1e-9).sum()),
               "fraud_top100_overlap": len(set(top.merchant_abn) & set(adjusted.merchant_abn)),
               "fraud_exit": top.loc[~top.merchant_abn.isin(adjusted.merchant_abn),
                    ["name", "rank_overall"]].to_dict("records"),
               "fraud_entry": adjusted.loc[~adjusted.merchant_abn.isin(top.merchant_abn),
                    ["name", "rank_adjusted"]].to_dict("records"),
               "fraud_top100_loss": float(adjusted.expected_fraud_loss.sum()), "diagnostics": metrics,
               "compatibility_aliases": aliases}
    (p / "results_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Reproduced results", "", "Generated by `scripts/compare_variants.py`; do not edit numerical values manually.", "",
             f"All production features use {FEATURE_START} to {FEATURE_END}, in UTC.",
             "September predictors are rebuilt through August; September is reused in exploratory comparisons.", "",
             f"Clean transactions: {results['clean_transactions']:,}; modelling-window transactions: {len(tx):,}; rankable merchants: {len(rank):,}.",
             f"Top 100 estimated revenue: ${results['top100_revenue']:,.2f} ({results['top100_revenue_share']:.4%} of ranked revenue).", "",
             "| Rank | Merchant | Estimated BNPL revenue |", "|---|---|---:|"]
    lines += [f"| {r['rank_overall']} | {r['name']} | ${r['estimated_bnpl_revenue']:,.2f} |" for r in results["top5"]]
    lines += ["", "| September picker | Spearman | Capture rate | Top 100 hits |", "|---|---:|---:|---:|"]
    lines += [f"| {r['picker']} | {r['spearman']:.6f} | {r['capture rate']:.4%} | {r['hits in realised top 100']} |" for r in results["holdout"]]
    lines += ["", "| Segment | Within-segment rank | Merchant | Revenue | Global rank |", "|---|---:|---|---:|---:|"]
    lines += [f"| {r['segment_name']} | {r['rank_in_segment']} | {r['name']} | ${r['estimated_bnpl_revenue']:,.2f} | {r['rank_overall']} |" for r in results["segment_top3"]]
    lines += ["", "| Fraud example | Commercial rank | Adjusted rank | Revenue | Expected loss | Loss/revenue |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['name']} | {r['rank_overall']} | {r['rank_adjusted']} | ${r['estimated_bnpl_revenue']:,.2f} | ${r['expected_fraud_loss']:,.2f} | {r['loss_to_revenue']:.4%} |" for r in results["fraud_examples"]]
    lines += ["", f"Sensitivity settings retain {int(sweep['top-100 overlap'].min())}–{int(sweep['top-100 overlap'].max())} commercial Top 100 members.",
              "", "| Diagnostic threshold | Consumer single-day average precision | Merchant out-of-fold average precision |", "|---|---:|---:|"]
    lines += [f"| ≥{r['cutoff']} | {r['consumer_average_precision']:.6f} | {r['merchant_oof_average_precision']:.6f} |" for r in metrics]
    lines += ["", f"Fraud-adjusted shortlist overlap: {results['fraud_top100_overlap']}/100; expected loss: ${results['fraud_top100_loss']:,.2f}.",
              f"Leaving commercial Top 100: {results['fraud_exit']}; entering: {results['fraud_entry']}.",
              "", "Authoritative files live in `data/tables/processed/`: `top100_hybrid.csv`, `top10_per_segment.csv`,",
              "`top100_hybrid_fraud_adjusted.csv`, `hybrid_holdout_results.csv` and `results_summary.json`.",
              "Legacy v2 filenames are byte-identical compatibility copies, refreshed after this audit passes.", "",
              "The supplied presentation PDF is external to the repository and has not been edited. Its values must be compared with this new feature snapshot."]
    readme = PROJECT_ROOT / "README.md"
    start, end = "<!-- reproduced-results:start -->", "<!-- reproduced-results:end -->"
    existing = readme.read_text().split(start)[0].rstrip()
    readme.write_text(existing + "\n\n" + start + "\n" + "\n".join(lines) + "\n" + end + "\n")
    print(json.dumps({k: results[k] for k in ["clean_transactions", "production_transactions", "rankable_merchants", "top100_revenue", "fraud_top100_overlap"]}, indent=2))
    print("PASS: ranks, shortlists, feature-window checks, fraud arithmetic and compatibility copies")



if __name__ == "__main__":
    main()
    audit_outputs()
