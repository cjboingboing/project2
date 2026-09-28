"""
Top 10 merchants in each behaviour segment, under three rankings.

The spec asks for a ranking system per segment, so each segment gets its own
weights (SEGMENT_WEIGHTS), set from how well each feature predicted realised
revenue inside that segment on the time split. Two references are printed
alongside: the global v2 score, and past revenue on its own.

Input:  data/tables/processed/merchant_ranking_v2.parquet (merchant_ranking_v2.ipynb)
        data/tables/processed/merchant_behaviour_segments (segment_behaviour.py)
Output: data/tables/processed/top10_per_segment.csv

Run with:
    python scripts/top10_per_segment.py
"""

from pathlib import Path

import pandas as pd

#Paths

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED = PROJECT_ROOT / "data" / "tables" / "processed"

RANKING_PATH = PROCESSED / "merchant_ranking_v2.parquet"
SEGMENTS_PATH = PROCESSED / "merchant_behaviour_segments"

TOP_N = 10

# display names for the clusters (segment_id is stable, KMeans is seeded)
NAMES = {
    1: "Established",
    3: "Rising",
    2: "Fading",
    0: "Intermittent",
}

# -1 means lower is better
ORIENT = {
    "estimated_bnpl_revenue": 1, "log_value_growth": 1, "transactions_per_customer": 1,
    "unique_customers": 1, "repeat_rate": 1, "active_month_ratio": 1,
    "value_cv": -1, "customer_hhi": -1, "customer_income": 1, "customer_irsad": 1,
}

# per-segment weights, proportional to each feature's Spearman against realised
# revenue within that segment (train Feb21-Feb22, realised Mar-Aug22).
# Established rewards breadth of customers; the thin segments have little signal
# beyond revenue itself, which is why revenue dominates there.
SEGMENT_WEIGHTS = {
    1: {  # Established
        "estimated_bnpl_revenue": 0.341, "customer_hhi": 0.147, "transactions_per_customer": 0.142,
        "repeat_rate": 0.141, "unique_customers": 0.141, "active_month_ratio": 0.053,
        "log_value_growth": 0.035,
    },
    3: {  # Rising
        "estimated_bnpl_revenue": 0.505, "customer_hhi": 0.131, "unique_customers": 0.119,
        "value_cv": 0.106, "repeat_rate": 0.055, "transactions_per_customer": 0.055,
        "active_month_ratio": 0.030,
    },
    2: {  # Fading
        "estimated_bnpl_revenue": 0.561, "customer_hhi": 0.122, "unique_customers": 0.116,
        "active_month_ratio": 0.098, "value_cv": 0.035, "repeat_rate": 0.034,
        "transactions_per_customer": 0.034,
    },
    0: {  # Intermittent
        "estimated_bnpl_revenue": 0.868, "value_cv": 0.067, "customer_irsad": 0.037,
        "customer_income": 0.028,
    },
}


def load():
    """Join the v2 scores onto the behaviour segments."""
    ranking = pd.read_parquet(RANKING_PATH)
    segments = pd.read_parquet(SEGMENTS_PATH)[["merchant_abn", "segment_id", "segment"]]

    df = ranking.merge(segments, on="merchant_abn", how="inner")
    assert len(df) == len(segments), f"join lost rows: {len(df)} vs {len(segments)}"

    df["segment_name"] = df["segment_id"].map(NAMES).fillna("Segment " + df["segment_id"].astype(str))
    return df


def segment_score(block, weights):
    """Weighted sum of within-segment percentile ranks, scaled 0-100."""
    ranks = pd.DataFrame(
        {f: block[f].astype(float).rank(pct=True) * ORIENT[f] for f in weights},
        index=block.index,
    ).fillna(0.5)
    return sum(ranks[f] * w for f, w in weights.items()) * 100


def add_ranks(df):
    """Score each merchant with its own segment's weights, plus the two references."""
    df["segment_score"] = 0.0
    for seg_id, weights in SEGMENT_WEIGHTS.items():
        mask = df.segment_id == seg_id
        df.loc[mask, "segment_score"] = segment_score(df[mask], weights)

    by = lambda col: df.groupby("segment_id")[col].rank(ascending=False, method="first")
    df["rank_in_segment"] = by("segment_score")
    df["rank_in_segment_v2"] = by("score")
    df["rank_in_segment_baseline"] = by("estimated_bnpl_revenue")
    return df


def report(df):
    """Print each segment's top 10 and how far it agrees with the baseline."""
    width = 104
    for seg_id in sorted(df["segment_id"].unique(), key=lambda s: -df.loc[df.segment_id == s, "score"].max()):
        block = df[df.segment_id == seg_id]
        name = block["segment_name"].iloc[0]
        top = block.nsmallest(TOP_N, "rank_in_segment")
        picks = lambda col: set(block.nsmallest(TOP_N, col)["merchant_abn"])
        mine = set(top["merchant_abn"])

        print(f"\n{f' {name}  ({len(block):,} merchants) ':-^{width}}")
        print(f"  {'#':<3}{'merchant':<38}{'score':>7}{'BNPL rev':>12}{'customers':>11}{'growth':>9}{'overall':>9}")
        for i, (_, r) in enumerate(top.iterrows(), 1):
            print(
                f"  {i:<3}{r['name'][:36]:<38}{r['segment_score']:>7.1f}"
                f"${r['estimated_bnpl_revenue']:>11,.0f}{r['unique_customers']:>11,}"
                f"{r['log_value_growth']:>9.2f}{int(r['rank_overall']):>9}"
            )
        print(f"  same picks as the global v2 score: {len(mine & picks('rank_in_segment_v2'))}/{TOP_N}"
              f"   as past revenue alone: {len(mine & picks('rank_in_segment_baseline'))}/{TOP_N}")


def main():
    df = add_ranks(load())

    print(f"{len(df):,} merchants ranked across {df['segment_id'].nunique()} segments")
    report(df)

    cols = [
        "segment_id", "segment_name", "segment", "rank_in_segment", "rank_in_segment_v2",
        "rank_in_segment_baseline", "merchant_abn", "name", "segment_score", "score",
        "rank_overall", "estimated_bnpl_revenue", "unique_customers", "log_value_growth",
        "active_month_ratio", "value_cv", "take_rate",
    ]
    out = df[df.rank_in_segment <= TOP_N].sort_values(["segment_id", "rank_in_segment"])[cols]
    out.to_csv(PROCESSED / "top10_per_segment.csv", index=False)
    print(f"\nWritten to {PROCESSED / 'top10_per_segment.csv'}  ({len(out)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
