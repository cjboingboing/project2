"""
Top 10 merchants in each behaviour segment using the validated v2 ranking logic.

Why this version exists
-----------------------
The previous script chose different weights for each segment using correlations
with realised revenue in the Mar-Aug 2022 test window. That makes the segment
weights test-informed and therefore too optimistic for an honest evaluation.

This version removes that leakage:
1. use the v2 weights that were fixed before evaluating the test window;
2. recompute feature percentile ranks *within each behaviour segment*;
3. apply the same v2 rank-space shrinkage for revenue and growth;
4. rank merchants by that segment-specific v2 score;
5. keep global-v2 and past-revenue rankings only as references.

Inputs
------
data/tables/processed/merchant_ranking_v2.parquet
    Full merchant-level v2 ranking/features from merchant_ranking_v2.ipynb.

data/tables/processed/merchant_behaviour_segments
    Behavioural segment assignment from segment_behaviour.py.

Output
------
data/tables/processed/top10_per_segment.csv

Run with:
    python scripts/top10_per_segment.py
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED = PROJECT_ROOT / "data" / "tables" / "processed"

RANKING_PATH = PROCESSED / "merchant_ranking_v2.parquet"
SEGMENTS_PATH = PROCESSED / "merchant_behaviour_segments"
OUTPUT_PATH = PROCESSED / "top10_per_segment.csv"

TOP_N = 10
K = 300


# Display names for the seeded K-means clusters.
NAMES = {
    1: "Established",
    3: "Rising",
    2: "Fading",
    0: "Intermittent",
}


# ---------------------------------------------------------------------------
# Leakage-free scoring configuration
# ---------------------------------------------------------------------------

# These are the final v2 weights from merchant_ranking_v2.ipynb.
WEIGHTS_V2 = {
    "estimated_bnpl_revenue": 0.70,
    "log_value_growth": 0.08,
    "transactions_per_customer": 0.04,
    "repeat_rate": 0.03,
    "stability": 0.03,
    "active_month_ratio": 0.02,
    "customer_spread": 0.05,
    "customer_income": 0.03,
    "customer_irsad": 0.02,
}

assert abs(sum(WEIGHTS_V2.values()) - 1.0) < 1e-9


# Revenue and growth use the same rank-space shrinkage as v2.
SHRINK_IN_RANK_SPACE = {
    "estimated_bnpl_revenue",
    "log_value_growth",
}


# Kept only for compatibility if segmentation.ipynb refers to it.
# Every segment now uses the same validated v2 weights.
SEGMENT_WEIGHTS = {
    seg_id: WEIGHTS_V2.copy()
    for seg_id in NAMES
}


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def load() -> pd.DataFrame:
    """Join v2 merchant features to behaviour-segment assignments."""

    ranking = pd.read_parquet(RANKING_PATH)

    segments = pd.read_parquet(SEGMENTS_PATH)[
        ["merchant_abn", "segment_id", "segment"]
    ]

    required = {
        "merchant_abn",
        "name",
        "score",
        "rank_overall",
        "transaction_count",
        "estimated_bnpl_revenue",
        "log_value_growth",
        "transactions_per_customer",
        "repeat_rate",
        "active_month_ratio",
        "value_cv",
        "customer_hhi",
        "customer_income",
        "customer_irsad",
        "unique_customers",
        "take_rate",
    }

    missing = sorted(required - set(ranking.columns))

    if missing:
        raise ValueError(
            "merchant_ranking_v2.parquet is missing required columns: "
            + ", ".join(missing)
        )

    if segments["merchant_abn"].duplicated().any():
        raise ValueError(
            "merchant_behaviour_segments has duplicate merchant_abn values"
        )

    df = ranking.merge(
        segments,
        on="merchant_abn",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(segments):
        raise ValueError(
            f"join lost segment rows: "
            f"{len(df):,} joined vs {len(segments):,} segments"
        )

    df["segment_name"] = df["segment_id"].map(NAMES)

    df["segment_name"] = df["segment_name"].fillna(
        "Segment " + df["segment_id"].astype(str)
    )

    return df


# ---------------------------------------------------------------------------
# Segment scoring
# ---------------------------------------------------------------------------

def derive(block: pd.DataFrame) -> pd.DataFrame:
    """
    Create positively oriented versions of stability and customer spread.

    Lower value_cv = more stable, therefore stability = -value_cv.
    Lower HHI = more diversified customers, therefore spread = 1 - HHI.
    """

    d = block.copy()

    d["stability"] = -d["value_cv"]

    d["customer_spread"] = 1.0 - d["customer_hhi"]

    return d


def segment_score_v2(
    block: pd.DataFrame,
    weights: dict[str, float] = WEIGHTS_V2,
    k: int = K,
) -> pd.Series:
    """
    Calculate a v2-style score within one behaviour segment.

    Feature percentiles are recalculated only against merchants in the same
    segment.

    Missing values receive a neutral score of 50.

    Revenue and growth are shrunk toward 50 when the merchant has relatively
    little transaction evidence.
    """

    d = derive(block)

    # Same evidence-based shrinkage used in v2.
    shrinkage = (
        d["transaction_count"]
        / (d["transaction_count"] + k)
    )

    # Calculate percentile rank within the current segment.
    ranks = pd.DataFrame(
        {
            feature: (
                d[feature]
                .rank(
                    pct=True,
                    na_option="keep",
                    method="average",
                )
                * 100.0
            )
            for feature in weights
        },
        index=d.index,
    )

    # Missing features get neutral percentile score.
    ranks = ranks.fillna(50.0)

    # Shrink revenue and growth toward neutral 50 when evidence is thin.
    for feature in SHRINK_IN_RANK_SPACE & set(weights):

        ranks[feature] = (
            shrinkage * ranks[feature]
            + (1.0 - shrinkage) * 50.0
        )

    # Weighted v2 score.
    score = sum(
        ranks[feature] * weight
        for feature, weight in weights.items()
    )

    return score


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three rankings within each segment.

    rank_in_segment
        Our final leakage-free segment score.
        Uses fixed v2 weights with percentile ranks recalculated within segment.

    rank_in_segment_v2
        Existing global v2 score, but re-ranked inside each segment.

    rank_in_segment_baseline
        Estimated BNPL revenue only, ranked within each segment.
    """

    df = df.copy()

    df["segment_score"] = pd.NA

    # Recalculate score independently within every behavioural segment.
    for seg_id in df["segment_id"].unique():

        mask = df["segment_id"] == seg_id

        df.loc[
            mask,
            "segment_score"
        ] = segment_score_v2(
            df.loc[mask]
        )

    df["segment_score"] = pd.to_numeric(
        df["segment_score"]
    )

    def rank_within_segment(column: str) -> pd.Series:

        return (
            df.groupby("segment_id")[column]
            .rank(
                ascending=False,
                method="first",
            )
        )

    # Final segment ranking.
    df["rank_in_segment"] = (
        rank_within_segment("segment_score")
        .astype(int)
    )

    # Reference: global v2 score.
    df["rank_in_segment_v2"] = (
        rank_within_segment("score")
        .astype(int)
    )

    # Reference: revenue-only baseline.
    df["rank_in_segment_baseline"] = (
        rank_within_segment(
            "estimated_bnpl_revenue"
        )
        .astype(int)
    )

    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(df: pd.DataFrame) -> None:
    """Print each segment's Top 10 and compare reference rankings."""

    width = 108

    segment_order = sorted(
        df["segment_id"].unique(),
        key=lambda seg_id: -df.loc[
            df["segment_id"] == seg_id,
            "score",
        ].max(),
    )

    for seg_id in segment_order:

        block = df[
            df["segment_id"] == seg_id
        ]

        name = block[
            "segment_name"
        ].iloc[0]

        top = block.nsmallest(
            TOP_N,
            "rank_in_segment"
        )

        def picks(rank_column: str) -> set:

            return set(
                block
                .nsmallest(
                    TOP_N,
                    rank_column
                )["merchant_abn"]
            )

        selected = set(
            top["merchant_abn"]
        )

        overlap_global_v2 = len(
            selected
            & picks("rank_in_segment_v2")
        )

        overlap_revenue = len(
            selected
            & picks("rank_in_segment_baseline")
        )

        print(
            f"\n"
            f"{f' {name} ({len(block):,} merchants) ':-^{width}}"
        )

        print(
            f"  {'#':<3}"
            f"{'merchant':<38}"
            f"{'seg score':>10}"
            f"{'BNPL rev':>13}"
            f"{'customers':>11}"
            f"{'growth':>10}"
            f"{'overall':>9}"
        )

        for _, row in top.iterrows():

            growth = row[
                "log_value_growth"
            ]

            growth_text = (
                "NA"
                if pd.isna(growth)
                else f"{growth:.2f}"
            )

            print(
                f"  "
                f"{int(row['rank_in_segment']):<3}"
                f"{str(row['name'])[:36]:<38}"
                f"{row['segment_score']:>10.1f}"
                f"${row['estimated_bnpl_revenue']:>12,.0f}"
                f"{row['unique_customers']:>11,.0f}"
                f"{growth_text:>10}"
                f"{int(row['rank_overall']):>9}"
            )

        print(
            f"  overlap with global-v2 Top {TOP_N}: "
            f"{overlap_global_v2}/{TOP_N}"
        )

        print(
            f"  overlap with revenue-only Top {TOP_N}: "
            f"{overlap_revenue}/{TOP_N}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    df = add_ranks(
        load()
    )

    print(
        f"{len(df):,} merchants ranked across "
        f"{df['segment_id'].nunique()} behaviour segments"
    )

    print(
        "Segment scores use the fixed v2 weights "
        "with percentile ranks recomputed within each segment; "
        "no future test-period weights are used."
    )

    report(df)

    cols = [
        "segment_id",
        "segment_name",
        "segment",
        "rank_in_segment",
        "rank_in_segment_v2",
        "rank_in_segment_baseline",
        "merchant_abn",
        "name",
        "segment_score",
        "score",
        "rank_overall",
        "estimated_bnpl_revenue",
        "transaction_count",
        "unique_customers",
        "log_value_growth",
        "transactions_per_customer",
        "repeat_rate",
        "customer_hhi",
        "active_month_ratio",
        "value_cv",
        "customer_income",
        "customer_irsad",
        "take_rate",
    ]

    out = (
        df[
            df["rank_in_segment"] <= TOP_N
        ]
        .sort_values(
            [
                "segment_id",
                "rank_in_segment"
            ]
        )[cols]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    expected_rows = (
        TOP_N
        * df["segment_id"].nunique()
    )

    if len(out) != expected_rows:

        raise ValueError(
            f"expected {expected_rows} "
            f"Top-{TOP_N} rows, "
            f"got {len(out)}"
        )

    print(
        f"\nWritten to {OUTPUT_PATH} "
        f"({len(out)} rows)"
    )

    print("Done.")


if __name__ == "__main__":
    main()