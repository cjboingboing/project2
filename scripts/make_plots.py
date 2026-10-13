"""
Presentation figures, saved as PNGs under plots/.

Redraws the figures shown in notebooks/final_findings.ipynb with one shared
style and colour palette, sized for a 16:9 slide.

Input:  data/tables/processed/ (transactions_clean, the fraud tables, the v2
        ranking and its fraud-adjusted version, behaviour segments, weights)
        data/tables/models/ (exported by fraud_models.ipynb)
Output: plots/eda/      transaction amount distribution
        plots/segments/ revenue by behaviour segment
        plots/ranking/  v1 and v2 weights
        plots/models/   what the fraud classifiers respond to
        plots/fraud/    penalty by ticket size, how a day's score is split across
                        purchases, raw scores, score against spend, expected
                        loss at three cutoffs

Run with:
    python scripts/make_plots.py              # everything
    python scripts/make_plots.py fraud eda    # only these folders
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#Paths

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED = PROJECT_ROOT / "data" / "tables" / "processed"
MODEL_DIR = PROJECT_ROOT / "data" / "tables" / "models"
PLOTS = PROJECT_ROOT / "plots"

#Palette: one meaning per colour, in every figure

BLUE = "#2a78d6"      # the main series: real scores, the shipped ranking, the later scoring rule
ORANGE = "#eb6834"    # the series it is compared against: imputed, flagged, the early scoring rule
GREY = "#b4b2a9"      # background or reference series
BLUE_DARK = "#104281" # line drawn over a blue fill
BLUE_LIGHT = "#9ec5f4"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e7e6e1"

DPI = 200
RANDOM_STATE = 42

SCORE_START, SCORE_END = "2021-02-28", "2022-02-27"   # fraud scores exist only in this window
REGIME_START = "2021-08-28"                           # first day of the second transaction snapshot
DATA_END = "2022-10-27"
CUTOFFS = [30, 50, 70]                                # lowest score counted as a fraud signal
BINS_PER_DECADE = 10                                  # log-spaced bands of a user's total spend that day
MAX_SPEND_BIN = int(np.floor(np.log10(20_000) * BINS_PER_DECADE))   # pool everything above about $20K


def set_style():
    sns.set_theme(style="white", rc={
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 13, "axes.labelcolor": INK_SOFT, "axes.edgecolor": GRID, "axes.linewidth": 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 12,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })


def save(fig, folder, name):
    out = PLOTS / folder / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved", out.relative_to(PROJECT_ROOT))


def solid_legend(ax, **kwargs):
    """Legend whose swatches are opaque even when the points are drawn translucent."""
    legend = ax.legend(**kwargs)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    return legend


_spark = None


def get_spark():
    global _spark
    if _spark is None:
        from pyspark.sql import SparkSession
        _spark = SparkSession.builder.config("spark.sql.session.timeZone", "Etc/UTC").config("spark.driver.memory", "6g").getOrCreate()
        _spark.sparkContext.setLogLevel("ERROR")
    return _spark


#eda

def plot_eda():
    from pyspark.sql import functions as F
    spark = get_spark()
    max_rows = 100_000
    value = F.col("dollar_value").cast("double")
    cost = (spark.read.parquet(str(PROCESSED / "transactions_clean"))
            .select(value.alias("dollar_value"))
            .where(value.isNotNull() & ~F.isnan(value) & (value >= 0)))
    # Mean and standard deviation of log1p(amount) over every transaction, for the fitted normal.
    n_rows, mu, sigma = cost.select(F.count("*"), F.mean(F.log1p("dollar_value")),
                                    F.stddev(F.log1p("dollar_value"))).first()
    fraction = min(1.0, max_rows / n_rows)
    values = (cost.sample(False, fraction, seed=RANDOM_STATE).limit(max_rows)
              .toPandas()["dollar_value"].to_numpy())
    log_values = np.log1p(values)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.histplot(x=log_values, bins=120, stat="density", color=BLUE_LIGHT, edgecolor="none", alpha=0.9, ax=ax)
    sns.kdeplot(x=log_values, color=BLUE_DARK, linewidth=2, label="observed", ax=ax)
    grid = np.linspace(log_values.min(), log_values.max(), 400)
    ax.plot(grid, np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)), color=ORANGE,
            label="normal with the same mean and variance")
    ax.legend(loc="upper right")
    ticks = np.log1p([1, 10, 100, 1_000, 10_000, 100_000])
    ticks = ticks[ticks <= log_values.max() + 0.5]
    ax.set_xticks(ticks, [f"${np.expm1(t):,.0f}" for t in ticks])
    ax.set(xlabel="Transaction amount (log scale)", ylabel="Density",
           title="Transaction amounts are roughly log-normal")
    ax.grid(axis="x", visible=False)
    print(f"log1p(amount): mean {mu:.3f}, sd {sigma:.3f} over {n_rows:,} transactions "
          f"(median-equivalent ${np.expm1(mu):,.2f})")
    save(fig, "eda", "transaction_amount_distribution")


#segments and ranking

SEGMENT_NAMES = {1: "Established", 3: "Rising", 2: "Fading", 0: "Intermittent"}


def plot_segments():
    segments = pd.read_parquet(PROCESSED / "merchant_behaviour_segments")[["merchant_abn", "segment_id"]]
    ranking = pd.read_parquet(PROCESSED / "merchant_ranking_hybrid.parquet")
    seg = ranking.merge(segments, on="merchant_abn", how="inner")
    log_rev = np.log10(seg["estimated_bnpl_revenue"])
    bins = np.linspace(log_rev.min(), log_rev.max(), 81)

    fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True)
    for ax, seg_id in zip(axes, [1, 3, 2, 0]):
        block = log_rev[seg["segment_id"] == seg_id]
        sns.histplot(x=block, bins=bins, stat="density", color=BLUE_LIGHT, edgecolor="none", alpha=0.9, ax=ax)
        sns.kdeplot(x=block, color=BLUE_DARK, linewidth=2, ax=ax)
        ax.axvline(block.median(), color=INK_SOFT, linewidth=1)
        ax.set_title(f"{SEGMENT_NAMES[seg_id]}", fontsize=13)
        ax.text(1.0, 1.04, f"{len(block):,} merchants, median ${10 ** block.median():,.0f}",
                transform=ax.transAxes, ha="right", fontsize=12, color=INK_SOFT)
        ax.set_ylabel("Density")
        ax.grid(axis="x", visible=False)
    ticks = np.arange(np.floor(log_rev.min()), np.ceil(log_rev.max()) + 1)
    axes[-1].set_xticks(ticks, [f"${10 ** t:,.0f}" for t in ticks])
    axes[-1].set_xlabel("Estimated BNPL revenue over the full period (log scale)")
    fig.tight_layout()
    save(fig, "segments", "revenue_by_segment")


# v1 weights as set in merchant_ranking_baseline.ipynb; v2 weights are saved by merchant_ranking_hybrid.ipynb.
WEIGHTS_V1 = {"estimated_bnpl_revenue": 0.30, "log_value_growth": 0.20, "transactions_per_customer": 0.10,
              "stability": 0.10, "unique_customers": 0.05, "active_month_ratio": 0.05, "repeat_rate": 0.05,
              "customer_income": 0.05, "customer_irsad": 0.05, "customer_spread": 0.05}


def plot_ranking():
    weights_hybrid = pd.read_csv(PROCESSED / "weights_hybrid.csv", index_col=0)["weight"]
    weights = pd.DataFrame({"v1": pd.Series(WEIGHTS_V1), "hybrid (shipped)": weights_hybrid}).fillna(0) * 100
    weights = weights.sort_values("hybrid (shipped)", ascending=False)
    long = weights.rename_axis("feature").reset_index().melt("feature", var_name="version", value_name="weight")
    long["feature"] = long["feature"].str.replace("_", " ")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.barplot(data=long, y="feature", x="weight", hue="version", palette=[GREY, BLUE],
                saturation=1, width=0.72, ax=ax)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.0f%%", padding=4, fontsize=11, color=INK_SOFT)
    ax.set(xlabel="Weight in the score (%)", ylabel="",
           title="Ranking weights before and after the time-split validation")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", title=None)
    save(fig, "ranking", "ranking_weights_v1_v2")


#models

def plot_models(flag_cutoff=30):
    from xgboost import XGBClassifier
    consumer_model = XGBClassifier()
    consumer_model.load_model(MODEL_DIR / f"consumer_fraud_ge{flag_cutoff}.json")
    consumer = pd.read_parquet(MODEL_DIR / "consumer_fraud_test.parquet")
    consumer["flagged"] = (consumer["fraud_probability"] >= flag_cutoff).astype(int)
    consumer["p_flagged"] = consumer_model.predict_proba(
        consumer[consumer_model.get_booster().feature_names])[:, 1]
    # Days with one transaction, where the score belongs to exactly that transaction.
    consumer = consumer[consumer["day_txn_count"] == 1]
    # The merchant model was fitted on all 114 scored days, so its out-of-fold predictions are plotted.
    merchant = pd.read_parquet(MODEL_DIR / "merchant_fraud_days.parquet")
    merchant["flagged"] = (merchant["fraud_probability"] >= flag_cutoff).astype(int)
    merchant["p_flagged"] = merchant[f"p_ge_{flag_cutoff}_oof"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    panels = [(axes[0], consumer, "dollar_value", 14, "Transaction value (log scale)",
               "Consumer: flagged when the purchase is large"),
              (axes[1], merchant, "revenue_ratio", 60, "Day's revenue ÷ merchant's median day (log scale)",
               "Merchant: flagged when the day is a spike")]
    for ax, frame, x, size, xlabel, title in panels:
        for flagged, colour, name in [(0, BLUE, f"below {flag_cutoff}"), (1, ORANGE, f"{flag_cutoff} or above")]:
            part = frame[frame["flagged"] == flagged]
            ax.scatter(part[x], part["p_flagged"], s=size, color=colour, alpha=0.55, linewidths=0, label=name)
        ax.set_xscale("log")
        ax.set(xlabel=xlabel, title=title)
    axes[0].xaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    axes[1].xaxis.set_major_formatter(lambda v, _: f"{v:g}×")
    axes[0].set_ylabel(f"Predicted chance the score is {flag_cutoff} or above")
    solid_legend(axes[0], loc="upper left", title="Actual score", markerscale=1.6)
    fig.tight_layout()
    save(fig, "models", "fraud_model_drivers")


#fraud

def plot_penalty():
    fraud = pd.read_parquet(PROCESSED / "merchant_ranking_hybrid_fraud_adjusted.parquet")
    fraud["score_change"] = fraud["score_adjusted"] - fraud["score"]
    penalised = fraud["expected_fraud_loss"] > 0

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(fraud.loc[~penalised, "avg_transaction_value"], fraud.loc[~penalised, "score_change"], s=14,
               color=GREY, alpha=0.6, linewidths=0, label="no expected fraud loss")
    ax.scatter(fraud.loc[penalised, "avg_transaction_value"], fraud.loc[penalised, "score_change"], s=22,
               color=ORANGE, alpha=0.65, linewidths=0, label="has expected fraud loss")
    ax.axhline(0, color=INK_SOFT, linewidth=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax.set(xlabel="Merchant's average transaction value (log scale)",
           ylabel="Change in score from the fraud penalty",
           title="The fraud penalty falls on big-ticket merchants")
    solid_legend(ax, loc="lower left", markerscale=1.6)
    save(fig, "fraud", "penalty_by_ticket_size")


def plot_fraud_weighting():
    """How one user-day score is shared across that day's purchases, drawn for a real user-day."""
    import textwrap
    from pyspark.sql import Window, functions as F
    spark = get_spark()
    scores = (spark.read.parquet(str(PROCESSED / "consumer_fraud_probability"))
              .where(F.col("fraud_probability").isNotNull())
              .select("user_id", F.to_date("order_datetime").alias("order_date"),
                      F.col("fraud_probability").alias("score")))
    day = Window.partitionBy("user_id", "order_date")
    scored_txns = (
        spark.read.parquet(str(PROCESSED / "transactions_clean"))
        .withColumn("order_date", F.to_date("order_datetime"))
        .where(F.col("order_date").between(REGIME_START, SCORE_END))
        .join(scores, ["user_id", "order_date"])
        .withColumn("day_spend", F.sum("dollar_value").over(day))
        .withColumn("purchases", F.count("*").over(day))
        .withColumn("share", F.col("dollar_value") / F.col("day_spend"))
        .withColumn("min_share", F.min("share").over(day))
        .withColumn("max_share", F.max("share").over(day))
    )
    # An example that reads well: three purchases, none of them a sliver of the day. The highest-scoring
    # such day is used, so the choice is repeatable.
    example = (scored_txns.where((F.col("purchases") == 3) & (F.col("min_share") >= 0.15) & (F.col("max_share") <= 0.6))
               .orderBy(F.desc("score"), "user_id", "order_date").limit(3)
               .join(spark.read.parquet(str(PROCESSED / "merchants_clean")).select("merchant_abn", "name"),
                     "merchant_abn", "left")
               .toPandas())
    example = example[(example["user_id"] == example["user_id"].iloc[0])
                      & (example["order_date"] == example["order_date"].iloc[0])]
    example = example.sort_values("dollar_value", ascending=False).reset_index(drop=True)
    assert len(example) == 3, "expected one user-day with three purchases"
    score, spend = example["score"].iloc[0], example["day_spend"].iloc[0]
    example["p"] = score * example["share"]                       # in percent
    example["loss"] = example["p"] / 100 * example["dollar_value"]

    colours = [BLUE, ORANGE, "#1baf7a"]
    left, width, gap = 0.17, 0.80, 0.004
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0, 0.98, "One fraud score per user per day, shared across that day's purchases by dollar value",
            fontsize=16, fontweight="bold", va="top")
    ax.text(0, 0.905, f"User {example['user_id'].iloc[0]} on {pd.Timestamp(example['order_date'].iloc[0]):%d %b %Y}: "
            f"three purchases totalling ${spend:,.0f}, and a fraud score of {score:.1f}% for the day",
            fontsize=13, color=INK_SOFT, va="top")

    rows = {"spend": 0.60, "score": 0.33}
    ax.text(left - 0.02, rows["spend"] + 0.045, "The day's spend", ha="right", va="center", fontsize=13)
    ax.text(left - 0.02, rows["spend"] + 0.005, f"${spend:,.0f}", ha="right", va="center", fontsize=13, color=INK_SOFT)
    ax.text(left - 0.02, rows["score"] + 0.045, "The day's score", ha="right", va="center", fontsize=13)
    ax.text(left - 0.02, rows["score"] + 0.005, f"{score:.1f}%", ha="right", va="center", fontsize=13, color=INK_SOFT)

    x = left
    for (_, t), colour in zip(example.iterrows(), colours):
        w = width * t["share"]
        mid = x + w / 2
        for y in rows.values():
            ax.add_patch(plt.Rectangle((x + gap, y), w - 2 * gap, 0.09, color=colour, linewidth=0))
        name = t["name"] if isinstance(t["name"], str) else f"Merchant {t['merchant_abn']}"
        ax.text(mid, rows["spend"] + 0.115, textwrap.fill(name, 22), ha="center", va="bottom", fontsize=12,
                color=INK_SOFT, linespacing=1.2)
        ax.text(mid, rows["spend"] - 0.02, f"${t['dollar_value']:,.0f}", ha="center", va="top", fontsize=14,
                fontweight="bold")
        ax.text(mid, rows["spend"] - 0.07, f"{t['share']:.0%} of the day", ha="center", va="top", fontsize=12,
                color=INK_SOFT)
        ax.annotate("", (mid, rows["score"] + 0.095), (mid, rows["spend"] - 0.125),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, linewidth=1.2))
        ax.text(mid, rows["score"] - 0.02, f"{t['p']:.1f}%", ha="center", va="top", fontsize=14, fontweight="bold")
        ax.text(mid, rows["score"] - 0.07, f"{t['share']:.0%} of the score", ha="center", va="top", fontsize=12,
                color=INK_SOFT)
        ax.text(mid, rows["score"] - 0.16, f"Expected loss\n{t['p']:.1f}% × \\${t['dollar_value']:,.0f} = \\${t['loss']:,.0f}",
                ha="center", va="top", fontsize=12, linespacing=1.4)
        x += w
    ax.text(0, 0.0, f"Allocation illustration before thresholding: ${example['loss'].sum():,.0f}. "
            "Default MIN_SCORE=30 excludes any score below 30.",
            fontsize=12, color=INK_SOFT, va="bottom")
    save(fig, "fraud", "score_split_across_purchases")
    print(example[["merchant_abn", "name", "dollar_value", "share", "score", "p", "loss"]].round(3).to_string())


def date_axis(ax, start, end, interval):
    ax.set_xlim(pd.Timestamp(start), pd.Timestamp(end))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def plot_fraud_scores():
    from pyspark.sql import functions as F
    spark = get_spark()
    break_day = pd.Timestamp(REGIME_START)

    def p_if_at_least(score_col, min_score):
        return F.when(F.col(score_col) >= min_score, F.col(score_col) / 100).otherwise(0.0)

    consumer_scores = (spark.read.parquet(str(PROCESSED / "consumer_fraud_probability"))
                       .where(F.col("fraud_probability").isNotNull())
                       .select("user_id", F.to_date("order_datetime").alias("order_date"),
                               F.col("fraud_probability").alias("consumer_score")))
    # One row per user per day: that day's total spend, and the supplied score where there is one.
    user_days = (
        spark.read.parquet(str(PROCESSED / "transactions_clean"))
        .withColumn("order_date", F.to_date("order_datetime"))
        .groupBy("user_id", "order_date").agg(F.sum("dollar_value").alias("day_spend"))
        .join(consumer_scores, ["user_id", "order_date"], "left")
        .withColumn("spend_bin", F.least(F.floor(F.log10("day_spend") * BINS_PER_DECADE), F.lit(MAX_SPEND_BIN)))
    ).cache()
    scored = user_days.where(F.col("consumer_score").isNotNull()).toPandas()
    scored["order_date"] = pd.to_datetime(scored["order_date"])
    merchant = (spark.read.parquet(str(PROCESSED / "merchant_fraud_probability"))
                .select(F.to_date("order_datetime").alias("order_date"),
                        F.col("fraud_probability").alias("merchant_score")).toPandas())
    merchant["order_date"] = pd.to_datetime(merchant["order_date"])

    #1. every supplied score through time
    # The x-axis runs to the last transaction date, but the months with no scores are squeezed into a
    # narrow strip behind an axis break: nothing happens there, and the strip only has to show that
    # the scores stop while transactions carry on.
    start, scores_end, last_day = pd.Timestamp(SCORE_START), pd.Timestamp(SCORE_END), pd.Timestamp(DATA_END)
    scored_span = (scores_end - start).days
    squeeze = 0.17                                    # width of a day after the scores stop, relative to before

    def x_of(dates):
        days = (pd.to_datetime(pd.Series(dates)) - start).dt.days.to_numpy(dtype=float)
        return np.where(days <= scored_span, days, scored_span + (days - scored_span) * squeeze)

    x_end = x_of([last_day])[0]
    strip_mid = (scored_span + x_end) / 2
    break_x = x_of([break_day])[0]
    per_day = scored.groupby("order_date").size().reindex(pd.date_range(start, last_day), fill_value=0)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, height_ratios=[3, 1.5, 2])
    axes[0].scatter(x_of(scored["order_date"]), scored["consumer_score"], s=5, color=BLUE, alpha=0.25, linewidths=0)
    axes[0].set(ylabel="Fraud probability (%)", ylim=(0, 100),
                title=f"Consumer scores: all {len(scored):,} supplied")
    axes[1].plot(x_of(per_day.index), per_day.values, color=BLUE)
    axes[1].set(ylabel="Scores per day", title="Consumer scores supplied per day")
    axes[1].set_ylim(0, None)
    axes[2].scatter(x_of(merchant["order_date"]), merchant["merchant_score"], s=70, color=BLUE, alpha=0.75,
                    edgecolor="white", linewidths=1.5)
    axes[2].set(ylabel="Fraud probability (%)", ylim=(0, 100),
                title=f"Merchant scores: all {len(merchant)} supplied")

    months = pd.date_range("2021-03-01", "2022-02-01", freq="MS")
    wave = np.linspace(-1, 1, 60)
    for ax in axes:
        ax.axvspan(scored_span + 0.5, x_end, color=GRID, alpha=0.55, linewidth=0)
        ax.axvline(break_x, color=INK_SOFT, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.set_xlim(0, x_end)
        ax.set_xticks(list(x_of(months)) + [x_end], [f"{m:%b %Y}" for m in months] + ["26 Oct\n2022"])
        ax.grid(axis="x", visible=False)
        # Axis break: a squiggle across the baseline in the middle of the squeezed strip.
        for colour, width in [("white", 7), (INK_SOFT, 1.6)]:
            ax.plot(strip_mid + wave * 9, 0.035 * np.sin(wave * 2.5 * np.pi), color=colour, linewidth=width,
                    transform=ax.get_xaxis_transform(), clip_on=False, zorder=5, solid_capstyle="round")
    for label in axes[-1].get_xticklabels()[:-1]:
        label.set_rotation(0)
    axes[-1].set_xticks(list(x_of(months[1::2])) + [x_end], [f"{m:%b %Y}" for m in months[1::2]] + ["26 Oct\n2022"])
    axes[0].annotate("28 Aug 2021:\nsecond data snapshot begins", (break_x, 90), xytext=(-8, 0),
                     textcoords="offset points", ha="right", va="center", fontsize=12, color=INK_SOFT)
    for ax in (axes[0], axes[2]):
        ax.text(strip_mid, 55, "no scores\nsupplied for\n8 months", ha="center", va="center",
                fontsize=12, color=INK_SOFT, linespacing=1.4)
    axes[1].text(strip_mid, 0.5, "not to scale", transform=axes[1].get_xaxis_transform(), ha="center", va="center",
                 fontsize=11, color=INK_SOFT, style="italic")
    fig.tight_layout()
    save(fig, "fraud", "raw_scores_through_time")

    #2. score against the day's spend, before and after the break
    late, early = scored[scored["order_date"] >= break_day], scored[scored["order_date"] < break_day]

    def spend_panel(ax, divisor, title):
        ax.scatter(late["day_spend"], late["consumer_score"], s=6, color=BLUE, alpha=0.18, linewidths=0,
                   label="28 Aug 2021 onwards")
        ax.scatter(early["day_spend"] / divisor, early["consumer_score"], s=6, color=ORANGE, alpha=0.4,
                   linewidths=0, label="before 28 Aug 2021")
        ax.set_xscale("log")
        ax.set_xticks([1500, 3000, 6000, 12000, 25000, 50000], ["$1.5k", "$3k", "$6k", "$12k", "$25k", "$50k"])
        ax.minorticks_off()
        ax.set(xlabel="User's total spend that day", title=title, xlim=(1200, 80_000), ylim=(0, 100))

    titles = {1: "As recorded", 2: "Earlier days drawn at half their spend"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, divisor in zip(axes, titles):
        spend_panel(ax, divisor, titles[divisor])
    axes[0].set_ylabel("Fraud probability (%)")
    solid_legend(axes[0], loc="upper left", markerscale=3.5)
    fig.tight_layout()
    save(fig, "fraud", "score_vs_spend_early_late")
    # The same two panels as separate slides, so the second can be revealed after the first.
    for divisor, name in [(1, "score_vs_spend_1_as_recorded"), (2, "score_vs_spend_2_half_spend")]:
        fig, ax = plt.subplots(figsize=(11, 6))
        spend_panel(ax, divisor, titles[divisor])
        ax.set_ylabel("Fraud probability (%)")
        solid_legend(ax, loc="upper left", markerscale=3.5)
        save(fig, "fraud", name)

    #3. expected loss through time at each cutoff: supplied scores against the spend-band imputation
    late_days = user_days.where(F.col("order_date").between(REGIME_START, SCORE_END))
    lookup = late_days.groupBy("spend_bin").agg(
        *[(F.sum(p_if_at_least("consumer_score", c) * F.col("day_spend")) / F.sum("day_spend")).alias(f"p{c}")
          for c in CUTOFFS])
    daily = (
        user_days.join(F.broadcast(lookup), "spend_bin", "left")
        .groupBy("order_date")
        .agg(F.sum("day_spend").alias("value"),
             *[F.sum(p_if_at_least("consumer_score", c) * F.col("day_spend")).alias(f"observed{c}") for c in CUTOFFS],
             *[F.sum(F.coalesce(F.col(f"p{c}"), F.lit(0.0)) * F.col("day_spend")).alias(f"imputed{c}")
               for c in CUTOFFS])
        .toPandas()
    )
    daily["order_date"] = pd.to_datetime(daily["order_date"])
    daily = daily.sort_values("order_date").set_index("order_date")
    in_window = (daily.index >= SCORE_START) & (daily.index <= SCORE_END)
    rolling = daily.rolling("7D").sum()
    window_end = pd.Timestamp(SCORE_END) + pd.Timedelta(days=1)

    fig, axes = plt.subplots(len(CUTOFFS), 1, figsize=(13, 9), sharex=True, sharey=True)
    for ax, c in zip(axes, CUTOFFS):
        ax.axvspan(window_end, pd.Timestamp(DATA_END), color=GRID, alpha=0.55, linewidth=0)
        ax.axvline(break_day, color=INK_SOFT, linewidth=1.2, linestyle=(0, (4, 3)))
        imputed = rolling[f"imputed{c}"] / rolling["value"] * 100
        observed = (rolling[f"observed{c}"] / rolling["value"] * 100).where(in_window)
        ax.plot(imputed.index, imputed, color=ORANGE, linewidth=1.6, label="imputed from the day's spend")
        ax.plot(observed.index, observed, color=BLUE, linewidth=2.2, label="supplied scores")
        ax.set(ylabel=r"Loss per \$100 (\$)", title=f"Counting only scores of {c} and above")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
        date_axis(ax, SCORE_START, DATA_END, 2)
    axes[0].set_ylim(0, None)
    top = axes[0].get_ylim()[1]
    for when, text in [(pd.Timestamp(SCORE_START) + (break_day - pd.Timestamp(SCORE_START)) / 2, "early scoring rule"),
                       (break_day + (window_end - break_day) / 2, "later scoring rule"),
                       (window_end + (pd.Timestamp(DATA_END) - window_end) / 2, "no scores supplied")]:
        axes[-1].text(when, top * 0.93, text, ha="center", va="top", fontsize=12, color=INK_SOFT)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, loc="upper right", bbox_to_anchor=(0.99, 0.99))
    fig.suptitle(r"Expected fraud loss per \$100 transacted, 7-day rolling", x=0.065, y=0.975, ha="left",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, "fraud", "expected_loss_by_cutoff")


GROUPS = {
    "eda": [plot_eda],
    "segments": [plot_segments],
    "ranking": [plot_ranking],
    "models": [plot_models],
    "fraud": [plot_penalty, plot_fraud_weighting, plot_fraud_scores],
}


def main():
    wanted = sys.argv[1:] or list(GROUPS)
    unknown = [g for g in wanted if g not in GROUPS]
    if unknown:
        sys.exit(f"Unknown group(s) {unknown}; choose from {list(GROUPS)}")
    set_style()
    for group in wanted:
        for plot in GROUPS[group]:
            plot()
    if _spark is not None:
        _spark.stop()


if __name__ == "__main__":
    main()
