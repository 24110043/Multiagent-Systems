#!/usr/bin/env python3
"""
06_creator_analysis.py
Stage 7: Creator-Level Aggregation & Statistical Analysis for Final Click Appeal Dataset (N=809).

Tasks:
1. Creator Aggregation:
   - Aggregate by creator channel and calendar month.
   - Calculate channel video counts, mean, std, min, max Click Appeal Scores.
   - Calculate Leave-One-Creator-Out (LOCO) mean scores across the 38 channels.
   - Save scores/creator_summary_scores.csv and scores/creator_monthly_scores.csv.

2. Visualizations:
   - analysis/02_channel_appeal_comparison.png: Ranked horizontal bar plot for all 38 creator channels.
   - analysis/03_score_components_breakdown.png: Title score vs. Thumbnail score scatter with regression line.
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "06_creator_analysis.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("creator_analysis")

VIDEO_SCORES_PATH = BASE_DIR / "scores" / "video_scores.csv"
SCORES_DIR = BASE_DIR / "scores"
ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

CREATOR_MONTHLY_CSV = SCORES_DIR / "creator_monthly_scores.csv"
CREATOR_SUMMARY_CSV = SCORES_DIR / "creator_summary_scores.csv"

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]


def run_creator_analysis():
    logger.info("Starting Stage 7: Creator-Level Aggregation & Analysis (Final N=809)...")

    if not VIDEO_SCORES_PATH.exists():
        logger.error(f"Scores CSV missing at {VIDEO_SCORES_PATH}. Run Stage 6 first.")
        sys.exit(1)

    df = pd.read_csv(VIDEO_SCORES_PATH)
    logger.info(f"Loaded {len(df)} scored video records across {df['channel_name'].nunique()} channels.")

    # Convert publication timestamp to calendar month
    df["published_at_dt"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["year_month"] = df["published_at_dt"].dt.to_period("M").astype(str)

    # 1. Channel-level aggregation & Leave-One-Creator-Out (LOCO)
    creator_summary = []
    overall_mean_weighted = float(df["score_weighted"].mean())

    for (cid, cname), group in df.groupby(["channel_id", "channel_name"]):
        v_count = len(group)
        loco_dataset = df[df["channel_id"] != cid]
        loco_mean_weighted = float(loco_dataset["score_weighted"].mean()) if len(loco_dataset) > 0 else np.nan

        creator_summary.append({
            "channel_id": cid,
            "channel_name": cname,
            "video_count": v_count,
            "subscribers": group["subscribers"].iloc[0] if "subscribers" in group.columns else np.nan,
            "mean_score_weighted": round(float(group["score_weighted"].mean()), 6),
            "std_score_weighted": round(float(group["score_weighted"].std()), 6) if v_count > 1 else 0.0,
            "min_score_weighted": round(float(group["score_weighted"].min()), 6),
            "max_score_weighted": round(float(group["score_weighted"].max()), 6),
            "mean_score_title": round(float(group["score_title"].mean()), 6),
            "mean_score_thumbnail": round(float(group["score_thumbnail"].mean()), 6),
            "mean_score_equal": round(float(group["score_equal_weight"].mean()), 6),
            "mean_score_pca": round(float(group["score_pca"].mean()), 6),
            "mean_views": round(float(group["views"].mean()), 2),
            "mean_likes": round(float(group["likes"].mean()), 2),
            "mean_comments": round(float(group["comments"].mean()), 2),
            "loco_mean_weighted": round(loco_mean_weighted, 6),
            "loco_difference": round(float(group["score_weighted"].mean()) - loco_mean_weighted, 6)
        })

    df_creator_summary = pd.DataFrame(creator_summary).sort_values(by="mean_score_weighted", ascending=False)
    df_creator_summary.to_csv(CREATOR_SUMMARY_CSV, index=False, encoding="utf-8")
    logger.info(f"Saved creator summary to {CREATOR_SUMMARY_CSV} ({len(df_creator_summary)} channels)")

    # 2. Monthly aggregation by channel
    monthly_records = []
    for (cid, cname, ym), group in df.groupby(["channel_id", "channel_name", "year_month"]):
        v_count = len(group)
        # Niche-month mean across all other creators in same month
        month_peers = df[(df["year_month"] == ym) & (df["channel_id"] != cid)]
        loco_month_mean = float(month_peers["score_weighted"].mean()) if len(month_peers) > 0 else np.nan

        monthly_records.append({
            "channel_id": cid,
            "channel_name": cname,
            "year_month": ym,
            "monthly_video_count": v_count,
            "mean_score_weighted": round(float(group["score_weighted"].mean()), 6),
            "std_score_weighted": round(float(group["score_weighted"].std()), 6) if v_count > 1 else 0.0,
            "mean_score_title": round(float(group["score_title"].mean()), 6),
            "mean_score_thumbnail": round(float(group["score_thumbnail"].mean()), 6),
            "mean_views": round(float(group["views"].mean()), 2),
            "loco_month_mean_weighted": round(loco_month_mean, 6) if not np.isnan(loco_month_mean) else np.nan
        })

    df_monthly = pd.DataFrame(monthly_records).sort_values(by=["channel_name", "year_month"])
    df_monthly.to_csv(CREATOR_MONTHLY_CSV, index=False, encoding="utf-8")
    logger.info(f"Saved monthly scores to {CREATOR_MONTHLY_CSV} ({len(df_monthly)} rows)")

    # 3. Channel Appeal Comparison Plot (Ranked Bar Chart for 38 Channels)
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    channel_ranked = df_creator_summary.sort_values(by="mean_score_weighted", ascending=True)
    y_pos = np.arange(len(channel_ranked))
    ax.barh(y_pos, channel_ranked["mean_score_weighted"], xerr=channel_ranked["std_score_weighted"], color="#2ca02c", alpha=0.85, edgecolor="#1e7e1e", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['channel_name']} (n={r['video_count']})" for _, r in channel_ranked.iterrows()], fontsize=9)
    ax.set_xlabel("Mean Click Appeal Score ($X_v$)", fontsize=11)
    ax.set_title("Mean Click Appeal Score across Creator Channels (Final N=809, 38 Channels)", fontsize=13, fontweight="bold", pad=15)
    ax.axvline(overall_mean_weighted, color="black", linestyle="--", linewidth=1.2, label=f"Overall Mean: {overall_mean_weighted:.3f}")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot_channel_path = ANALYSIS_DIR / "02_channel_appeal_comparison.png"
    plt.savefig(plot_channel_path, dpi=300)
    plt.close()
    logger.info(f"Saved {plot_channel_path}")

    # 4. Score Components Breakdown Plot (Title vs Thumbnail)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    scatter = ax.scatter(df["score_title"], df["score_thumbnail"], c=df["score_weighted"], cmap="viridis", alpha=0.7, s=40, edgecolors="none")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Click Appeal Score ($X_v = 0.4 X_{\\mathrm{title}} + 0.6 X_{\\mathrm{thumb}}$)", fontsize=10)
    m, b = np.polyfit(df["score_title"], df["score_thumbnail"], 1)
    x_vals = np.linspace(df["score_title"].min(), df["score_title"].max(), 100)
    ax.plot(x_vals, m * x_vals + b, color="red", linestyle="--", linewidth=1.5, label=f"Trend Line (slope={m:.2f})")
    ax.set_title("Title Score vs. Thumbnail Score Breakdown (N=809)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Title Click Appeal Score ($X_{\\mathrm{title}}$)", fontsize=11)
    ax.set_ylabel("Thumbnail Click Appeal Score ($X_{\\mathrm{thumb}}$)", fontsize=11)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plot_breakdown_path = ANALYSIS_DIR / "03_score_components_breakdown.png"
    plt.savefig(plot_breakdown_path, dpi=300)
    plt.close()
    logger.info(f"Saved {plot_breakdown_path}")

    print("\n================== STAGE 7 CREATOR ANALYSIS SUMMARY ==================")
    print(f"  Total channels analyzed      : {len(df_creator_summary)}")
    print(f"  Overall mean Click Appeal    : {overall_mean_weighted:.4f}")
    print(f"  Top 3 highest appeal creators:")
    for _, r in df_creator_summary.head(3).iterrows():
        print(f"    - {r['channel_name']:<24}: {r['mean_score_weighted']:.4f} (n={r['video_count']})")
    print(f"  Bottom 3 lowest appeal creators:")
    for _, r in df_creator_summary.tail(3).iterrows():
        print(f"    - {r['channel_name']:<24}: {r['mean_score_weighted']:.4f} (n={r['video_count']})")
    print("======================================================================\n")


if __name__ == "__main__":
    run_creator_analysis()
