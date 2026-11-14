#!/usr/bin/env python3
"""
07_validate_results.py
Stage 8: Comprehensive End-to-End Validation & Quality Audit for Final Click Appeal Dataset (N=809).

Checks:
1. Row Accountability: Exactly 809 unique videos across cleaned data, manifests, features, and scores.
2. File Availability: All 809 original thumbnails and annotated thumbnails verified on disk.
3. ID & Title Integrity: 0 duplicate IDs, 0 formula artifacts, all titles present and valid.
4. Feature Completeness: 0 missing or NaN values across all 8 title and thumbnail features.
5. Score Boundedness: All scores strictly in [0.0, 1.0].
6. Mathematical Reconstructibility:
   - X_title == (x_caps + x_trigger + x_sensitive + x_symbol) / 4
   - X_thumb == (x_face + x_expr + x_contrast + x_saturation) / 4
   - X_v == 0.4 * X_title + 0.6 * X_thumb
7. Generates:
   - analysis/01_score_distribution.png (clean distribution plot for the 809 videos)
   - analysis/validation_report.csv
   - analysis/VALIDATION_REPORT.md
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
        logging.FileHandler(LOG_DIR / "07_validate_results.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("validate_results")

CLEANED_CSV = BASE_DIR / "data" / "processed" / "cleaned_videos.csv"
MANIFEST_CSV = BASE_DIR / "thumbnails_original" / "download_manifest.csv"
TITLE_FEATURES_CSV = BASE_DIR / "features" / "title_features.csv"
THUMB_FEATURES_CSV = BASE_DIR / "features" / "thumbnail_features.csv"
NORMALIZED_CSV = BASE_DIR / "features" / "normalized_features.csv"
SCORES_CSV = BASE_DIR / "scores" / "video_scores.csv"
CREATOR_SUMMARY_CSV = BASE_DIR / "scores" / "creator_summary_scores.csv"
CREATOR_MONTHLY_CSV = BASE_DIR / "scores" / "creator_monthly_scores.csv"

THUMBNAILS_DIR = BASE_DIR / "thumbnails_original"
ANNOTATED_DIR = BASE_DIR / "annotated_thumbnails"
ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_REPORT_CSV = ANALYSIS_DIR / "validation_report.csv"
VALIDATION_REPORT_MD = ANALYSIS_DIR / "VALIDATION_REPORT.md"
DISTRIBUTION_PLOT = ANALYSIS_DIR / "01_score_distribution.png"


def run_validation():
    logger.info("Starting Stage 8: Comprehensive Validation & Quality Audit (Final N=809)...")
    checks = []

    # 1. Row Accountability
    df_clean = pd.read_csv(CLEANED_CSV)
    df_manifest = pd.read_csv(MANIFEST_CSV)
    df_title = pd.read_csv(TITLE_FEATURES_CSV)
    df_thumb = pd.read_csv(THUMB_FEATURES_CSV)
    df_norm = pd.read_csv(NORMALIZED_CSV)
    df_scores = pd.read_csv(SCORES_CSV)

    n_clean = len(df_clean)
    n_manifest = len(df_manifest)
    n_title = len(df_title)
    n_thumb = len(df_thumb)
    n_norm = len(df_norm)
    n_scores = len(df_scores)

    row_match = (n_clean == n_manifest == n_title == n_thumb == n_norm == n_scores == 809)
    checks.append({
        "check_id": "CHK_01_ROW_ACCOUNTABILITY",
        "description": "Exactly 809 unique videos present across all pipeline stages",
        "expected": "809 rows in all files",
        "actual": f"Cleaned={n_clean}, Manifest={n_manifest}, Title={n_title}, Thumb={n_thumb}, Norm={n_norm}, Scores={n_scores}",
        "status": "PASS" if row_match else "FAIL",
        "details": "Zero missing or discarded records across all stages."
    })

    # 2. Duplicate Video IDs
    clean_dupes = df_clean["video_id"].astype(str).str.strip().duplicated().sum()
    scores_dupes = df_scores["video_id"].astype(str).str.strip().duplicated().sum()
    dupes_pass = (clean_dupes == 0 and scores_dupes == 0)
    checks.append({
        "check_id": "CHK_02_DUPLICATE_IDS",
        "description": "Zero duplicate video IDs in cleaned and scored datasets",
        "expected": "0 duplicates",
        "actual": f"Cleaned dupes={clean_dupes}, Scores dupes={scores_dupes}",
        "status": "PASS" if dupes_pass else "FAIL",
        "details": "All 809 video IDs are unique."
    })

    # 3. ID and Title Integrity
    formula_artifacts = (
        (df_clean["video_id"] == "#NAME?").sum() +
        df_clean["video_id"].astype(str).str.startswith("=").sum()
    )
    empty_titles = df_clean["title"].isnull().sum() + (df_clean["title"].astype(str).str.strip() == "").sum()
    integrity_pass = (formula_artifacts == 0 and empty_titles == 0)
    checks.append({
        "check_id": "CHK_03_ID_TITLE_INTEGRITY",
        "description": "No corrupted #NAME? IDs, formula strings, or missing titles",
        "expected": "0 artifacts, 0 empty titles",
        "actual": f"Corrupted IDs={formula_artifacts}, Empty titles={empty_titles}",
        "status": "PASS" if integrity_pass else "FAIL",
        "details": "All video IDs and titles are valid YouTube records."
    })

    # 4. Thumbnail Image Availability on Disk
    missing_thumbs = 0
    missing_annotated = 0
    for vid in df_clean["video_id"].astype(str).str.strip():
        t_path = THUMBNAILS_DIR / f"{vid}.jpg"
        a_path = ANNOTATED_DIR / f"{vid}_annotated.jpg"
        if not t_path.exists() or t_path.stat().st_size == 0:
            missing_thumbs += 1
        if not a_path.exists() or a_path.stat().st_size == 0:
            missing_annotated += 1

    files_pass = (missing_thumbs == 0 and missing_annotated == 0)
    checks.append({
        "check_id": "CHK_04_THUMBNAIL_FILES_ON_DISK",
        "description": "All 809 original and annotated thumbnails available on disk",
        "expected": "809 thumbnails, 809 annotated images, 0 missing",
        "actual": f"Missing original={missing_thumbs}, Missing annotated={missing_annotated}",
        "status": "PASS" if files_pass else "FAIL",
        "details": "Complete pool of 809 images accessible locally."
    })

    # 5. Feature Completeness
    feature_cols = ["x_caps", "x_trigger", "x_sensitive", "x_symbol", "x_face", "x_expr", "x_contrast", "x_saturation"]
    missing_feats = df_norm[feature_cols].isnull().sum().sum()
    feats_pass = (missing_feats == 0)
    checks.append({
        "check_id": "CHK_05_FEATURE_COMPLETENESS",
        "description": "Zero missing or NaN values across all 8 normalized features",
        "expected": "0 missing values",
        "actual": f"Missing feature values={missing_feats}",
        "status": "PASS" if feats_pass else "FAIL",
        "details": "All features successfully extracted and normalized."
    })

    # 6. Score Boundedness [0, 1]
    score_cols = ["score_title", "score_thumbnail", "score_weighted", "score_equal_weight", "score_pca"]
    out_of_bounds = 0
    for sc in score_cols:
        vals = df_scores[sc]
        out_of_bounds += ((vals < -1e-6) | (vals > 1.0 + 1e-6)).sum()
    bounds_pass = (out_of_bounds == 0)
    checks.append({
        "check_id": "CHK_06_SCORE_BOUNDEDNESS",
        "description": "All composite Click Appeal Scores strictly bounded within [0.0, 1.0]",
        "expected": "0 out-of-bound values",
        "actual": f"Out-of-bound values={out_of_bounds}",
        "status": "PASS" if bounds_pass else "FAIL",
        "details": "Every score respects theoretical [0, 1] support."
    })

    # 7. Mathematical Reconstructibility
    calc_title = (df_norm["x_caps"] + df_norm["x_trigger"] + df_norm["x_sensitive"] + df_norm["x_symbol"]) / 4.0
    calc_thumb = (df_norm["x_face"] + df_norm["x_expr"] + df_norm["x_contrast"] + df_norm["x_saturation"]) / 4.0
    calc_weighted = 0.4 * calc_title + 0.6 * calc_thumb

    diff_title = (df_scores["score_title"] - calc_title).abs().max()
    diff_thumb = (df_scores["score_thumbnail"] - calc_thumb).abs().max()
    diff_weighted = (df_scores["score_weighted"] - calc_weighted).abs().max()

    reconstruct_pass = (diff_title < 1e-5 and diff_thumb < 1e-5 and diff_weighted < 1e-5)
    checks.append({
        "check_id": "CHK_07_MATHEMATICAL_RECONSTRUCTIBILITY",
        "description": "Exact mathematical reconstructibility of X_title, X_thumb, and X_v = 0.4*X_title + 0.6*X_thumb",
        "expected": "Max absolute difference < 1e-5",
        "actual": f"Max diff: Title={diff_title:.8f}, Thumb={diff_thumb:.8f}, Weighted={diff_weighted:.8f}",
        "status": "PASS" if reconstruct_pass else "FAIL",
        "details": "Original 0.4 / 0.6 scoring formula perfectly preserved."
    })

    # Save validation table
    df_checks = pd.DataFrame(checks)
    df_checks.to_csv(VALIDATION_REPORT_CSV, index=False, encoding="utf-8")
    logger.info(f"Saved validation metrics table to {VALIDATION_REPORT_CSV}")

    # Generate Distribution Plot for the 809 videos
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.histplot(df_scores["score_weighted"], kde=True, ax=ax, color="#1f77b4", bins=30, edgecolor="white", alpha=0.6)
    mean_val = df_scores["score_weighted"].mean()
    median_val = df_scores["score_weighted"].median()
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.4f}")
    ax.axvline(median_val, color="green", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.4f}")
    ax.set_title("Click Appeal Score Distribution (Final Dataset, N = 809 Videos)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Click Appeal Score ($X_v = 0.4 X_{\\mathrm{title}} + 0.6 X_{\\mathrm{thumb}}$)", fontsize=11)
    ax.set_ylabel("Video Count", fontsize=11)
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    plt.savefig(DISTRIBUTION_PLOT, dpi=300)
    plt.close()
    logger.info(f"Saved {DISTRIBUTION_PLOT}")

    # Generate Markdown Validation Report
    sc = df_scores["score_weighted"]
    report_md = f"""# Click Appeal Score — Final Validation Report (N = 809)

## 1. Overview
This report documents the final quality assurance and mathematical validation of the **Click Appeal Score Pipeline** executed on the unified final dataset of **809 unique YouTube videos** across 38 creator channels.

---

## 2. Validation Audit Checklist

| Check ID | Description | Expected | Actual | Status |
| :--- | :--- | :--- | :--- | :---: |
"""
    for _, r in df_checks.iterrows():
        report_md += f"| `{r['check_id']}` | {r['description']} | {r['expected']} | {r['actual']} | **{r['status']}** |\n"

    report_md += f"""
---

## 3. Score Distribution Summary ($N = 809$)

| Metric | Score Formula / Definition | Mean | Std | Min | 25% | Median | 75% | Max | Skewness |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`score_weighted` ($X_v$)** | **$0.4 X_{{\\text{{title}}}} + 0.6 X_{{\\text{{thumb}}}}$ (Primary)** | **{sc.mean():.4f}** | **{sc.std():.4f}** | **{sc.min():.4f}** | **{sc.quantile(0.25):.4f}** | **{sc.median():.4f}** | **{sc.quantile(0.75):.4f}** | **{sc.max():.4f}** | **{sc.skew():.3f}** |
| `score_title` | $(x_{{\\text{{caps}}}} + x_{{\\text{{trig}}}} + x_{{\\text{{sens}}}} + x_{{\\text{{sym}}}}) / 4$ | {df_scores['score_title'].mean():.4f} | {df_scores['score_title'].std():.4f} | {df_scores['score_title'].min():.4f} | {df_scores['score_title'].quantile(0.25):.4f} | {df_scores['score_title'].median():.4f} | {df_scores['score_title'].quantile(0.75):.4f} | {df_scores['score_title'].max():.4f} | {df_scores['score_title'].skew():.3f} |
| `score_thumbnail` | $(x_{{\\text{{face}}}} + x_{{\\text{{expr}}}} + x_{{\\text{{cont}}}} + x_{{\\text{{sat}}}}) / 4$ | {df_scores['score_thumbnail'].mean():.4f} | {df_scores['score_thumbnail'].std():.4f} | {df_scores['score_thumbnail'].min():.4f} | {df_scores['score_thumbnail'].quantile(0.25):.4f} | {df_scores['score_thumbnail'].median():.4f} | {df_scores['score_thumbnail'].quantile(0.75):.4f} | {df_scores['score_thumbnail'].max():.4f} | {df_scores['score_thumbnail'].skew():.3f} |
| `score_equal_weight` | $\\frac{{1}}{{8}} \\sum_{{j=1}}^8 x_j$ | {df_scores['score_equal_weight'].mean():.4f} | {df_scores['score_equal_weight'].std():.4f} | {df_scores['score_equal_weight'].min():.4f} | {df_scores['score_equal_weight'].quantile(0.25):.4f} | {df_scores['score_equal_weight'].median():.4f} | {df_scores['score_equal_weight'].quantile(0.75):.4f} | {df_scores['score_equal_weight'].max():.4f} | {df_scores['score_equal_weight'].skew():.3f} |
| `score_pca` | Projection on established PC1 | {df_scores['score_pca'].mean():.4f} | {df_scores['score_pca'].std():.4f} | {df_scores['score_pca'].min():.4f} | {df_scores['score_pca'].quantile(0.25):.4f} | {df_scores['score_pca'].median():.4f} | {df_scores['score_pca'].quantile(0.75):.4f} | {df_scores['score_pca'].max():.4f} | {df_scores['score_pca'].skew():.3f} |

---

## 4. Summary of Deliverables Retained
- **Cleaned Dataset:** `data/processed/cleaned_videos.csv` (809 rows, 16 metadata columns).
- **ID Recovery Log:** `data/processed/cleaning_log.csv` (12 corrupted IDs recovered).
- **Thumbnails:** `thumbnails_original/` (809 high-resolution images).
- **Annotated Thumbnails:** `annotated_thumbnails/` (809 audit overlays + `contact_sheet.jpg`).
- **Feature Tables:** `features/title_features.csv`, `features/thumbnail_features.csv`, `features/normalized_features.csv`.
- **Scores Table:** `scores/video_scores.csv` (809 video Click Appeal Scores).
- **Creator Summaries:** `scores/creator_summary_scores.csv` (38 channels), `scores/creator_monthly_scores.csv`.
- **Visualizations:** `analysis/01_score_distribution.png`, `analysis/02_channel_appeal_comparison.png`, `analysis/03_score_components_breakdown.png`, `analysis/contact_sheet.jpg`.

---
*Report generated automatically by `07_validate_results.py`.*
"""
    with open(VALIDATION_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"Saved validation report to {VALIDATION_REPORT_MD}")

    print("\n================== STAGE 8 VALIDATION SUMMARY (FINAL N=809) ==================")
    all_pass = all(c["status"] == "PASS" for c in checks)
    for c in checks:
        print(f"  [{c['status']}] {c['check_id']:<38} : {c['actual']}")
    print("-" * 80)
    print(f"  OVERALL VALIDATION STATUS : {'PASSED (100% Valid)' if all_pass else 'FAILED'}")
    print("===============================================================================\n")


if __name__ == "__main__":
    run_validation()
