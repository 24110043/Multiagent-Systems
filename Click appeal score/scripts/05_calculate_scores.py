#!/usr/bin/env python3
"""
05_calculate_scores.py
Stage 6: Feature Normalization & Click Appeal Score Calculation (Final N=809).

Tasks:
1. Feature Normalization:
   - Uses the established scaling parameters (1st and 99th percentiles) from scores/scaling_parameters.csv.
   - Normalizes continuous features: caps, trigger, sensitive, symbol, contrast, saturation.
   - Preserves face presence (x_face) and expression (x_expr) directly in [0, 1].
   - Saves features/normalized_features.csv.

2. Click Appeal Score Calculation:
   - Title score: X_title = (x_caps + x_trigger + x_sensitive + x_symbol) / 4
   - Thumbnail score: X_thumb = (x_face + x_expr + x_contrast + x_saturation) / 4
   - Primary Weighted score: X_v = 0.4 * X_title + 0.6 * X_thumb
   - Equal-weight alternative: X_equal = (1/8) * sum(x_j)
   - PCA alternative: First principal component projection using established loadings.
   - Sensitivity alternatives excluding expression:
     X_thumb_no_expr = (x_face + x_contrast + x_saturation) / 3
     X_weighted_no_expr = 0.4 * X_title + 0.6 * X_thumb_no_expr
     X_equal_no_expr = (1/7) * sum(7 features)
   - Saves scores/video_scores.csv and scores/score_methodology.csv.
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Configure logging
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "05_calculate_scores.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("calculate_scores")

CLEANED_CSV_PATH = BASE_DIR / "data" / "processed" / "cleaned_videos.csv"
TITLE_FEATURES_PATH = BASE_DIR / "features" / "title_features.csv"
THUMB_FEATURES_PATH = BASE_DIR / "features" / "thumbnail_features.csv"
FEATURES_DIR = BASE_DIR / "features"
SCORES_DIR = BASE_DIR / "scores"
SCORES_DIR.mkdir(parents=True, exist_ok=True)

NORMALIZED_FEATURES_PATH = FEATURES_DIR / "normalized_features.csv"
SCALING_PARAMETERS_PATH = SCORES_DIR / "scaling_parameters.csv"
VIDEO_SCORES_PATH = SCORES_DIR / "video_scores.csv"
SCORE_METHODOLOGY_PATH = SCORES_DIR / "score_methodology.csv"
PCA_LOADINGS_PATH = SCORES_DIR / "pca_loadings.csv"


def calculate_scores():
    logger.info("Starting Stage 6: Feature Normalization & Click Appeal Score Calculation (Final N=809)...")

    if not TITLE_FEATURES_PATH.exists() or not THUMB_FEATURES_PATH.exists():
        logger.error("Feature CSVs not found! Run Stage 4 and Stage 5 first.")
        sys.exit(1)

    df_cleaned = pd.read_csv(CLEANED_CSV_PATH)
    df_title = pd.read_csv(TITLE_FEATURES_PATH)
    df_thumb = pd.read_csv(THUMB_FEATURES_PATH)

    logger.info(f"Loaded {len(df_cleaned)} cleaned videos, {len(df_title)} title feature rows, {len(df_thumb)} thumb feature rows.")

    # Merge on video_id
    df_merged = df_cleaned.merge(
        df_title[["video_id", "f_caps", "f_trigger", "f_sensitive", "f_symbol"]],
        on="video_id",
        how="left"
    )
    df_merged = df_merged.merge(
        df_thumb[["video_id", "detected_face_count", "f_face", "f_expr", "f_contrast", "f_saturation", "expression_status"]],
        on="video_id",
        how="left"
    )

    logger.info(f"Merged dataset shape: {df_merged.shape}")

    # Verify no missing data in features
    feature_raw_cols = ["f_caps", "f_trigger", "f_sensitive", "f_symbol", "f_face", "f_expr", "f_contrast", "f_saturation"]
    missing = df_merged[feature_raw_cols].isnull().sum()
    if missing.sum() > 0:
        logger.error(f"Missing values found in raw features:\n{missing}")
        raise ValueError("Missing feature values detected!")

    # 1. Feature Normalization using Established Scaling Parameters
    logger.info(f"Loading scaling parameters from {SCALING_PARAMETERS_PATH}...")
    df_scaling = pd.read_csv(SCALING_PARAMETERS_PATH)
    scaling_dict = {r["feature"]: r for _, r in df_scaling.iterrows()}

    df_normalized = pd.DataFrame()
    df_normalized["video_id"] = df_merged["video_id"]
    df_normalized["channel_name"] = df_merged["channel_name"]
    df_normalized["title"] = df_merged["title"]

    continuous_features = ["f_caps", "f_trigger", "f_sensitive", "f_symbol", "f_contrast", "f_saturation"]
    for feat in continuous_features:
        params = scaling_dict[feat]
        q01 = float(params["q_0.01"])
        q99 = float(params["q_0.99"])
        denom = float(params["denominator"])

        scaled = (np.clip(df_merged[feat], q01, q99) - q01) / denom
        norm_col = feat.replace("f_", "x_")
        df_normalized[norm_col] = np.clip(scaled, 0.0, 1.0).round(6)

    # Face and expression identity preservation
    df_normalized["x_face"] = df_merged["f_face"].astype(float).round(6)
    df_normalized["x_expr"] = df_merged["f_expr"].astype(float).round(6)

    feature_cols = ["x_caps", "x_trigger", "x_sensitive", "x_symbol", "x_face", "x_expr", "x_contrast", "x_saturation"]
    df_normalized_final = df_normalized[["video_id", "channel_name", "title"] + feature_cols]
    df_normalized_final.to_csv(NORMALIZED_FEATURES_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved normalized features to {NORMALIZED_FEATURES_PATH} ({len(df_normalized_final)} rows)")

    # 2. Composite Click Appeal Scores
    # Title score: X_title = (x_caps + x_trigger + x_sensitive + x_symbol) / 4
    x_title = (df_normalized["x_caps"] + df_normalized["x_trigger"] + df_normalized["x_sensitive"] + df_normalized["x_symbol"]) / 4.0

    # Thumbnail score: X_thumb = (x_face + x_expr + x_contrast + x_saturation) / 4
    x_thumb = (df_normalized["x_face"] + df_normalized["x_expr"] + df_normalized["x_contrast"] + df_normalized["x_saturation"]) / 4.0

    # Primary Weighted score: X_v = 0.4 * X_title + 0.6 * X_thumb
    x_weighted = 0.4 * x_title + 0.6 * x_thumb

    # Equal-weight alternative: X_equal = (1/8) * sum(x_j)
    x_equal = df_normalized[feature_cols].mean(axis=1)

    # Sensitivity scores excluding expression
    x_thumb_no_expr = df_normalized[["x_face", "x_contrast", "x_saturation"]].mean(axis=1)
    x_weighted_no_expr = 0.4 * x_title + 0.6 * x_thumb_no_expr
    x_equal_no_expr = df_normalized[[
        "x_caps", "x_trigger", "x_sensitive", "x_symbol",
        "x_face", "x_contrast", "x_saturation"
    ]].mean(axis=1)

    # 3. PCA Alternative Score (Pure NumPy Projection using established loadings)
    logger.info(f"Loading PCA loadings from {PCA_LOADINGS_PATH}...")
    df_pca = pd.read_csv(PCA_LOADINGS_PATH)
    w = df_pca["pca_loading"].values

    # Established baseline standardization parameters
    mu_orig = np.array([
        0.1994494779698769, 0.2959623274301839, 0.0918740411059861, 0.11410676416140551,
        0.9354838709677419, 0.20510752688172043, 0.46717893689885925, 0.5635236978289196
    ])
    sigma_orig = np.array([
        0.30019731553688395, 0.2904645402293648, 0.23940412104696973, 0.19244605705970305,
        0.2456701001891583, 0.25828266429346136, 0.19756265208354157, 0.2282288082853319
    ])
    pc1_min_orig = -3.667747
    pc1_max_orig = 3.438617

    X_std = (df_normalized[feature_cols].values - mu_orig) / sigma_orig
    pc1_raw = np.dot(X_std, w)
    x_pca = np.clip((pc1_raw - pc1_min_orig) / (pc1_max_orig - pc1_min_orig + 1e-6), 0.0, 1.0)

    # 4. Assemble Full Video Scores Dataset
    df_video_scores = df_merged[[
        "video_id", "channel_name", "channel_id", "title", "published_at",
        "views", "likes", "comments", "duration_sec", "age_days", "tags_count",
        "category_id", "subscribers", "channel_video_count", "face_count", "detected_face_count"
    ]].copy()

    # Raw features
    for col in feature_raw_cols:
        df_video_scores[col] = df_merged[col]

    # Normalized features
    for col in feature_cols:
        df_video_scores[col] = df_normalized[col]

    # Composite scores
    df_video_scores["score_title"] = x_title.round(6)
    df_video_scores["score_thumbnail"] = x_thumb.round(6)
    df_video_scores["score_weighted"] = x_weighted.round(6)
    df_video_scores["score_equal_weight"] = x_equal.round(6)
    df_video_scores["score_pca"] = x_pca.round(6)
    df_video_scores["score_weighted_no_expr"] = x_weighted_no_expr.round(6)
    df_video_scores["score_equal_no_expr"] = x_equal_no_expr.round(6)

    df_video_scores.to_csv(VIDEO_SCORES_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved complete video scores to {VIDEO_SCORES_PATH} ({len(df_video_scores)} rows)")

    # Save methodology documentation
    methodology_records = [
        {"score_name": "score_title", "description": "Average of 4 normalized title features", "formula": "(x_caps + x_trigger + x_sensitive + x_symbol) / 4", "range": "[0, 1]", "role": "Sub-score"},
        {"score_name": "score_thumbnail", "description": "Average of 4 normalized thumbnail features", "formula": "(x_face + x_expr + x_contrast + x_saturation) / 4", "range": "[0, 1]", "role": "Sub-score"},
        {"score_name": "score_weighted", "description": "Primary Click Appeal Score (baseline)", "formula": "0.4 * score_title + 0.6 * score_thumbnail", "range": "[0, 1]", "role": "Primary metric (X_v)"},
        {"score_name": "score_equal_weight", "description": "Equal-weight average across all 8 features", "formula": "(1/8) * sum(x_j)", "range": "[0, 1]", "role": "Robustness check"},
        {"score_name": "score_pca", "description": "First principal component projection onto established PC1 loadings", "formula": "min_max_scale((X_std . w))", "range": "[0, 1]", "role": "Dimensionality reduction"},
        {"score_name": "score_weighted_no_expr", "description": "Weighted score excluding expression from thumbnail", "formula": "0.4 * score_title + 0.6 * mean(x_face, x_contrast, x_saturation)", "range": "[0, 1]", "role": "Sensitivity check"},
        {"score_name": "score_equal_no_expr", "description": "Equal-weight score excluding expression", "formula": "mean(7 features excluding x_expr)", "range": "[0, 1]", "role": "Sensitivity check"}
    ]
    pd.DataFrame(methodology_records).to_csv(SCORE_METHODOLOGY_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved methodology description to {SCORE_METHODOLOGY_PATH}")

    # Summary
    print("\n================== STAGE 6 CLICK APPEAL SCORES SUMMARY (FINAL N=809) ==================")
    print(f"  Total videos scored : {len(df_video_scores)}")
    print(f"  score_title         -> Min: {df_video_scores['score_title'].min():.4f}, Mean: {df_video_scores['score_title'].mean():.4f}, Max: {df_video_scores['score_title'].max():.4f}, Std: {df_video_scores['score_title'].std():.4f}")
    print(f"  score_thumbnail     -> Min: {df_video_scores['score_thumbnail'].min():.4f}, Mean: {df_video_scores['score_thumbnail'].mean():.4f}, Max: {df_video_scores['score_thumbnail'].max():.4f}, Std: {df_video_scores['score_thumbnail'].std():.4f}")
    print(f"  score_weighted (X_v)-> Min: {df_video_scores['score_weighted'].min():.4f}, Mean: {df_video_scores['score_weighted'].mean():.4f}, Max: {df_video_scores['score_weighted'].max():.4f}, Std: {df_video_scores['score_weighted'].std():.4f}")
    print(f"  score_equal_weight  -> Min: {df_video_scores['score_equal_weight'].min():.4f}, Mean: {df_video_scores['score_equal_weight'].mean():.4f}, Max: {df_video_scores['score_equal_weight'].max():.4f}, Std: {df_video_scores['score_equal_weight'].std():.4f}")
    print(f"  score_pca           -> Min: {df_video_scores['score_pca'].min():.4f}, Mean: {df_video_scores['score_pca'].mean():.4f}, Max: {df_video_scores['score_pca'].max():.4f}, Std: {df_video_scores['score_pca'].std():.4f}")
    print("=======================================================================================\n")


if __name__ == "__main__":
    calculate_scores()
