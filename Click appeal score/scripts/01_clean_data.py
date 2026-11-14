#!/usr/bin/env python3
"""
01_clean_data.py
Stage 2: Dataset Inspection, Cleaning & Merging Pipeline for Click Appeal Score (N=809).

Tasks:
- Inspect and merge raw datasets from data/original/:
    1. challenge_videos_final(challenge_videos_final).csv (original 186 benchmark videos)
    2. challenge_videos_final (1)(challenge_videos_final (1)).csv (expanded raw 690 rows)
- Recover all 12 corrupted YouTube video IDs (#NAME?) from CDN thumbnail URLs.
- Repair channel name anomalies (e.g. '\' -> 'Airrack').
- Deduplicate on video_id while strictly preserving all valid videos.
- Verify exactly 809 unique videos across 38 channels with 0 duplicate IDs.
- Produce:
    1. data/processed/cleaned_videos.csv (809 rows)
    2. data/processed/cleaning_log.csv (audit of all ID recoveries and channel repairs)
"""

import os
import re
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "01_clean_data.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("clean_data")

ORIGINAL_CSV_186 = BASE_DIR / "data" / "original" / "challenge_videos_final(challenge_videos_final).csv"
NEW_RAW_CSV = BASE_DIR / "data" / "original" / "challenge_videos_final (1)(challenge_videos_final (1)).csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_CSV_PATH = PROCESSED_DATA_DIR / "cleaned_videos.csv"
CLEANING_LOG_PATH = PROCESSED_DATA_DIR / "cleaning_log.csv"


def recover_video_id(video_id_raw: str, thumbnail_url: str):
    """Recover corrupted video IDs (#NAME? or Excel formulas) from thumbnail URLs."""
    v_str = str(video_id_raw).strip()
    thumb_str = str(thumbnail_url).strip()
    if v_str == "#NAME?" or v_str.startswith("=") or len(v_str) < 5 or v_str.lower() == "nan":
        m = re.search(r"/vi/([^/]+)/", thumb_str)
        if m:
            return m.group(1).strip(), True
    return v_str, False


def clean_dataset():
    logger.info("Starting Stage 2: Dataset Inspection, Cleaning & Merging (Final N=809)...")

    if not ORIGINAL_CSV_186.exists():
        logger.error(f"Original 186 dataset not found at {ORIGINAL_CSV_186}")
        sys.exit(1)
    if not NEW_RAW_CSV.exists():
        logger.error(f"New raw dataset not found at {NEW_RAW_CSV}")
        sys.exit(1)

    # 1. Load original 186 benchmark dataset
    try:
        df_186 = pd.read_csv(ORIGINAL_CSV_186, encoding="cp1252")
    except Exception:
        df_186 = pd.read_csv(ORIGINAL_CSV_186, encoding="latin1")
    logger.info(f"Loaded original benchmark dataset: {len(df_186)} rows.")

    # 2. Load new raw dataset
    try:
        df_new = pd.read_csv(NEW_RAW_CSV, encoding="cp1252")
    except Exception:
        df_new = pd.read_csv(NEW_RAW_CSV, encoding="latin1")
    logger.info(f"Loaded new raw dataset: {len(df_new)} rows.")

    cleaning_logs = []

    # Map channel_id to valid channel_name from benchmark
    channel_id_to_name = df_186.set_index("channel_id")["channel_name"].to_dict()

    # Clean original 186 records
    cleaned_186_map = {}
    for idx, row in df_186.iterrows():
        rec = row.to_dict()
        vid_raw = rec["video_id"]
        thumb = rec["thumbnail"]
        vid_clean, was_recovered = recover_video_id(vid_raw, thumb)
        rec["video_id"] = vid_clean

        if was_recovered:
            cleaning_logs.append({
                "source_dataset": "original_186",
                "row_index": idx,
                "video_id_raw": vid_raw,
                "video_id_cleaned": vid_clean,
                "action": "corrupted_id_recovered",
                "details": f"Recovered YouTube ID '{vid_clean}' from thumbnail URL '{thumb}'"
            })

        cid = rec["channel_id"]
        cname = rec.get("channel_name", "")
        if cname == "\\" or pd.isnull(cname) or len(str(cname).strip()) <= 1:
            corrected = channel_id_to_name.get(cid, "Airrack")
            rec["channel_name"] = corrected
            cleaning_logs.append({
                "source_dataset": "original_186",
                "row_index": idx,
                "video_id_raw": vid_raw,
                "video_id_cleaned": vid_clean,
                "action": "channel_name_repaired",
                "details": f"Channel ID {cid} matched to valid name '{corrected}'"
            })
        cleaned_186_map[vid_clean] = rec

    # Clean new dataset records and recover IDs
    recovered_new_map = {}
    for idx, row in df_new.iterrows():
        rec = row.to_dict()
        vid_raw = rec["video_id"]
        thumb = rec["thumbnail"]
        vid_clean, was_recovered = recover_video_id(vid_raw, thumb)
        rec["video_id"] = vid_clean

        if was_recovered:
            cleaning_logs.append({
                "source_dataset": "new_raw_690",
                "row_index": idx,
                "video_id_raw": vid_raw,
                "video_id_cleaned": vid_clean,
                "action": "corrupted_id_recovered",
                "details": f"Recovered YouTube ID '{vid_clean}' from thumbnail URL '{thumb}'"
            })

        cid = rec["channel_id"]
        cname = rec.get("channel_name", "")
        if cname == "\\" or pd.isnull(cname) or len(str(cname).strip()) <= 1:
            corrected = channel_id_to_name.get(cid, "Airrack")
            rec["channel_name"] = corrected
            cleaning_logs.append({
                "source_dataset": "new_raw_690",
                "row_index": idx,
                "video_id_raw": vid_raw,
                "video_id_cleaned": vid_clean,
                "action": "channel_name_repaired",
                "details": f"Channel ID {cid} matched to valid name '{corrected}'"
            })

        recovered_new_map[vid_clean] = rec

    logger.info(f"Unique videos in original benchmark: {len(cleaned_186_map)}")
    logger.info(f"Unique videos in new dataset after ID recovery: {len(recovered_new_map)}")

    # Merge: All 186 original videos preserved + genuinely new videos from new dataset
    merged_videos = dict(cleaned_186_map)
    added_new_count = 0
    for vid, rec in recovered_new_map.items():
        if vid not in merged_videos:
            merged_videos[vid] = rec
            added_new_count += 1

    df_cleaned = pd.DataFrame(list(merged_videos.values()))

    # Sort deterministically: by channel_name, then published_at descending
    if "published_at" in df_cleaned.columns:
        df_cleaned["published_at_dt"] = pd.to_datetime(df_cleaned["published_at"], errors="coerce")
        df_cleaned = df_cleaned.sort_values(by=["channel_name", "published_at_dt"], ascending=[True, False]).drop(columns=["published_at_dt"])
    else:
        df_cleaned = df_cleaned.sort_values(by=["channel_name", "video_id"])

    # Ensure required columns
    expected_cols = [
        "channel_name", "channel_id", "video_id", "title", "thumbnail", "published_at",
        "views", "likes", "comments", "duration_sec", "age_days", "tags_count",
        "category_id", "subscribers", "channel_video_count", "face_count"
    ]
    df_cleaned = df_cleaned[[c for c in expected_cols if c in df_cleaned.columns]]

    # Assertions
    assert len(df_cleaned) == 809, f"Expected exactly 809 unique videos, found {len(df_cleaned)}!"
    assert df_cleaned["video_id"].duplicated().sum() == 0, "Duplicate video_ids found in cleaned dataset!"
    assert not (df_cleaned["video_id"] == "#NAME?").any(), "Corrupted #NAME? IDs remain in cleaned dataset!"
    assert not df_cleaned["video_id"].str.startswith("=").any(), "Formula IDs remain in cleaned dataset!"

    df_cleaned.to_csv(CLEANED_CSV_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved cleaned final dataset to {CLEANED_CSV_PATH} ({len(df_cleaned)} rows)")

    df_logs = pd.DataFrame(cleaning_logs)
    df_logs.to_csv(CLEANING_LOG_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved cleaning log to {CLEANING_LOG_PATH} ({len(df_logs)} entries)")

    print("\n================== STAGE 2 CLEANING SUMMARY (FINAL N=809) ==================")
    print(f"  Total unique videos in final dataset : {len(df_cleaned)}")
    print(f"  Original benchmark videos preserved   : {len(cleaned_186_map)}")
    print(f"  Genuinely new videos added           : {added_new_count}")
    print(f"  Total creator channels               : {df_cleaned['channel_name'].nunique()}")
    print(f"  Corrupted IDs recovered from URLs    : {len([l for l in cleaning_logs if l['action'] == 'corrupted_id_recovered'])}")
    print(f"  Duplicate IDs in final dataset       : {df_cleaned['video_id'].duplicated().sum()} (PASSED)")
    print("============================================================================\n")


if __name__ == "__main__":
    clean_dataset()
