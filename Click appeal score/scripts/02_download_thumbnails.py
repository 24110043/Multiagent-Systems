#!/usr/bin/env python3
"""
02_download_thumbnails.py
Stage 3: Thumbnail Retrieval & Verification Pipeline for Click Appeal Score (N=809).

Tasks:
- Inspect all 809 videos in data/processed/cleaned_videos.csv.
- Verify existing local thumbnails in thumbnails_original/ or download missing ones from YouTube CDN.
- Record HTTP status, dimensions (width, height), image format, and byte size.
- Save download manifest to:
    1. thumbnails_original/download_manifest.csv
    2. data/processed/download_manifest.csv
"""

import sys
import os
import time
import logging
from pathlib import Path
from io import BytesIO
import requests
import pandas as pd
from PIL import Image

# Configure logging
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "02_download_thumbnails.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("download_thumbnails")

CLEANED_CSV_PATH = BASE_DIR / "data" / "processed" / "cleaned_videos.csv"
THUMBNAILS_DIR = BASE_DIR / "thumbnails_original"
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = THUMBNAILS_DIR / "download_manifest.csv"
PROCESSED_MANIFEST_PATH = BASE_DIR / "data" / "processed" / "download_manifest.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def download_thumbnails():
    logger.info("Starting Stage 3: Thumbnail Retrieval & Verification (Final N=809)...")
    if not CLEANED_CSV_PATH.exists():
        logger.error(f"Cleaned CSV not found at {CLEANED_CSV_PATH}. Run scripts/01_clean_data.py first.")
        sys.exit(1)

    df = pd.read_csv(CLEANED_CSV_PATH)
    total_videos = len(df)
    logger.info(f"Loaded {total_videos} videos from {CLEANED_CSV_PATH}")

    manifest_records = []
    success_count = 0
    failure_count = 0
    cached_count = 0

    session = requests.Session()
    session.headers.update(HEADERS)

    for idx, row in df.iterrows():
        video_id = str(row["video_id"]).strip()
        url = str(row["thumbnail"]).strip()
        channel_name = str(row["channel_name"]).strip()

        target_path = THUMBNAILS_DIR / f"{video_id}.jpg"

        # Check existing file
        if target_path.exists() and target_path.stat().st_size > 0:
            try:
                with Image.open(target_path) as img:
                    width, height = img.size
                    img_format = img.format
                manifest_records.append({
                    "video_id": video_id,
                    "channel_name": channel_name,
                    "thumbnail_url": url,
                    "status": "SUCCESS",
                    "http_status": 200,
                    "width": width,
                    "height": height,
                    "format": img_format,
                    "file_size_bytes": target_path.stat().st_size,
                    "file_path": str(target_path.relative_to(BASE_DIR)).replace("\\", "/"),
                    "failure_reason": "None (cached/existing)"
                })
                success_count += 1
                cached_count += 1
                continue
            except Exception as e:
                logger.warning(f"Existing file {target_path} corrupted: {e}. Re-downloading.")

        # Download from CDN
        download_success = False
        attempts = [url]
        if "maxresdefault.jpg" in url:
            attempts.append(url.replace("maxresdefault.jpg", "hqdefault.jpg"))

        for req_url in attempts:
            try:
                resp = session.get(req_url, timeout=12)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    with Image.open(BytesIO(resp.content)) as img:
                        img.verify()
                    with open(target_path, "wb") as f:
                        f.write(resp.content)
                    with Image.open(target_path) as img:
                        width, height = img.size
                        img_format = img.format
                    manifest_records.append({
                        "video_id": video_id,
                        "channel_name": channel_name,
                        "thumbnail_url": req_url,
                        "status": "SUCCESS",
                        "http_status": 200,
                        "width": width,
                        "height": height,
                        "format": img_format,
                        "file_size_bytes": len(resp.content),
                        "file_path": str(target_path.relative_to(BASE_DIR)).replace("\\", "/"),
                        "failure_reason": "None"
                    })
                    success_count += 1
                    download_success = True
                    break
            except Exception as e:
                continue

        if not download_success:
            failure_count += 1
            manifest_records.append({
                "video_id": video_id,
                "channel_name": channel_name,
                "thumbnail_url": url,
                "status": "FAILURE",
                "http_status": 404,
                "width": 0,
                "height": 0,
                "format": "UNKNOWN",
                "file_size_bytes": 0,
                "file_path": "",
                "failure_reason": "Download failed across all CDN variants"
            })

        if (idx + 1) % 100 == 0 or idx + 1 == total_videos:
            logger.info(f"Progress: [{idx+1}/{total_videos}] Cached={cached_count}, Downloaded={success_count - cached_count}, Failed={failure_count}")

    df_manifest = pd.DataFrame(manifest_records)
    df_manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8")
    df_manifest.to_csv(PROCESSED_MANIFEST_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved manifest to {MANIFEST_PATH} and {PROCESSED_MANIFEST_PATH}")

    print("\n================== STAGE 3 THUMBNAIL DOWNLOAD SUMMARY ==================")
    print(f"  Total thumbnails verified    : {len(df_manifest)}")
    print(f"  Cached / already existing   : {cached_count}")
    print(f"  Newly downloaded             : {success_count - cached_count}")
    print(f"  Failed downloads             : {failure_count}")
    print("========================================================================\n")


if __name__ == "__main__":
    download_thumbnails()
