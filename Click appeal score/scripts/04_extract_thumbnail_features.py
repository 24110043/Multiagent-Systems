#!/usr/bin/env python3
"""
04_extract_thumbnail_features.py
Stage 5: Thumbnail Feature Extraction & Visual Audit for Click Appeal Score (N=809).

Tasks:
A. Face Detection:
   - MediaPipe 0.10.21: mp.solutions.face_detection
   - Parameters: model_selection=1, min_detection_confidence=0.5
   - f_face = 1 if at least one face detected else 0
   - Record face count, bounding boxes, detection confidences, areas

B. Expression Detection:
   - FER 25.10.3 + TensorFlow 2.16.1 backend
   - MediaPipe face bounding box for largest face by Area = w * h
   - f_expr = max(P(surprise), P(fear))
   - If no face detected, assign 0.0
   - If face detected but classification fails, record missing data and processing error (not zero)

C. Contrast:
   - RGB to Luminance: Y = 0.299R + 0.587G + 0.114B in [0, 1]
   - f_contrast = sigma(Y) (population standard deviation)

D. Saturation:
   - HSV Saturation S in [0, 1]
   - f_saturation = sigma(S) (population standard deviation)

E. Visual Audit:
   - Annotated thumbnails saved in annotated_thumbnails/
   - 24-image representative contact sheet saved in annotated_thumbnails/ and analysis/
"""

import sys
import os
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Suppress excessive TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "04_extract_thumbnail_features.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("thumbnail_features")

CLEANED_CSV_PATH = BASE_DIR / "data" / "processed" / "cleaned_videos.csv"
THUMBNAILS_DIR = BASE_DIR / "thumbnails_original"
ANNOTATED_DIR = BASE_DIR / "annotated_thumbnails"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR = BASE_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FEATURES_CSV = FEATURES_DIR / "thumbnail_features.csv"
CONTACT_SHEET_ANNOTATED = ANNOTATED_DIR / "contact_sheet.jpg"
CONTACT_SHEET_ANALYSIS = ANALYSIS_DIR / "contact_sheet.jpg"


def annotate_image(img_bgr, faces_data, largest_face_idx, f_expr, p_fear, p_surprise, video_id, channel_name):
    """Draw visual audit annotations on thumbnail."""
    annotated = img_bgr.copy()
    H, W, _ = annotated.shape

    # Draw header bar
    cv2.rectangle(annotated, (0, 0), (W, 40), (20, 20, 20), -1)
    expr_str = f"{f_expr:.3f}" if not np.isnan(f_expr) else "ERR"
    header_text = f"{channel_name} | {video_id} | Detected Faces: {len(faces_data)} | f_expr: {expr_str} (Fear:{p_fear:.2f}, Surpr:{p_surprise:.2f})"
    cv2.putText(annotated, header_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    if not faces_data:
        # No face banner
        cv2.rectangle(annotated, (W // 4, H // 2 - 30), (3 * W // 4, H // 2 + 30), (0, 0, 180), -1)
        cv2.putText(annotated, "NO FACE DETECTED (f_face=0, f_expr=0.0)", (W // 4 + 20, H // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return annotated

    for idx, face in enumerate(faces_data):
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        conf = face["confidence"]
        is_largest = (idx == largest_face_idx)

        if is_largest:
            color = (0, 215, 255)  # Gold in BGR
            thickness = 4
            label = f"LARGEST FACE #{idx+1} [Conf: {conf:.2f}] Area: {w*h}px"
            expr_label = f"f_expr: {f_expr:.2f} (Fear: {p_fear:.2f}, Surpr: {p_surprise:.2f})"
        else:
            color = (0, 255, 0)
            thickness = 2
            label = f"Face #{idx+1} [Conf: {conf:.2f}]"
            expr_label = None

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

        label_y = max(20, y - 10)
        cv2.rectangle(annotated, (x, label_y - 18), (x + len(label) * 11, label_y + 4), color, -1)
        cv2.putText(annotated, label, (x + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        if expr_label:
            cv2.rectangle(annotated, (x, y + h + 4), (x + len(expr_label) * 11, y + h + 24), (0, 0, 0), -1)
            cv2.putText(annotated, expr_label, (x + 2, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

    return annotated


def create_contact_sheet(df_results, representative_ids, output_paths):
    """Generate a clean 24-image contact sheet of representative thumbnails."""
    logger.info(f"Generating contact sheet with {len(representative_ids)} representative thumbnails...")
    cols = 4
    rows = (len(representative_ids) + cols - 1) // cols
    tile_w = 400
    tile_h = 225

    sheet_w = cols * tile_w
    sheet_h = rows * tile_h + 80
    contact_sheet = Image.new("RGB", (sheet_w, sheet_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(contact_sheet)

    banner_text = "Click Appeal Score — Visual Audit Contact Sheet (Zero, Single & Multiple Faces)"
    draw.text((sheet_w // 2 - 350, 25), banner_text, fill=(255, 255, 255))

    for idx, vid in enumerate(representative_ids):
        ann_path = ANNOTATED_DIR / f"{vid}_annotated.jpg"
        if not ann_path.exists():
            ann_path = THUMBNAILS_DIR / f"{vid}.jpg"

        if ann_path.exists():
            try:
                with Image.open(ann_path) as tile_img:
                    tile_resized = tile_img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                    r = idx // cols
                    c = idx % cols
                    pos_x = c * tile_w
                    pos_y = 80 + r * tile_h
                    contact_sheet.paste(tile_resized, (pos_x, pos_y))
                    draw.rectangle([pos_x, pos_y, pos_x + tile_w - 1, pos_y + tile_h - 1], outline=(60, 60, 60), width=1)
            except Exception as e:
                logger.warning(f"Could not load image for contact sheet: {vid} ({e})")

    for p in output_paths:
        contact_sheet.save(p, "JPEG", quality=90)
        logger.info(f"Saved contact sheet to {p}")


def extract_thumbnail_features():
    logger.info("Starting Stage 5: Thumbnail Feature Extraction & Visual Audit (Final N=809)...")

    if not CLEANED_CSV_PATH.exists():
        logger.error(f"Cleaned CSV not found at {CLEANED_CSV_PATH}")
        sys.exit(1)

    df_cleaned = pd.read_csv(CLEANED_CSV_PATH)
    total_videos = len(df_cleaned)
    logger.info(f"Loaded {total_videos} videos for visual processing.")

    # Check resume cache
    existing_records = {}
    if OUTPUT_FEATURES_CSV.exists():
        try:
            df_exist = pd.read_csv(OUTPUT_FEATURES_CSV)
            for _, r in df_exist.iterrows():
                existing_records[str(r["video_id"]).strip()] = r.to_dict()
            logger.info(f"Found {len(existing_records)} existing records in {OUTPUT_FEATURES_CSV}.")
        except Exception as e:
            logger.warning(f"Could not read existing features CSV: {e}")

    # Determine which videos need processing
    needs_processing = []
    for _, row in df_cleaned.iterrows():
        vid = str(row["video_id"]).strip()
        ann_path = ANNOTATED_DIR / f"{vid}_annotated.jpg"
        if vid in existing_records and ann_path.exists():
            continue
        needs_processing.append(row)

    logger.info(f"Cached/Already Processed: {total_videos - len(needs_processing)} | Remaining to process: {len(needs_processing)}")

    # Initialize ML models only if needed
    face_detector = None
    fer_detector = None
    if len(needs_processing) > 0:
        import mediapipe as mp
        from fer.fer import FER

        logger.info("Initializing MediaPipe FaceDetection (model_selection=1, min_detection_confidence=0.5)...")
        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

        logger.info("Initializing FER Emotion Detector (TensorFlow backend)...")
        fer_detector = FER()

    results_map = dict(existing_records)

    for count, row in enumerate(needs_processing, 1):
        video_id = str(row["video_id"]).strip()
        channel_name = str(row["channel_name"]).strip()
        csv_face_count = int(row["face_count"]) if pd.notnull(row["face_count"]) else 0

        img_path = THUMBNAILS_DIR / f"{video_id}.jpg"
        if not img_path.exists():
            logger.error(f"Thumbnail image missing at {img_path}!")
            results_map[video_id] = {
                "video_id": video_id,
                "channel_name": channel_name,
                "thumbnail_path": None,
                "image_width": None,
                "image_height": None,
                "csv_face_count": csv_face_count,
                "detected_face_count": None,
                "face_count_disagreement": None,
                "f_face": None,
                "faces_detected_json": json.dumps([]),
                "largest_face_id": None,
                "largest_face_box": None,
                "largest_face_area": None,
                "p_angry": None,
                "p_disgust": None,
                "p_fear": None,
                "p_happy": None,
                "p_sad": None,
                "p_surprise": None,
                "p_neutral": None,
                "f_expr": np.nan,
                "expression_status": "MISSING_IMAGE",
                "f_contrast": np.nan,
                "f_saturation": np.nan,
                "annotated_image_path": None
            }
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.error(f"cv2.imread failed on {img_path}")
            continue

        H, W, _ = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Contrast
        R_norm = img_rgb[:, :, 0].astype(np.float64) / 255.0
        G_norm = img_rgb[:, :, 1].astype(np.float64) / 255.0
        B_norm = img_rgb[:, :, 2].astype(np.float64) / 255.0
        Y = 0.299 * R_norm + 0.587 * G_norm + 0.114 * B_norm
        f_contrast = float(np.std(Y, ddof=0))

        # Saturation
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        S_norm = img_hsv[:, :, 1].astype(np.float64) / 255.0
        f_saturation = float(np.std(S_norm, ddof=0))

        # Face Detection
        mp_results = face_detector.process(img_rgb)
        detections = mp_results.detections or []
        detected_face_count = len(detections)
        f_face = 1 if detected_face_count >= 1 else 0

        faces_data = []
        for f_idx, d in enumerate(detections):
            score = float(d.score[0])
            bb = d.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * W))
            y = max(0, int(bb.ymin * H))
            w = min(W - x, max(1, int(bb.width * W)))
            h = min(H - y, max(1, int(bb.height * H)))
            area = w * h
            faces_data.append({
                "face_id": f_idx + 1,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "confidence": round(score, 4),
                "area": area,
                "is_largest": False
            })

        # Expression Detection
        p_angry = 0.0
        p_disgust = 0.0
        p_fear = 0.0
        p_happy = 0.0
        p_sad = 0.0
        p_surprise = 0.0
        p_neutral = 0.0
        f_expr = 0.0
        largest_face_idx = None
        largest_face_id = None
        largest_face_box = None
        largest_face_area = None
        expr_status = "NO_FACE"

        if detected_face_count == 0:
            f_expr = 0.0
            expr_status = "NO_FACE"
        else:
            largest_face_idx = max(range(len(faces_data)), key=lambda i: faces_data[i]["area"])
            faces_data[largest_face_idx]["is_largest"] = True
            largest_face = faces_data[largest_face_idx]
            largest_face_id = largest_face["face_id"]
            largest_face_box = f"({largest_face['x']},{largest_face['y']},{largest_face['w']},{largest_face['h']})"
            largest_face_area = largest_face["area"]

            rect = (largest_face["x"], largest_face["y"], largest_face["w"], largest_face["h"])
            try:
                fer_output = fer_detector.detect_emotions(img_bgr, face_rectangles=[rect])
                if fer_output and len(fer_output) > 0 and "emotions" in fer_output[0]:
                    emotions = fer_output[0]["emotions"]
                    p_angry = float(emotions.get("angry", 0.0))
                    p_disgust = float(emotions.get("disgust", 0.0))
                    p_fear = float(emotions.get("fear", 0.0))
                    p_happy = float(emotions.get("happy", 0.0))
                    p_sad = float(emotions.get("sad", 0.0))
                    p_surprise = float(emotions.get("surprise", 0.0))
                    p_neutral = float(emotions.get("neutral", 0.0))
                    f_expr = float(max(p_surprise, p_fear))
                    expr_status = "SUCCESS"
                else:
                    f_expr = np.nan
                    expr_status = "ERROR_CLASSIFICATION_FAILED"
            except Exception as e:
                f_expr = np.nan
                expr_status = f"ERROR_{str(e)[:50]}"

        # Visual audit annotation
        annotated_img = annotate_image(
            img_bgr, faces_data, largest_face_idx, f_expr, p_fear, p_surprise, video_id, channel_name
        )
        ann_save_path = ANNOTATED_DIR / f"{video_id}_annotated.jpg"
        cv2.imwrite(str(ann_save_path), annotated_img)

        # Record result
        results_map[video_id] = {
            "video_id": video_id,
            "channel_name": channel_name,
            "thumbnail_path": f"thumbnails_original/{video_id}.jpg",
            "image_width": W,
            "image_height": H,
            "csv_face_count": csv_face_count,
            "detected_face_count": detected_face_count,
            "face_count_disagreement": detected_face_count - csv_face_count,
            "f_face": f_face,
            "faces_detected_json": json.dumps(faces_data),
            "largest_face_id": largest_face_id,
            "largest_face_box": largest_face_box,
            "largest_face_area": largest_face_area,
            "p_angry": round(p_angry, 4),
            "p_disgust": round(p_disgust, 4),
            "p_fear": round(p_fear, 4),
            "p_happy": round(p_happy, 4),
            "p_sad": round(p_sad, 4),
            "p_surprise": round(p_surprise, 4),
            "p_neutral": round(p_neutral, 4),
            "f_expr": round(f_expr, 6) if not np.isnan(f_expr) else np.nan,
            "expression_status": expr_status,
            "f_contrast": round(f_contrast, 6),
            "f_saturation": round(f_saturation, 6),
            "annotated_image_path": f"annotated_thumbnails/{video_id}_annotated.jpg",
        }

        if count % 25 == 0 or count == len(needs_processing):
            logger.info(f"Progress: [{count}/{len(needs_processing)}] Processed {video_id} | Faces: {detected_face_count} | Status: {expr_status}")

    if face_detector is not None:
        face_detector.close()

    # Reorder according to df_cleaned
    ordered_results = []
    for vid in df_cleaned["video_id"]:
        vid_str = str(vid).strip()
        rec = results_map[vid_str]
        # Ensure path points to thumbnails_original and annotated_thumbnails
        rec["thumbnail_path"] = f"thumbnails_original/{vid_str}.jpg"
        rec["annotated_image_path"] = f"annotated_thumbnails/{vid_str}_annotated.jpg"
        ordered_results.append(rec)

    df_features = pd.DataFrame(ordered_results)
    df_features.to_csv(OUTPUT_FEATURES_CSV, index=False, encoding="utf-8")
    logger.info(f"Saved complete thumbnail features to {OUTPUT_FEATURES_CSV} ({len(df_features)} rows)")

    # Generate Contact Sheet (24 Representative Examples: 8 zero, 8 single, 8 multi)
    zero_face_vids = df_features[df_features["detected_face_count"] == 0]["video_id"].tolist()
    one_face_vids = df_features[df_features["detected_face_count"] == 1]["video_id"].tolist()
    multi_face_vids = df_features[df_features["detected_face_count"] >= 2]["video_id"].tolist()

    selected_zero = zero_face_vids[:8]
    selected_one = one_face_vids[:8]
    selected_multi = multi_face_vids[:8]
    representative_vids = selected_zero + selected_one + selected_multi

    create_contact_sheet(df_features, representative_vids, [CONTACT_SHEET_ANNOTATED, CONTACT_SHEET_ANALYSIS])

    # Summary
    valid_expr = df_features["f_expr"].dropna()
    print("\n================== STAGE 5 THUMBNAIL FEATURES SUMMARY ==================")
    print(f"  Total thumbnails analyzed    : {len(df_features)}")
    print(f"  Faces detected               : Total {df_features['detected_face_count'].sum()} faces across {len(df_features)} images")
    print(f"    - Zero-face thumbnails     : {(df_features['detected_face_count'] == 0).sum()} ({(df_features['detected_face_count'] == 0).mean()*100:.1f}%)")
    print(f"    - Single-face thumbnails   : {(df_features['detected_face_count'] == 1).sum()} ({(df_features['detected_face_count'] == 1).mean()*100:.1f}%)")
    print(f"    - Multi-face thumbnails    : {(df_features['detected_face_count'] >= 2).sum()} ({(df_features['detected_face_count'] >= 2).mean()*100:.1f}%)")
    print(f"  Expression detection status:")
    for status, cnt in df_features["expression_status"].value_counts().items():
        print(f"    - {status:<28} : {cnt}")
    print(f"  f_expr (valid)               -> Min: {valid_expr.min():.4f}, Mean: {valid_expr.mean():.4f}, Max: {valid_expr.max():.4f}")
    print(f"  f_contrast                   -> Min: {df_features['f_contrast'].min():.4f}, Mean: {df_features['f_contrast'].mean():.4f}, Max: {df_features['f_contrast'].max():.4f}")
    print(f"  f_saturation                 -> Min: {df_features['f_saturation'].min():.4f}, Mean: {df_features['f_saturation'].mean():.4f}, Max: {df_features['f_saturation'].max():.4f}")
    print("========================================================================\n")


if __name__ == "__main__":
    extract_thumbnail_features()
