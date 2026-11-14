# Click Appeal Score — Final Validation Report (N = 809)

## 1. Overview
This report documents the final quality assurance and mathematical validation of the **Click Appeal Score Pipeline** executed on the unified final dataset of **809 unique YouTube videos** across 38 creator channels.

---

## 2. Validation Audit Checklist

| Check ID | Description | Expected | Actual | Status |
| :--- | :--- | :--- | :--- | :---: |
| `CHK_01_ROW_ACCOUNTABILITY` | Exactly 809 unique videos present across all pipeline stages | 809 rows in all files | Cleaned=809, Manifest=809, Title=809, Thumb=809, Norm=809, Scores=809 | **PASS** |
| `CHK_02_DUPLICATE_IDS` | Zero duplicate video IDs in cleaned and scored datasets | 0 duplicates | Cleaned dupes=0, Scores dupes=0 | **PASS** |
| `CHK_03_ID_TITLE_INTEGRITY` | No corrupted #NAME? IDs, formula strings, or missing titles | 0 artifacts, 0 empty titles | Corrupted IDs=0, Empty titles=0 | **PASS** |
| `CHK_04_THUMBNAIL_FILES_ON_DISK` | All 809 original and annotated thumbnails available on disk | 809 thumbnails, 809 annotated images, 0 missing | Missing original=0, Missing annotated=0 | **PASS** |
| `CHK_05_FEATURE_COMPLETENESS` | Zero missing or NaN values across all 8 normalized features | 0 missing values | Missing feature values=0 | **PASS** |
| `CHK_06_SCORE_BOUNDEDNESS` | All composite Click Appeal Scores strictly bounded within [0.0, 1.0] | 0 out-of-bound values | Out-of-bound values=0 | **PASS** |
| `CHK_07_MATHEMATICAL_RECONSTRUCTIBILITY` | Exact mathematical reconstructibility of X_title, X_thumb, and X_v = 0.4*X_title + 0.6*X_thumb | Max absolute difference < 1e-5 | Max diff: Title=0.00000050, Thumb=0.00000050, Weighted=0.00000050 | **PASS** |

---

## 3. Score Distribution Summary ($N = 809$)

| Metric | Score Formula / Definition | Mean | Std | Min | 25% | Median | 75% | Max | Skewness |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`score_weighted` ($X_v$)** | **$0.4 X_{\text{title}} + 0.6 X_{\text{thumb}}$ (Primary)** | **0.3924** | **0.1037** | **0.0054** | **0.3337** | **0.3914** | **0.4562** | **0.6843** | **-0.274** |
| `score_title` | $(x_{\text{caps}} + x_{\text{trig}} + x_{\text{sens}} + x_{\text{sym}}) / 4$ | 0.1659 | 0.1395 | 0.0000 | 0.0500 | 0.1453 | 0.2500 | 0.7326 | 0.872 |
| `score_thumbnail` | $(x_{\text{face}} + x_{\text{expr}} + x_{\text{cont}} + x_{\text{sat}}) / 4$ | 0.5434 | 0.1377 | 0.0091 | 0.4752 | 0.5563 | 0.6291 | 0.8898 | -0.744 |
| `score_equal_weight` | $\frac{1}{8} \sum_{j=1}^8 x_j$ | 0.3546 | 0.1023 | 0.0045 | 0.2878 | 0.3524 | 0.4133 | 0.6620 | 0.045 |
| `score_pca` | Projection on established PC1 | 0.5340 | 0.1681 | 0.0000 | 0.4341 | 0.5325 | 0.6431 | 1.0000 | -0.125 |

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
