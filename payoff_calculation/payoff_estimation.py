#!/usr/bin/env python3
"""
payoff_estimation.py
Master Pipeline Execution for Empirical Payoff Function Estimation.

CS 329: Foundations of AI: Multi-Agent Systems (Autumn 2026)
Project: Click Appeal as a Potential Game: Modeling Creator Incentives,
         Attention Congestion, and Social Welfare in YouTube
Author: Bhavya Shah (24110324) - Payoff Function Estimation & Statistical Evaluation

Executes sequential empirical stages (01 to 10):
1. scripts/01_prepare_payoff_data.py
2. scripts/02_create_stratified_split.py
3. scripts/03_train_models.py
4. scripts/04_evaluate_loco_cv.py
5. scripts/05_validate_and_freeze.py
6. scripts/06_evaluate_test_holdout.py
7. scripts/07_post_test_structural_calibration.py
8. scripts/08_generate_plots.py
9. scripts/09_backward_looking_robustness.py
10. scripts/10_congestion_test.py
"""

import sys
import os
import time
import subprocess
import logging
from pathlib import Path

PAYOFF_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PAYOFF_DIR / "scripts"
LOGS_DIR = PAYOFF_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure master pipeline logger (overwrite mode for clean run)
log_file = LOGS_DIR / "master_pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("master_pipeline")

STAGES = [
    ("01_prepare_payoff_data.py", "Stage 1: Data Preparation & Integrity Audit"),
    ("02_create_stratified_split.py", "Stage 2: Channel-Stratified Splitting (26/6/6)"),
    ("03_train_models.py", "Stage 3: Primary Model Development (Training Sample)"),
    ("04_evaluate_loco_cv.py", "Stage 4: 26-Fold LOCO-CV on Training Cohort"),
    ("05_validate_and_freeze.py", "Stage 5: Validation Evaluation & Model Locking"),
    ("06_evaluate_test_holdout.py", "Stage 6: Final Holdout Test Evaluation"),
    ("07_post_test_structural_calibration.py", "Stage 7: Full-Sample Calibration & Bootstrap"),
    ("08_generate_plots.py", "Stage 8: Publication-Quality Econometric Figures"),
    ("09_backward_looking_robustness.py", "Stage 9: Backward-Looking Peer Window Robustness"),
    ("10_congestion_test.py", "Stage 10: State-Dependent Peer Congestion Test E(x)")
]


def run_stage(script_name, description):
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info(f"STARTING STAGE: {description} ({script_name})")
    logger.info("=" * 70)

    start_time = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PAYOFF_DIR),
        capture_output=False
    )
    elapsed = time.time() - start_time

    if result.returncode != 0:
        logger.error(f"STAGE FAILED with exit code {result.returncode}: {script_name}")
        sys.exit(result.returncode)
    else:
        logger.info(f"STAGE COMPLETED: {description} in {elapsed:.2f} seconds.\n")


def main():
    start_total = time.time()
    logger.info("=" * 70)
    logger.info(" EMPIRICAL PAYOFF FUNCTION ESTIMATION - MASTER RESEARCH PIPELINE")
    logger.info(" CS 329: Multi-Agent Systems | IIT Gandhinagar (Autumn 2026)")
    logger.info(" Bhavya Shah (24110324) - Payoff Estimation & Statistical Evaluation")
    logger.info("=" * 70 + "\n")

    for script_name, description in STAGES:
        run_stage(script_name, description)

    elapsed_total = time.time() - start_total
    logger.info("=" * 70)
    logger.info(f" ALL 10 PIPELINE STAGES COMPLETED SUCCESSFULLY IN {elapsed_total:.2f}s")
    logger.info(f" Master log saved to: {log_file}")
    logger.info(f" Deliverables directory: {PAYOFF_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
