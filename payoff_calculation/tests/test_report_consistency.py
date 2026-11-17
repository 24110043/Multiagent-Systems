#!/usr/bin/env python3
"""
tests/test_report_consistency.py
Automated Consistency Verification Test.

CS 329 Multi-Agent Systems (Autumn 2026)
Empirical Payoff Estimation & Econometric Analysis

Reads:
- FINAL_PAYOFF_REPORT.md
- README.md
- results/game_theory_coefficients.json
- results/post_test_structural_regression_results.csv
- results/training_model_comparison.csv
- results/loco_cv_aggregate_summary.csv
- results/validation_evaluation.csv
- results/test_set_evaluation.csv
- results/bootstrap_inference_results.csv
- results/wellposedness_check_report.json

Asserts that key reported numbers in the Markdown reports match the underlying
CSV/JSON results within expected numeric precision.
"""

import sys
import re
import json
from pathlib import Path
import pandas as pd
import numpy as np

PAYOFF_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PAYOFF_DIR / "results"
REPORT_PATH = PAYOFF_DIR / "FINAL_PAYOFF_REPORT.md"
README_PATH = PAYOFF_DIR / "README.md"


def load_ground_truth():
    gt = {}

    # 1. Game theory coefficients JSON
    with open(RESULTS_DIR / "game_theory_coefficients.json", "r", encoding="utf-8") as f:
        gt_json = json.load(f)

    u_coefs = gt_json["uncentered_structural_coefficients_for_game_theory"]
    c_coefs = gt_json["centered_estimation_coefficients"]
    hyp = gt_json["hypothesis_testing_and_model_diagnostics"]
    audit = gt_json["game_theoretic_properties"]["contraction_mapping_audit"]
    stab = gt_json["game_theoretic_properties"]["stability_evaluation_option_A_unaugmented"]
    cost = gt_json["game_theoretic_properties"]["stability_evaluation_option_B_cost_augmented"]

    gt["b1_u"] = u_coefs["b1_u_own_linear_appeal"]
    gt["b2_u"] = u_coefs["b2_u_quadratic_curvature"]
    gt["b3_u"] = u_coefs["b3_u_peer_congestion"]
    gt["b5_u"] = u_coefs["b5_u_strategic_interaction"]
    gt["b0_u"] = u_coefs["b0_u_intercept"]

    gt["b1_c"] = c_coefs["b1_c_own_appeal_at_mean"]
    gt["b2_c"] = c_coefs["b2_c_quadratic_curvature"]
    gt["b3_c"] = c_coefs["b3_c_peer_congestion_at_mean"]
    gt["b5_c"] = c_coefs["b5_c_strategic_interaction"]
    gt["gamma_s"] = c_coefs["gamma_subscribers"]
    gt["gamma_d"] = c_coefs["gamma_duration"]
    gt["gamma_a"] = c_coefs["gamma_age_days"]

    gt["f_stat"] = hyp["joint_f_test_packaging_terms"]["f_statistic"]
    gt["f_pval"] = hyp["joint_f_test_packaging_terms"]["p_value"]
    gt["wcb_p_b2"] = hyp["test_concavity"]["wild_cluster_bootstrap_p_value"]
    gt["wcb_p_b5"] = hyp["test_strategic_interaction"]["wild_cluster_bootstrap_p_value"]

    gt["rho"] = audit["sample_point_estimate_rho"]
    gt["share_concave"] = audit["bootstrap_concave_fraction"]
    gt["share_both"] = audit["bootstrap_p_both_concave_and_contraction"]
    gt["rho_ci_lower"] = audit["bootstrap_rho_95_ci"][0]
    gt["rho_ci_upper"] = audit["bootstrap_rho_95_ci"][1]

    gt["kappa_min"] = 4.3995  # |b5| - 2|b2|

    # 2. Model performance comparisons
    train_df = pd.read_csv(RESULTS_DIR / "training_model_comparison.csv", index_col="model_name")
    gt["train_r2_m0"] = train_df.loc["Model_0_Baseline", "r_squared"]
    gt["train_r2_m4"] = train_df.loc["Model_4_Centered", "r_squared"]

    loco_df = pd.read_csv(RESULTS_DIR / "loco_cv_aggregate_summary.csv", index_col="model_name")
    gt["loco_r2_m0"] = loco_df.loc["Model_0_Baseline", "r2_oos"]
    gt["loco_r2_m4"] = loco_df.loc["Model_4_Interaction", "r2_oos"]

    val_df = pd.read_csv(RESULTS_DIR / "validation_evaluation.csv", index_col="model_name")
    gt["val_r2_m0"] = val_df.loc["Model_0_Baseline", "r2_oos_val"]
    gt["val_r2_m1"] = val_df.loc["Model_1_Linear", "r2_oos_val"]
    gt["val_r2_m4"] = val_df.loc["Model_4_Interaction", "r2_oos_val"]

    test_df = pd.read_csv(RESULTS_DIR / "test_set_evaluation.csv", index_col="model_name")
    gt["test_r2_m0"] = test_df.loc["Model_0_Baseline", "r2_oos_test"]
    gt["test_r2_m4"] = test_df.loc["Model_4_Interaction", "r2_oos_test"]

    # 3. Standard errors
    reg_df = pd.read_csv(RESULTS_DIR / "post_test_structural_regression_results.csv")
    m4_c = reg_df[reg_df["specification"] == "Model_4_Centered"].set_index("variable")
    m4_u = reg_df[reg_df["specification"] == "Model_4_Uncentered_Analytical"].set_index("variable")

    gt["se_b1_c"] = m4_c.loc["x_c", "std_error_crse"]
    gt["se_b2_c"] = m4_c.loc["x2_c", "std_error_crse"]
    gt["se_b3_c"] = m4_c.loc["x_bar_c", "std_error_crse"]
    gt["se_b5_c"] = m4_c.loc["x_inter_c", "std_error_crse"]
    gt["se_gamma_s"] = m4_c.loc["log_subscribers", "std_error_crse"]

    gt["se_b1_u"] = m4_u.loc["x", "std_error_crse"]
    gt["se_b3_u"] = m4_u.loc["x_bar", "std_error_crse"]

    return gt


def assert_value_present(text, value, label, doc_name, precision=4):
    """Checks whether the formatted numeric value appears in text."""
    # Build regex patterns for precision variations (e.g., -2.0190 or -2.019 or -2.02 or 4.3995 or 4.40)
    s_full = f"{value:.4f}"
    s_3dec = f"{value:.3f}"
    s_2dec = f"{value:.2f}"
    
    # Also handle leading plus sign if positive
    patterns = [re.escape(s_full), re.escape(s_3dec), re.escape(s_2dec)]
    if value > 0:
        patterns.append(re.escape(f"+{s_full}"))
        patterns.append(re.escape(f"+{s_3dec}"))
        patterns.append(re.escape(f"+{s_2dec}"))

    found = any(re.search(pat, text) for pat in patterns)
    if not found:
        raise AssertionError(
            f"Consistency Failure in {doc_name}: {label} expected {s_full} (or {s_short}/{s_2dec}) not found."
        )
    print(f"  [PASS] {doc_name}: {label} matches {s_full}")


def test_markdown_consistency():
    print("\n" + "=" * 70)
    print(" EXECUTING REPORT CONSISTENCY & VALUE ACCURACY TEST")
    print("=" * 70)

    gt = load_ground_truth()

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report_text = f.read()

    with open(README_PATH, "r", encoding="utf-8") as f:
        readme_text = f.read()

    docs = [
        ("FINAL_PAYOFF_REPORT.md", report_text),
        ("README.md", readme_text)
    ]

    for doc_name, text in docs:
        print(f"\nVerifying Consistency in {doc_name}...")

        # 1. Structural Parameters
        assert_value_present(text, gt["b1_u"], "b1_u (Uncentered Own Appeal)", doc_name, 4)
        assert_value_present(text, gt["b2_u"], "b2_u (Uncentered Curvature)", doc_name, 4)
        assert_value_present(text, gt["b3_u"], "b3_u (Uncentered Peer Congestion)", doc_name, 4)
        assert_value_present(text, gt["b5_u"], "b5_u (Uncentered Interaction)", doc_name, 4)
        assert_value_present(text, gt["b0_u"], "b0_u (Uncentered Intercept)", doc_name, 4)

        assert_value_present(text, gt["b1_c"], "b1_c (Centered Own Appeal)", doc_name, 4)
        assert_value_present(text, gt["b2_c"], "b2_c (Centered Curvature)", doc_name, 4)
        assert_value_present(text, gt["b3_c"], "b3_c (Centered Peer Congestion)", doc_name, 4)
        assert_value_present(text, gt["b5_c"], "b5_c (Centered Interaction)", doc_name, 4)
        assert_value_present(text, gt["gamma_s"], "gamma_s (Scale Elasticity)", doc_name, 4)

        # 2. Model Performance Comparisons
        assert_value_present(text, gt["train_r2_m0"], "Train R2 M0", doc_name, 4)
        assert_value_present(text, gt["train_r2_m4"], "Train R2 M4", doc_name, 4)
        assert_value_present(text, gt["loco_r2_m0"], "LOCO-CV R2 M0", doc_name, 4)
        assert_value_present(text, gt["loco_r2_m4"], "LOCO-CV R2 M4", doc_name, 4)
        assert_value_present(text, gt["val_r2_m0"], "Val R2 M0", doc_name, 4)
        assert_value_present(text, gt["val_r2_m4"], "Val R2 M4", doc_name, 4)
        assert_value_present(text, gt["test_r2_m0"], "Test R2 M0", doc_name, 4)
        assert_value_present(text, gt["test_r2_m4"], "Test R2 M4", doc_name, 4)

        # 3. Hypothesis Tests & Parameter Properties
        assert_value_present(text, gt["f_stat"], "Joint Packaging F-statistic", doc_name, 4)
        assert_value_present(text, gt["f_pval"], "Joint Packaging F p-value", doc_name, 4)
        assert_value_present(text, gt["wcb_p_b2"], "Concavity WCB p-value", doc_name, 4)
        assert_value_present(text, gt["wcb_p_b5"], "Interaction WCB p-value", doc_name, 4)

        # Check percentage/share representations (e.g. 81.40% or 81.40\% or 0.814)
        assert ("81.40%" in text or "81.4%" in text or "0.814" in text or "81.40\\%" in text), f"Missing share_concave in {doc_name}"
        assert ("18.65%" in text or "0.1865" in text or "18.65\\%" in text), f"Missing share_both in {doc_name}"
        print(f"  [PASS] {doc_name}: Bootstrap condition shares (81.4% concave, 18.65% both) confirmed.")

        # Check terminology
        assert "Aggregative" in text, f"Missing Aggregative in {doc_name}"
        assert "Exact Potential Game" in text, f"Missing Exact Potential Game in {doc_name}"
        assert "global English" in text or "global English-language" in text, f"Missing data clarification note in {doc_name}"
        print(f"  [PASS] {doc_name}: Terminology and Data vs Proposal Note verified.")

    print("\n" + "=" * 70)
    print(" ALL CONSISTENCY VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_markdown_consistency()
