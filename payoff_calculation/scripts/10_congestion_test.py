#!/usr/bin/env python3
"""
scripts/10_congestion_test.py
Evaluation of the State-Dependent Peer Congestion Function E(x) = b3_u + b5_u * x.

CS 329 Multi-Agent Systems (Autumn 2026)
Empirical Payoff Estimation & Econometric Analysis

The marginal effect of competitor/peer appeal on log views is:
    d ln(V) / d x_bar = b3_u + b5_u * x = E(x)

This script computes E(x) at the sample mean x = mu_x (0.392395) and across the
observed range x in [0.00, 0.68], calculating:
1. Analytical point estimate and delta-method standard errors / 95% CIs via Sigma_u.
2. Cluster-pairs bootstrap (B=2000) mean, SE, percentile 95% CIs, and share of draws with E(x) < 0.

Outputs:
- results/congestion_test.csv
- plots/07_congestion_test.png
- logs/10_congestion_test.log
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
LOGS_DIR = BASE_DIR / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging
log_file = LOGS_DIR / "10_congestion_test.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("10_congestion_test")


def main():
    logger.info("=" * 75)
    logger.info("STARTING CONGESTION FUNCTION EVALUATION: E(x) = b3_u + b5_u * x")
    logger.info("=" * 75)

    # 1. Load Coefficients and Covariance
    json_path = RESULTS_DIR / "game_theory_coefficients.json"
    assert json_path.exists(), f"Missing {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    u_params = gt_data["uncentered_structural_coefficients_for_game_theory"]
    mu_x = float(gt_data["sample_summary"]["centering_mu_x"])
    mu_xbar = float(gt_data["sample_summary"]["centering_mu_xbar"])
    b3_u = float(u_params["b3_u_peer_congestion"])
    b5_u = float(u_params["b5_u_strategic_interaction"])
    
    cov_order = u_params["covariance_matrix_order"]
    cov_u = np.array(u_params["covariance_matrix_delta_method"])
    idx_b3 = cov_order.index("b3_u")
    idx_b5 = cov_order.index("b5_u")

    var_b3 = float(cov_u[idx_b3, idx_b3])
    var_b5 = float(cov_u[idx_b5, idx_b5])
    cov_b3_b5 = float(cov_u[idx_b3, idx_b5])

    logger.info(f"Loaded Uncentered Parameters: b3_u = {b3_u:.6f}, b5_u = {b5_u:.6f}")
    logger.info(f"Centering Means: mu_x = {mu_x:.6f}, mu_xbar = {mu_xbar:.6f}")
    logger.info(f"Covariance Matrix Terms: Var(b3_u)={var_b3:.4f}, Var(b5_u)={var_b5:.4f}, Cov(b3_u, b5_u)={cov_b3_b5:.4f}")

    # 2. Load Bootstrap Draws
    npz_path = RESULTS_DIR / "bootstrap_draws.npz"
    assert npz_path.exists(), f"Missing {npz_path}"
    npz_data = np.load(npz_path)
    beta_u_boot = npz_data["beta_u_boot"]  # Shape (B, 4): b1_u, b2_u, b3_u, b5_u
    b3_boot = beta_u_boot[:, 2]
    b5_boot = beta_u_boot[:, 3]
    B = len(b3_boot)
    logger.info(f"Loaded {B} cluster-pairs bootstrap draws.")

    df_t = 37  # G - 1 clusters
    t_crit = float(stats.t.ppf(0.975, df=df_t))

    # 3. Key Evaluation Points
    key_points = [0.0, 0.1, 0.2, 0.3, mu_x, 0.4, 0.5, 0.6, 0.628, 0.68]
    # Build complete grid of x from 0.0 to 0.68
    grid_fine = np.linspace(0.0, 0.68, 69)
    # Ensure mu_x and key points are in evaluation grid
    all_x = np.unique(np.sort(np.concatenate([grid_fine, key_points])))

    records = []
    logger.info("\nEvaluating E(x) across representative appeal levels:")
    logger.info(f"{'x':>7} | {'E(x)':>8} {'(SE_delta)':>11} {'[Delta 95% CI]':>25} {'p_val':>7} | {'Boot Mean':>10} {'[Boot 95% CI]':>25} {'P(E<0)':>8}")
    logger.info("-" * 105)

    for x_val in all_x:
        # Point estimate & Delta method
        e_pt = b3_u + b5_u * x_val
        var_e = var_b3 + (x_val ** 2) * var_b5 + 2.0 * x_val * cov_b3_b5
        se_delta = float(np.sqrt(max(0.0, var_e)))
        t_stat = (e_pt / se_delta) if se_delta > 1e-12 else 0.0
        p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df_t)))
        delta_ci_low = e_pt - t_crit * se_delta
        delta_ci_high = e_pt + t_crit * se_delta

        # Bootstrap
        e_b = b3_boot + b5_boot * x_val
        boot_mean = float(np.mean(e_b))
        boot_se = float(np.std(e_b, ddof=1))
        boot_ci_low = float(np.percentile(e_b, 2.5))
        boot_ci_high = float(np.percentile(e_b, 97.5))
        p_neg = float(np.mean(e_b < 0.0))

        is_mean = abs(x_val - mu_x) < 1e-5
        rec = {
            "x": round(float(x_val), 6),
            "is_sample_mean": is_mean,
            "e_point_estimate": round(e_pt, 6),
            "delta_se": round(se_delta, 6),
            "delta_t_stat": round(t_stat, 6),
            "delta_p_value": round(p_val, 6),
            "delta_ci_95_lower": round(delta_ci_low, 6),
            "delta_ci_95_upper": round(delta_ci_high, 6),
            "boot_mean": round(boot_mean, 6),
            "boot_se": round(boot_se, 6),
            "boot_ci_95_lower": round(boot_ci_low, 6),
            "boot_ci_95_upper": round(boot_ci_high, 6),
            "boot_share_negative": round(p_neg, 6),
        }
        records.append(rec)

        if x_val in key_points or is_mean:
            tag = " (mu_x)" if is_mean else ""
            logger.info(
                f"{x_val:7.4f}{tag:<7} | {e_pt:8.4f} ({se_delta:9.4f}) "
                f"[{delta_ci_low:9.4f}, {delta_ci_high:9.4f}] {p_val:7.4f} | "
                f"{boot_mean:10.4f} [{boot_ci_low:9.4f}, {boot_ci_high:9.4f}] {p_neg:8.4f}"
            )

    df_results = pd.DataFrame(records)
    csv_out = RESULTS_DIR / "congestion_test.csv"
    df_results.to_csv(csv_out, index=False)
    logger.info(f"\nSaved congestion evaluation results to: {csv_out}")

    # 4. Generate Plot 07_congestion_test.png
    plot_x = df_results["x"].values
    plot_e = df_results["e_point_estimate"].values
    plot_d_low = df_results["delta_ci_95_lower"].values
    plot_d_high = df_results["delta_ci_95_upper"].values
    plot_b_low = df_results["boot_ci_95_lower"].values
    plot_b_high = df_results["boot_ci_95_upper"].values
    plot_p_neg = df_results["boot_share_negative"].values

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )

    # Upper panel: E(x) with CIs
    ax1.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="Neutral Threshold E = 0")
    ax1.axvline(mu_x, color="#2ca02c", linestyle=":", linewidth=1.5, label=f"Sample Mean Own Appeal $\\mu_x = {mu_x:.4f}$")
    ax1.axvline(0.628, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.8, label="99th Percentile $x = 0.628$")

    # Shaded CIs
    ax1.fill_between(
        plot_x, plot_d_low, plot_d_high, color="#1f77b4", alpha=0.18, label="Delta-Method 95% CI (CRSE df=37)"
    )
    ax1.fill_between(
        plot_x, plot_b_low, plot_b_high, color="#ff7f0e", alpha=0.18, label="Bootstrap Percentile 95% CI (B=2000)"
    )
    ax1.plot(plot_x, plot_e, color="#1f77b4", linewidth=2.2, label=r"Point Estimate $\hat{E}(x) = b_{3\_u} + b_{5\_u} x$")

    # Annotate key points
    e_at_mean = float(df_results.loc[df_results["is_sample_mean"], "e_point_estimate"].iloc[0])
    se_at_mean = float(df_results.loc[df_results["is_sample_mean"], "delta_se"].iloc[0])
    p_at_mean = float(df_results.loc[df_results["is_sample_mean"], "delta_p_value"].iloc[0])
    
    ax1.scatter([mu_x], [e_at_mean], color="#2ca02c", s=60, zorder=5)
    ax1.annotate(
        f"At $\\mu_x = {mu_x:.3f}$:\n$E = +{e_at_mean:.2f} \\equiv b_{{3c}}$\nSE = {se_at_mean:.2f} ($p = {p_at_mean:.2f}$)",
        xy=(mu_x, e_at_mean),
        xytext=(mu_x - 0.18, e_at_mean + 12),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#2ca02c", alpha=0.9),
        fontsize=9,
    )

    e_at_zero = float(df_results.loc[df_results["x"] == 0.0, "e_point_estimate"].iloc[0])
    se_at_zero = float(df_results.loc[df_results["x"] == 0.0, "delta_se"].iloc[0])
    p_at_zero = float(df_results.loc[df_results["x"] == 0.0, "delta_p_value"].iloc[0])
    
    ax1.scatter([0.0], [e_at_zero], color="#d62728", s=60, zorder=5)
    ax1.annotate(
        f"At $x = 0$:\n$E = {e_at_zero:.2f} \\equiv b_{{3\\_u}}$\nSE = {se_at_zero:.2f} ($p = {p_at_zero:.2f}$)",
        xy=(0.0, e_at_zero),
        xytext=(0.04, e_at_zero - 18),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="#fdedec", ec="#d62728", alpha=0.9),
        fontsize=9,
    )

    ax1.set_ylabel(r"Peer Congestion Effect: $E(x) = \frac{\partial \ln(V)}{\partial \bar{x}}$", fontsize=11)
    ax1.set_title(
        r"State-Dependent Peer Congestion Function: $E(x) = b_{3\_u} + b_{5\_u} x$" + "\n" +
        r"Tested on $E(x)$ Across Observed Range $x \in [0.00, 0.68]$ (Insigificant Everywhere)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_ylim(-60, 60)

    # Lower panel: Probability of negative congestion P(E < 0)
    ax2.plot(plot_x, plot_p_neg * 100.0, color="#d62728", linewidth=2.0, label=r"Bootstrap $\Pr(E(x) < 0)$ (%)")
    ax2.axhline(50.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="50% Threshold")
    ax2.axvline(mu_x, color="#2ca02c", linestyle=":", linewidth=1.5)
    ax2.axvline(0.628, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.8)
    
    p_neg_mean = float(df_results.loc[df_results["is_sample_mean"], "boot_share_negative"].iloc[0]) * 100.0
    ax2.scatter([mu_x], [p_neg_mean], color="#2ca02c", s=45, zorder=5)
    ax2.annotate(
        f"{p_neg_mean:.1f}% at $\\mu_x$",
        xy=(mu_x, p_neg_mean),
        xytext=(mu_x + 0.03, p_neg_mean + 10),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.0),
        fontsize=8.5,
    )

    ax2.set_xlabel("Creator Own Click Appeal Score ($x$)", fontsize=11)
    ax2.set_ylabel(r"$\Pr(E < 0)$ (%)", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_out = PLOTS_DIR / "07_congestion_test.png"
    plt.savefig(plot_out, dpi=300)
    plt.close()
    logger.info(f"Saved congestion plot to: {plot_out}")

    # Log summary for reporting
    logger.info("\n" + "=" * 75)
    logger.info("CONGESTION TEST SUMMARY RESULT (FOR DOWNSTREAM HANDOFF):")
    logger.info(f"  At sample mean x = mu_x ({mu_x:.6f}):")
    logger.info(f"    E(mu_x) = {e_at_mean:+.4f} (SE_delta = {se_at_mean:.4f}, p = {p_at_mean:.4f})")
    logger.info(f"    Delta 95% CI: [{df_results.loc[df_results['is_sample_mean'], 'delta_ci_95_lower'].iloc[0]:+.4f}, {df_results.loc[df_results['is_sample_mean'], 'delta_ci_95_upper'].iloc[0]:+.4f}]")
    logger.info(f"    Bootstrap 95% Percentile CI: [{df_results.loc[df_results['is_sample_mean'], 'boot_ci_95_lower'].iloc[0]:+.4f}, {df_results.loc[df_results['is_sample_mean'], 'boot_ci_95_upper'].iloc[0]:+.4f}]")
    logger.info(f"    Bootstrap share with E < 0: {df_results.loc[df_results['is_sample_mean'], 'boot_share_negative'].iloc[0]*100:.2f}%")
    logger.info(f"  At zero appeal x = 0.0:")
    logger.info(f"    E(0) = {e_at_zero:+.4f} (SE_delta = {se_at_zero:.4f}, p = {p_at_zero:.4f})")
    logger.info(f"    Bootstrap share with E < 0: {df_results.loc[df_results['x'] == 0.0, 'boot_share_negative'].iloc[0]*100:.2f}%")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
