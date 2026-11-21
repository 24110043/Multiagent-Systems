"""
Parameter Sweeps and Visualisation for Learning & Game Sensitivity
CS 329: Foundations of AI: Multi-Agent Systems (2026)
Indian Institute of Technology Gandhinagar

Executes Task G parameter sweeps:
  - sigma in [0, 0.05, 0.1, 0.25, 0.5, 1.14]
  - eta in [0.01, 0.05, 0.1, 0.2, 0.4]
  - beta_5 in [-10, 0, 10, 20.79] (Regime A)
  - n_creators in [5, 10, 20, 38, 100]
  - kappa in [4.5, 6, 8, 12, 16] (Regime A)

Produces:
  results/sweeps/learning_<param>.csv
  results/sweeps/learning_<param>_summary.csv
  plots/09_noise_sweep.png
  plots/10_step_size_sweep.png
  plots/11_kappa_sensitivity.png
"""

import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_GT_SCRIPTS = os.path.join(_ROOT, "game_theoretic_formulation", "scripts")
for p in [_GT_SCRIPTS, _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sim_learning import sweep
from sim_regime_compare import PALETTE, REGIME_LABELS

SWEEPS_DIR = os.path.abspath(os.path.join(_HERE, "..", "results", "sweeps"))
PLOTS_DIR  = os.path.abspath(os.path.join(_HERE, "..", "plots"))
DPI = 300


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------
def run_all_sweeps(n_trials: int = 10, t_horizon: int = 500):
    print("=" * 60)
    print("Running Task G Parameter Sweeps (n_trials=10, T=500, init=both)")
    print("=" * 60)

    cfg = {"T_horizon": t_horizon, "init": "both"}

    # 1. Sigma sweep
    print("\n[1/5] Sweeping sigma...")
    sigma_vals = [0.0, 0.05, 0.1, 0.25, 0.5, 1.14]
    sweep("sigma", sigma_vals, base_regime="A", n_trials=n_trials, output_dir=SWEEPS_DIR, config=cfg)

    # 2. Eta sweep
    print("\n[2/5] Sweeping eta...")
    eta_vals = [0.01, 0.05, 0.1, 0.2, 0.4]
    sweep("eta", eta_vals, base_regime="A", n_trials=n_trials, output_dir=SWEEPS_DIR, config=cfg)

    # 3. Beta_5 sweep
    print("\n[3/5] Sweeping beta_5...")
    b5_vals_A = [-10.0, 0.0, 10.0, 20.7942]
    sweep("beta_5", b5_vals_A, base_regime="A", n_trials=n_trials, output_dir=SWEEPS_DIR, config=cfg)

    # 4. N_creators sweep
    print("\n[4/5] Sweeping n_creators...")
    n_vals = [5, 10, 20, 38, 100]
    sweep("n_creators", n_vals, base_regime="A", n_trials=n_trials, output_dir=SWEEPS_DIR, config=cfg)

    # 5. Kappa sweep (Regime A only)
    print("\n[5/5] Sweeping kappa (Regime A)...")
    kappa_vals = [4.5, 6.0, 8.0, 12.0, 16.0]
    sweep("kappa", kappa_vals, base_regime="A", n_trials=n_trials, output_dir=SWEEPS_DIR, config=cfg)

    print("\nAll sweeps completed successfully.")


# ---------------------------------------------------------------------------
# Plot 09 — Noise Sweep (D_T and Delta_T vs sigma)
# ---------------------------------------------------------------------------
def plot_09_noise_sweep(outdir: str = PLOTS_DIR):
    csv_path = os.path.join(SWEEPS_DIR, "learning_sigma_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows_A = [r for r in rows if r.get("regime") == "A"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Noise Sensitivity: Individual Dispersion & Mean Gap vs Payoff Noise $\\sigma$",
                 fontsize=13, fontweight="bold")

    for ax, metric, title, ylab in [
        (ax1, "D_T", r"Individual Dispersion $D_T = \max_i |x_i - x_{\mathrm{Nash}}|$", "$D_T$ (Sup-Norm Distance)"),
        (ax2, "Delta_T", r"Population Mean Gap $\Delta_T = |\bar{x}_T - x_{\mathrm{Nash}}|$", r"$\Delta_T$ (Mean Distance)"),
    ]:
        for r_rows, reg_key, color, label in [
            (rows_A, "A", PALETTE["A"], "Regime A: Calibrated Baseline"),
        ]:
            if not r_rows:
                continue
            has_init = any(r.get("init") for r in r_rows)
            if has_init:
                for im, ls, marker, init_lbl in [("low", "--", "o", "low init [0, 0.3]"), ("high", "-", "s", "high init [0.7, 1.0]")]:
                    im_rows = [r for r in r_rows if r.get("init") == im]
                    if not im_rows:
                        continue
                    sigmas = [float(r["param_value"]) for r in im_rows]
                    means = [float(r[f"{metric}_mean"]) for r in im_rows]
                    p10s = [float(r[f"{metric}_p10"]) for r in im_rows]
                    p90s = [float(r[f"{metric}_p90"]) for r in im_rows]
                    ax.plot(sigmas, means, linestyle=ls, marker=marker, color=color, linewidth=1.8, label=f"{label} ({init_lbl})")
                    ax.fill_between(sigmas, p10s, p90s, color=color, alpha=0.12)
            else:
                sigmas = [float(r["param_value"]) for r in r_rows]
                means = [float(r[f"{metric}_mean"]) for r in r_rows]
                p10s = [float(r[f"{metric}_p10"]) for r in r_rows]
                p90s = [float(r[f"{metric}_p90"]) for r in r_rows]
                ax.plot(sigmas, means, "o-", color=color, linewidth=2.0, label=label)
                ax.fill_between(sigmas, p10s, p90s, color=color, alpha=0.18)

        ax.axvline(0.05, color="gray", linestyle=":", label=r"Baseline $\sigma=0.05$")
        ax.axvline(1.14, color="darkred", linestyle="--", alpha=0.7, label=r"Empirical RMSE $\sigma=1.14$")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"Payoff Noise $\sigma$ (Gaussian std)", fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "09_noise_sweep.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 10 — Step Size Sweep (D_T and Delta_T vs eta)
# ---------------------------------------------------------------------------
def plot_10_step_size_sweep(outdir: str = PLOTS_DIR):
    csv_path = os.path.join(SWEEPS_DIR, "learning_eta_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows_A = [r for r in rows if r.get("regime") == "A"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"Step-Size Sensitivity: Learning Dynamics vs Base Step Size $\eta$",
                 fontsize=13, fontweight="bold")

    for ax, metric, title, ylab in [
        (ax1, "D_T", r"Individual Dispersion $D_T = \max_i |x_i - x_{\mathrm{Nash}}|$", "$D_T$ (Sup-Norm Distance)"),
        (ax2, "Delta_T", r"Population Mean Gap $\Delta_T = |\bar{x}_T - x_{\mathrm{Nash}}|$", r"$\Delta_T$ (Mean Distance)"),
    ]:
        for r_rows, reg_key, color, label in [
            (rows_A, "A", PALETTE["A"], "Regime A: Calibrated Baseline"),
        ]:
            if not r_rows:
                continue
            has_init = any(r.get("init") for r in r_rows)
            if has_init:
                for im, ls, marker, init_lbl in [("low", "--", "o", "low init [0, 0.3]"), ("high", "-", "s", "high init [0.7, 1.0]")]:
                    im_rows = [r for r in r_rows if r.get("init") == im]
                    if not im_rows:
                        continue
                    etas = [float(r["param_value"]) for r in im_rows]
                    means = [float(r[f"{metric}_mean"]) for r in im_rows]
                    p10s = [float(r[f"{metric}_p10"]) for r in im_rows]
                    p90s = [float(r[f"{metric}_p90"]) for r in im_rows]
                    ax.plot(etas, means, linestyle=ls, marker=marker, color=color, linewidth=1.8, label=f"{label} ({init_lbl})")
                    ax.fill_between(etas, p10s, p90s, color=color, alpha=0.12)
            else:
                etas = [float(r["param_value"]) for r in r_rows]
                means = [float(r[f"{metric}_mean"]) for r in r_rows]
                p10s = [float(r[f"{metric}_p10"]) for r in r_rows]
                p90s = [float(r[f"{metric}_p90"]) for r in r_rows]
                ax.plot(etas, means, "s-", color=color, linewidth=2.0, label=label)
                ax.fill_between(etas, p10s, p90s, color=color, alpha=0.18)

        ax.axvline(0.1, color="gray", linestyle=":", label=r"Baseline $\eta=0.1$")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"Base Step Size $\eta$ ($\eta_t = \eta/\sqrt{t}$)", fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "10_step_size_sweep.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 11 — Kappa Sensitivity (Regime A)
# ---------------------------------------------------------------------------
def plot_11_kappa_sensitivity(outdir: str = PLOTS_DIR):
    csv_path = os.path.join(SWEEPS_DIR, "learning_kappa_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows_low = [r for r in rows if r.get("init") == "low"]
    rows_high = [r for r in rows if r.get("init") == "high"]
    base_rows = rows_high if rows_high else rows

    kappas = [float(r["param_value"]) for r in base_rows]
    x_nash = [float(r["x_nash"]) for r in base_rows]
    x_so   = [float(r["x_so"]) for r in base_rows]
    k_vals = [float(r["k"]) for r in base_rows]
    poas   = [float(r["poa"]) for r in base_rows]
    lams   = [float(r["lambda_max_eps_0.01"]) for r in base_rows]
    rs     = [float(r["welfare_recovery_at_lambda_max_eps_0.01"]) for r in base_rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(r"Regime A Sensitivity to Assumed Quadratic Cost Parameter $\kappa$",
                 fontsize=14, fontweight="bold")

    # Panel 1: Equilibria
    ax = axes[0, 0]
    ax.plot(kappas, x_nash, "o-", color="#1f77b4", linewidth=2.0, label=r"$x_{\mathrm{Nash}}$")
    ax.plot(kappas, x_so, "s--", color="#2ca02c", linewidth=2.0, label=r"$x_{\mathrm{SO}}$")
    ax.axvline(8.0, color="red", linestyle=":", linewidth=1.5, label=r"Calibrated $\kappa=8.0$")
    ax.set_title("Equilibrium Click Appeal", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Assumed Cost Parameter $\kappa$", fontsize=10)
    ax.set_ylabel("Click Appeal in [0, 1]", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 2: Contraction & PoA
    ax = axes[0, 1]
    color_k = "#9467bd"
    color_poa = "#d62728"
    ax.plot(kappas, k_vals, "d-", color=color_k, linewidth=2.0, label=r"Contraction Rate $k$")
    ax.set_xlabel(r"Assumed Cost Parameter $\kappa$", fontsize=10)
    ax.set_ylabel(r"Contraction Rate $k = |\beta_5|/(2|\beta_2^{\mathrm{eff}}|)$", color=color_k, fontsize=10)
    ax.tick_params(axis="y", labelcolor=color_k)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="$k=1$ Boundary")

    ax_poa = ax.twinx()
    ax_poa.plot(kappas, poas, "^-", color=color_poa, linewidth=2.0, label=r"$\mathrm{PoA}$")
    ax_poa.set_ylabel(r"Price of Anarchy $\mathrm{PoA}$", color=color_poa, fontsize=10)
    ax_poa.tick_params(axis="y", labelcolor=color_poa)
    ax.set_title("Contraction Rate $k$ & Price of Anarchy", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel 3: Mediator Effectiveness
    ax = axes[1, 0]
    ax.plot(kappas, lams, "o-", color="#8c564b", linewidth=2.0, label=r"$\lambda_{\max}(\epsilon=0.01)$")
    ax.plot(kappas, rs, "s--", color="#e377c2", linewidth=2.0, label=r"$R(\lambda_{\max})$")
    ax.axvline(8.0, color="red", linestyle=":", linewidth=1.5, label=r"Calibrated $\kappa=8.0$")
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_title(r"Platform Mediator Effectiveness ($\epsilon=0.01$)", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Assumed Cost Parameter $\kappa$", fontsize=10)
    ax.set_ylabel("Mediator Metrics", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 4: Learning Performance
    ax = axes[1, 1]
    if rows_low and rows_high:
        d_t_low = [float(r["D_T_mean"]) for r in rows_low]
        d_t_high = [float(r["D_T_mean"]) for r in rows_high]
        delta_t_low = [float(r["Delta_T_mean"]) for r in rows_low]
        delta_t_high = [float(r["Delta_T_mean"]) for r in rows_high]
        ax.plot(kappas, d_t_high, "o-", color="#17becf", linewidth=1.8, label=r"$D_T$ (high start)")
        ax.plot(kappas, d_t_low, "o--", color="#17becf", alpha=0.7, linewidth=1.8, label=r"$D_T$ (low start)")
        ax.plot(kappas, delta_t_high, "s-", color="#bcbd22", linewidth=1.8, label=r"$\Delta_T$ (high start)")
        ax.plot(kappas, delta_t_low, "s--", color="#bcbd22", alpha=0.7, linewidth=1.8, label=r"$\Delta_T$ (low start)")
    else:
        d_t = [float(r["D_T_mean"]) for r in base_rows]
        delta_t = [float(r["Delta_T_mean"]) for r in base_rows]
        ax.plot(kappas, d_t, "o-", color="#17becf", linewidth=2.0, label=r"Individual Dispersion $D_T$")
        ax.plot(kappas, delta_t, "s--", color="#bcbd22", linewidth=2.0, label=r"Population Mean Gap $\Delta_T$")
    ax.axvline(8.0, color="red", linestyle=":", linewidth=1.5, label=r"Calibrated $\kappa=8.0$")
    ax.set_title("Payoff-Based Learning Convergence at Horizon $T=500$", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Assumed Cost Parameter $\kappa$", fontsize=10)
    ax.set_ylabel("Convergence Error", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "11_kappa_sensitivity.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    run_all_sweeps(n_trials=10, t_horizon=500)
    print("\nGenerating sweep plots (09, 10, 11)...")
    plot_09_noise_sweep(PLOTS_DIR)
    plot_10_step_size_sweep(PLOTS_DIR)
    plot_11_kappa_sensitivity(PLOTS_DIR)
    print("Sweeps and plots complete.")


if __name__ == "__main__":
    main()
