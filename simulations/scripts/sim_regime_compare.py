"""
Simulation: Cross-Regime Comparison & Plot Generation
CS 329: Foundations of AI: Multi-Agent Systems (2026)
Indian Institute of Technology Gandhinagar

Experiment 4 — Cross-Regime Comparison:
  Runs dynamics and mediator sweep for all three regimes (A, B, B') and
  assembles a summary table comparing key metrics.

Master plot generator — produces 4 (or 6 with --with-learning) publication plots:
  01_br_convergence.png
  02_welfare_trajectory.png
  03_mediator_sweep.png
  04_regime_comparison.png
  05_learning_convergence.png   [--with-learning only]
  06_learning_welfare.png       [--with-learning only]

Usage:
    python simulations/scripts/sim_regime_compare.py
    python simulations/scripts/sim_regime_compare.py --with-learning

Outputs:
    simulations/results/regime_comparison_summary.csv
    simulations/results/mediator_sweep.csv
    simulations/results/dynamics_regime_*.csv
    simulations/plots/*.png
"""

import sys
import os
import csv
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_GT_SCRIPTS = os.path.join(_ROOT, "game_theoretic_formulation", "scripts")
for p in [_GT_SCRIPTS, _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)

from game_theory_solver import PotentialGame       # noqa: E402
from sim_dynamics import simulate_regime, N_TRIALS, make_regime  # noqa: E402
from sim_mediator import mediator_sweep, EPSILONS   # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
JSON_PATH   = os.path.join(_ROOT, "payoff_calculation", "results", "game_theory_coefficients.json")
if not os.path.exists(JSON_PATH):
    JSON_PATH = os.path.join(_ROOT, "payoff_estimation", "results", "game_theory_coefficients.json")
RESULTS_DIR = os.path.abspath(os.path.join(_HERE, "..", "results"))
PLOTS_DIR   = os.path.abspath(os.path.join(_HERE, "..", "plots"))
N_CREATORS  = 38
SEED_BASE   = 42
DPI         = 300

# Shared style
PALETTE = {
    "A":  "#1f77b4",   # blue
}
REGIME_LABELS = {
    "A":  "Calibrated (N=809, assumed kappa=8)",
}


# ---------------------------------------------------------------------------
# Build games
# ---------------------------------------------------------------------------
def build_games(json_path: str = None):
    game_A = make_regime("A")
    return {"A": game_A}


# ---------------------------------------------------------------------------
# Plot 01 — BR Convergence (unchanged from original)
# ---------------------------------------------------------------------------
def plot_01_br_convergence(dynamics_data: dict, outdir: str):
    """
    Log-scale error ‖x_t − x_Nash‖_inf vs iteration.
    Thin lines = individual trials, bold = mean, dashed = theoretical decay k^t.
    """
    n_reg = len(dynamics_data)
    fig, axes = plt.subplots(1, n_reg, figsize=(6.5 * n_reg, 5), sharey=False, squeeze=False)
    axes = axes[0]
    fig.suptitle("Best-Response Dynamics: Convergence to Nash Equilibrium",
                 fontsize=14, fontweight="bold")

    for ax, (key, (game, results)) in zip(axes, dynamics_data.items()):
        color = PALETTE[key]
        label = REGIME_LABELS[key]
        k = game.contraction_rate

        trajs = [r["error_trajectory"] for r in results]
        max_len = max(len(t) for t in trajs)
        traj_arr = np.full((len(trajs), max_len), np.nan)
        for i, t in enumerate(trajs):
            traj_arr[i, : len(t)] = t

        iters = np.arange(max_len)

        for i in range(min(20, len(trajs))):
            ax.semilogy(iters, traj_arr[i], color=color, alpha=0.15, linewidth=0.7)

        mean_traj = np.nanmean(traj_arr, axis=0)
        ax.semilogy(iters, mean_traj, color=color, linewidth=2.2, label="Mean error")

        init_err = np.nanmean(traj_arr[:, 0])
        theory_decay = init_err * (k ** iters)
        mask = theory_decay > 1e-10
        ax.semilogy(iters[mask], theory_decay[mask], "k--", linewidth=1.4,
                    label=f"Theory $k^t$ (k={k:.3f})")

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Iteration t")
        ax.set_ylabel(r"$\|x_t - x^{Nash}\|_\infty$")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(left=0)

    plt.tight_layout()
    path = os.path.join(outdir, "01_br_convergence.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 02 — Welfare Trajectory (Task 5: footnote + note on initial welfare)
# ---------------------------------------------------------------------------
def plot_02_welfare_trajectory(dynamics_data: dict, outdir: str):
    """
    Symmetric welfare π(mean(x_t), mean(x_t)) vs iteration t.
    Dashed horizontal lines for Nash and SO welfare levels.
    """
    n_reg = len(dynamics_data)
    fig, axes = plt.subplots(1, n_reg, figsize=(6.5 * n_reg, 5), sharey=False, squeeze=False)
    axes = axes[0]
    fig.suptitle("Welfare Trajectory During Best-Response Dynamics",
                 fontsize=14, fontweight="bold")

    for ax, (key, (game, results)) in zip(axes, dynamics_data.items()):
        color = PALETTE[key]
        label = REGIME_LABELS[key]

        trajs = [r["welfare_trajectory"] for r in results]
        max_len = max(len(t) for t in trajs)
        traj_arr = np.full((len(trajs), max_len), np.nan)
        for i, t in enumerate(trajs):
            traj_arr[i, : len(t)] = t

        iters = np.arange(max_len)
        mean_traj = np.nanmean(traj_arr, axis=0)

        for i in range(min(20, len(trajs))):
            ax.plot(iters, traj_arr[i], color=color, alpha=0.15, linewidth=0.7)

        ax.plot(iters, mean_traj, color=color, linewidth=2.2, label="Mean welfare")

        w_nash = game.social_welfare_symmetric(game.x_nash)
        w_so   = game.social_welfare_symmetric(game.x_so)
        ax.axhline(w_nash, color="gray", linestyle="--", linewidth=1.4,
                   label=f"Nash welfare ({w_nash:.3f})")
        ax.axhline(w_so, color="black", linestyle="-.", linewidth=1.4,
                   label=f"SO welfare ({w_so:.3f})")

        ax.fill_between([0, max_len - 1], w_nash, w_so,
                        alpha=0.08, color="orange", label="PoA gap")

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Iteration t")
        ax.set_ylabel(r"$\pi(\bar{x}_t, \bar{x}_t)$")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    # Footnotes (Task 5)
    fig.text(0.5, -0.04,
             "* Welfare π(x,x) = (β₁+β₃)x + (β₂+β₅)x² excludes channel constant c (not absolute log-views).\n"
             "  Random starts average x̄₀ ≈ 0.5; in Regime A, initial welfare "
             "can exceed Nash welfare before converging.",
             ha="center", va="top", fontsize=8, style="italic", color="gray")

    plt.tight_layout()
    path = os.path.join(outdir, "02_welfare_trajectory.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 03 — Mediator Sweep (Dual-axis G(lambda) and R(lambda))
# ---------------------------------------------------------------------------
def plot_03_mediator_sweep(sweep_data: dict, outdir: str):
    """
    Dual-axis plot: G(λ) deviation gain (left) and R(λ) welfare recovery (right).
    For Regime A, R(λ) axis extends negative to show convex welfare dip and R=0 crossing.
    Vertical markers at λ_max for each ε.
    """
    n_reg = len(sweep_data)
    fig, axes = plt.subplots(1, n_reg, figsize=(7.0 * n_reg, 5), squeeze=False)
    axes = axes[0]
    fig.suptitle("Platform Mediator: Deviation Gain G(λ) and Welfare Recovery R(λ)",
                 fontsize=14, fontweight="bold")

    eps_colors = {0.001: "#9467bd", 0.01: "#8c564b", 0.05: "#e377c2", 0.10: "#bcbd22"}
    eps_labels = {0.001: "ε=0.001",  0.01: "ε=0.01",  0.05: "ε=0.05",  0.10: "ε=0.10"}

    for ax, (key, (game, rows)) in zip(axes, sweep_data.items()):
        color = PALETTE[key]
        label = REGIME_LABELS[key]

        lambdas = np.array([r["lambda"] for r in rows])
        G = np.array([r["deviation_gain_definition"] for r in rows])
        R = np.array([r["welfare_recovery_R"] for r in rows])

        # Left axis: G(λ)
        ax.plot(lambdas, G, color=color, linewidth=2.2, label="G(λ) deviation gain")
        ax.set_xlabel("λ (mediator strength)")
        ax.set_ylabel("G(λ) — Deviation Gain", color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.set_ylim(bottom=0)   # Task 5: start at 0

        # Right axis: R(λ)
        ax2 = ax.twinx()
        ax2.plot(lambdas, R, color="black", linewidth=1.8, linestyle="--",
                 label="R(λ) welfare recovery")
        ax2.set_ylabel("R(λ) — Welfare Recovery", color="black")
        if key == "A" and np.any(R < 0):
            ax2.set_ylim(min(R) - 0.05, 1.05)
            ax2.axhline(0, color="k", linestyle=":", alpha=0.6, linewidth=1.0)
            # Find and mark R=0 crossing
            for idx in range(len(R) - 1):
                if R[idx] < 0 and R[idx+1] >= 0:
                    lam_zero = lambdas[idx] + (0 - R[idx]) * (lambdas[idx+1] - lambdas[idx]) / (R[idx+1] - R[idx])
                    ax2.axvline(lam_zero, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
                    ax2.annotate(f"R=0 at λ={lam_zero:.3f}", xy=(lam_zero, 0.02),
                                xytext=(lam_zero + 0.08, min(R) * 0.5),
                                fontsize=7, color="gray",
                                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
                    break
            ax.text(0.5, 0.02, "welfare convex: R(λ) ≠ 2λ-λ²",
                    transform=ax.transAxes, fontsize=7, ha="center", style="italic", color="darkred")
        else:
            ax2.set_ylim(0, 1.05)

        # Mark λ_max for each ε
        for eps, ecol in eps_colors.items():
            lm = game.mediator_max_lambda(eps)
            ax.axvline(lm, color=ecol, linewidth=1.1, linestyle=":", alpha=0.9,
                       label=f"λ_max({eps_labels[eps]})={lm:.3f}")

        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper left")

        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "03_mediator_sweep.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 04 — Regime Comparison (Task 5: Mean iters on separate subplot)
# ---------------------------------------------------------------------------
def plot_04_regime_comparison(summary_rows: list, outdir: str):
    """
    Two subplots:
      Left:  grouped bars for x_Nash, x_SO, PoA, λ_max(ε=0.01)
      Right: grouped bars for Mean iters (separate y-axis scale)
    """
    regime_short = [r["regime_key"] for r in summary_rows]
    colors = [PALETTE[k] for k in regime_short]

    main_metrics = {
        "x_Nash": [r["x_nash"] for r in summary_rows],
        "x_SO":   [r["x_so"]   for r in summary_rows],
        "PoA":    [r["poa"]    for r in summary_rows],
        r"$\lambda_{max}$(ε=0.01)": [r["lambda_max_eps_0.01"] for r in summary_rows],
    }
    iters_vals = [r["mean_iters"] for r in summary_rows]

    fig, (ax_main, ax_iters) = plt.subplots(
        1, 2, figsize=(12, 5.5),
        gridspec_kw={"width_ratios": [3.5, 1.2]},
    )
    fig.suptitle("Key Equilibrium & Mediator Metrics (Regime A Calibrated Baseline)",
                 fontsize=13, fontweight="bold")

    # --- Left: main metrics ---
    n_metrics = len(main_metrics)
    x = np.arange(n_metrics)
    width = 0.35
    for i, (rshort, color) in enumerate(zip(regime_short, colors)):
        vals = [list(main_metrics.values())[m][i] for m in range(n_metrics)]
        offset = (i - (len(regime_short) - 1) / 2.0) * width
        bars = ax_main.bar(x + offset, vals, width,
                           label=REGIME_LABELS[rshort],
                           color=color, alpha=0.82)
        for bar, v in zip(bars, vals):
            ax_main.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.01,
                         f"{v:.3f}", ha="center", va="bottom",
                         fontsize=8.5, fontweight="bold")

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(list(main_metrics.keys()), fontsize=11)
    ax_main.set_ylabel("Value", fontsize=12)
    ax_main.legend(fontsize=9)
    ax_main.grid(axis="y", alpha=0.3)
    ax_main.set_ylim(0, max(max(v) for v in main_metrics.values()) * 1.25)

    # --- Right: mean iterations ---
    xi = np.arange(len(iters_vals))
    bars2 = ax_iters.bar(xi, iters_vals, color=colors, alpha=0.82)
    for bar, v, rshort in zip(bars2, iters_vals, regime_short):
        ax_iters.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.05,
                      f"{v:.1f}", ha="center", va="bottom",
                      fontsize=9, fontweight="bold")
    ax_iters.set_xticks(xi)
    ax_iters.set_xticklabels(regime_short, fontsize=11)
    ax_iters.set_ylabel("Mean Iterations to Convergence", fontsize=10)
    ax_iters.set_title("BR Iterations", fontsize=11)
    ax_iters.grid(axis="y", alpha=0.3)
    ax_iters.set_ylim(0, max(iters_vals) * 1.4)

    plt.tight_layout()
    path = os.path.join(outdir, "04_regime_comparison.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 05 — Learning Convergence (Task 1)
# ---------------------------------------------------------------------------
def plot_05_learning_convergence(learning_traj_data: dict, outdir: str):
    """
    xbar_t (mean x over agents) vs round t for each regime.
    Thin = individual trials; bold = mean; dashed = x_nash reference.
    """
    n_reg = len(learning_traj_data)
    fig, axes = plt.subplots(1, n_reg, figsize=(6.5 * n_reg, 5), sharey=False, squeeze=False)
    axes = axes[0]
    fig.suptitle("RQ2: Payoff-Based Learning — Convergence of Mean Appeal",
                 fontsize=14, fontweight="bold")

    for ax, (key, (game, traj_rows)) in zip(axes, learning_traj_data.items()):
        color = PALETTE[key]
        label = REGIME_LABELS[key]
        x_nash = game.x_nash

        # Group by trial
        trials = {}
        for row in traj_rows:
            tr = row["trial"]
            if tr not in trials:
                trials[tr] = []
            trials[tr].append((row["t"], row["xbar_t"]))

        for tr_id, pts in list(trials.items())[:20]:
            ts = [p[0] for p in pts]
            xs = [p[1] for p in pts]
            ax.plot(ts, xs, color=color, alpha=0.15, linewidth=0.7)

        # Mean across trials
        all_t = sorted({row["t"] for row in traj_rows})
        mean_xbar = []
        for t in all_t:
            vals = [row["xbar_t"] for row in traj_rows if row["t"] == t]
            mean_xbar.append(np.mean(vals))
        ax.plot(all_t, mean_xbar, color=color, linewidth=2.2, label="Mean x̄_t")
        ax.axhline(x_nash, color="k", linestyle="--", linewidth=1.4,
                   label=f"x_Nash={x_nash:.4f}")

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Round t")
        ax.set_ylabel(r"$\bar{x}_t$ (mean appeal)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(outdir, "05_learning_convergence.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 06 — Learning Welfare (Task 1)
# ---------------------------------------------------------------------------
def plot_06_learning_welfare(learning_traj_data: dict, outdir: str):
    """
    W_V_t = sum_i exp(pi_i) (noise-free) vs round t.
    Dashed lines for Nash and SO reference W_V.
    """
    n_reg = len(learning_traj_data)
    fig, axes = plt.subplots(1, n_reg, figsize=(6.5 * n_reg, 5), sharey=False, squeeze=False)
    axes = axes[0]
    fig.suptitle("RQ2: Payoff-Based Learning — View Welfare Trajectory",
                 fontsize=14, fontweight="bold")

    for ax, (key, (game, traj_rows)) in zip(axes, learning_traj_data.items()):
        color = PALETTE[key]
        label = REGIME_LABELS[key]

        # Reference W_V at Nash and SO
        from sim_numeric import make_payoff_fn as _mpfn
        pfn = _mpfn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)
        n   = game.n_creators
        W_V_nash = n * np.exp(pfn(game.x_nash, game.x_nash))
        W_V_so   = n * np.exp(pfn(game.x_so,   game.x_so))

        trials = {}
        for row in traj_rows:
            tr = row["trial"]
            if tr not in trials:
                trials[tr] = []
            trials[tr].append((row["t"], row["W_V_t"]))

        for tr_id, pts in list(trials.items())[:20]:
            ts = [p[0] for p in pts]
            ws = [p[1] for p in pts]
            ax.plot(ts, ws, color=color, alpha=0.15, linewidth=0.7)

        all_t = sorted({row["t"] for row in traj_rows})
        mean_W = [np.mean([r["W_V_t"] for r in traj_rows if r["t"] == t]) for t in all_t]
        ax.plot(all_t, mean_W, color=color, linewidth=2.2, label="Mean W_V(t)")

        ax.axhline(W_V_nash, color="gray", linestyle="--", linewidth=1.4,
                   label=f"Nash W_V ({W_V_nash:.1f})")
        ax.axhline(W_V_so, color="black", linestyle="-.", linewidth=1.4,
                   label=f"SO W_V ({W_V_so:.1f})")

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Round t")
        ax.set_ylabel(r"$W_V = \sum_i \exp(\pi_i)$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "06_learning_welfare.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 07 — Learning Dispersion (Task A5)
# ---------------------------------------------------------------------------
def plot_07_learning_dispersion(learning_traj_data: dict, outdir: str):
    """
    Plot 07 — Learning Dispersion:
    Per regime, mean over trials of max_i |x_i - x_nash| (err_sup_to_nash) vs round t
    with a 10-90% band across trials.
    Shows that while the mean action converges, individual creators keep hovering around Nash.
    """
    n_reg = len(learning_traj_data)
    fig, axes = plt.subplots(1, n_reg, figsize=(6.5 * n_reg, 5), sharey=True, squeeze=False)
    axes = axes[0]
    fig.suptitle(
        "RQ2: Payoff-Based Learning — Dispersion Around Nash Profile\n"
        r"(Mean $\max_i |x_i - x_{\mathrm{Nash}}|$ with 10–90% Band Across Trials)",
        fontsize=13, fontweight="bold",
    )

    for ax, (key, (game, traj_rows)) in zip(axes, learning_traj_data.items()):
        color = PALETTE[key]
        label = REGIME_LABELS[key]

        all_t = sorted({row["t"] for row in traj_rows})
        t_to_errs = {}
        for row in traj_rows:
            t = row["t"]
            err = row["err_sup_to_nash"]
            if t not in t_to_errs:
                t_to_errs[t] = []
            t_to_errs[t].append(err)

        means = [float(np.mean(t_to_errs[t])) for t in all_t]
        p10s = [float(np.percentile(t_to_errs[t], 10)) for t in all_t]
        p90s = [float(np.percentile(t_to_errs[t], 90)) for t in all_t]

        ax.fill_between(all_t, p10s, p90s, color=color, alpha=0.22, label="10–90% Band")
        ax.plot(all_t, means, color=color, linewidth=2.2, label=r"Mean $\max_i |x_i - x_{\mathrm{Nash}}|$")

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Round t", fontsize=10)
        ax.set_ylabel(r"$\max_i |x_i - x_{\mathrm{Nash}}|$", fontsize=10)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "07_learning_dispersion.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Save regime comparison CSV (Task 3: add view_recovery columns)
# ---------------------------------------------------------------------------
def save_regime_summary_csv(summary_rows, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    keys = list(summary_rows[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Saved: {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-learning", action="store_true",
                        help="Also run RQ2 payoff-based learning experiment (slower)")
    parser.add_argument("--with-bootstrap", action="store_true",
                        help="Also run parametric bootstrap uncertainty analysis (slower)")
    parser.add_argument("--with-sweeps", action="store_true",
                        help="Also run Task G parameter sweeps and generate plots 09-11")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 4: Cross-Regime Comparison + All Plots")
    if args.with_learning:
        print("  [--with-learning] RQ2 learning experiment ENABLED")
    if args.with_bootstrap:
        print("  [--with-bootstrap] Parametric bootstrap uncertainty ENABLED")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    games = build_games(JSON_PATH)

    # --- Run dynamics for all regimes ---
    print("\n[1/2] Running best-response dynamics for all regimes...")
    dynamics_data = {}
    from sim_dynamics import save_results_csv
    for key, game in games.items():
        print(f"  Regime {key}: {game.name}")
        results = simulate_regime(game, n_trials=N_TRIALS, seed_base=SEED_BASE)
        dynamics_data[key] = (game, results)
        label_map = {"A": "A_empirical", "B": "B_hypothetical", "Bp": "Bprime_hypothetical"}
        csv_path = os.path.join(RESULTS_DIR, f"dynamics_regime_{label_map[key]}.csv")
        save_results_csv(results, csv_path)

    # --- Run mediator sweep for all regimes ---
    print("\n[2/2] Running mediator sweep for all regimes...")
    from sim_mediator import save_mediator_csv
    sweep_data = {}
    all_sweep_rows = []
    for key, game in games.items():
        rows = mediator_sweep(game)
        sweep_data[key] = (game, rows)
        all_sweep_rows.extend(rows)
    save_mediator_csv(all_sweep_rows, os.path.join(RESULTS_DIR, "mediator_sweep.csv"))

    # --- Build regime comparison summary (Task 3: add view_recovery) ---
    summary_rows = []
    for key, game in games.items():
        _, results = dynamics_data[key]
        mean_iters    = float(np.mean([r["n_iters"]        for r in results]))
        mean_error    = float(np.mean([r["error_to_nash"]  for r in results]))
        mean_poa_sim  = float(np.mean([r["poa_simulated"]  for r in results]))
        n_converged   = sum(1 for r in results if r["converged"])

        row = {
            "regime":             game.name,
            "regime_key":         key,
            "x_nash":             game.x_nash,
            "x_so":               game.x_so,
            "delta":              game.nash_so_gap,
            "marginal_E":         game.marginal_external_congestion,
            "poa":                game.price_of_anarchy,
            "contraction_rate_k": game.contraction_rate,
            "strategic_type":     "complements" if game.beta_5 > 0 else "substitutes",
            "n_trials":           N_TRIALS,
            "n_converged":        n_converged,
            "mean_iters":         mean_iters,
            "mean_error_to_nash": mean_error,
            "poa_simulated_mean": mean_poa_sim,
        }
        for eps in EPSILONS:
            lm = game.mediator_max_lambda(eps)
            row[f"lambda_max_eps_{eps}"] = lm
            row[f"welfare_recovery_at_lambda_max_eps_{eps}"]  = game.welfare_recovery(lm)
            row[f"view_recovery_at_lambda_max_eps_{eps}"]     = game.view_welfare_recovery(lm)   # Task 3

        summary_rows.append(row)

        print(f"\nRegime {key}: {game.name}")
        print(f"  x_Nash={game.x_nash:.4f}, x_SO={game.x_so:.4f}, PoA={game.price_of_anarchy:.4f}")
        print(f"  Converged: {n_converged}/{N_TRIALS}, Mean iters: {mean_iters:.1f}")
        print(f"  Simulated PoA: {mean_poa_sim:.4f} (theory: {game.price_of_anarchy:.4f})")
        for eps in EPSILONS:
            lm = game.mediator_max_lambda(eps)
            rv = game.view_welfare_recovery(lm)
            rw = game.welfare_recovery(lm)
            print(f"  eps={eps}: lambda_max={lm:.4f}, R(log-view)={rw:.4f}, R_V(view)={rv:.4f}")

    save_regime_summary_csv(summary_rows, os.path.join(RESULTS_DIR, "regime_comparison_summary.csv"))

    # --- Generate plots 01-04 ---
    print("\nGenerating plots...")
    plot_01_br_convergence(dynamics_data, PLOTS_DIR)
    plot_02_welfare_trajectory(dynamics_data, PLOTS_DIR)
    plot_03_mediator_sweep(sweep_data, PLOTS_DIR)
    plot_04_regime_comparison(summary_rows, PLOTS_DIR)

    # --- Learning experiment (Task 1, --with-learning) ---
    if args.with_learning:
        print("\nRunning RQ2 learning experiment...")
        from sim_learning import (
            run_learning_regime, LEARNING_CONFIG,
            save_learning_csv, save_learning_trajectories_csv,
        )

        learning_traj_data = {}
        for key, game in games.items():
            label_map = {"A": "A_empirical"}
            lbl = label_map[key]
            # Main run 1: sigma = 0.05
            print(f"  Regime {key}: {game.name} (sigma=0.05)")
            summaries, trajs = run_learning_regime(
                beta_1=game.beta_1, beta_2=game.beta_2,
                beta_3=game.beta_3, beta_5=game.beta_5,
                x_nash=game.x_nash, x_so=game.x_so,
                n=game.n_creators, config=dict(LEARNING_CONFIG, sigma=0.05, init="both"),
                seed_base=SEED_BASE,
            )
            save_learning_csv(summaries, os.path.join(RESULTS_DIR, f"learning_{lbl}.csv"))
            save_learning_trajectories_csv(trajs, os.path.join(RESULTS_DIR, f"learning_trajectories_{lbl}.csv"))
            learning_traj_data[key] = (game, trajs)

            # Main run 2: sigma = 1.14 (empirical payoff RMSE)
            print(f"  Regime {key}: {game.name} (sigma=1.14)")
            summaries_high, trajs_high = run_learning_regime(
                beta_1=game.beta_1, beta_2=game.beta_2,
                beta_3=game.beta_3, beta_5=game.beta_5,
                x_nash=game.x_nash, x_so=game.x_so,
                n=game.n_creators, config=dict(LEARNING_CONFIG, sigma=1.14, init="both"),
                seed_base=SEED_BASE,
            )
            save_learning_csv(summaries_high, os.path.join(RESULTS_DIR, f"learning_{lbl}_sigma_1.14.csv"))
            save_learning_trajectories_csv(trajs_high, os.path.join(RESULTS_DIR, f"learning_trajectories_{lbl}_sigma_1.14.csv"))

        print("\nGenerating learning plots (05, 06, 07)...")
        plot_05_learning_convergence(learning_traj_data, PLOTS_DIR)
        plot_06_learning_welfare(learning_traj_data, PLOTS_DIR)
        plot_07_learning_dispersion(learning_traj_data, PLOTS_DIR)

    # --- Bootstrap uncertainty (Part B, --with-bootstrap) ---
    if args.with_bootstrap:
        print("\nRunning parametric bootstrap uncertainty analysis...")
        from sim_bootstrap import run_bootstrap_simulation
        run_bootstrap_simulation(
            n_draws=10000, seed=SEED_BASE,
            json_path=JSON_PATH,
            results_dir=RESULTS_DIR,
            plots_dir=PLOTS_DIR,
        )

    # --- Parameter sweeps (Task G, --with-sweeps) ---
    if args.with_sweeps:
        print("\nRunning Task G parameter sweeps...")
        from sim_sweeps import main as run_sweeps_main
        run_sweeps_main()

    print("\n" + "=" * 60)
    print("All experiments and plots complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
