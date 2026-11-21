"""
Bootstrap Uncertainty for Mediator Sweep and BR Dynamics
CS 329: Foundations of AI: Multi-Agent Systems (2026)
Indian Institute of Technology Gandhinagar

Generates 10,000 parametric bootstrap draws from the cluster-robust parameter
covariance matrix of Model 5 (uncentered).
Evaluates numeric mediator metrics and BR dynamics iterations for all well-posed draws.
Produces:
    - results/bootstrap_draws.csv
    - results/bootstrap_mediator_summary.csv
    - plots/08_bootstrap_mediator.png
"""

import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_GT_SCRIPTS = os.path.join(_ROOT, "game_theoretic_formulation", "scripts")
for p in [_GT_SCRIPTS, _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)

from run_game_theoretic_analysis import load_empirical_inputs  # noqa: E402
from sim_dynamics import make_regime, run_br_dynamics          # noqa: E402
from sim_numeric import (                                      # noqa: E402
    make_payoff_fn, make_sym_welfare_fn,
    numeric_nash, numeric_so,
    numeric_R, numeric_view_recovery,
    numeric_lambda_max,
)
from sim_mediator import EPSILONS                              # noqa: E402

JSON_PATH = os.path.join(_ROOT, "payoff_calculation", "results", "game_theory_coefficients.json")
if not os.path.exists(JSON_PATH):
    JSON_PATH = os.path.join(_ROOT, "payoff_estimation", "results", "game_theory_coefficients.json")
RESULTS_DIR = os.path.join(_HERE, "..", "results")
PLOTS_DIR = os.path.join(_HERE, "..", "plots")
DPI = 300


# ---------------------------------------------------------------------------
# Single draw metric computer (shared with point-estimate verification)
# ---------------------------------------------------------------------------
def compute_draw_metrics(b1: float, b2: float, b3: float, b5: float,
                         n_creators: int = 38, seed: int = 42,
                         epsilons: list = None) -> dict:
    """
    Compute game-theoretic and mediator metrics for a given parameter tuple (b1, b2, b3, b5)
    using independent numeric helpers (no closed forms).
    """
    if epsilons is None:
        epsilons = EPSILONS

    game = make_regime("A", beta_1=b1, beta_2=b2, beta_3=b3, beta_5=b5, n_creators=n_creators)
    pfn = make_payoff_fn(b1, b2, b3, b5)
    wfn = make_sym_welfare_fn(b1, b2, b3, b5)

    k = abs(b5) / (2.0 * abs(b2))
    x_n = numeric_nash(pfn)
    x_s = numeric_so(wfn)
    delta = x_s - x_n
    marginal_E = b3 + b5 * x_n

    # PoA by definition with numeric welfare: exp(W(x_SO) - W(x_Nash))
    poa = float(np.exp(wfn(x_s) - wfn(x_n)))
    x_n_clipped = bool(x_n <= 1e-6 or x_n >= 1.0 - 1e-6)
    x_s_clipped = bool(x_s <= 1e-6 or x_s >= 1.0 - 1e-6)

    res = {
        "k": float(k),
        "x_nash": float(x_n),
        "x_so": float(x_s),
        "delta": float(delta),
        "marginal_E": float(marginal_E),
        "poa": float(poa),
        "x_nash_clipped": x_n_clipped,
        "x_so_clipped": x_s_clipped,
    }

    # Mediator metrics across epsilons
    for eps in epsilons:
        lam = numeric_lambda_max(pfn, x_n, x_s, eps)
        r_w = numeric_R(wfn, x_n, x_s, lam)
        r_v = numeric_view_recovery(wfn, x_n, x_s, lam)
        res[f"lambda_max_eps_{eps}"] = float(lam)
        res[f"welfare_recovery_at_lambda_max_eps_{eps}"] = float(r_w)
        res[f"view_recovery_at_lambda_max_eps_{eps}"] = float(r_v)

    # BR dynamics: 1 trial with given seed
    x0 = np.random.default_rng(seed).uniform(0.0, 1.0, size=n_creators)
    br_res = run_br_dynamics(game, x0, tol=1e-6, max_iters=500)
    res["n_iters"] = int(br_res["n_iters"])
    res["converged"] = bool(br_res["converged"])

    return res


# ---------------------------------------------------------------------------
# Bootstrap simulation
# ---------------------------------------------------------------------------
def run_bootstrap_simulation(n_draws: int = 10000, seed: int = 42,
                             json_path: str = JSON_PATH,
                             results_dir: str = RESULTS_DIR,
                             plots_dir: str = PLOTS_DIR):
    """
    Run parametric bootstrap over the cluster-robust parameter covariance.
    Outputs:
      results/bootstrap_draws.csv
      results/bootstrap_mediator_summary.csv
      plots/08_bootstrap_mediator.png
    """
    uncentered, cov_sub, _, _ = load_empirical_inputs(json_path)
    mean_vec = np.array([
        uncentered["beta_1_own_click_appeal"],
        uncentered["beta_2_quadratic_own_appeal"],
        uncentered["beta_3_competitor_click_appeal"],
        uncentered["beta_5_strategic_interaction"],
    ], dtype=float)

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean_vec, cov_sub, size=n_draws)

    b1_all = draws[:, 0]
    b2_all = draws[:, 1]
    b3_all = draws[:, 2]
    b5_all = draws[:, 3]

    # Well-posedness conditions (Proposal Eq. 3)
    cond_b2_neg = b2_all < 0.0
    cond_B_pos = (np.abs(b2_all) - b5_all) > 0.0
    cond_contraction = np.abs(b5_all) < (2.0 * np.abs(b2_all))
    cond_well_posed = cond_b2_neg & cond_B_pos & cond_contraction

    n_valid = int(np.sum(cond_well_posed))
    print(f"Bootstrap draws: {n_draws} total, {n_valid} well-posed ({n_valid/n_draws*100:.2f}%)")

    draw_rows = []
    well_posed_rows = []

    for i in range(n_draws):
        b1, b2, b3, b5 = float(b1_all[i]), float(b2_all[i]), float(b3_all[i]), float(b5_all[i])
        wp = bool(cond_well_posed[i])

        row = {
            "draw_idx": i,
            "beta_1": b1,
            "beta_2": b2,
            "beta_3": b3,
            "beta_5": b5,
            "well_posed": wp,
            "cond_b2_neg": bool(cond_b2_neg[i]),
            "cond_B_pos": bool(cond_B_pos[i]),
            "cond_contraction": bool(cond_contraction[i]),
        }

        if wp:
            m = compute_draw_metrics(b1, b2, b3, b5, seed=seed + i)
            row.update(m)
            well_posed_rows.append(row)
        else:
            row.update({
                "k": float("nan"),
                "x_nash": float("nan"),
                "x_so": float("nan"),
                "delta": float("nan"),
                "marginal_E": float("nan"),
                "poa": float("nan"),
                "x_nash_clipped": False,
                "x_so_clipped": False,
                "n_iters": float("nan"),
                "converged": False,
            })
            for eps in EPSILONS:
                row[f"lambda_max_eps_{eps}"] = float("nan")
                row[f"welfare_recovery_at_lambda_max_eps_{eps}"] = float("nan")
                row[f"view_recovery_at_lambda_max_eps_{eps}"] = float("nan")

        draw_rows.append(row)

    # -----------------------------------------------------------------------
    # Compute metrics for ALL contraction-valid draws (including non-well-posed)
    # -----------------------------------------------------------------------
    cond_contraction_valid = cond_b2_neg & cond_contraction  # beta_2 < 0 AND |beta_5| < 2|beta_2|, NO B>0 requirement
    all_cv_rows = []  # metrics for every contraction-valid draw (superset of well-posed)
    for i in range(n_draws):
        if cond_contraction_valid[i]:
            if cond_well_posed[i]:
                # Already computed — reuse from draw_rows
                all_cv_rows.append(draw_rows[i])
            else:
                # Contraction-valid but NOT well-posed (B <= 0): compute metrics now
                b1, b2, b3, b5 = float(b1_all[i]), float(b2_all[i]), float(b3_all[i]), float(b5_all[i])
                m = compute_draw_metrics(b1, b2, b3, b5, seed=seed + i)
                cv_row = dict(draw_rows[i])
                cv_row.update(m)
                all_cv_rows.append(cv_row)

    n_cv = int(np.sum(cond_contraction_valid))
    print(f"  Contraction-valid draws: {n_cv} ({n_cv/n_draws*100:.2f}%)")

    # -----------------------------------------------------------------------
    # Save results/bootstrap_draws.csv
    # -----------------------------------------------------------------------
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        draws_csv = os.path.join(results_dir, "bootstrap_draws.csv")
        with open(draws_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(draw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(draw_rows)
        print(f"  Saved: {draws_csv}")

    # -----------------------------------------------------------------------
    # Summary stats over well-posed draws
    # -----------------------------------------------------------------------
    metrics_to_summarize = [
        "k", "x_nash", "x_so", "delta", "marginal_E", "poa",
    ]
    for eps in EPSILONS:
        metrics_to_summarize.extend([
            f"lambda_max_eps_{eps}",
            f"welfare_recovery_at_lambda_max_eps_{eps}",
            f"view_recovery_at_lambda_max_eps_{eps}",
        ])

    summary_rows = []
    cv_converged_rows = [r for r in all_cv_rows if r.get("converged", False)]
    n_cv_converged = len(cv_converged_rows)

    for m in metrics_to_summarize:
        arr_wp = np.array([r[m] for r in well_posed_rows], dtype=float)
        arr_cv = np.array([r[m] for r in all_cv_rows], dtype=float)
        summary_rows.append({
            "metric": m,
            # Conditional on well-posed (lower tail of beta_5; the point estimate itself is excluded)
            "mean": float(np.mean(arr_wp)),
            "std": float(np.std(arr_wp)),
            "p2_5": float(np.percentile(arr_wp, 2.5)),
            "p25": float(np.percentile(arr_wp, 25.0)),
            "median": float(np.median(arr_wp)),
            "p75": float(np.percentile(arr_wp, 75.0)),
            "p97_5": float(np.percentile(arr_wp, 97.5)),
            "n_valid": n_valid,
            # Contraction-valid summary (beta_2 < 0 AND |beta_5| < 2|beta_2|; point estimate belongs to this set)
            "mean_contraction_valid": float(np.mean(arr_cv)),
            "std_contraction_valid": float(np.std(arr_cv)),
            "p2_5_contraction_valid": float(np.percentile(arr_cv, 2.5)),
            "p25_contraction_valid": float(np.percentile(arr_cv, 25.0)),
            "median_contraction_valid": float(np.median(arr_cv)),
            "p75_contraction_valid": float(np.percentile(arr_cv, 75.0)),
            "p97_5_contraction_valid": float(np.percentile(arr_cv, 97.5)),
            "n_valid_contraction_valid": n_cv,
        })

    # BR dynamics n_iters summarized over converged draws only
    converged_rows = [r for r in well_posed_rows if r["converged"]]
    n_converged = len(converged_rows)
    frac_br_converged = float(n_converged / n_valid) if n_valid > 0 else 0.0

    if n_converged > 0:
        arr_iters = np.array([r["n_iters"] for r in converged_rows], dtype=float)
        arr_cv_iters = np.array([r["n_iters"] for r in cv_converged_rows], dtype=float) if n_cv_converged > 0 else np.array([float("nan")])
        summary_rows.append({
            "metric": "n_iters",
            "mean": float(np.mean(arr_iters)),
            "std": float(np.std(arr_iters)),
            "p2_5": float(np.percentile(arr_iters, 2.5)),
            "p25": float(np.percentile(arr_iters, 25.0)),
            "median": float(np.median(arr_iters)),
            "p75": float(np.percentile(arr_iters, 75.0)),
            "p97_5": float(np.percentile(arr_iters, 97.5)),
            "n_valid": n_converged,
            "mean_contraction_valid": float(np.mean(arr_cv_iters)),
            "std_contraction_valid": float(np.std(arr_cv_iters)),
            "p2_5_contraction_valid": float(np.percentile(arr_cv_iters, 2.5)),
            "p25_contraction_valid": float(np.percentile(arr_cv_iters, 25.0)),
            "median_contraction_valid": float(np.median(arr_cv_iters)),
            "p75_contraction_valid": float(np.percentile(arr_cv_iters, 75.0)),
            "p97_5_contraction_valid": float(np.percentile(arr_cv_iters, 97.5)),
            "n_valid_contraction_valid": n_cv_converged,
        })
    else:
        summary_rows.append({
            "metric": "n_iters",
            "mean": float("nan"), "std": float("nan"),
            "p2_5": float("nan"), "p25": float("nan"),
            "median": float("nan"), "p75": float("nan"),
            "p97_5": float("nan"), "n_valid": 0,
            "mean_contraction_valid": float("nan"), "std_contraction_valid": float("nan"),
            "p2_5_contraction_valid": float("nan"), "p25_contraction_valid": float("nan"),
            "median_contraction_valid": float("nan"), "p75_contraction_valid": float("nan"),
            "p97_5_contraction_valid": float("nan"), "n_valid_contraction_valid": 0,
        })

    # Summary diagnostic fractions
    so_clipped_wp = np.array([r["x_so_clipped"] for r in well_posed_rows], dtype=bool)
    e_neg_wp = np.array([r["marginal_E"] < 0 for r in well_posed_rows], dtype=bool)

    # All-draws computations for x_so, x_nash, E
    x_so_all = np.zeros(n_draws)
    x_nash_all = np.zeros(n_draws)
    for i in range(n_draws):
        b1, b2, b3, b5 = b1_all[i], b2_all[i], b3_all[i], b5_all[i]
        denom_so = 2.0 * (b2 + b5)
        if denom_so < -1e-12:
            x_s_unc = -(b1 + b3) / denom_so
            x_s = np.clip(x_s_unc, 0.0, 1.0)
        elif denom_so > 1e-12:
            w1 = (b1 + b3) + (b2 + b5)
            x_s = 1.0 if w1 > 0.0 else 0.0
        else:
            x_s = 1.0 if (b1 + b3) > 0.0 else 0.0
        x_so_all[i] = x_s

        if b2 < -1e-12:
            denom_n = 2.0 * abs(b2) - b5
            if abs(denom_n) > 1e-12:
                x_n_unc = b1 / denom_n
                x_n = np.clip(x_n_unc, 0.0, 1.0)
            else:
                x_n = 1.0 if b1 > 0 else 0.0
        else:
            diff1 = b1 + b2 + b5
            diff0 = b1 + b2
            if diff1 > 0 and diff0 > 0:
                x_n = 1.0
            elif diff1 < 0 and diff0 < 0:
                x_n = 0.0
            else:
                x_n = 1.0 if diff1 > 0 else 0.0
        x_nash_all[i] = x_n

    e_all = b3_all + b5_all * x_nash_all
    so_clipped_all = (x_so_all <= 1e-6) | (x_so_all >= 1.0 - 1e-6)
    nash_clipped_all = (x_nash_all <= 1e-6) | (x_nash_all >= 1.0 - 1e-6)

    def _diag_row(metric_name, val, n_v):
        return {
            "metric": metric_name, "mean": float(val), "std": 0.0,
            "p2_5": float(val), "p25": float(val),
            "median": float(val), "p75": float(val),
            "p97_5": float(val), "n_valid": n_v,
            "mean_contraction_valid": float(val), "std_contraction_valid": 0.0,
            "p2_5_contraction_valid": float(val), "p25_contraction_valid": float(val),
            "median_contraction_valid": float(val), "p75_contraction_valid": float(val),
            "p97_5_contraction_valid": float(val), "n_valid_contraction_valid": n_v,
        }

    diagnostic_fractions = [
        _diag_row("fraction_not_well_posed", 1.0 - (n_valid / n_draws), n_draws),
        _diag_row("fraction_non_contractive", np.mean(~cond_contraction), n_draws),
        _diag_row("fraction_x_so_clipped_among_well_posed", np.mean(so_clipped_wp), n_valid),
        _diag_row("fraction_E_negative_among_well_posed", np.mean(e_neg_wp), n_valid),
        _diag_row("fraction_br_converged", frac_br_converged, n_valid),
        # All-draws probabilities (Task C)
        _diag_row("P_b2_neg_all", np.mean(cond_b2_neg), n_draws),
        _diag_row("P_B_pos_all", np.mean(cond_B_pos), n_draws),
        _diag_row("P_contraction_all", np.mean(cond_contraction), n_draws),
        _diag_row("P_well_posed_all", np.mean(cond_well_posed), n_draws),
        _diag_row("P_E_neg_all", np.mean(e_all < 0), n_draws),
        _diag_row("P_x_so_clipped_all", np.mean(so_clipped_all), n_draws),
        _diag_row("P_x_nash_clipped_all", np.mean(nash_clipped_all), n_draws),
    ]
    summary_rows.extend(diagnostic_fractions)

    if results_dir:
        summary_csv = os.path.join(results_dir, "bootstrap_mediator_summary.csv")
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Saved: {summary_csv}")

    # -----------------------------------------------------------------------
    # Plot 08 — Bootstrap Mediator Uncertainty (plots/08_bootstrap_mediator.png)
    # -----------------------------------------------------------------------
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        # Point estimate metrics
        pt = compute_draw_metrics(mean_vec[0], mean_vec[1], mean_vec[2], mean_vec[3])

        lam_vals = [r["lambda_max_eps_0.01"] for r in well_posed_rows]
        r_vals = [r["welfare_recovery_at_lambda_max_eps_0.01"] for r in well_posed_rows]
        rv_vals = [r["view_recovery_at_lambda_max_eps_0.01"] for r in well_posed_rows]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(
            r"Bootstrap Uncertainty Distribution for Mediator Intervention ($\epsilon=0.01$)" "\n"
            f"($N={n_valid}$ Well-Posed Draws from Cluster-Robust Covariance)",
            fontsize=13, fontweight="bold",
        )

        panels = [
            (axes[0], lam_vals, pt["lambda_max_eps_0.01"], r"$\lambda_{\max}(\epsilon=0.01)$", "#2b5c8f"),
            (axes[1], r_vals, pt["welfare_recovery_at_lambda_max_eps_0.01"], r"Log-View Recovery $R(\lambda_{\max})$", "#3d8b37"),
            (axes[2], rv_vals, pt["view_recovery_at_lambda_max_eps_0.01"], r"View Recovery $R_V(\lambda_{\max})$", "#d95f02"),
        ]

        for ax, data, pt_val, title, col in panels:
            p2_5 = np.percentile(data, 2.5)
            p97_5 = np.percentile(data, 97.5)
            med = np.median(data)

            ax.hist(data, bins=40, color=col, alpha=0.7, edgecolor="white", density=True)
            ax.axvline(pt_val, color="red", linestyle="--", linewidth=2.0,
                       label=f"Point Est: {pt_val:.3f}")
            ax.axvline(med, color="black", linestyle="-", linewidth=1.5,
                       label=f"Median: {med:.3f}")
            ax.axvspan(p2_5, p97_5, color=col, alpha=0.15,
                       label=f"95% CI: [{p2_5:.3f}, {p97_5:.3f}]")

            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Value", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.legend(fontsize=8.5, loc="upper right")
            ax.grid(alpha=0.25)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "08_bootstrap_mediator.png")
        plt.savefig(plot_path, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {plot_path}")

    return draw_rows, summary_rows


if __name__ == "__main__":
    run_bootstrap_simulation()
