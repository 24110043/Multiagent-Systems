"""
Simulation: Mediator Intervention Sweep
CS 329: Foundations of AI: Multi-Agent Systems (2026)
Indian Institute of Technology Gandhinagar

Experiment 3 — Platform Mediator λ-Sweep:
  For λ ∈ [0, 1] (50 steps):
    x_λ = (1-λ) · x_Nash + λ · x_SO
    G(λ) = deviation gain (incentive-compatibility measure)
    R(λ) = 2λ − λ²     (welfare recovery fraction)
    R_V(λ) = view recovery
  Marks ε-IC thresholds λ_max for ε ∈ {0.001, 0.01, 0.05, 0.10}.

Runs for all three regimes (A, B, B').

Usage:
    python simulations/scripts/sim_mediator.py
Outputs:
    simulations/results/mediator_sweep.csv
"""

import sys
import os
import csv
import numpy as np
import warnings

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_GT_SCRIPTS = os.path.join(_ROOT, "game_theoretic_formulation", "scripts")
if _GT_SCRIPTS not in sys.path:
    sys.path.insert(0, _GT_SCRIPTS)
_SIM_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SIM_SCRIPTS not in sys.path:
    sys.path.insert(0, _SIM_SCRIPTS)

from game_theory_solver import PotentialGame  # noqa: E402
from sim_dynamics import make_regime          # noqa: E402
from sim_numeric import (                      # noqa: E402
    make_payoff_fn, make_sym_welfare_fn,
    numeric_nash, numeric_so,
    numeric_G, numeric_R, numeric_view_recovery,
    numeric_lambda_max,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
JSON_PATH = os.path.join(_ROOT, "payoff_calculation", "results", "game_theory_coefficients.json")
if not os.path.exists(JSON_PATH):
    JSON_PATH = os.path.join(_ROOT, "payoff_estimation", "results", "game_theory_coefficients.json")
RESULTS_DIR = os.path.join(_HERE, "..", "results")
N_LAMBDA = 50
EPSILONS = [0.001, 0.01, 0.05, 0.10]


# ---------------------------------------------------------------------------
# Mediator sweep
# ---------------------------------------------------------------------------
def mediator_sweep(game: PotentialGame, n_lambda: int = N_LAMBDA):
    """
    Sweep λ from 0 to 1 and compute mediator quantities.

    Existing columns are preserved exactly (< 1e-12 regression tolerance).
    New appended columns (Task 2):
      welfare_recovery_numeric  — R_num(λ) = (W(x_λ)-W(x_N))/(W(x_SO)-W(x_N))
      view_recovery_numeric     — R_V_num(λ) = view analogue of R_num

    If x_so_unclipped or x_nash_unclipped is outside (0,1), warns that
    R=2λ-λ² is not guaranteed and relies on numeric R.

    Returns list of dicts, one per λ value.
    """
    # Build numeric helpers (independent of closed forms)
    pfn = make_payoff_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)
    wfn = make_sym_welfare_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)

    # Warn if boundary clipping applies
    x_nash_unc = game.x_nash_unclipped
    x_so_unc   = game.x_so_unclipped
    if not (0.0 < x_nash_unc < 1.0) or not (0.0 < x_so_unc < 1.0):
        import warnings
        warnings.warn(
            f"[mediator_sweep] Equilibria at boundary (x_nash_unc={x_nash_unc:.4f}, "
            f"x_so_unc={x_so_unc:.4f}). R=2λ-λ² not guaranteed; using numeric R.",
            stacklevel=2,
        )

    # Use closed-form equilibria (clipped) as the reference points for mediator
    x_n = game.x_nash
    x_s = game.x_so

    lambdas = np.linspace(0.0, 1.0, n_lambda)
    rows = []

    # Pre-compute lambda_max for each epsilon
    lam_max_dict = {eps: game.mediator_max_lambda(eps) for eps in EPSILONS}

    for lam in lambdas:
        x_rec = game.mediator_recommendation(lam)
        G_def = float(game.mediator_deviation_gain(lam))
        G_cf_raw = float(game.mediator_deviation_gain_closed_form(lam))
        # Task B: NaN unless interior equilibria AND concave welfare
        equilibria_interior = (0.0 < game.x_nash_unclipped < 1.0) and (0.0 < game.x_so_unclipped < 1.0)
        welfare_concave = (game.beta_2 + game.beta_5) < 0
        G_cf = G_cf_raw if (equilibria_interior and welfare_concave) else float("nan")
        R     = float(game.welfare_recovery(lam))
        R_V   = float(game.view_welfare_recovery(lam))

        # IC flag for each epsilon: G(λ) ≤ ε
        ic_flags = {eps: bool(G_def <= eps) for eps in EPSILONS}

        # Numeric welfare recovery (new appended columns — independent check)
        R_num   = float(numeric_R(wfn, x_n, x_s, lam))
        R_V_num = float(numeric_view_recovery(wfn, x_n, x_s, lam))

        row = {
            # --- Existing columns (order preserved) ---
            "regime": game.name,
            "lambda": float(lam),
            "x_recommendation": float(x_rec),
            "deviation_gain_definition": G_def,
            "deviation_gain_closed_form": G_cf,
            "welfare_recovery_R": R,
            "view_recovery_R_V": R_V,
            "x_nash": x_n,
            "x_so": x_s,
            "poa": game.price_of_anarchy,
            "delta": game.nash_so_gap,
        }
        for eps in EPSILONS:
            row[f"ic_at_eps_{eps}"]     = ic_flags[eps]
            row[f"lambda_max_eps_{eps}"] = lam_max_dict[eps]

        # --- New appended columns (Task 2) ---
        row["welfare_recovery_numeric"]  = R_num
        row["view_recovery_numeric"]     = R_V_num
        row["welfare_concave"]           = welfare_concave
        row["equilibria_interior"]       = equilibria_interior

        rows.append(row)

    return rows


def save_mediator_csv(all_rows, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not all_rows:
        return
    keys = list(all_rows[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Saved: {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Experiment 3: Mediator Intervention Sweep")
    print("=" * 60)

    game_A = make_regime("A")

    all_rows = []
    for game in [game_A]:
        rows = mediator_sweep(game)
        all_rows.extend(rows)

        # Print IC thresholds
        print(f"\nRegime: {game.name}")
        print(f"  x_Nash={game.x_nash:.4f}, x_SO={game.x_so:.4f}, Delta={game.nash_so_gap:.4f}")
        for eps in EPSILONS:
            lm = game.mediator_max_lambda(eps)
            R_at_lm = game.welfare_recovery(lm)
            print(f"  eps={eps}: lambda_max={lm:.4f}, R(lambda_max)={R_at_lm:.4f}")

    outfile = os.path.join(RESULTS_DIR, "mediator_sweep.csv")
    save_mediator_csv(all_rows, outfile)
    print("\nDone.")


if __name__ == "__main__":
    main()
