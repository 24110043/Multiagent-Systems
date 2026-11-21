"""
Simulation: Best-Response Dynamics & Welfare Trajectory
CS 329: Foundations of AI: Multi-Agent Systems (2026)
Indian Institute of Technology Gandhinagar

Experiment 1 — Best-Response Dynamics (Convergence):
  - n = 38 agents iterate synchronous best-response from random starting profiles.
  - Guarantees convergence to x_Nash via contraction mapping (k < 1).
  - Records: n_iter, final mean x, error to x_Nash, convergence trajectory.

Experiment 2 — Welfare Trajectory & PoA Verification:
  - Tracks per-iteration symmetric welfare π(x_t, x_t) alongside error.
  - Verifies that simulated PoA matches closed-form theory.

Usage:
    python simulations/scripts/sim_dynamics.py
Outputs:
    simulations/results/dynamics_regime_A_empirical.csv
    simulations/results/dynamics_regime_B_hypothetical.csv
    simulations/results/dynamics_regime_Bprime_hypothetical.csv
    (also exposes run_br_dynamics() for the plot script and test suite)
"""

import sys
import os
import csv
import json
import warnings
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow import of PotentialGame from Part 3
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
from sim_numeric import make_payoff_fn, numeric_nash  # noqa: E402

# ---------------------------------------------------------------------------
# Constants & Factory
# ---------------------------------------------------------------------------
JSON_PATH = os.path.join(_ROOT, "payoff_calculation", "results", "game_theory_coefficients.json")
if not os.path.exists(JSON_PATH):
    JSON_PATH = os.path.join(_ROOT, "payoff_estimation", "results", "game_theory_coefficients.json")
RESULTS_DIR = os.path.join(_HERE, "..", "results")
N_TRIALS = 100          # random starting profiles per regime
MAX_ITERS = 500         # maximum BR iteration steps
TOL = 1e-6              # convergence tolerance: max_i |x^{t+1}_i - x^t_i|
N_CREATORS = 38

_EMPIRICAL_GAME = None


def _get_empirical_game() -> PotentialGame:
    global _EMPIRICAL_GAME
    if _EMPIRICAL_GAME is None:
        _EMPIRICAL_GAME = PotentialGame.from_json(JSON_PATH, n_creators=N_CREATORS)
    return _EMPIRICAL_GAME


def make_regime(base_key: str, **overrides) -> PotentialGame:
    """
    Factory for PotentialGame instances for regimes 'A', 'B', 'Bprime'.

    Accepts overrides for: beta_1, beta_2, beta_3, beta_5, n_creators, name.
    Raises ValueError unless:
      - beta_2 < 0 (individual concavity)
      - |beta_5| < 2 * |beta_2| (contraction condition)
    Warns when x_nash or x_so is clipped to [0, 1] or welfare concavity is violated.
    """
    base_game = _get_empirical_game()
    key = base_key.strip()

    if key in ("A", "A_empirical"):
        b1 = base_game.beta_1
        b2 = base_game.beta_2
        b3 = base_game.beta_3
        b5 = base_game.beta_5
        name = "Calibrated (N=809, assumed kappa=8)"
    elif key in ("B", "B_hypothetical"):
        b1 = 8.301959
        b2 = -9.167702
        b3 = -3.802120
        b5 = 0.782244
        name = "Illustrative B (legacy N=186 coefficients)"
    elif key in ("Bprime", "B'", "Bprime_hypothetical", "B_prime"):
        b1 = 8.301959
        b2 = -9.167702
        b3 = -3.802120
        b5 = -0.782244
        name = "Illustrative B' (legacy N=186 coefficients)"
    else:
        raise ValueError(f"Unknown base_key '{base_key}'. Expected one of 'A', 'B', 'Bprime'.")

    n_creators = base_game.n_creators

    b1 = overrides.get("beta_1", b1)
    b2 = overrides.get("beta_2", b2)
    b3 = overrides.get("beta_3", b3)
    b5 = overrides.get("beta_5", b5)
    n_creators = overrides.get("n_creators", n_creators)
    name = overrides.get("name", name)

    abs_b2 = abs(b2)
    if not (b2 < 0.0 and abs(b5) < (2.0 * abs_b2)):
        raise ValueError(
            f"Game not well-posed under conditions (beta_2 < 0, |beta_5| < 2|beta_2|): "
            f"beta_2={b2}, beta_5={b5}"
        )
    if (b2 + b5) >= 0.0:
        warnings.warn(
            f"Welfare concavity violated (beta_2 + beta_5 = {b2 + b5:.4f} >= 0). Social welfare is convex; x_so is boundary maximizer.",
            UserWarning,
            stacklevel=2,
        )

    game = PotentialGame(
        beta_1=b1,
        beta_2=b2,
        beta_3=b3,
        beta_5=b5,
        n_creators=n_creators,
        name=name,
    )

    if (game.x_nash_unclipped < 0.0 or game.x_nash_unclipped > 1.0 or
            game.x_so_unclipped < 0.0 or game.x_so_unclipped > 1.0):
        warnings.warn(
            f"Equilibrium clipped in regime '{base_key}': "
            f"x_nash_unclipped={game.x_nash_unclipped:.4f}, "
            f"x_so_unclipped={game.x_so_unclipped:.4f}",
            UserWarning,
            stacklevel=2,
        )

    return game


# ---------------------------------------------------------------------------
# Core BR simulation
# ---------------------------------------------------------------------------
def run_br_dynamics(
    game: PotentialGame,
    x0: np.ndarray,
    max_iters: int = MAX_ITERS,
    tol: float = TOL,
):
    """
    Run synchronous best-response dynamics for one starting profile x0.

    Parameters
    ----------
    game      : PotentialGame instance
    x0        : (n,) initial profile, values in [0, 1]
    max_iters : maximum iterations
    tol       : convergence threshold on max_i |Δx_i|

    Returns
    -------
    dict with keys:
        'converged'       : bool
        'n_iters'         : int — iterations to convergence (or max_iters)
        'x_final'         : (n,) final profile
        'mean_x_final'    : float
        'error_to_nash'   : float — |mean(x_final) - x_nash|
        'error_trajectory': list[float] — ‖x_t - x_nash‖_inf at each t
        'welfare_trajectory': list[float] — symmetric welfare π(mean(x_t), mean(x_t)) at each t
    """
    x = x0.copy().astype(float)
    n = len(x)
    x_nash_target = game.x_nash

    error_traj = []
    welfare_traj = []

    for t in range(max_iters):
        # Record pre-update state
        mean_x = np.mean(x)
        err = float(np.max(np.abs(x - x_nash_target)))
        error_traj.append(err)
        welfare_traj.append(float(game.social_welfare_symmetric(mean_x)))

        # Synchronous update: each agent best-responds to current x_bar_{-i}
        # In symmetric population x_bar_{-i} = (sum_j x_j - x_i) / (n-1)
        sum_x = np.sum(x)
        x_bar_minus_i = (sum_x - x) / (n - 1)   # shape (n,)
        x_new = game.best_response(x_bar_minus_i, clip=True)

        # Check convergence
        delta = np.max(np.abs(x_new - x))
        x = x_new

        if delta < tol:
            # Record final state
            mean_x_f = np.mean(x)
            error_traj.append(float(np.max(np.abs(x - x_nash_target))))
            welfare_traj.append(float(game.social_welfare_symmetric(mean_x_f)))
            return {
                "converged": True,
                "n_iters": t + 1,
                "x_final": x,
                "mean_x_final": float(np.mean(x)),
                "error_to_nash": float(abs(np.mean(x) - x_nash_target)),
                "error_trajectory": error_traj,
                "welfare_trajectory": welfare_traj,
            }

    return {
        "converged": False,
        "n_iters": max_iters,
        "x_final": x,
        "mean_x_final": float(np.mean(x)),
        "error_to_nash": float(abs(np.mean(x) - x_nash_target)),
        "error_trajectory": error_traj,
        "welfare_trajectory": welfare_traj,
    }


def simulate_regime(game: PotentialGame, n_trials: int = N_TRIALS, seed_base: int = 42):
    """
    Run n_trials random-start BR simulations for one PotentialGame.

    Returns list of result dicts (one per trial) with added fields:
        'trial_seed', 'x_so', 'welfare_nash', 'welfare_so', 'poa_simulated',
        'poa_from_numeric_nash' (numeric verification, independent of closed forms)
    """
    # Lazily import numeric helper to avoid circular import at module level
    from sim_numeric import make_payoff_fn, numeric_nash as _numeric_nash

    n = game.n_creators  # use game's own n, not module-level constant
    results = []
    x_so = game.x_so
    w_nash_theory = game.social_welfare_symmetric(game.x_nash)
    w_so_theory = game.social_welfare_symmetric(x_so)
    poa_theory = game.price_of_anarchy

    # Compute numeric Nash once per regime (independent check)
    _pfn = make_payoff_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)
    _x_nash_num = _numeric_nash(_pfn)

    for trial in range(n_trials):
        rng = np.random.default_rng(seed_base + trial)
        x0 = rng.uniform(0.0, 1.0, size=n)
        result = run_br_dynamics(game, x0)

        # PoA from simulated equilibrium (using closed-form x_so as reference)
        w_nash_sim = game.social_welfare_symmetric(result["mean_x_final"])
        w_so_sim = game.social_welfare_symmetric(x_so)
        poa_sim = float(np.exp(w_so_sim - w_nash_sim))

        # PoA using numeric Nash (independent verification, appended column)
        w_nash_num = game.social_welfare_symmetric(_x_nash_num)
        poa_from_numeric_nash = float(np.exp(w_so_sim - w_nash_num))

        result.update(
            {
                "trial_seed": seed_base + trial,
                "x_nash_theory": game.x_nash,
                "x_so_theory": x_so,
                "welfare_nash_theory": w_nash_theory,
                "welfare_so_theory": w_so_theory,
                "poa_theory": poa_theory,
                "poa_simulated": poa_sim,
                "poa_from_numeric_nash": poa_from_numeric_nash,  # NEW column
            }
        )
        results.append(result)

    return results


def save_results_csv(results, filepath: str):
    """Save scalar summary (one row per trial) to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    scalar_keys = [
        "trial_seed", "converged", "n_iters", "mean_x_final",
        "error_to_nash", "x_nash_theory", "x_so_theory",
        "welfare_nash_theory", "welfare_so_theory",
        "poa_theory", "poa_simulated",
        "poa_from_numeric_nash",   # NEW — appended column
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in scalar_keys})
    print(f"  Saved: {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Experiment 1 & 2: Best-Response Dynamics & Welfare")
    print("=" * 60)

    # --- Construct regimes via factory ---
    game_A = make_regime("A")
    game_B = make_regime("B")
    game_Bp = make_regime("Bprime")

    regimes = [
        ("A_empirical", game_A),
        ("B_hypothetical", game_B),
        ("Bprime_hypothetical", game_Bp),
    ]

    for label, game in regimes:
        print(f"\nRegime {label}: x_Nash={game.x_nash:.4f}, x_SO={game.x_so:.4f}, "
              f"PoA={game.price_of_anarchy:.4f}, k={game.contraction_rate:.4f}")
        results = simulate_regime(game, n_trials=N_TRIALS)

        n_converged = sum(1 for r in results if r["converged"])
        mean_iters = np.mean([r["n_iters"] for r in results])
        mean_error = np.mean([r["error_to_nash"] for r in results])
        poa_sim_mean = np.mean([r["poa_simulated"] for r in results])

        print(f"  Converged:         {n_converged}/{N_TRIALS}")
        print(f"  Mean iterations:   {mean_iters:.1f}")
        print(f"  Mean error to Nash:{mean_error:.6f}")
        print(f"  Simulated PoA:     {poa_sim_mean:.6f}  (theory: {results[0]['poa_theory']:.6f})")

        outfile = os.path.join(RESULTS_DIR, f"dynamics_regime_{label}.csv")
        save_results_csv(results, outfile)

    print("\nDone.")


if __name__ == "__main__":
    main()
