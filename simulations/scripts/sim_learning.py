"""
Simulation: RQ2 Payoff-Based Trial-and-Error Learning
CS 329: Foundations of AI: Multi-Agent Systems (2026)
Indian Institute of Technology Gandhinagar

RQ2 — Does decentralised payoff-based learning converge to the Nash equilibrium?

Algorithm (paper Eq. 16, payoff-based trial-and-error):
  Agents DO NOT know betas, best responses, or analytical formulas.
  They only observe their own noisy payoff each round.

  Initialisation (round t=0):
    x_i^0 ~ U[0,1], d_i^0 ~ Uniform({-1, +1})
    y_i^0 = payoff_fn(x_i^0, x_bar_{-i}^0) + N(0, sigma^2)
    (y^0 is the payoff at the round-0 profile; this is documented as the
     boundary condition so that the round-t=1 update has a valid comparison.)

  Update rule for t = 1, 2, ...:
    eta_t = eta / sqrt(t)
    d_i^t = d_i^{t-1}       if y_i^t >= y_i^{t-1}
           -d_i^{t-1}        otherwise
    x_i^{t+1} = clip(x_i^t + eta_t * d_i^t, 0, 1)
    y_i^{t+1} = payoff_fn(x_i^{t+1}, x_bar_{-i}^{t+1}) + N(0, sigma^2)

  Creators DO NOT receive x_bar; the environment evaluates the payoff callable.

Outputs (per regime):
  results/learning_<regime>.csv           — per-trial summary (one row per trial)
  results/learning_trajectories_<regime>.csv — per-round means (one row per round)
  plots/05_learning_convergence.png
  plots/06_learning_welfare.png

Usage:
    python simulations/scripts/sim_learning.py
Or wired into sim_regime_compare.py via --with-learning.
"""

import sys
import os
import csv
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_GT_SCRIPTS = os.path.join(_ROOT, "game_theoretic_formulation", "scripts")
if _GT_SCRIPTS not in sys.path:
    sys.path.insert(0, _GT_SCRIPTS)

from sim_numeric import make_payoff_fn  # noqa: E402
from sim_dynamics import make_regime    # noqa: E402

RESULTS_DIR = os.path.abspath(os.path.join(_HERE, "..", "results"))
DEFAULT_SWEEPS_DIR = os.path.join(RESULTS_DIR, "sweeps")

# ---------------------------------------------------------------------------
# LEARNING_CONFIG
# ---------------------------------------------------------------------------
LEARNING_CONFIG = {
    "eta": 0.1,
    "sigma": 0.05,
    "T_horizon": 500,
    "n_trials": 20,
    "delta": 0.05,
    "k_window": 50,
    "seed_base": 42,
    "init": "uniform",
}


# ---------------------------------------------------------------------------
# T_delta and t_first_hit helper (paper definition)
# ---------------------------------------------------------------------------
def compute_t_delta_and_first_hit(xbar_history, x_nash, delta_thr):
    """
    Computes (T_delta, t_first_hit) for a sequence of xbar values.

    t_first_hit: first round t (1-indexed) where |xbar_t - x_nash| < delta_thr; NaN if never.
    T_delta: min { t : |xbar_{t'} - x_nash| < delta_thr for ALL t' >= t up to T };
             NaN if not within delta_thr at round T.
    """
    t_first_hit = None
    for idx, xb in enumerate(xbar_history):
        if abs(xb - x_nash) < delta_thr:
            t_first_hit = idx + 1
            break
    t_first_hit = float(t_first_hit) if t_first_hit is not None else float("nan")

    if len(xbar_history) == 0 or abs(xbar_history[-1] - x_nash) >= delta_thr:
        T_delta = float("nan")
    else:
        earliest_idx = len(xbar_history) - 1
        for idx in range(len(xbar_history) - 1, -1, -1):
            if abs(xbar_history[idx] - x_nash) < delta_thr:
                earliest_idx = idx
            else:
                break
        T_delta = float(earliest_idx + 1)

    return T_delta, t_first_hit


# ---------------------------------------------------------------------------
# Payoff-based trial-and-error learner
# ---------------------------------------------------------------------------
def run_learning(
    payoff_fn,          # callable(x_i, x_bar_minus_i) -> float; no betas exposed
    x_nash: float,      # reference for metric computation only (not used by learner)
    x_so: float,        # reference for metric computation only (not used by learner)
    n: int = 38,
    config: dict = None,
    seed: int = 42,
):
    """
    Run one trial of payoff-based trial-and-error learning for n creators.

    Parameters
    ----------
    payoff_fn   : callable(x_i, x_bar) -> float
                  ONLY the environment uses this; creators never see betas.
    x_nash      : theoretical Nash (used ONLY for metric recording)
    x_so        : theoretical SO (used ONLY for metric recording)
    n           : number of creators
    config      : LEARNING_CONFIG dict
    seed        : RNG seed

    Returns
    -------
    summary : dict of scalar metrics
    traj    : list of per-round dicts
    """
    if config is None:
        config = LEARNING_CONFIG

    eta       = config["eta"]
    sigma     = config["sigma"]
    T         = config["T_horizon"]
    delta_thr = config["delta"]
    k_win     = config["k_window"]

    rng = np.random.default_rng(seed)

    # --- Initialisation (t=0) ---
    init = config.get("init", "uniform")
    if init == "low":
        x = rng.uniform(0.0, 0.3, size=n)
    elif init == "high":
        x = rng.uniform(0.7, 1.0, size=n)
    elif init == "uniform":
        x = rng.uniform(0.0, 1.0, size=n)
    else:
        raise ValueError(f"Invalid init mode '{init}'. Expected 'uniform', 'low', or 'high'.")
    d = rng.choice([-1.0, 1.0], size=n)                  # d_i^0

    # Evaluate noisy payoffs at round-0 profile (boundary condition)
    # x_bar_{-i}^0 = (sum(x) - x_i) / (n-1) for each i
    sum_x = np.sum(x)
    x_bar = (sum_x - x) / (n - 1)
    noise = rng.normal(0.0, sigma, size=n)
    y = np.array([payoff_fn(x[i], x_bar[i]) for i in range(n)]) + noise

    # Trajectory accumulators
    traj = []

    # Metrics
    xbar_history = []          # mean x over rounds 1..T
    total_reversals = 0

    for t in range(1, T + 1):
        # --- Compute step size ---
        eta_t = eta / np.sqrt(t)

        # --- Update: each creator plays x_i^t = clip(x_i^{t-1} + eta_t * d_i^{t-1}) ---
        x_new = np.clip(x + eta_t * d, 0.0, 1.0)

        # --- Environment evaluates payoffs (creators observe noisy payoff only) ---
        sum_x_new = np.sum(x_new)
        x_bar_new = (sum_x_new - x_new) / (n - 1)
        noise_new = rng.normal(0.0, sigma, size=n)
        y_new = np.array([payoff_fn(x_new[i], x_bar_new[i]) for i in range(n)]) + noise_new

        # --- Direction update: reverse if payoff did not improve ---
        improved = y_new >= y
        reversals_this_round = int(np.sum(~improved))
        d_new = np.where(improved, d, -d)
        total_reversals += reversals_this_round

        # --- Record pre-update metrics (state at start of round t) ---
        xbar_t = float(np.mean(x_new))
        err_t  = float(np.max(np.abs(x_new - x_nash)))   # sup-norm to x_nash

        # Noise-free welfare W_V = sum_i exp(pi_i(x_i, xbar_{-i}))
        x_bar_nf = (sum_x_new - x_new) / (n - 1)
        pi_nf = np.array([payoff_fn(x_new[i], x_bar_nf[i]) for i in range(n)])
        W_V_t = float(np.sum(np.exp(pi_nf)))

        xbar_history.append(xbar_t)

        traj.append({
            "t": t,
            "xbar_t": xbar_t,
            "err_sup_to_nash": err_t,
            "reversals_this_round": reversals_this_round,
            "W_V_t": W_V_t,
            "payoff_std_across_creators": float(np.std(y_new)),
        })

        # Update state
        x, d, y = x_new, d_new, y_new

    # --- Final metrics ---
    xbar_last_k = np.array(xbar_history[-k_win:])
    T_delta, t_first_hit = compute_t_delta_and_first_hit(xbar_history, x_nash, delta_thr)

    # Noise-free welfare at Nash and SO (for reference)
    x_bar_nash_nf = x_nash
    x_bar_so_nf   = x_so
    W_V_nash = float(n * np.exp(payoff_fn(x_nash, x_bar_nash_nf)))
    W_V_so   = float(n * np.exp(payoff_fn(x_so,   x_bar_so_nf)))
    W_V_T    = traj[-1]["W_V_t"] if traj else float("nan")

    summary = {
        "seed": seed,
        "eta": eta,
        "sigma": sigma,
        "T_horizon": T,
        "delta_thr": delta_thr,
        "k_window": k_win,
        # Paper metrics
        "Delta_T": float(abs(np.mean(x) - x_nash)),          # |xbar_T - x_nash|
        "D_T": float(np.max(np.abs(x - x_nash))),            # sup-norm at T
        "T_delta": T_delta,
        "S_T": float(np.var(xbar_last_k)),                    # variance of xbar in last k
        "total_reversals": total_reversals,
        # Welfare at T relative to Nash and SO
        "W_V_T": W_V_T,
        "W_V_nash": W_V_nash,
        "W_V_so": W_V_so,
        "W_V_T_rel_nash": W_V_T / W_V_nash if W_V_nash > 0 else float("nan"),
        "W_V_T_rel_so":  W_V_T / W_V_so  if W_V_so  > 0 else float("nan"),
        # Reference
        "x_nash_ref": x_nash,
        "x_so_ref": x_so,
        "frac_creators_within_delta_T": float(np.mean(np.abs(x - x_nash) < delta_thr)),
        "std_across_creators_T": float(np.std(x)),
        "init": init,
        "trial": 0,
        "t_first_hit": t_first_hit,
    }

    return summary, traj


# ---------------------------------------------------------------------------
# Regime runner
# ---------------------------------------------------------------------------
def run_learning_regime(
    beta_1, beta_2, beta_3, beta_5,
    x_nash: float,
    x_so: float,
    n: int = 38,
    config: dict = None,
    seed_base: int = 42,
):
    """
    Run n_trials learning simulations for one regime.
    When config['init'] is 'both', runs n_trials for 'low' AND n_trials for 'high'
    and reports them separately (each tagged with its init mode).
    Returns (summaries, all_trajs).
    """
    if config is None:
        config = LEARNING_CONFIG
    n_trials = config["n_trials"]
    init_mode = config.get("init", "both")

    payoff_fn = make_payoff_fn(beta_1, beta_2, beta_3, beta_5)

    if init_mode == "both":
        modes = ["low", "high"]
    elif init_mode in ("uniform", "low", "high"):
        modes = [init_mode]
    else:
        raise ValueError(f"Invalid init mode '{init_mode}'. Expected 'uniform', 'low', 'high', or 'both'.")

    summaries = []
    all_trajs = []
    for im in modes:
        cfg = dict(config, init=im)
        for trial in range(n_trials):
            seed = seed_base + trial
            summary, traj = run_learning(
                payoff_fn=payoff_fn,
                x_nash=x_nash,
                x_so=x_so,
                n=n,
                config=cfg,
                seed=seed,
            )
            summary["trial"] = trial
            summary["init"] = im
            tfh = summary.pop("t_first_hit", float("nan"))
            summary["t_first_hit"] = tfh
            summaries.append(summary)
            for row in traj:
                row["trial"] = trial
                row["init"] = im
            all_trajs.extend(traj)

    return summaries, all_trajs


# ---------------------------------------------------------------------------
# Parameter sweep helper
# ---------------------------------------------------------------------------
def sweep(
    param_name: str,
    values: list,
    base_regime: str = "A",
    n_trials: int = None,
    seed_base: int = 42,
    output_dir: str = DEFAULT_SWEEPS_DIR,
    **kwargs,
):
    """
    Sweep one parameter over a list of values.
    Accepts param_name in: 'eta', 'sigma', 'beta_5', 'T_horizon', 'delta', 'k_window', 'n_creators', 'kappa'.

    For every value:
      - Rebuilds the game with make_regime (or PotentialGame.from_json for kappa)
        and recomputes x_nash, x_so, and W_V references.
      - Catches ValueError from make_regime and records the row as skipped.
      - Runs n_trials (each trial t using seed_base + t).
      - If output_dir is not None, writes:
          results/sweeps/learning_<param_name>.csv
          results/sweeps/learning_<param_name>_summary.csv
        If output_dir is None, writes nothing to disk.

    Returns:
        (all_trial_rows, summary_rows)
    """
    valid_params = {"eta", "sigma", "beta_5", "T_horizon", "delta", "k_window", "n_creators", "kappa"}
    if param_name not in valid_params:
        raise ValueError(
            f"Invalid param_name '{param_name}'. Must be one of: {sorted(list(valid_params))}"
        )

    all_trial_rows = []
    summary_rows = []

    metrics = [
        "Delta_T", "D_T", "T_delta", "t_first_hit", "S_T",
        "total_reversals", "W_V_T", "W_V_nash", "W_V_so",
        "W_V_T_rel_nash", "W_V_T_rel_so",
        "frac_creators_within_delta_T", "std_across_creators_T",
    ]

    # Learner config
    cfg_base = dict(LEARNING_CONFIG)
    if "config" in kwargs and isinstance(kwargs["config"], dict):
        cfg_base.update(kwargs["config"])
    for k in ["T_horizon", "eta", "sigma", "delta", "k_window", "init"]:
        if k in kwargs:
            cfg_base[k] = kwargs[k]

    init_mode = cfg_base.get("init", "both")
    if init_mode == "both":
        modes = ["low", "high"]
    elif init_mode in ("uniform", "low", "high"):
        modes = [init_mode]
    else:
        raise ValueError(f"Invalid init mode '{init_mode}'. Expected 'uniform', 'low', 'high', or 'both'.")

    num_trials = n_trials if n_trials is not None else cfg_base.get("n_trials", 10)

    for val in values:
        try:
            regime_kwargs = {k: v for k, v in kwargs.items() if k in ["beta_1", "beta_2", "beta_3", "n_creators", "name"]}
            if param_name == "beta_5":
                regime_kwargs["beta_5"] = val
            elif param_name == "n_creators":
                regime_kwargs["n_creators"] = int(val)

            if param_name == "kappa":
                from game_theory_solver import PotentialGame
                json_path = os.path.join(_ROOT, "payoff_calculation", "results", "game_theory_coefficients.json")
                if not os.path.exists(json_path):
                    json_path = os.path.join(_ROOT, "payoff_estimation", "results", "game_theory_coefficients.json")
                n_c = regime_kwargs.get("n_creators", kwargs.get("n_creators", 38))
                game = PotentialGame.from_json(json_path, kappa=float(val), n_creators=n_c)
            else:
                game = make_regime(base_regime, **regime_kwargs)
        except ValueError:
            for im in modes:
                sum_row = {
                    "regime": base_regime,
                    "param_name": param_name,
                    "param_value": val,
                    "init": im,
                    "n_trials": 0,
                    "frac_T_delta_nan": 1.0,
                    "skipped": True,
                }
                if param_name == "kappa":
                    sum_row["x_nash"] = float("nan")
                    sum_row["x_so"] = float("nan")
                    sum_row["k"] = float("nan")
                    sum_row["poa"] = float("nan")
                    sum_row["lambda_max_eps_0.01"] = float("nan")
                    sum_row["welfare_recovery_at_lambda_max_eps_0.01"] = float("nan")
                for m in metrics:
                    sum_row[f"{m}_mean"] = float("nan")
                    sum_row[f"{m}_std"]  = float("nan")
                    sum_row[f"{m}_p10"]  = float("nan")
                    sum_row[f"{m}_p50"]  = float("nan")
                    sum_row[f"{m}_p90"]  = float("nan")
                summary_rows.append(sum_row)
            continue

        pfn = make_payoff_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)
        x_n = game.x_nash
        x_s = game.x_so

        cfg = dict(cfg_base)
        if param_name in cfg:
            cfg[param_name] = val

        for im in modes:
            cfg_im = dict(cfg, init=im)
            val_trial_rows = []
            for t in range(num_trials):
                seed = seed_base + t
                summ, _ = run_learning(
                    payoff_fn=pfn,
                    x_nash=x_n,
                    x_so=x_s,
                    n=game.n_creators,
                    config=cfg_im,
                    seed=seed,
                )
                row_dict = {
                    "regime": base_regime,
                    "param_name": param_name,
                    "param_value": val,
                    "init": im,
                }
                for k, v in summ.items():
                    if k not in ["trial", "t_first_hit"]:
                        row_dict[k] = v
                row_dict["trial"] = t
                row_dict["t_first_hit"] = summ.get("t_first_hit", float("nan"))

                val_trial_rows.append(row_dict)
                all_trial_rows.append(row_dict)

            # 3. Summary row for this value and init mode
            t_delta_vals = np.array([r["T_delta"] for r in val_trial_rows], dtype=float)
            frac_nan = float(np.mean(np.isnan(t_delta_vals)))

            sum_row = {
                "regime": base_regime,
                "param_name": param_name,
                "param_value": val,
                "init": im,
                "n_trials": len(val_trial_rows),
                "frac_T_delta_nan": frac_nan,
                "skipped": False,
            }
            if param_name == "kappa":
                sum_row["x_nash"] = game.x_nash
                sum_row["x_so"] = game.x_so
                sum_row["k"] = game.contraction_rate
                sum_row["poa"] = game.price_of_anarchy
                lm = game.mediator_max_lambda(0.01)
                sum_row["lambda_max_eps_0.01"] = lm
                sum_row["welfare_recovery_at_lambda_max_eps_0.01"] = game.welfare_recovery(lm)

            for m in metrics:
                m_vals = np.array([r[m] for r in val_trial_rows], dtype=float)
                if len(m_vals) == 0 or np.all(np.isnan(m_vals)):
                    sum_row[f"{m}_mean"] = float("nan")
                    sum_row[f"{m}_std"]  = float("nan")
                    sum_row[f"{m}_p10"]  = float("nan")
                    sum_row[f"{m}_p50"]  = float("nan")
                    sum_row[f"{m}_p90"]  = float("nan")
                else:
                    sum_row[f"{m}_mean"] = float(np.nanmean(m_vals))
                    sum_row[f"{m}_std"]  = float(np.nanstd(m_vals))
                    sum_row[f"{m}_p10"]  = float(np.nanpercentile(m_vals, 10))
                    sum_row[f"{m}_p50"]  = float(np.nanpercentile(m_vals, 50))
                    sum_row[f"{m}_p90"]  = float(np.nanpercentile(m_vals, 90))

            summary_rows.append(sum_row)

    if output_dir and all_trial_rows and summary_rows:
        os.makedirs(output_dir, exist_ok=True)
        append_mode = kwargs.get("append", False)
        t_path = os.path.join(output_dir, f"learning_{param_name}.csv")
        file_exists = os.path.exists(t_path) and append_mode
        with open(t_path, "a" if append_mode else "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_trial_rows[0].keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerows(all_trial_rows)
        s_path = os.path.join(output_dir, f"learning_{param_name}_summary.csv")
        s_file_exists = os.path.exists(s_path) and append_mode
        with open(s_path, "a" if append_mode else "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            if not s_file_exists:
                writer.writeheader()
            writer.writerows(summary_rows)

    return all_trial_rows, summary_rows


# ---------------------------------------------------------------------------
# CSV savers
# ---------------------------------------------------------------------------
def save_learning_csv(summaries, filepath):
    if not summaries:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    keys = list(summaries[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"  Saved: {filepath}")


def save_learning_trajectories_csv(all_trajs, filepath):
    if not all_trajs:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    keys = list(all_trajs[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_trajs)
    print(f"  Saved: {filepath}")


# ---------------------------------------------------------------------------
# Main (standalone)
# ---------------------------------------------------------------------------
def main():
    RESULTS_DIR = os.path.abspath(os.path.join(_HERE, "..", "results"))

    game_A = make_regime("A")

    regimes = [("A_empirical", game_A)]

    for label, game in regimes:
        print(f"\nLearning — Regime {label}")
        summaries, trajs = run_learning_regime(
            beta_1=game.beta_1, beta_2=game.beta_2,
            beta_3=game.beta_3, beta_5=game.beta_5,
            x_nash=game.x_nash, x_so=game.x_so,
            n=game.n_creators, config=LEARNING_CONFIG,
        )
        save_learning_csv(summaries, os.path.join(RESULTS_DIR, f"learning_{label}.csv"))
        save_learning_trajectories_csv(trajs, os.path.join(RESULTS_DIR, f"learning_trajectories_{label}.csv"))


if __name__ == "__main__":
    main()
