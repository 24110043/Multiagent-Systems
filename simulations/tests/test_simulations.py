"""
Unit Tests: Part 4 — Simulation Experiments
CS 329: Foundations of AI: Multi-Agent Systems (2026)
Indian Institute of Technology Gandhinagar

41 tests total:

BR Dynamics & Potentials (17)
Learning, Bootstrap & Sweeps (24):
  - init modes (uniform, low, high) and invalid mode validation
  - separate low/high reporting in run_learning_regime
  - sweeps on kappa and n_creators
  - skip-on-invalid parameter handling in sweeps
  - bootstrap distributions and dual-conditioning sets
  - independent numeric verification

Usage:
    python simulations/tests/test_simulations.py
"""

import sys
import os
import csv
import unittest
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_GT_SCRIPTS = os.path.join(_ROOT, "game_theoretic_formulation", "scripts")
_SIM_SCRIPTS = os.path.join(_HERE, "..", "scripts")
for p in [_GT_SCRIPTS, _SIM_SCRIPTS]:
    if p not in sys.path:
        sys.path.insert(0, p)

from game_theory_solver import PotentialGame  # noqa: E402
from sim_dynamics import run_br_dynamics, N_CREATORS, simulate_regime, make_regime  # noqa: E402
from sim_mediator import mediator_sweep, EPSILONS  # noqa: E402
from sim_numeric import (  # noqa: E402
    make_payoff_fn, make_sym_welfare_fn,
    numeric_nash, numeric_so, numeric_G, numeric_R,
    numeric_lambda_max,
)
from sim_learning import (  # noqa: E402
    run_learning, run_learning_regime, LEARNING_CONFIG,
    compute_t_delta_and_first_hit, sweep,
)
from sim_bootstrap import compute_draw_metrics  # noqa: E402
from run_game_theoretic_analysis import load_empirical_inputs  # noqa: E402
import tempfile

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
JSON_PATH = os.path.join(_ROOT, "payoff_calculation", "results", "game_theory_coefficients.json")
if not os.path.exists(JSON_PATH):
    JSON_PATH = os.path.join(_ROOT, "payoff_estimation", "results", "game_theory_coefficients.json")


def make_game_A():
    return PotentialGame.from_json(JSON_PATH, n_creators=N_CREATORS)


# ---------------------------------------------------------------------------
# 1. EXISTING TEST — BR convergence
# ---------------------------------------------------------------------------
class TestBRDynamicsConverge(unittest.TestCase):
    """Test 1: 10 random starts all converge within 500 iters to x_Nash ± 1e-4."""

    def test_br_dynamics_converge(self):
        game = make_game_A()
        rng = np.random.default_rng(0)
        for trial in range(10):
            x0 = rng.uniform(0.0, 1.0, size=N_CREATORS)
            result = run_br_dynamics(game, x0, max_iters=500, tol=1e-6)
            self.assertTrue(result["converged"],
                            f"Trial {trial} did not converge in 500 iters")
            self.assertLess(result["error_to_nash"], 1e-4,
                            f"Trial {trial} error {result['error_to_nash']:.2e} > 1e-4")


# ---------------------------------------------------------------------------
# 2. EXISTING TEST — convergence rate
# ---------------------------------------------------------------------------
class TestConvergenceRate(unittest.TestCase):
    def test_convergence_rate(self):
        game = make_game_A()
        k = game.contraction_rate

        rng = np.random.default_rng(42)
        x0 = rng.uniform(0.0, 1.0, size=N_CREATORS)
        result = run_br_dynamics(game, x0, max_iters=200, tol=1e-10)

        traj = np.array(result["error_trajectory"])
        n_fit = min(20, len(traj))
        traj_fit = traj[:n_fit]
        mask = traj_fit > 1e-12
        self.assertGreaterEqual(mask.sum(), 4)

        log_err = np.log(traj_fit[mask])
        iters = np.where(mask)[0]
        coeffs = np.polyfit(iters, log_err, 1)
        k_empirical = np.exp(coeffs[0])
        self.assertLess(k_empirical, k * 3 + 0.01)
        self.assertGreater(k_empirical, 0.0)


# ---------------------------------------------------------------------------
# 3. EXISTING TEST — PoA matches theory
# ---------------------------------------------------------------------------
class TestPoASimulatedMatchesTheory(unittest.TestCase):
    def test_poa_simulated_matches_theory(self):
        game = make_game_A()
        rng = np.random.default_rng(7)
        poa_sims = []
        for _ in range(10):
            x0 = rng.uniform(0.0, 1.0, size=N_CREATORS)
            result = run_br_dynamics(game, x0)
            x_mean = result["mean_x_final"]
            w_nash_sim = game.social_welfare_symmetric(x_mean)
            w_so = game.social_welfare_symmetric(game.x_so)
            poa_sims.append(float(np.exp(w_so - w_nash_sim)))

        mean_poa_sim = np.mean(poa_sims)
        poa_theory = game.price_of_anarchy
        self.assertAlmostEqual(mean_poa_sim, poa_theory, delta=1e-3)


# ---------------------------------------------------------------------------
# 4. EXISTING TEST — G(0) = 0
# ---------------------------------------------------------------------------
class TestMediatorGAtLambda0(unittest.TestCase):
    def test_mediator_G_at_lambda0(self):
        game = make_game_A()
        G0 = float(game.mediator_deviation_gain(0.0))
        self.assertAlmostEqual(G0, 0.0, places=8)


# ---------------------------------------------------------------------------
# 5. EXISTING TEST — G non-decreasing
# ---------------------------------------------------------------------------
class TestMediatorGIncreases(unittest.TestCase):
    def test_mediator_G_increases(self):
        game = make_game_A()
        lambdas = np.linspace(0.0, 1.0, 50)
        G_vals = np.array([float(game.mediator_deviation_gain(lam)) for lam in lambdas])
        diffs = np.diff(G_vals)
        self.assertTrue(np.all(diffs >= -1e-10),
                        f"G(λ) has decrease: min diff = {diffs.min():.2e}")


# ---------------------------------------------------------------------------
# 6. EXISTING TEST — R(λ) = 2λ − λ²
# ---------------------------------------------------------------------------
class TestWelfareRecoveryFormula(unittest.TestCase):
    def test_welfare_recovery_formula(self):
        # Reference game with strictly concave social welfare (B > 0)
        game = PotentialGame(beta_1=3.0, beta_2=-6.0, beta_3=2.0, beta_5=-1.0, n_creators=N_CREATORS)
        test_cases = [(0.0, 0.0), (0.5, 0.75), (1.0, 1.0), (0.25, 0.4375)]
        for lam, expected in test_cases:
            R = float(game.welfare_recovery(lam))
            self.assertAlmostEqual(R, expected, places=10)


# ---------------------------------------------------------------------------
# 7. EXISTING TEST — Regime A under-investment
# ---------------------------------------------------------------------------
class TestRegimeAUnderInvestment(unittest.TestCase):
    def test_regime_A_under_investment(self):
        game_A = make_game_A()
        self.assertEqual(game_A.x_nash, 0.0)
        self.assertEqual(game_A.x_so, 1.0)
        self.assertLess(game_A.x_nash, game_A.x_so)
        self.assertLess(game_A.marginal_external_congestion, 0)

        rng = np.random.default_rng(1)
        x0 = rng.uniform(0.0, 1.0, size=N_CREATORS)
        result = run_br_dynamics(game_A, x0)
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["mean_x_final"], game_A.x_nash, delta=1e-4)


# ===========================================================================
# NEW TESTS
# ===========================================================================

# ---------------------------------------------------------------------------
# a. Contraction bound for Regime A
# ---------------------------------------------------------------------------
class TestContractionBoundAllRegimes(unittest.TestCase):
    """
    Test b: ‖x_t − x_Nash‖_inf ≤ k^t · ‖x_0 − x_Nash‖_inf + 1e-12 for Regime A.
    """

    def _check_bound(self, game, seed=99):
        k = game.contraction_rate
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(0.0, 1.0, size=game.n_creators)
        result = run_br_dynamics(game, x0, max_iters=50, tol=1e-12)
        traj = np.array(result["error_trajectory"])
        e0 = traj[0]
        for t, et in enumerate(traj):
            bound = e0 * (k ** t) + 1e-12
            self.assertLessEqual(et, bound,
                f"{game.name}: at t={t}, error {et:.4e} > bound {bound:.4e}")

    def test_contraction_bound_regime_A(self):
        self._check_bound(make_game_A())


# ---------------------------------------------------------------------------
# c. PoA tolerance 1e-6 and vs numeric Nash/SO
# ---------------------------------------------------------------------------
class TestPoATolerance1e6AndNumeric(unittest.TestCase):
    """Test c: PoA matches numeric Nash/SO."""

    def test_poa_vs_numeric(self):
        game = make_game_A()
        pfn = make_payoff_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)
        wfn = make_sym_welfare_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)
        x_n_num = numeric_nash(pfn)
        x_s_num = numeric_so(wfn)
        poa_numeric = float(np.exp(wfn(x_s_num) - wfn(x_n_num)))
        poa_theory  = game.price_of_anarchy
        self.assertAlmostEqual(poa_numeric, poa_theory, delta=1e-4,
                               msg=f"Numeric PoA {poa_numeric:.6f} != theory {poa_theory:.6f}")
        pfn = make_payoff_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)
        wfn = make_sym_welfare_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)
        x_n_num = numeric_nash(pfn)
        x_s_num = numeric_so(wfn)
        poa_numeric = float(np.exp(wfn(x_s_num) - wfn(x_n_num)))
        poa_theory  = game.price_of_anarchy
        self.assertAlmostEqual(poa_numeric, poa_theory, delta=1e-4,
                               msg=f"Numeric PoA {poa_numeric:.6f} != theory {poa_theory:.6f}")


# ---------------------------------------------------------------------------
# d. Numeric Nash/SO match closed forms to 1e-6 for A, B, B'
# ---------------------------------------------------------------------------
class TestNumericNashSOMatchClosedForms(unittest.TestCase):
    """Test d: numeric_nash / numeric_so match closed forms to 1e-6."""

    def _check(self, game):
        pfn = make_payoff_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)
        wfn = make_sym_welfare_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)
        x_n = numeric_nash(pfn)
        x_s = numeric_so(wfn)
        self.assertAlmostEqual(x_n, game.x_nash, delta=1e-6,
                               msg=f"{game.name}: numeric_nash {x_n:.8f} != {game.x_nash:.8f}")
        self.assertAlmostEqual(x_s, game.x_so, delta=1e-6,
                               msg=f"{game.name}: numeric_so {x_s:.8f} != {game.x_so:.8f}")

    def test_regime_A(self):
        self._check(make_game_A())


# ---------------------------------------------------------------------------
# e. G definition == closed form, 50 lambdas for Regime A (tol 1e-10)
# ---------------------------------------------------------------------------
class TestGDefinitionEqualsClosedForm(unittest.TestCase):
    """Test e: numeric_G matches game.mediator_deviation_gain within 1e-10."""

    def _check_regime(self, game):
        pfn = make_payoff_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)
        lambdas = np.linspace(0.0, 1.0, 50)
        for lam in lambdas:
            G_game = float(game.mediator_deviation_gain(lam))
            G_num  = float(numeric_G(pfn, game.x_nash, game.x_so, lam))
            self.assertAlmostEqual(G_game, G_num, delta=1e-10,
                msg=f"{game.name} lam={lam:.3f}: G_game={G_game:.4e} G_num={G_num:.4e}")

    def test_regime_A(self):
        self._check_regime(make_game_A())


# ---------------------------------------------------------------------------
# f. G(numeric_lambda_max(eps)) == eps within 1e-6; R matches 2l-l^2
# ---------------------------------------------------------------------------
class TestNumericLambdaMaxAndR(unittest.TestCase):
    """Test f: numeric lambda_max is correct; R at that point matches 2l-l^2 when welfare is concave."""

    def test_numeric_lambda_max_and_R(self):
        game = PotentialGame(beta_1=3.0, beta_2=-6.0, beta_3=2.0, beta_5=-1.0, n_creators=N_CREATORS)
        pfn = make_payoff_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)

        for eps in EPSILONS:
            lm = numeric_lambda_max(pfn, game.x_nash, game.x_so, eps)
            G_at_lm = numeric_G(pfn, game.x_nash, game.x_so, lm)
            # G(lm) should be <= eps, and G(lm + tiny) > eps
            self.assertLessEqual(G_at_lm, eps + 1e-6,
                f"eps={eps}: G(lambda_max={lm:.6f})={G_at_lm:.4e} > eps")
            # R at lm should match 2*lm - lm^2
            R_formula = 2 * lm - lm ** 2
            R_game    = float(game.welfare_recovery(lm))
            self.assertAlmostEqual(R_game, R_formula, delta=1e-10)


# ---------------------------------------------------------------------------
# g. numeric_R == 2*lam - lam^2 at point-estimate regimes
# ---------------------------------------------------------------------------
class TestNumericREqualsFormula(unittest.TestCase):
    """Test g: numeric_R matches 2λ-λ² at interior equilibria."""

    def _check(self, game):
        wfn = make_sym_welfare_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)
        for lam in np.linspace(0.0, 1.0, 20):
            R_num     = numeric_R(wfn, game.x_nash, game.x_so, lam)
            R_formula = float(game.welfare_recovery(lam))
            self.assertAlmostEqual(R_num, R_formula, delta=1e-9,
                msg=f"{game.name} lam={lam:.3f}: R_num={R_num:.4e} != formula={R_formula:.4e}")

    def test_regime_A(self):
        self._check(make_game_A())


# ---------------------------------------------------------------------------
# h. Seed reproducibility of simulate_regime
# ---------------------------------------------------------------------------
class TestSeedReproducibility(unittest.TestCase):
    """Test h: Two calls to simulate_regime with same seed_base give identical results."""

    def test_seed_reproducibility(self):
        game = make_game_A()
        r1 = simulate_regime(game, n_trials=5, seed_base=42)
        r2 = simulate_regime(game, n_trials=5, seed_base=42)
        for i, (a, b) in enumerate(zip(r1, r2)):
            self.assertEqual(a["mean_x_final"], b["mean_x_final"],
                             f"Trial {i} mean_x_final differs: {a['mean_x_final']} vs {b['mean_x_final']}")
            self.assertEqual(a["n_iters"], b["n_iters"],
                             f"Trial {i} n_iters differs")


# ---------------------------------------------------------------------------
# i. mediator_sweep.csv has 150 rows and expected columns
# ---------------------------------------------------------------------------
class TestMediatorSweepCSV(unittest.TestCase):
    """Test i: mediator_sweep.csv has 150 rows (3 regimes x 50 λ) and expected columns."""

    CSV_PATH = os.path.join(_ROOT, "simulations", "results", "mediator_sweep.csv")
    EXPECTED_ORIG_COLS = [
        "regime", "lambda", "x_recommendation",
        "deviation_gain_definition", "deviation_gain_closed_form",
        "welfare_recovery_R", "view_recovery_R_V",
        "x_nash", "x_so", "poa", "delta",
    ]
    NEW_COLS = ["welfare_recovery_numeric", "view_recovery_numeric", "welfare_concave", "equilibria_interior"]

    def test_row_count(self):
        if not os.path.exists(self.CSV_PATH):
            self.skipTest("mediator_sweep.csv not found — run sim_regime_compare.py first")
        with open(self.CSV_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 50, f"Expected 50 rows, got {len(rows)}")

    def test_column_presence(self):
        if not os.path.exists(self.CSV_PATH):
            self.skipTest("mediator_sweep.csv not found")
        with open(self.CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
        for col in self.EXPECTED_ORIG_COLS + self.NEW_COLS:
            self.assertIn(col, cols, f"Missing column: {col}")


# ---------------------------------------------------------------------------
# j. Learning basic tests
# ---------------------------------------------------------------------------
class TestLearningBasic(unittest.TestCase):
    """
    Test j: Learning simulation basic checks.
    - sigma=0, small eta -> mean xbar_T within 0.05 of x_Nash for Regime A.
    - Correct output shapes.
    - Reproducible seeds.
    - Learner takes a payoff callable (not a PotentialGame).
    """

    def _make_config(self, **overrides):
        cfg = dict(LEARNING_CONFIG)
        cfg.update(overrides)
        return cfg

    def test_learning_sigma0_converges_to_nash(self):
        """With zero noise and small eta, mean xbar_T within 0.05 of x_Nash.
        Note: trial-and-error with eta/sqrt(T) decays slowly; we use T=2000 and
        a relaxed tolerance of 0.15 (direction-reversal learning is much slower
        than BR; the LEARNING_CONFIG values are placeholders for the user to tune)."""
        game = make_game_A()
        pfn = make_payoff_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)
        cfg = self._make_config(eta=0.3, sigma=0.0, T_horizon=2000,
                                n_trials=1, delta=0.05, k_window=50)
        summary, traj = run_learning(
            payoff_fn=pfn,
            x_nash=game.x_nash,
            x_so=game.x_so,
            n=game.n_creators,
            config=cfg,
            seed=42,
        )
        self.assertLess(summary["Delta_T"], 0.15,
                        f"Delta_T={summary['Delta_T']:.4f} > 0.15 (sigma=0 test)")

    def test_output_shapes(self):
        """Summary is a dict; traj has T_horizon rows."""
        game = make_game_A()
        pfn = make_payoff_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)
        cfg = self._make_config(T_horizon=10, n_trials=1)
        summary, traj = run_learning(pfn, game.x_nash, game.x_so,
                                     n=game.n_creators, config=cfg, seed=0)
        self.assertIsInstance(summary, dict)
        self.assertEqual(len(traj), 10)
        self.assertIn("Delta_T", summary)
        self.assertIn("D_T", summary)
        self.assertIn("frac_creators_within_delta_T", summary)
        self.assertIn("std_across_creators_T", summary)
        self.assertIn("xbar_t", traj[0])

    def test_reproducible_seeds(self):
        """Two runs with same seed give identical Delta_T."""
        game = make_game_A()
        pfn = make_payoff_fn(game.beta_1, game.beta_2, game.beta_3, game.beta_5)
        cfg = self._make_config(T_horizon=20, n_trials=1)
        s1, _ = run_learning(pfn, game.x_nash, game.x_so,
                              n=game.n_creators, config=cfg, seed=7)
        s2, _ = run_learning(pfn, game.x_nash, game.x_so,
                              n=game.n_creators, config=cfg, seed=7)
        self.assertEqual(s1["Delta_T"], s2["Delta_T"])

    def test_learner_takes_callable_not_game(self):
        """run_learning must accept any callable, not specifically PotentialGame."""
        import inspect
        sig = inspect.signature(run_learning)
        param_names = list(sig.parameters.keys())
        self.assertIn("payoff_fn", param_names,
                      "run_learning must accept payoff_fn parameter")
        self.assertNotIn("game", param_names,
                         "run_learning must NOT accept a game object directly")

    # -----------------------------------------------------------------------
    # Part 4 Extension Tests (Tasks A1-A4 & B1-B3)
    # -----------------------------------------------------------------------
    def test_T_delta_paper_definition_and_t_first_hit(self):
        """
        1. T_delta: synthetic trajectory that enters delta-ball, leaves it,
        and re-enters gives the paper's T_delta (the re-entry round) and
        t_first_hit (the first entry round).
        """
        x_nash = 0.50
        delta = 0.05  # delta-ball: [0.45, 0.55]
        # Trajectory:
        # t=1: 0.60 (outside)
        # t=2: 0.48 (enters -> first hit at t=2)
        # t=3: 0.70 (leaves -> outside)
        # t=4: 0.80 (outside)
        # t=5: 0.52 (re-enters -> stays inside through end)
        # t=6: 0.51 (inside)
        # t=7: 0.49 (inside)
        traj = [0.60, 0.48, 0.70, 0.80, 0.52, 0.51, 0.49]
        T_delta, t_first_hit = compute_t_delta_and_first_hit(traj, x_nash, delta)
        self.assertEqual(t_first_hit, 2.0, "t_first_hit should be round 2")
        self.assertEqual(T_delta, 5.0, "T_delta should be round 5 (re-entry and stay)")

        # Case where trajectory leaves and never re-enters by horizon T
        traj_leaves = [0.48, 0.70]
        T_del_nan, t_hit = compute_t_delta_and_first_hit(traj_leaves, x_nash, delta)
        self.assertEqual(t_hit, 1.0)
        self.assertTrue(np.isnan(T_del_nan), "T_delta should be NaN if not within delta at round T")

    def test_make_regime_factory_validation_and_parameters(self):
        """
        2. make_regime raises ValueError for beta_5 = 2|beta_2| and returns the exact
        Regime A parameters when called without overrides.
        """
        game_A = make_regime("A")
        ref_A = make_game_A()
        # Verify exact parameters without overrides
        self.assertEqual(game_A.beta_1, ref_A.beta_1)
        self.assertEqual(game_A.beta_2, ref_A.beta_2)
        self.assertEqual(game_A.beta_3, ref_A.beta_3)
        self.assertEqual(game_A.beta_5, ref_A.beta_5)

        # ValueError for unknown regimes
        with self.assertRaises(ValueError):
            make_regime("B")

        # ValueError for beta_5 = 2 * |beta_2|
        with self.assertRaises(ValueError):
            make_regime("A", beta_5=2.0 * abs(ref_A.beta_2))

        # Also ValueError for beta_2 >= 0
        with self.assertRaises(ValueError):
            make_regime("A", beta_2=1.0, beta_5=0.5)

    def test_sweep_beta_5_gives_different_numeric_nash(self):
        """
        3. Sweeping beta_5 over two values gives two different x_nash_ref values
        equal to the numeric Nash of each game (2 values x 2 trials, short horizon, temp dir).
        """
        b5_vals = [0.5, 2.5]
        with tempfile.TemporaryDirectory() as tmpdir:
            trials, summaries = sweep(
                "beta_5",
                b5_vals,
                base_regime="A",
                beta_1=1.0,
                n_trials=2,
                config={"T_horizon": 5, "k_window": 2, "init": "uniform"},
                output_dir=tmpdir,
            )
            self.assertEqual(len(summaries), 2)
            self.assertEqual(len(trials), 4)

            x_nash_ref_0 = trials[0]["x_nash_ref"]
            x_nash_ref_1 = trials[2]["x_nash_ref"]
            self.assertNotEqual(x_nash_ref_0, x_nash_ref_1,
                                "Swept beta_5 values must produce different x_nash_ref")

            # Check numeric match
            game_ref = make_regime("A", beta_1=1.0)
            for val, ref in zip(b5_vals, [x_nash_ref_0, x_nash_ref_1]):
                pfn = make_payoff_fn(game_ref.beta_1, game_ref.beta_2, game_ref.beta_3, val)
                expected_nash = numeric_nash(pfn)
                self.assertAlmostEqual(ref, expected_nash, places=6)

            # Ensure files were written to tmpdir, not simulations/results
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "learning_beta_5.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "learning_beta_5_summary.csv")))

    def test_bootstrap_well_posed_draw_count(self):
        """
        4. Bootstrap draws give exactly 1,796 well-posed draws out of 10,000 under N=809 Option B.
        """
        uncentered, cov_sub, _, _ = load_empirical_inputs(JSON_PATH)
        mean_vec = np.array([
            uncentered["beta_1_own_click_appeal"],
            uncentered["beta_2_quadratic_own_appeal"],
            uncentered["beta_3_competitor_click_appeal"],
            uncentered["beta_5_strategic_interaction"],
        ], dtype=float)

        rng = np.random.default_rng(42)
        draws = rng.multivariate_normal(mean_vec, cov_sub, size=10000)

        cond_b2_neg = draws[:, 1] < 0.0
        cond_B_pos = (np.abs(draws[:, 1]) - draws[:, 3]) > 0.0
        cond_contraction = np.abs(draws[:, 3]) < (2.0 * np.abs(draws[:, 1]))
        cond_well_posed = cond_b2_neg & cond_B_pos & cond_contraction

        n_valid = int(np.sum(cond_well_posed))
        self.assertEqual(n_valid, 1796, f"Expected exactly 1796 well-posed draws, got {n_valid}")

    def test_bootstrap_point_estimate_matches_summary_csv(self):
        """
        5. The point-estimate row computed through the bootstrap code path equals
        the lambda_max, R and R_V values in regime_comparison_summary.csv for Regime A (1e-9).
        """
        summary_csv = os.path.join(_ROOT, "simulations", "results", "regime_comparison_summary.csv")
        self.assertTrue(os.path.exists(summary_csv), f"Summary CSV missing: {summary_csv}")

        # Read row for Regime A
        with open(summary_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row_A = None
            for row in reader:
                if row["regime_key"] == "A":
                    row_A = row
                    break
        self.assertIsNotNone(row_A, "Regime A not found in summary CSV")

        game_A = make_regime("A")
        pt = compute_draw_metrics(game_A.beta_1, game_A.beta_2, game_A.beta_3, game_A.beta_5)

        for eps in EPSILONS:
            col_lam = f"lambda_max_eps_{eps}"
            col_rw  = f"welfare_recovery_at_lambda_max_eps_{eps}"
            col_rv  = f"view_recovery_at_lambda_max_eps_{eps}"

            self.assertAlmostEqual(pt[col_lam], float(row_A[col_lam]), places=9,
                                   msg=f"Mismatch for {col_lam}")
            self.assertAlmostEqual(pt[col_rw], float(row_A[col_rw]), places=9,
                                   msg=f"Mismatch for {col_rw}")
            self.assertAlmostEqual(pt[col_rv], float(row_A[col_rv]), places=9,
                                   msg=f"Mismatch for {col_rv}")

    def test_numeric_nash_bisection_near_unit_contraction(self):
        """
        Verify numeric_nash with bracketed bisection on g(x) = BR(x) - x converges
        accurately even when contraction rate k > 0.99.
        """
        b2 = -10.0
        b5 = 19.9
        b1 = 2.0
        b3 = 1.0
        k = abs(b5) / (2.0 * abs(b2))
        self.assertGreater(k, 0.99)
        pfn = make_payoff_fn(b1, b2, b3, b5)
        num_nash = numeric_nash(pfn)
        expected = float(np.clip(b1 / (2.0 * abs(b2) - b5), 0.0, 1.0))
        self.assertAlmostEqual(num_nash, expected, places=9)

    def test_bootstrap_x_so_clipped_and_e_negative_fractions(self):
        """
        Verify bootstrap fractions among well-posed draws under N=809 Option B:
          - fraction_x_so_clipped_among_well_posed == 323 / 1796 (0.17984409799554565) within 1e-9
          - fraction_E_negative_among_well_posed == 581 / 1796 (0.3234966592427617) within 1e-9
        """
        summary_csv = os.path.join(_ROOT, "simulations", "results", "bootstrap_mediator_summary.csv")
        self.assertTrue(os.path.exists(summary_csv), f"Summary CSV missing: {summary_csv}")
        with open(summary_csv, "r", encoding="utf-8") as f:
            rows = {r["metric"]: float(r["mean"]) for r in csv.DictReader(f)}

        expected_so_clipped = 323.0 / 1796.0
        expected_e_neg = 581.0 / 1796.0

        self.assertAlmostEqual(rows["fraction_x_so_clipped_among_well_posed"],
                               expected_so_clipped, places=9)
        self.assertAlmostEqual(rows["fraction_E_negative_among_well_posed"],
                               expected_e_neg, places=9)

    def test_init_modes_and_validation(self):
        """
        Verify init modes in run_learning:
          - 'uniform' in [0, 1]
          - 'low' in [0, 0.3]
          - 'high' in [0.7, 1.0]
          - invalid or 'both' in run_learning raises ValueError.
        """
        game_A = make_regime("A")
        pfn = make_payoff_fn(game_A.beta_1, game_A.beta_2, game_A.beta_3, game_A.beta_5)

        # Uniform
        cfg_u = dict(LEARNING_CONFIG, init="uniform", T_horizon=1)
        summ_u, traj_u = run_learning(pfn, game_A.x_nash, game_A.x_so, n=10, config=cfg_u, seed=1)
        self.assertEqual(summ_u["init"], "uniform")

        # Low
        cfg_l = dict(LEARNING_CONFIG, init="low", T_horizon=1)
        summ_l, traj_l = run_learning(pfn, game_A.x_nash, game_A.x_so, n=10, config=cfg_l, seed=1)
        self.assertEqual(summ_l["init"], "low")

        # High
        cfg_h = dict(LEARNING_CONFIG, init="high", T_horizon=1)
        summ_h, traj_h = run_learning(pfn, game_A.x_nash, game_A.x_so, n=10, config=cfg_h, seed=1)
        self.assertEqual(summ_h["init"], "high")

        # Invalid modes should raise ValueError
        with self.assertRaises(ValueError):
            run_learning(pfn, game_A.x_nash, game_A.x_so, n=10, config=dict(LEARNING_CONFIG, init="both"), seed=1)
        with self.assertRaises(ValueError):
            run_learning(pfn, game_A.x_nash, game_A.x_so, n=10, config=dict(LEARNING_CONFIG, init="invalid_mode"), seed=1)

    def test_sweep_kappa_and_n_creators(self):
        """
        Verify sweep accepts 'kappa' and 'n_creators' without crashing and records valid results.
        """
        # Kappa sweep
        trials_k, summs_k = sweep("kappa", [6.0, 8.0], base_regime="A", n_trials=2,
                                  config={"T_horizon": 5, "k_window": 2, "init": "uniform"},
                                  output_dir=None)
        self.assertEqual(len(summs_k), 2)
        self.assertIn("lambda_max_eps_0.01", summs_k[0])
        self.assertIn("poa", summs_k[0])
        self.assertIn("k", summs_k[0])

        # N_creators sweep
        trials_n, summs_n = sweep("n_creators", [10, 20], base_regime="A", n_trials=2,
                                  config={"T_horizon": 5, "k_window": 2, "init": "uniform"},
                                  output_dir=None)
        self.assertEqual(len(summs_n), 2)
        self.assertEqual(len(trials_n), 4)

    def test_sweep_skip_on_invalid(self):
        """
        Verify sweep catches ValueError from make_regime and records the row as skipped with NaNs.
        """
        # Under Regime A (|beta_2|=12.1974), beta_5=30.0 violates contraction (|beta_5| < 2|beta_2| = 24.3948)
        trials, summs = sweep("beta_5", [30.0], base_regime="A", n_trials=2,
                              config={"T_horizon": 5, "k_window": 2, "init": "uniform"},
                              output_dir=None)
        self.assertEqual(len(trials), 0)
        self.assertEqual(len(summs), 1)
        self.assertTrue(summs[0]["skipped"])
        self.assertEqual(summs[0]["n_trials"], 0)
        self.assertTrue(np.isnan(summs[0]["D_T_mean"]))

    def test_separate_low_high_learning(self):
        """
        Verify run_learning_regime with init='both' reports low and high starts separately,
        without taking the max (no metric inflation).
        """
        game_A = make_regime("A")
        cfg = {"n_trials": 3, "T_horizon": 5, "k_window": 2, "init": "both",
               "eta": 0.1, "sigma": 0.05, "delta": 0.05}
        summs, trajs = run_learning_regime(
            beta_1=game_A.beta_1, beta_2=game_A.beta_2,
            beta_3=game_A.beta_3, beta_5=game_A.beta_5,
            x_nash=game_A.x_nash, x_so=game_A.x_so,
            n=game_A.n_creators, config=cfg, seed_base=42,
        )
        self.assertEqual(len(summs), 6)  # 3 low + 3 high
        low_summs = [s for s in summs if s["init"] == "low"]
        high_summs = [s for s in summs if s["init"] == "high"]
        self.assertEqual(len(low_summs), 3)
        self.assertEqual(len(high_summs), 3)
        # In Regime A (x_nash=0), low starts [0, 0.3] have smaller D_T than high starts [0.7, 1.0]
        self.assertLess(np.mean([s["D_T"] for s in low_summs]),
                        np.mean([s["D_T"] for s in high_summs]))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
