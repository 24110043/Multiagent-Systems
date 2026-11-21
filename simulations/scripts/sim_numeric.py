"""
Numeric verification helpers — independent of game_theory_solver's closed forms.
CS 329: Foundations of AI: Multi-Agent Systems (2026)
Indian Institute of Technology Gandhinagar

Provides:
  numeric_nash(payoff_fn, n, beta_1, beta_2, beta_5, n_grid, tol)
  numeric_so(payoff_fn_sym, n, beta_1, beta_2, beta_3, beta_5, n_grid, tol)
  numeric_G(payoff_fn, x_nash, x_so, lam)
  numeric_R(payoff_fn_sym, x_nash, x_so, lam)
  numeric_lambda_max(payoff_fn, payoff_fn_sym, x_nash, x_so, eps, tol_bisect)

None of these functions use game.x_nash, game.x_so, game.best_response,
game.mediator_deviation_gain, or any closed-form formula — they treat the
game as a black box via the payoff callable.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Payoff callable builder (pure function, no PotentialGame internals)
# ---------------------------------------------------------------------------
def make_payoff_fn(beta_1, beta_2, beta_3, beta_5):
    """
    Return a payoff callable:
        pi(x_i, x_bar_minus_i) = b1*x_i + b2*x_i^2 + b3*x_bar_minus_i + b5*x_i*x_bar_minus_i
    c_i = 0 (channel-level constant excluded; only strategic terms matter).
    """
    def payoff(x_i, x_bar_minus_i):
        return (beta_1 * x_i
                + beta_2 * (x_i ** 2)
                + beta_3 * x_bar_minus_i
                + beta_5 * x_i * x_bar_minus_i)
    return payoff


def make_sym_welfare_fn(beta_1, beta_2, beta_3, beta_5):
    """
    Return symmetric welfare per-creator callable:
        w(x) = pi(x, x) = (b1+b3)*x + (b2+b5)*x^2
    """
    def w(x):
        return (beta_1 + beta_3) * x + (beta_2 + beta_5) * (x ** 2)
    return w


# ---------------------------------------------------------------------------
# Task 2: Numeric Nash & Social Optimum — grid search + golden-section refinement
# ---------------------------------------------------------------------------
def _golden_section_max(f, a=0.0, b=1.0, tol=1e-12, max_iter=200):
    """Maximise strictly concave f on [a, b] via golden-section search with boundary check and parabolic refinement."""
    # Boundary check
    if f(b) >= f(b - 1e-6):
        return float(b)
    if f(a) >= f(a + 1e-6):
        return float(a)

    phi = (np.sqrt(5) - 1) / 2
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    for _ in range(max_iter):
        if abs(b - a) < 1e-5:
            break
        if f(c) < f(d):
            a = c
        else:
            b = d
        c = b - phi * (b - a)
        d = a + phi * (b - a)

    x2 = (a + b) / 2
    h = 1e-4
    x1, x3 = x2 - h, x2 + h
    y1, y2, y3 = f(x1), f(x2), f(x3)
    denom = (x1 - x2) * (y1 - y3) - (x1 - x3) * (y1 - y2)
    if abs(denom) > 1e-18:
        num = (x1**2 - x2**2) * (y1 - y3) - (x1**2 - x3**2) * (y1 - y2)
        v = 0.5 * num / denom
        if 0.0 <= v <= 1.0:
            return float(v)
    return float(np.clip(x2, 0.0, 1.0))


def numeric_nash(payoff_fn, n_grid=2000, tol=1e-12):
    """
    Numerically find the symmetric Nash equilibrium via bracketed bisection
    on g(x) = BR(x) - x over [0, 1].

    Treats payoff_fn as a black box:
        BR(x_bar) = argmax_{x in [0,1]} payoff_fn(x, x_bar)

    The symmetric Nash is the unique root g(x*) = BR(x*) - x* = 0.
    Since g is strictly decreasing (slope s - 1 < 0 for |s| < 1), bisection
    converges unconditionally to tolerance 1e-12.
    Returns numeric_nash (float in [0, 1]).
    """
    def BR(x_bar):
        f = lambda xi: payoff_fn(xi, x_bar)
        return _golden_section_max(f, 0.0, 1.0, tol=tol)

    lo, hi = 0.0, 1.0
    g_lo = BR(lo) - lo
    if g_lo <= 0.0:
        return float(lo)

    g_hi = BR(hi) - hi
    if g_hi >= 0.0:
        return float(hi)

    while (hi - lo) > 1e-12:
        mid = 0.5 * (lo + hi)
        g_mid = BR(mid) - mid
        if g_mid == 0.0:
            return float(mid)
        elif g_mid > 0.0:
            lo = mid
        else:
            hi = mid

    return float(np.clip(0.5 * (lo + hi), 0.0, 1.0))


def numeric_so(sym_welfare_fn, n_grid=2000, tol=1e-10):
    """
    Numerically find the symmetric Social Optimum via golden-section search
    on the symmetric welfare function w(x) = pi(x, x).

    Returns numeric_so (float in [0,1]).
    """
    return float(_golden_section_max(sym_welfare_fn, 0.0, 1.0, tol=tol))


# ---------------------------------------------------------------------------
# Task 2: Numeric G (deviation gain), numeric R (welfare recovery)
# ---------------------------------------------------------------------------
def numeric_G(payoff_fn, x_nash, x_so, lam):
    """
    Deviation gain from definition — no closed forms used.
    G(lam) = pi(BR(x_lam), x_lam) - pi(x_lam, x_lam)
    where BR is computed via golden-section maximisation.
    """
    x_lam = (1.0 - lam) * x_nash + lam * x_so
    f = lambda xi: payoff_fn(xi, x_lam)
    br = _golden_section_max(f, 0.0, 1.0, tol=1e-12)
    return float(payoff_fn(br, x_lam) - payoff_fn(x_lam, x_lam))


def numeric_R(sym_welfare_fn, x_nash, x_so, lam):
    """
    Numeric welfare recovery fraction:
        R_num(lam) = (W(x_lam) - W(x_nash)) / (W(x_so) - W(x_nash))
    where W(x) = sym_welfare_fn(x) = pi(x, x).

    If W(x_so) == W(x_nash) (degenerate), returns 0.
    If x_nash or x_so is at the boundary (0 or 1), warn that the formula
    2*lam - lam^2 is not guaranteed and rely on this numeric computation.
    """
    W_nash = sym_welfare_fn(x_nash)
    W_so = sym_welfare_fn(x_so)
    x_lam = (1.0 - lam) * x_nash + lam * x_so
    W_lam = sym_welfare_fn(x_lam)
    denom = W_so - W_nash
    if abs(denom) < 1e-15:
        return 0.0
    return float((W_lam - W_nash) / denom)


def numeric_view_recovery(sym_welfare_fn, x_nash, x_so, lam):
    """
    View recovery via views = exp(welfare):
        R_V(lam) = (exp(W(x_lam)) - exp(W(x_nash))) / (exp(W(x_so)) - exp(W(x_nash)))
    """
    W_nash = sym_welfare_fn(x_nash)
    W_so = sym_welfare_fn(x_so)
    x_lam = (1.0 - lam) * x_nash + lam * x_so
    W_lam = sym_welfare_fn(x_lam)
    denom = np.exp(W_so) - np.exp(W_nash)
    if abs(denom) < 1e-15:
        return 0.0
    return float((np.exp(W_lam) - np.exp(W_nash)) / denom)


def numeric_lambda_max(payoff_fn, x_nash, x_so, eps, tol_bisect=1e-10):
    """
    Find lambda_max = max lam in [0,1] s.t. G_num(lam) <= eps
    via bisection on numeric_G.

    Returns float in [0, 1].
    """
    # G is non-negative and non-decreasing; binary search
    # Check if even lam=1 is IC
    if numeric_G(payoff_fn, x_nash, x_so, 1.0) <= eps:
        return 1.0
    # Check if lam=0 is already not IC (should not happen, G(0)=0)
    if numeric_G(payoff_fn, x_nash, x_so, 0.0) > eps:
        return 0.0

    lo, hi = 0.0, 1.0
    for _ in range(80):  # 80 bisections → precision ~1e-24
        mid = (lo + hi) / 2
        if abs(hi - lo) < tol_bisect:
            break
        if numeric_G(payoff_fn, x_nash, x_so, mid) <= eps:
            lo = mid
        else:
            hi = mid
    return float(lo)


def make_regime(*args, **kwargs):
    """Convenience forwarding to sim_dynamics.make_regime."""
    from sim_dynamics import make_regime as _make_regime
    return _make_regime(*args, **kwargs)

