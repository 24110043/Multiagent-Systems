"""
Plotting Script: Publication-Quality Theoretical Game Theory Figures
CS 329: Foundations of AI: Multi-Agent Systems (2026)
Indian Institute of Technology Gandhinagar

Component: Game-Theoretic Formulation & Price of Anarchy Visualizations
Regime A Calibrated Empirical Model (N=809, kappa=8.0)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from game_theory_solver import PotentialGame

# Set publication style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "figure.dpi": 300,
})


def plot_01_best_response_and_nash(empirical_game: PotentialGame, save_path: str):
    """Plot 1: Best-Response function and Nash equilibrium fixed point."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x_grid = np.linspace(0, 1, 300)

    # Panel A: Best-Response Function vs 45-degree Line
    ax = axes[0]
    br_empirical = empirical_game.best_response(x_grid)
    ax.plot(x_grid, x_grid, "k--", alpha=0.6, label="45° Line (Symmetric Profile $x_i = \\bar{x}_{-i}$)")
    ax.plot(x_grid, br_empirical, color="#1f77b4", linewidth=2.5,
            label=f"Best Response $\\text{{BR}}_i(\\bar{{x}}_{{-i}})$ (slope $s = {empirical_game.best_response_slope:.3f}$)")
    ax.scatter([empirical_game.x_nash], [empirical_game.x_nash], color="#d62728", s=110, zorder=5,
               label=f"Nash Equilibrium $x^* = {empirical_game.x_nash:.4f}$")
    ax.scatter([empirical_game.x_so], [empirical_game.x_so], color="#2ca02c", s=90, marker="s", zorder=5,
               label=f"Social Optimum $x^{{\\text{{SO}}}} = {empirical_game.x_so:.4f}$")
    ax.axvline(empirical_game.x_nash, color="#d62728", linestyle=":", alpha=0.6)
    ax.axvline(empirical_game.x_so, color="#2ca02c", linestyle=":", alpha=0.6)
    ax.set_title(f"Panel A: Calibrated Empirical Best Response (Option B, $\\kappa=8$)\n"
                 f"$\\beta_1={empirical_game.beta_1:.3f}, \\beta_2={empirical_game.beta_2:.3f}, \\beta_5={empirical_game.beta_5:.3f}$")
    ax.set_xlabel("Competitor Average Click Appeal $\\bar{x}_{-i}$")
    ax.set_ylabel("Focal Creator Optimal Strategy $x_i$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", framealpha=0.95)

    # Panel B: Best-Response Cobweb / Contraction Trajectory
    ax = axes[1]
    ax.plot(x_grid, x_grid, "k--", alpha=0.6, label="45° Line")
    ax.plot(x_grid, br_empirical, color="#1f77b4", linewidth=2.0, label="Best Response")
    
    # Trace contraction trajectory from x0 = 0.8
    x_curr = 0.8
    traj_x = [x_curr]
    traj_y = [0.0]
    for _ in range(8):
        x_next = float(empirical_game.best_response(np.array([x_curr]))[0])
        traj_x.extend([x_curr, x_next])
        traj_y.extend([x_next, x_next])
        x_curr = x_next
    
    ax.plot(traj_x, traj_y, color="#e377c2", linewidth=1.8, linestyle="-", marker=".",
            label=f"Contraction Iteration ($k = {empirical_game.contraction_rate:.4f} < 1$)")
    ax.scatter([empirical_game.x_nash], [empirical_game.x_nash], color="#d62728", s=110, zorder=5,
               label=f"Stable Fixed Point $x^* = {empirical_game.x_nash:.1f}$")
    ax.set_title(f"Panel B: Contraction Dynamics to Unique Nash\nConvergence Rate $k = 0.8524 < 1.0$")
    ax.set_xlabel("State at Iteration $t$ ($x_t$)")
    ax.set_ylabel("State at Iteration $t+1$ ($x_{t+1}$)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Generated Plot 1: {save_path}")


def plot_02_welfare_nash_vs_social_optimum(empirical_game: PotentialGame, save_path: str):
    """Plot 2: Symmetric Welfare curve, Nash vs Social Optimum, and Price of Anarchy."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x_grid = np.linspace(0, 1, 300)

    # Panel A: Symmetric Welfare curve
    ax = axes[0]
    w_emp = empirical_game.social_welfare_symmetric(x_grid)
    w_nash = empirical_game.social_welfare_symmetric(empirical_game.x_nash)
    w_so = empirical_game.social_welfare_symmetric(empirical_game.x_so)
    ax.plot(x_grid, w_emp, color="#1f77b4", linewidth=2.5, label="Symmetric Welfare $\\pi(x, x)$")
    ax.scatter([empirical_game.x_nash], [w_nash], color="#d62728", s=100, zorder=5,
               label=f"Nash Equilibrium ($x^*={empirical_game.x_nash:.2f}, \\pi={w_nash:.2f}$)")
    ax.scatter([empirical_game.x_so], [w_so], color="#2ca02c", s=100, marker="s", zorder=5,
               label=f"Social Optimum ($x^{{\\text{{SO}}}}={empirical_game.x_so:.2f}, \\pi={w_so:.2f}$)")
    ax.axvline(empirical_game.x_nash, color="#d62728", linestyle=":", alpha=0.7)
    ax.axvline(empirical_game.x_so, color="#2ca02c", linestyle=":", alpha=0.7)
    gap_a = w_so - w_nash
    ax.fill_between(x_grid, w_nash, w_emp, where=(x_grid >= empirical_game.x_nash) & (x_grid <= empirical_game.x_so),
                    color="#2ca02c", alpha=0.15,
                    label=f"Welfare Gap $\\Delta W = {gap_a:.3f}$\n$\\text{{PoA}} = {empirical_game.price_of_anarchy:.3f}$")
    ax.set_title("Panel A: Symmetric Welfare Landscape\nUnder-Investment Gap: $x^* = 0.0$ vs $x^{\\text{SO}} = 1.0$")
    ax.set_xlabel("Common Click Appeal Score $x$")
    ax.set_ylabel("Expected Log-Views Welfare $\\pi(x, x)$")
    ax.legend(loc="upper left", framealpha=0.95)

    # Panel B: Mediator Intervention Recovery Curve
    ax = axes[1]
    lambdas = np.linspace(0, 1, 200)
    R_vals = [empirical_game.welfare_recovery(lam) for lam in lambdas]
    ax.plot(lambdas, R_vals, color="#2ca02c", linewidth=2.5, label="Welfare Recovery $R(\\lambda)$")
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.7)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Full Recovery ($R=1.0$)")
    
    # Mark R=0 crossing and minimum dip
    min_R_idx = np.argmin(R_vals)
    ax.scatter([lambdas[min_R_idx]], [R_vals[min_R_idx]], color="#d62728", s=80, zorder=5,
               label=f"Convex Dip ($R={R_vals[min_R_idx]:.3f}$ at $\\lambda={lambdas[min_R_idx]:.2f}$)")
    
    ax.set_title("Panel B: Platform Mediator Recovery Curve $R(\\lambda)$\n" + r"Convex Dip for Weak Interventions ($\lambda \leq 0.07$)")
    ax.set_xlabel("Mediator Intervention Strength $\\lambda \\in [0, 1]$")
    ax.set_ylabel("Welfare Recovery Fraction $R(\\lambda)$")
    ax.legend(loc="lower right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Generated Plot 2: {save_path}")


def plot_03_potential_landscape(empirical_game: PotentialGame, save_path: str):
    """Plot 3: 2D Contour map of Exact Potential Function Phi(x1, x2)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    grid = np.linspace(0, 1, 200)
    X1, X2 = np.meshgrid(grid, grid)
    
    # Evaluate potential on 2-agent slice
    b1 = empirical_game.beta_1
    b2 = empirical_game.beta_2
    b5 = empirical_game.beta_5
    
    # Phi(x_1, x_2) = (b1*x1 + b2*x1^2) + (b1*x2 + b2*x2^2) + b5*x1*x2
    Phi = (b1 * X1 + b2 * (X1 ** 2)) + (b1 * X2 + b2 * (X2 ** 2)) + b5 * X1 * X2
    
    cp = ax.contourf(X1, X2, Phi, levels=25, cmap="viridis")
    fig.colorbar(cp, ax=ax, label="Potential Function $\\Phi(x_1, x_2)$")
    
    # Nash equilibrium point
    x_n = empirical_game.x_nash
    ax.scatter([x_n], [x_n], color="#d62728", s=120, edgecolors="white", linewidths=1.5, zorder=6,
               label=f"Nash Maximizer $(x^*, x^*) = ({x_n:.2f}, {x_n:.2f})$")
    ax.scatter([empirical_game.x_so], [empirical_game.x_so], color="#2ca02c", s=100, marker="s", edgecolors="white", linewidths=1.5, zorder=6,
               label=f"Social Optimum $(x^{{\\text{{SO}}}}, x^{{\\text{{SO}}}}) = ({empirical_game.x_so:.2f}, {empirical_game.x_so:.2f})$")
    ax.plot([0, 1], [0, 1], "w--", alpha=0.7, label="Symmetric Diagonal ($x_1 = x_2$)")
    
    ax.set_title(r"Exact Potential Game Landscape $\Phi(x_1, x_2)$" + "\n" + f"$\\beta_1={b1:.2f}, \\beta_2={b2:.2f}, \\beta_5={b5:.2f}$")
    ax.set_xlabel("Creator 1 Click Appeal $x_1$")
    ax.set_ylabel("Creator 2 Click Appeal $x_2$")
    ax.legend(loc="upper left", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Generated Plot 3: {save_path}")


def plot_04_poa_sensitivity_heatmap(save_path: str):
    """Plot 4: 2D Heatmaps of Price of Anarchy across (beta_2, beta_5) for empirical parameters."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    b2_vals = np.linspace(-25, -5, 100)
    b5_vals = np.linspace(0, 30, 100)
    B2, B5 = np.meshgrid(b2_vals, b5_vals)
    b1 = -2.0190

    panels = [
        {"ax": axes[0], "b3": -5.5424, "title": r"Panel A: Calibrated Negative Congestion ($\beta_3 = -5.54$)" + "\n" + r"Empirical Calibrated Baseline ($\text{PoA} = 2.817$)", "show_marker": True},
        {"ax": axes[1], "b3": 0.0, "title": r"Panel B: Zero Peer Congestion ($\beta_3 = 0.0$)" + "\n" + r"Baseline Strategic Interaction Landscape", "show_marker": False},
    ]

    for p in panels:
        ax = p["ax"]
        b3 = p["b3"]
        PoA = np.zeros_like(B2)
        for i in range(len(b5_vals)):
            for j in range(len(b2_vals)):
                b2 = B2[i, j]
                b5 = B5[i, j]
                abs_b2 = abs(b2)
                B = abs_b2 - b5
                # Stability & concavity
                if abs(b5) < (2.0 * abs_b2):
                    denom_n = 2.0 * abs_b2 - b5
                    x_n = np.clip(b1 / denom_n, 0.0, 1.0)
                    denom_s = 2.0 * B if B != 0 else 1.0
                    x_s = np.clip((b1 + b3) / denom_s, 0.0, 1.0) if B > 0 else 1.0
                    w_s = (b1 + b3) * x_s + (b2 + b5) * (x_s ** 2)
                    w_n = (b1 + b3) * x_n + (b2 + b5) * (x_n ** 2)
                    PoA[i, j] = np.exp(np.clip(w_s - w_n, 0.0, 10.0))
                else:
                    PoA[i, j] = np.nan

        im = ax.imshow(PoA, extent=[b2_vals[0], b2_vals[-1], b5_vals[0], b5_vals[-1]],
                       origin="lower", aspect="auto", cmap="magma", vmin=1.0, vmax=5.0)
        fig.colorbar(im, ax=ax, label=r"Price of Anarchy ($\text{PoA} = \exp(W(x^{\text{SO}}) - W(x^*))$)")

        if p["show_marker"]:
            ax.scatter([-12.1974], [20.7942], color="#00ffcc", s=140, marker="o", edgecolors="black", linewidths=1.8, zorder=6,
                       label=r"Calibrated Baseline ($\beta_2^{\mathrm{eff}} = -12.20, \beta_5 = 20.79$)")
            ax.legend(loc="upper left", framealpha=0.9)

        ax.set_title(p["title"])
        ax.set_xlabel(r"Curvature Parameter $\beta_2^{\mathrm{eff}} = \beta_2 - \kappa/2$")
        ax.set_ylabel(r"Strategic Interaction $\beta_5$")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Generated Plot 4: {save_path}")


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    json_path = os.path.abspath(os.path.join(base_dir, "..", "payoff_calculation", "results", "game_theory_coefficients.json"))
    if not os.path.exists(json_path):
        json_path = os.path.abspath(os.path.join(base_dir, "..", "payoff_estimation", "results", "game_theory_coefficients.json"))
    
    empirical_game = PotentialGame.from_json(json_path, n_creators=38)
    
    plot_01_best_response_and_nash(empirical_game, os.path.join(plots_dir, "01_best_response_and_nash.png"))
    plot_02_welfare_nash_vs_social_optimum(empirical_game, os.path.join(plots_dir, "02_welfare_nash_vs_social_optimum.png"))
    plot_03_potential_landscape(empirical_game, os.path.join(plots_dir, "03_potential_function_landscape.png"))
    plot_04_poa_sensitivity_heatmap(os.path.join(plots_dir, "04_poa_sensitivity_heatmap.png"))
    
    print("\nAll Part 3 theoretical figures generated successfully!")


if __name__ == "__main__":
    main()
