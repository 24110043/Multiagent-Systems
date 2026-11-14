# Empirical Payoff Function Estimation

**CS 329: Foundations of AI: Multi-Agent Systems (2026)**  
*Indian Institute of Technology Gandhinagar*  
**Project:** Click Appeal as a Potential Game: Modeling Creator Incentives, Attention Congestion, and Social Welfare in YouTube Recommendation  
**Author:** Bhavya Shah (Roll: 24110324) — *Payoff Function Estimation, Regression Analysis & Statistical Evaluation*  
**Collaborators:**  
- Ankita Kushwaha (24110043) — Data Preprocessing & Click Appeal Score  
- Khushi Shah (24110326) — Game-Theoretic Formulation & Social Welfare Analysis  
- Harshita Singh (24110342) — Platform Mediation & Multi-Agent Learning Dynamics  

---

## 1. Executive Summary & Purpose

This component provides the empirical econometric foundation for modeling YouTube competition as an exact potential game. Rather than assuming hypothetical or synthetic creator payoffs, we estimate the parameters of the empirical payoff function directly from real-world multimodal video data ($N = 186$ videos across 38 creator channels).

Our focus is on **primary payoff-function estimation and statistical inference**, not building a black-box predictive model. We estimate the primary models on the full available dataset, evaluate five progressively richer OLS specifications, assess generalization via grouped cross-validation (keeping all videos from each creator together), and export the fitted parameters with full uncertainty metrics for game-theoretic analysis.

---

## 2. Model Specifications (Progressive OLS Hierarchy)

We estimate and compare five progressively richer OLS models to evaluate how each theoretical component contributes to explaining creator payoffs:

$$\log(V_{it}) = \text{Payoff of Creator } i \text{ on Video } t$$

1. **Model 1 (Own Click Appeal Score):**
   $$\log(V_{it}) = \beta_0 + \beta_1 x_{it} + \varepsilon_{it}$$
   *Tests whether own packaging intensity has a linear bivariate association with view counts.*

2. **Model 2 (Quadratic Own-Score — Diminishing Returns):**
   $$\log(V_{it}) = \beta_0 + \beta_1 x_{it} + \beta_2 x_{it}^2 + \varepsilon_{it}$$
   *Tests the core theoretical hypothesis of diminishing returns and strict concavity ($\beta_1 > 0, \beta_2 < 0$), establishing an interior optimal strategy.*

3. **Model 3 (Competitor Average Appeal — Additive Externality):**
   $$\log(V_{it}) = \beta_0 + \beta_1 x_{it} + \beta_2 x_{it}^2 + \beta_3 \bar{x}_{-i,t} + \varepsilon_{it}$$
   *Tests the attention-congestion hypothesis ($\beta_3 < 0$), where heightened competitor packaging shifts down the focal creator's payoff level.*

4. **Model 4 (Strategic Interaction Term — Game-Theoretic Core):**
   $$\log(V_{it}) = \beta_0 + \beta_1 x_{it} + \beta_2 x_{it}^2 + \beta_3 \bar{x}_{-i,t} + \beta_5 (x_{it} \times \bar{x}_{-i,t}) + \varepsilon_{it}$$
   *Tests whether competitor packaging alters the marginal return to own click appeal ($\beta_5 > 0$, strategic complements).*

5. **Model 5 (Full Proposed Model with Exogenous Controls — Proposal Eq. 1):**
   $$\log(V_{it}) = \beta_0 + \beta_1 x_{it} + \beta_2 x_{it}^2 + \beta_3 \bar{x}_{-i,t} + \beta_5 (x_{it} \times \bar{x}_{-i,t}) + \beta_4 \log(s_i) + \gamma_1 \log(\text{duration}_{it}) + \gamma_2 \text{age}_{it} + \varepsilon_{it}$$
   *Controls for creator subscriber base $\log(s_i)$, video duration, and elapsed publication days.*

---

## 3. Data Integration & Feasibility Audit

The estimation pipeline ingests `Click appeal score/scores/video_scores.csv` without fabricating missing data or modifying upstream assets:

| Dimension | Empirical Finding | Methodological Implication |
| :--- | :--- | :--- |
| **Sample Size** | $N = 186$ videos | Full dataset utilized for primary regression estimation. |
| **Creator Panels** | 38 channels (37 with 5 videos, 1 with 1 video) | Unbalanced panel with creator-level clustering. Requires **Channel-Clustered Standard Errors**. |
| **Subscriber Counts** | 100% available ($1.18 \times 10^5 \le s_i \le 5.24 \times 10^8$) | Proposed full model with $\beta_4 \log(s_i)$ is fully estimable without imputation. |
| **Dependent Variable** | Min $= 1,132$, Max $= 215,604,900$, Median $= 1.38 \times 10^6$ | Zero non-positive views; $\log(V_{it})$ is smooth and approximately normally distributed. |
| **Temporal Span** | 2024-10-02 to 2026-10-01 (728 days; 24 calendar months) | Unequal temporal distribution: 2024-10 has 32 videos, 2026-09 has 31 videos, while other months have as few as 1 video. |

---

## 4. Definition and Defense of the Competition Variable

Defining competitor average click appeal ($\bar{x}_{-i,t}$) requires careful consideration of YouTube's discovery mechanics and the available data:

1. **Why Strict Calendar Months Are Not Defensible:**
   In our sample of 186 videos, there are 182 distinct (creator, month) pairs. In 3 calendar months, only a single video was published; in several others, only 1 channel was active. A strict monthly grouping results in undefined (NaN) peer averages for isolated videos and extreme small-sample noise.
2. **Chosen Specification: Rolling 90-Day Window LOCO ($t \pm 45$ days):**
   Viewer attention on YouTube is contested primarily among videos actively circulating in discovery algorithms concurrently. We define the competitor pool $\mathcal{C}_{-i,t}$ as all videos published by competing creators $j \neq i$ within a 90-day window centered at video $t$:
   $$\bar{x}_{-i,t} = \frac{1}{|\mathcal{C}_{-i,t}|} \sum_{k \in \mathcal{C}_{-i,t}} x_k$$
   - **Exclusion of Focal Creator:** Strictly satisfied ($\partial \bar{x}_{-i,t} / \partial x_{it} = 0$).
   - **Defensibility Check:** Across the 186 videos, the rolling 90-day peer pool contains a median of 24.5 competitor videos (mean $= 24.5$, IQR $= [19, 32]$). For 4 boundary cases where $|\mathcal{C}_{-i,t}| < 3$, a smooth fallback to Global LOCO prevents erratic peer outliers.
3. **Robustness Check: Global Niche LOCO:**
   Averages all 181 competing videos from the other 37 creators across the full two-year sample.

---

## 5. Full-Sample Econometric Estimation Results

All models are estimated using Ordinary Least Squares on the full dataset ($N=186$) with **Cluster-Robust Standard Errors** clustered at the channel level (38 clusters), guarding against deflated standard errors from repeated observations.

### Model Comparison Table (Full Dataset In-Sample vs. LOCO Cross-Validation)

| Model | Specification | In-Sample $R^2$ | In-Sample Adj. $R^2$ | AIC | In-Sample RMSE | OOS RMSE (LOCO-CV) | OOS $R^2$ (LOCO-CV) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1** | $x$ | $0.0000$ | $-0.0054$ | $861.0$ | $2.4295$ | $2.5567$ | $-0.1134$ |
| **Model 2** | $x, x^2$ | $0.0828$ | $0.0728$ | $847.0$ | $2.3267$ | $2.4665$ | $-0.0363$ |
| **Model 3** | $x, x^2, \bar{x}$ | $0.0844$ | $0.0693$ | $848.6$ | $2.3248$ | $2.4719$ | $-0.0379$ |
| **Model 4** | $x, x^2, \bar{x}, x\bar{x}$ | $0.0850$ | $0.0647$ | $850.5$ | $2.3239$ | $2.4832$ | $-0.0468$ |
| **Model 5** | $x, x^2, \bar{x}, x\bar{x}, \log(s), \log(\text{dur}), \text{age}$ | **$0.6479$** | **$0.6340$** | **$678.9$** | **$1.4219$** | **$1.5817$** | **$+0.5739$** |
| **Model 5 (Centered)** | $x_c, x_c^2, \bar{x}_c, x_c\bar{x}_c, \log(s), \log(\text{dur}), \text{age}$ | **$0.6479$** | **$0.6340$** | **$678.9$** | **$1.4219$** | **$1.5817$** | **$+0.5739$** |

### Detailed Coefficients for Progressive Specifications (Cluster-Robust SE)

```text
Model 1 (Linear Own-Score):
  log(V) = 14.07 + 0.16 x                 (x SE=3.89, p=0.966)

Model 2 (Quadratic Own-Score — Significant Concavity Without Controls):
  log(V) = 8.52 + 33.31 x - 44.34 x^2     (x SE=13.86, p=0.016; x^2 SE=16.02, p=0.0056)
  --> Unconditional concavity (beta_2 < 0) significant at 1%; peak at x* = 0.3756.
  --> NOTE: When subscriber scale is omitted, packaging acts as an indirect proxy for creator scale.

Model 3 (Add Competitor Average):
  log(V) = 6.25 + 33.94 x - 45.07 x^2 + 5.74 x_bar   (x SE=13.66, p=0.013; x^2 SE=15.85, p=0.0045)

Model 4 (Add Strategic Interaction):
  log(V) = 12.63 + 17.37 x - 44.59 x^2 - 10.24 x_bar + 40.60 (x * x_bar)

Model 5 (Full Proposed Payoff Specification — Uncentered):
  log(V) = -7.9837 + 8.3020 x - 9.1677 x^2 + 3.8021 x_bar - 0.7822 (x * x_bar) 
           + 1.0668 log(s) + 0.2744 log(dur) + 0.0001 age
  --> R^2 = 64.79%, log(s) SE=0.1141 (p = 8.8e-21) [STATISTICALLY SIGNIFICANT]
  --> beta_1 SE=20.255, p=0.6819; beta_2 SE=8.438, p=0.2773; beta_5 SE=46.475, p=0.9866

Model 5 (Full Proposed Payoff Specification — Mean-Centered):
  log(V) = -4.7531 + 0.7349 x_c - 9.1677 x_c^2 + 3.4925 x_bar_c - 0.7822 (x_c * x_bar_c)
           + 1.0668 log(s) + 0.2744 log(dur) + 0.0001 age
  --> beta_1^c SE=1.7746, p=0.6788 (standard error drops by 91% due to collinearity removal)
  --> beta_2 and beta_5 are mathematically identical to uncentered model.
```

---

## 6. Robustness & Generalization via Grouped Cross-Validation

To guard against data leakage caused by repeated observations from the same creator, we implement **strict Leave-One-Channel-Out (LOCO) Cross-Validation** (38 folds). In each fold, all videos from one creator channel are held out, competitor peer averages for both training and test sets are calculated strictly using only the training fold, the model is trained on the remaining 37 channels, and predictions are evaluated strictly out-of-sample:

### Findings & Sample Constraints:
1. **Models 1–4 Out-of-Sample Performance:**
   Models 1–4 have negative out-of-sample $R^2$ ($R^2_{OOS} \approx -0.04$ to $-0.11$). This occurs because between-channel view scales range from 50,000 to 50,000,000 views. A model relying solely on title/thumbnail appeal cannot guess the baseline audience level of an unseen creator.
2. **Model 5 Out-of-Sample Generalization:**
   Model 5 achieves an Out-of-Sample $R^2_{OOS} = \mathbf{+0.5739}$ (57.4% of out-of-sample view variance explained across completely unseen channels!), with OOS RMSE dropping from $2.48$ to $1.58$.
3. **Sample Limitations:** With $N=186$ across 38 channels, estimating 8 parameters in a grouped cross-validation scheme reduces the effective sample size per fold to 37 clusters. The out-of-sample stability of Model 5 confirms that audience scale $\log(s_i)$ is statistically essential for cross-creator generalization.

---

## 7. Multicollinearity & Centering Resolution

In polynomial and interaction models, uncentered product terms ($x^2$ and $x\bar{x}$) correlate with linear terms, inflating Variance Inflation Factors:
- **Raw Uncentered Model 5:** $\text{VIF}(x) = 824.1$, $\text{VIF}(x\bar{x}) = 770.9$.
- **Mean-Centered Model 5 ($x_c = x - \bar{x}$):** All VIFs drop below **1.25** ($\text{VIF}(x_c) = 1.16$, $\text{VIF}(x_c^2) = 1.25$, $\text{VIF}(\bar{x}_c) = 1.07$, $\text{VIF}(x_c\bar{x}_c) = 1.11$).
- **Invariance:** Mean-centering produces **identical** curvature ($\beta_2 = -9.1677$) and interaction ($\beta_5 = -0.7822$), proving that concavity is structural and not an artifact of multicollinearity. Centering changes the linear slope $\beta_1^c = +0.7349$ to represent the marginal effect at the sample mean ($x = 0.3958, \bar{x} = 0.3954$).

---

## 8. Machine-Readable Game-Theory Handover (For Khushi Shah)

Per project design, equilibrium derivation (Nash equilibrium, social optimum, Price of Anarchy) is reserved for the game-theory component. We export the empirical payoff function parameters and their uncertainty structure to:
- [`results/game_theory_coefficients.json`](file:///c:/Users/bhavy/OneDrive/Documents/GitHub/Multiagent-Systems/payoff_estimation/results/game_theory_coefficients.json)
- [`results/game_theory_coefficients.csv`](file:///c:/Users/bhavy/OneDrive/Documents/GitHub/Multiagent-Systems/payoff_estimation/results/game_theory_coefficients.csv)

### Mathematical Well-Posedness vs. Statistical Significance:

| Property | Formula | Point Estimate | Mathematical Status | Statistical Status (95% CI) |
| :--- | :--- | :---: | :---: | :---: |
| **Strict Concavity** | $\beta_2 < 0$ | $-9.1677$ | **Holds** (strictly concave) | $[-25.71, +7.37]$ (includes zero; $p=0.28$) |
| **Contraction Mapping / Unique Nash** | $|\beta_5| < 2\|\beta_2\|$ | $\|-0.7822\| < 18.335$ | **Holds** (rate $= 0.0427 < 1$) | $[-91.87, +90.31]$ (includes zero; $p=0.99$) |
| **Social Welfare Strict Concavity** | $\beta_2 + \beta_5 < 0$ | $-9.9499 < 0$ | **Holds** (strictly concave) | Wide CI spanning positive and negative |

> **Crucial Guidance for Downstream Teammate (Khushi Shah):**  
> 1. **Calibrated Baseline:** The empirical point estimates provide a mathematically well-behaved foundation for computing an interior symmetric Nash equilibrium ($x^* \approx 0.435$) and Social Optimum ($x^{SO} \approx 0.417$) within the observed domain $[0.07, 0.66]$.  
> 2. **Sensitivity Analysis Required:** Because the confidence intervals for $\beta_1, \beta_2, \beta_3, \beta_5$ in Model 5 include zero, downstream analysis MUST NOT treat these point estimates as undisputed ground truth. The game-theory module should present results across a sensitivity grid (e.g., varying $\beta_2 \in [-20, -5]$ and $\beta_5 \in [-5, +5]$) and transparently state that packaging coefficients have high estimation uncertainty in the 186-video sample.

---

## 9. Publication-Quality Visualizations

Generated plots in `plots/`:
1. `01_model_progression_comparison.png`: Explanatory power progression ($R^2$ and RMSE) comparing in-sample vs LOCO cross-validation across Models 1–5.
2. `02_actual_vs_predicted.png`: Scatter plot of observed vs fitted $\log(V_{it})$ for Model 5 ($R^2 = 0.648, \text{RMSE} = 1.42$).
3. `03_residual_diagnostics.png`: 4-panel residual diagnostics (Residuals vs Fitted, Normal Q-Q, Histogram/KDE, Channel cluster dispersion).
4. `04_payoff_response_curves.png`: Empirical payoff function $\pi_i(x_i, \bar{x}_{-i})$ across low, median, and high competitor levels, illustrating quadratic concavity ($\beta_2 = -9.168$).
5. `05_peer_interaction_heatmap.png`: 2D strategic payoff landscape contour map across own click appeal and competitor average.
6. `06_variable_importance_vif.png`: Multicollinearity resolution comparing uncentered vs centered VIFs.
7. `07_grouped_cv_channel_errors.png`: Leave-One-Channel-Out Cross-Validation error distribution across all 38 creator channels.

---

## 10. Research Limitations & Cautionary Framing

1. **Association vs. Causation:** This model estimates observational statistical associations on YouTube. We do not claim exogenous random assignment or direct causal identification.
2. **Views vs. CTR:** Cumulative views are an aggregate platform outcome governed by algorithmic recommendations, subscribers, and watch time—not pure click-through rate (CTR). Our dataset contains views, not impressions or CTR.
3. **Dominance of Subscriber Scale:** Channel subscriber scale accounts for over $60\%$ of view variance. Title/thumbnail packaging operates as a fine-tuning marginal incentive ($x_{it}$), not a substitute for audience distribution.

---

## 11. Pipeline Execution

To reproduce all results end-to-end:
```bash
python payoff_estimation/payoff_estimation.py
```
