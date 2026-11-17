# Empirical Payoff Function Estimation: Final Comprehensive Research Report

**CS 329: Foundations of AI: Multi-Agent Systems (Autumn 2026)**  
*Indian Institute of Technology Gandhinagar*  
**Research Project:** Click Appeal as a Potential Game: Modeling Creator Incentives, Attention Congestion, and Social Welfare in YouTube  
**Author:** Bhavya Shah (Roll: 24110324) — *Empirical Payoff Function Estimation & Econometric Analysis*  
**Co-Investigators:**  
- Ankita Kushwaha (24110043) — *Multimodal Feature Extraction & Scoring*  
- Harshita Singh (24110342) — *Platform Mediation & Algorithmic Simulation*  
- Khushi Shah (24110326) — *Game Theory, Potential Function Analysis & Social Welfare*  

**Report Status:** Post-Audit Final Econometric Calibration & Verification  
**Authoritative Artifact Directory:** `payoff_calculation/`  
**Latest Pipeline Run Timestamp:** November 2026  

---

## 1. Executive Summary

This report establishes the empirical foundation for the project *"Click Appeal as a Potential Game"*. The central purpose of this econometric component is to estimate the structural payoff function of YouTube video creators directly from multimodal feature data ($N = 809$ videos across $G = 38$ distinct YouTube channels), replacing arbitrary or synthetic payoff functions with real-world empirical parameters.

### Core Achievements & Key Empirical Findings
1. **Full-Pipeline Execution:** Successfully implemented, audited, and executed all 10 stages of the empirical payoff estimation architecture with zero data leakage.
2. **Channel-Stratified Partitioning:** Partitioned the 38 channels into 26 Training channels ($N = 541$), 6 Validation channels ($N = 136$), and 6 Held-Out Test channels ($N = 132$), preserving exact subscriber-tier distributions.
3. **Honest Out-of-Sample Performance Assessment:**
   - **Test Set Evaluation ($N = 132$, $G = 6$):** The baseline controls-only model (Model 0) achieves $R^2_{\mathrm{OOS}} = 0.7535$, while the full strategic model (Model 4) achieves $R^2_{\mathrm{OOS}} = 0.7537$ ($\Delta R^2_{\mathrm{OOS}} = +0.0002$). Packaging terms add only $0.02$ percentage points of predictive power out-of-sample.
   - **26-Fold LOCO-CV on Training Cohort ($N = 541$):** Model 0 achieves $R^2_{\mathrm{OOS}} = 0.5741$, outperforming Model 4 ($R^2_{\mathrm{OOS}} = 0.5565$).
   - **Validation Cohort ($N = 136$, $G = 6$):** Model 0 achieves $R^2_{\mathrm{OOS}} = 0.4768$ and Model 1 achieves $0.4035$, both outperforming Model 4 ($R^2_{\mathrm{OOS}} = 0.3965$).
   - **Specification Locking Decision:** Model 4-Centered was locked **strictly by theoretical necessity**, not because it achieved the highest predictive score. It is the minimal specification that incorporates the required game-theoretic terms: quadratic own curvature ($\beta_2$), peer attention congestion ($\beta_3$), and strategic cross-packaging interaction ($\beta_5$).
4. **Variance Decomposition & Joint Hypothesis Testing:**
   - Creator scale (subscribers) and production controls explain $99.3\%$ of the total explained variance ($R^2_{\mathrm{controls}} = 0.6308$ vs. $R^2_{\mathrm{full}} = 0.6353$, $\Delta R^2 = 0.0045$).
   - A joint Wald/F-test on all four packaging parameters in the full model yields $F = 0.4739$ ($p = 0.7546$), confirming that packaging variables are jointly statistically insignificant.
5. **Analytical Uncentering of Payoff Parameters:**
   - In accordance with the project paper (Equation 2), the uncentered structural payoff function parameters for click appeal $x \in [0, 1]$ were derived analytically:
     $$\hat{\beta}_{1\_u} = -2.0190, \quad \hat{\beta}_{2\_u} = -8.1974, \quad \hat{\beta}_{3\_u} = -5.5424, \quad \hat{\beta}_{5\_u} = +20.7942, \quad \hat{\beta}_{0\_u} = -1.7604$$
   - Delta-method covariance matrices $\mathbf{\Sigma}_u$ and clustered standard errors were derived and exported for downstream error propagation.
6. **Hypothesis Tests & Identification Challenges:**
   - None of the proposal's original hypotheses (H1–H4) are statistically supported by the data:
     - *H1 ($\beta_1 > 0$, positive own return):* not supported ($p = 0.92$) (uncentered point estimate $\hat{\beta}_{1\_u} = -2.0190$, $\mathrm{SE} = 20.9201$).
     - *H2 ($\beta_2 < 0$, diminishing returns):* Point estimate is concave ($\hat{\beta}_2 = -8.1974$), but statistically insignificant ($p_{\mathrm{CRSE}} = 0.3297$; Wild Cluster Bootstrap $p_{\mathrm{WCB}} = 0.2600$). Concavity is a calibrated modeling choice, not an established empirical fact.
     - *H3 ($\beta_3 < 0$, attention congestion):* Uncentered point estimate is negative ($\hat{\beta}_{3\_u} = -5.5424$), but statistically insignificant ($p = 0.8097$).
     - *H4 ($\beta_5 < 0$, strategic substitutes):* Point estimate is positive ($\hat{\beta}_5 = +20.7942$, suggesting strategic complements) and statistically insignificant ($p_{\mathrm{CRSE}} = 0.7318$; $p_{\mathrm{WCB}} = 0.7795$).
   - *Weak Identification of Peer Effects:* Across the 809 videos, own click appeal exhibits substantial variation ($\mathrm{SD}(x) = 0.1037$), but peer click appeal exhibits almost zero variation ($\mathrm{SD}(\bar{x}) = 0.0074$).Averaging over the ~20–30 active competing channels within a rolling 60-day window washes out peer variance, causing high standard errors and sample sensitivity ($\hat{\beta}_2 = +1.56, \hat{\beta}_5 = -44.88$ on training vs. $\hat{\beta}_2 = -8.20, \hat{\beta}_5 = +20.79$ on full sample).
7. **Corrected Bootstrap Inference ($B = 2{,}000$):**
   - Corrected cluster-pairs bootstrap (without within-cluster residual demeaning) yields standard errors that match cluster-robust standard errors across all regressors ($0.95 \le \mathrm{SE}_{\mathrm{boot}} / \mathrm{SE}_{\mathrm{CRSE}} \le 1.06$).
   - Across $2{,}000$ draws, $81.40\%$ ($1{,}628/2{,}000$) satisfy concavity ($\beta_2 < 0$).
   - Across $2{,}000$ draws, only $18.65\%$ ($373/2{,}000$) satisfy both concavity ($\beta_2 < 0$) and the contraction mapping condition ($\rho < 1$). The claim that the empirical estimates are "100% concave" was an artifact of flawed within-cluster demeaning and has been retracted.
8. **Output for Downstream Use (Game-Theoretic Parameters):**
   The uncentered structural payoff function parameters (for paper Eq. 2) and centering sample means derived from the full sample ($N = 809, G = 38$) are:
   $$b_{1\_u} = -2.0190, \quad b_{2\_u} = -8.1974, \quad b_{3\_u} = -5.5424, \quad b_{5\_u} = +20.7942$$
   $$b_{0\_u} = -1.7604, \quad \gamma_s = 1.0417, \quad \gamma_d = 0.0454, \quad \gamma_a = 0.0003$$
   $$\mu_x = 0.392395, \quad \mu_{\bar{x}} = 0.391390$$
   with delta-method covariance matrix $\mathbf{\Sigma}_u$ exported for downstream error propagation. Factually, $b_{5\_u} > 2|b_{2\_u}|$ and $b_{2\_u} + b_{5\_u} > 0$ at the point estimates, and only $18.65\%$ of bootstrap draws satisfy $|b_5| < 2|b_2|$. 
---

## 2. Step 3 Audit Reconciliation & Methodological Clarifications

Before finalizing Phase 2, two outstanding verification items from the Step 3 audit were resolved:

### 2.1 Resolution of Specification C Interaction $p$-Value Discrepancy
In the Step 3 exploratory audit, an apparent discrepancy was noted between a reported interaction $p$-value of $0.3868$ and $0.3950$. The exact mathematical origin has been verified:

For Specification C (Channel Fixed Effects, within-channel estimation):
- Point estimate: $\hat{\beta}_5 = -30.070106$
- Cluster-robust standard error: $\widehat{\mathrm{SE}}_{\mathrm{CRSE}} = 34.743048$
- Empirical $t$-statistic:
  $$t = \frac{-30.070106}{34.743048} = -0.865500$$

The two values reflect different reference distributions:
1. **Asymptotic Standard Normal ($z$-test):**
   $$p_z = 2 \times (1 - \Phi(|-0.865500|)) = 2 \times (1 - 0.806618) = \mathbf{0.386764} \approx 0.3868$$
   This is the default asymptotic $p$-value emitted by `statsmodels` when computing cluster-robust covariances (`cov_type='cluster'`).
2. **Finite-Sample Student-$t$ Distribution ($df = G - 1 = 25$):**
   $$p_t = 2 \times (1 - F_{t(25)}(|-0.865500|)) = 2 \times (1 - 0.802503) = \mathbf{0.394994} \approx 0.3950$$
   This is the small-sample adjusted $p$-value recommended by Cameron and Miller (2015) for moderate cluster counts ($G = 26$).

**Conclusion:** Both calculations are mathematically consistent and lead to the identical substantive conclusion: the strategic interaction term under within-channel fixed effects is statistically indistinguishable from zero ($p \approx 0.39$).

### 2.2 Backward-Looking Peer Window $[t - 60\text{d}, t)$ Robustness Analysis
We investigated whether defining competitor packaging contemporaneously ($[t - 30\text{d}, t + 30\text{d}]$) versus retrospectively ($[t - 60\text{d}, t)$) alters the empirical estimates. This robustness check was fully implemented in `scripts/09_backward_looking_robustness.py` and exported to `results/backward_looking_peer_robustness.csv`:

```
+--------------------------------+--------------------------------+------------+------------+------------+------------+--------+
| Sample                         | Peer Window Definition         |   beta_1   |   beta_2   |   beta_3   |   beta_5   |   R²   |
+--------------------------------+--------------------------------+------------+------------+------------+------------+--------+
| Training Cohort (N=541, G=26)  | Contemporaneous [t-30d, t+30d] |    +1.5980 |    +1.5578 |    -3.1497 |   -44.8756 | 0.6383 |
| Training Cohort (N=541, G=26)  | Backward-Looking [t-60d, t)    |    +1.7018 |    +2.2462 |    +1.3644 |   -24.2018 | 0.6379 |
| Full Sample (N=809, G=38)      | Contemporaneous [t-30d, t+30d] |    -0.3135 |    -8.1974 |    +2.6172 |   +20.7942 | 0.6353 |
| Full Sample (N=809, G=38)      | Backward-Looking [t-60d, t)    |    -0.3175 |    -8.2613 |    +0.4679 |   +32.4845 | 0.6356 |
+--------------------------------+--------------------------------+------------+------------+------------+------------+--------+
```

**Key Findings:**
1. In the full sample, $\beta_2$ remains strictly negative under both specifications ($-8.1974$ vs. $-8.2613$), and $\beta_5$ remains statistically insignificant ($p = 0.73$ vs. $p = 0.56$).
2. Creator scale elasticity ($\gamma_s$) remains stable at $1.0417$ (contemporaneous) vs. $1.0423$ (backward-looking), with $p < 0.0001$.
3. In the training sample (peers drawn only from the other 25 training channels), $\beta_2$ stays positive ($+1.5578$ vs. $+2.2462$) and $\beta_5$ stays negative ($-44.8756$ vs. $-24.2018$), but $\beta_3$ changes sign ($-3.1497$ vs. $+1.3644$); all four are statistically insignificant under both windows, so the training-sample sign pattern is not robust to the peer window.
4. The contemporaneous window $[t - 30\text{d}, t + 30\text{d}]$ is retained as the primary specification because YouTube's recommendation algorithm evaluates viewer impressions against competing videos actively circulating within the same 60-day window.

### 2.3 Strict Concavity vs. Interior Optimality & Cost Function Role
We clarify the exact theoretical distinction between concavity and interior optimality:
- **Strict Concavity** of the revenue function ($\beta_2 < 0$) guarantees that any unconstrained critical point is a unique global maximum on $\mathbb{R}$.
- **Interior Optimality** ($x^* \in (0, 1)$) requires that marginal net payoff crosses zero strictly within $(0, 1)$:
  $$\left. \frac{\partial \Pi_i}{\partial x_i} \right|_{x_i = 0} > 0 \quad \text{and} \quad \left. \frac{\partial \Pi_i}{\partial x_i} \right|_{x_i = 1} < 0$$
- **Role of Linear Costs ($C(x) = cx$):** Linear cost shifts the intercept and gradient of the marginal payoff curve downwards by $c$, but does NOT alter the second derivative ($\frac{\partial^2 \Pi_i}{\partial x_i^2} = 2 \beta_2$). Linear costs cannot alter the contraction ratio $\rho = \frac{|\beta_5|}{2|\beta_2|}$ and **cannot restore a contraction mapping**.

---

## 3. Data Ingestion & Sample Stratification (Steps 1 & 2)

### 3.1 Sample Dimensions and Validation
The data pipeline merges multimodal packaging metrics from `video_scores.csv` with YouTube metadata from `cleaned_videos.csv`:
- **Total Sample:** Exactly $N = 809$ unique videos across $G = 38$ distinct channels (zero lost records; zero duplicates).
- **Dependent Variable:** Video view counts range from $\min = 898$ to $\max = 336{,}331{,}041$ (median $= 1{,}297{,}727$). Zero non-positive views; $\ln(V_{it})$ is smooth, approximately Gaussian, and requires zero clipping.
- **Creator Scale:** Channel subscriber counts range range from $118{,}000$ to $526{,}000{,}000$. Standardized to channel-level median to eliminate minor temporal snapshot drift.
- **Video Duration:** Corrected data dictionary range: $\min = 181$ seconds (3.0 minutes) to $\max = 59{,}251$ seconds (16.46 hours, Ludwig charity livestream).

> [!NOTE]
> **Data vs. Proposal Scope Clarification:**  
> The empirical dataset covers 38 channels from **global English-language digital entertainment** (Gaming, Comedy, General Entertainment, Food/Tech Lifestyle) rather than the Indian tech-review niche originally proposed. The analyzed sample comprises 809 videos across creators spanning from $118\text{K}$ to $526\text{M}$ subscribers, capturing real-world cross-channel competition across global attention categories.

### 3.2 Channel-Level Stratified Partition Map
To prevent data leakage caused by repeated observations from the same creator, partitioning was executed strictly at the **channel level** ($seed = 42$) across three subscriber tiers:

| Subscriber Tier | Total Channels | Training Cohort ($G=26$) | Validation Cohort ($G=6$) | Held-Out Test Cohort ($G=6$) |
| :--- | :---: | :---: | :---: | :---: |
| **Tier 1: Low (<5M)** | 12 | 8 channels (174 videos) | 2 channels (50 videos) | 2 channels (48 videos) |
| **Tier 2: Mid (5M–20M)** | 16 | 11 channels (242 videos) | 3 channels (63 videos) | 2 channels (37 videos) |
| **Tier 3: Mega (>20M)** | 10 | 7 channels (125 videos) | 1 channel (23 videos) | 2 channels (47 videos) |
| **Total** | **38** | **26 channels (541 videos)** | **6 channels (136 videos)** | **6 channels (132 videos)** |

**Channel Allocations:**
- **Training Cohort (26 channels, 541 videos):** Adley, Brent Rivera, Chloe Ting, Clean Bandit, Cocomelon, DanTDM, FBE, Gourmet Makes, Grand Kitchen, Jennxpenn, Jesse, Karolina Protsenko, Kiara Madisen, Lucas and Marcus, Mark Wiens, Penguinz0, Rosanna Pansino, Ryan Trahan, Smosh, Supercar Blondie, The Food Ranger, The Slow Mo Guys, The Try Guys, Thinknoodles, Unspeakable, Zach King.
- **Validation Cohort (6 channels, 136 videos):** Max Fosh, Michelle Khare, Nick DiGiovanni, Randy Santel, Sorted Food, Wake'N'Bite.
- **Test Cohort (6 channels, 132 videos):** Airrack, Foodie We, How Ridiculous, Joel Hansen, Matt Stonie, Stokes Twins.

**Peer Pool Isolation Protocol:**
- **Training peer averages:** Calculated strictly within the other 25 training channels ($\mathrm{Train} \setminus \{i\}$).
- **Validation peer averages:** Calculated strictly using the 26 training channels.
- **Test peer averages:** Calculated strictly using the combined 32 training + validation channels.
Zero test or validation click appeals contaminated training feature construction.

---

## 4. Primary Model Development on Training Sample (Step 3)

The primary econometric hierarchy was fitted on the 26 training channels ($N = 541$):

$$\begin{aligned}
\text{Model 0 (Controls):} \quad & \ln(V_{it}) = \beta_0 + \gamma_s \ln(s_i) + \gamma_d \ln(\text{dur}_{it}) + \gamma_a \text{age}_{it} + \varepsilon_{it} \\
\text{Model 1 (Linear):} \quad & \ln(V_{it}) = \text{Model 0} + \beta_1^* \tilde{x}_{it} \\
\text{Model 2 (Quadratic):} \quad & \ln(V_{it}) = \text{Model 1} + \beta_2^* \tilde{x}_{it}^2 \\
\text{Model 3 (Peer Congestion):} \quad & \ln(V_{it}) = \text{Model 2} + \beta_3^* \tilde{\bar{x}}_{-i,t} \\
\text{Model 4 (Interaction):} \quad & \ln(V_{it}) = \text{Model 3} + \beta_5^* (\tilde{x}_{it} \cdot \tilde{\bar{x}}_{-i,t}) \\
\text{Model 5 (Full Tags):} \quad & \ln(V_{it}) = \text{Model 4} + \gamma_t \ln(\text{tags}_{it} + 1)
\end{aligned}$$

Where $\tilde{x}_{it} = x_{it} - \mu_x^{\mathrm{Train}}$ and $\tilde{\bar{x}}_{-i,t} = \bar{x}_{-i,t} - \mu_{\bar{x}}^{\mathrm{Train}}$ are centered at the training means ($\mu_x = 0.388862, \mu_{\bar{x}} = 0.388553$).

### 4.1 Training Estimation Summary Table ($N = 541, G = 26$)

| Parameter | Model 0 (Base) | Model 1 (Linear) | Model 2 (Quad) | Model 3 (Peer) | Model 4 (M4-Centered) | Model 5 (Tags) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intercept ($\beta_0^*$)** | $-1.7492$ | $-2.5995$ | $-2.0864$ | $-2.0805$ | $-2.0640$ | $-1.5429$ |
| (CRSE) | $(2.9507)$ | $(3.0238)$ | $(2.8053)$ | $(2.8078)$ | $(2.8020)$ | $(3.1280)$ |
| **Own Appeal ($\beta_1^*$)** | — | $+1.6116$ | $+1.7103$ | $+1.6403$ | $+1.5980$ | $+1.6087$ |
| (CRSE) | — | $(1.5512)$ | $(1.5958)$ | $(1.5958)$ | $(1.5855)$ | $(1.5404)$ |
| **Curvature ($\beta_2^*$)** | — | — | $+2.3245$ | $+2.2504$ | $+1.5578$ | $+0.9139$ |
| (CRSE) | — | — | $(6.0719)$ | $(6.0574)$ | $(6.2838)$ | $(6.3035)$ |
| **Peer Appeal ($\beta_3^*$)** | — | — | — | $-3.5146$ | $-3.1497$ | $-2.6288$ |
| (CRSE) | — | — | — | $(5.3634)$ | $(5.3997)$ | $(5.3401)$ |
| **Interaction ($\beta_5^*$)** | — | — | — | — | $-44.8756$ | $-41.2563$ |
| (CRSE) | — | — | — | — | $(59.3614)$ | $(58.9060)$ |
| **Log Subscribers ($\gamma_s$)** | $+1.0012^{***}$ | $+1.0157^{***}$ | $+1.0208^{***}$ | $+1.0221^{***}$ | $+1.0230^{***}$ | $+1.0114^{***}$ |
| (CRSE) | $(0.1303)$ | $(0.1267)$ | $(0.1250)$ | $(0.1243)$ | $(0.1240)$ | $(0.1251)$ |
| **Log Duration ($\gamma_d$)** | $-0.0006$ | $-0.0060$ | $-0.0047$ | $-0.0040$ | $-0.0081$ | $-0.0251$ |
| **Age Days ($\gamma_a$)** | $-0.0001$ | $-0.0000$ | $-0.0000$ | $-0.0001$ | $-0.0001$ | $-0.0001$ |
| **Log Tags+1 ($\gamma_t$)** | — | — | — | — | — | $-0.0918$ |
| **In-Sample $R^2$** | $0.6324$ | $0.6373$ | $0.6376$ | $0.6378$ | **$0.6383$** | $0.6405$ |
| **Akaike IC (AIC)** | $1914.2$ | $1908.9$ | $1910.4$ | $1912.1$ | **$1913.4$** | $1912.0$ |
| **Bayesian IC (BIC)** | $1931.3$ | $1930.4$ | $1936.2$ | $1942.1$ | **$1947.7$** | $1950.7$ |
| **Max VIF** | $1.12$ | $1.14$ | $1.19$ | $1.26$ | **$1.26$** | $1.27$ |

*Significance markers: $^{***} p < 0.001$. Standard errors are clustered by channel ($G = 26$). Models 2–4 are shown with mean-centered packaging variables ($x_c = x - \mu_x, \bar{x}_c = \bar{x} - \mu_{\bar{x}}$ with $\mu_x = 0.3892, \mu_{\bar{x}} = 0.3906$). Uncentered CSV estimates for linear $x$ in Models 2 and 3 are $-0.0975$ ($\mathrm{SE} = 4.5320$) and $-0.1099$ ($\mathrm{SE} = 4.5139$).*

### 4.2 Exact Economic Interpretation of Point Estimates
For log-transformed view counts, the exact percentage change in views resulting from a change $\Delta X$ is:
$$\% \Delta V = 100 \times \left( \exp(\beta \cdot \Delta X) - 1 \right)$$

1. **Own Click Appeal ($\hat{\beta}_1^* = +1.5980$, $p = 0.3135$):**
   A hypothetical $+0.10$ increase in own click appeal score from the mean predicts an estimated view change of:
   $$\% \Delta V = 100 \times (\exp(1.5980 \times 0.10) - 1) = +17.33\%$$
   *Statistical Caveat:* Because $\mathrm{SE} = 1.5855$ ($p = 0.31$), this estimate carries substantial sampling variance and is not statistically established at $\alpha = 0.05$.
2. **Peer Attention Congestion ($\hat{\beta}_3^* = -3.1497$, $p = 0.5597$):**
   Peer appeal exhibits very small cross-channel variation ($\mathrm{SD}(\bar{x}) \approx 0.0074$); a $+0.10$ hypothetical shift would represent approximately $13.5$ standard deviations of the peer variable and is not an empirically supported counterfactual. At the point estimate $\hat{\beta}_3^* = -3.1497$, the coefficient is statistically indistinguishable from zero ($\mathrm{SE} = 5.3997, p = 0.56$). Because of the interaction term $\beta_5 x \bar{x}$, peer congestion cannot be evaluated from $\beta_3$ alone, but must be assessed via the state-dependent marginal effect $E(x) = \beta_3 + \beta_5 x$.
3. **Strategic Interaction ($\hat{\beta}_5^* = -44.8756$, $p = 0.4497$):**
   Suggests negative interaction (strategic substitutability) on the training cohort, but carries a large standard error ($\mathrm{SE} = 59.3614$).
4. **Creator Scale Elasticity ($\hat{\gamma}_s = +1.0230$, $p < 0.0001$):**
   A $1\%$ increase in subscriber count is associated with a $+1.02\%$ increase in views. Creator scale is the single dominant predictive driver ($t = 8.25$, $\mathrm{SE} = 0.1240$).

### 4.3 Multicollinearity Resolution: Uncentered vs. Centered Model 4
In uncentered Model 4, structural collinearity between $x$, $x^2$, and $x \cdot \bar{x}$ produced severe variance inflation:
- Uncentered VIFs: $x = 1{,}334.90$, $x^2 = 281.44$, $x \cdot \bar{x} = 1{,}281.33$.
- Centered Model 4 VIFs: $x_c = 1.13$, $x_c^2 = 1.15$, $x_c \cdot \bar{x}_c = 1.14$.

Mean-centering completely eliminates artificial collinearity ($VIF < 1.20$ for all variables) while preserving identical explanatory power, residuals, $R^2$, and curvature coefficients.

---

## 5. Model Evaluation: 26-Fold LOCO-CV & Validation Freezing (Steps 4 & 5)

### 5.1 Step 4: 26-Fold LOCO-CV on Training Cohort ($N = 541$)
To evaluate out-of-sample predictive power without data leakage, we executed 26-fold Leave-One-Channel-Out Cross-Validation strictly within the training cohort. Total out-of-sample sum of squares: $\mathrm{SST}_{\mathrm{OOS}} = 3{,}135.25$.

```
+--------------------------+-------------+-------------+------------+------------+------------+
| Candidate Model          |   SSE_OOS   |   SST_OOS   |   R²_OOS   |  RMSE_OOS  |  MAE_OOS   |
+--------------------------+-------------+-------------+------------+------------+------------+
| Model 0 (Controls Base)  |   1335.20   |   3135.25   |   0.5741   |   1.5710   |   1.2963   |
| Model 1 (Linear Appeal)  |   1345.52   |   3135.25   |   0.5708   |   1.5771   |   1.3056   |
| Model 2 (Quadratic)      |   1379.62   |   3135.25   |   0.5600   |   1.5969   |   1.3215   |
| Model 3 (Peer Congestion)|   1382.87   |   3135.25   |   0.5589   |   1.5988   |   1.3225   |
| Model 4 (Interaction)    |   1390.40   |   3135.25   |   0.5565   |   1.6031   |   1.3280   |
| Model 5 (Full Tags)      |   1429.81   |   3135.25   |   0.5440   |   1.6257   |   1.3401   |
+--------------------------+-------------+-------------+------------+------------+------------+
```

**Key Finding:** Out-of-sample predictive accuracy is highest in **Model 0** ($R^2_{\mathrm{OOS}} = 0.5741$). Adding packaging terms systematically reduces generalization accuracy ($R^2_{\mathrm{OOS}}$ drops to $0.5565$ in Model 4). This confirms that packaging variables carry negligible out-of-sample signal compared to channel scale.

### 5.2 Step 5: Validation Evaluation & Specification Freezing (6 Channels, $N = 136$)
We evaluated candidate models on the 6 validation channels (Max Fosh, Michelle Khare, Nick DiGiovanni, Randy Santel, Sorted Food, Wake'N'Bite). Centering means and peer pools were drawn strictly from the 26 training channels:

$$\mathrm{SST}_{\mathrm{Val}} = \sum_{j \in \mathrm{Val}} (y_j - \bar{y}_{\mathrm{Train}})^2 = 644.22$$

```
+--------------------------+-------------+-------------+------------+------------+------------+
| Model                    |   SSE_Val   |   SST_Val   | R²_OOS_Val |  RMSE_Val  |  MAE_Val   |
+--------------------------+-------------+-------------+------------+------------+------------+
| Model 0 (Controls Base)  |   337.08    |   644.22    |   0.4768   |   1.5743   |   1.2290   |
| Model 1 (Linear Appeal)  |   384.29    |   644.22    |   0.4035   |   1.6810   |   1.3022   |
| Model 2 (Quadratic)      |   397.10    |   644.22    |   0.3836   |   1.7087   |   1.3209   |
| Model 3 (Peer Congestion)|   394.22    |   644.22    |   0.3881   |   1.7025   |   1.3180   |
| Model 4 (Interaction)    |   388.81    |   644.22    | **0.3965** | **1.6908** | **1.3126** |
| Model 5 (Full Tags)      |   372.98    |   644.22    |   0.4210   |   1.6560   |   1.2633   |
+--------------------------+-------------+-------------+------------+------------+------------+
```

#### Specification Locking Decision: Model 4-Centered
We formally locked **Model 4-Centered** prior to testing. Crucially, this choice was dictated by **theoretical necessity**, not predictive dominance:
1. **Theoretical Necessity:** Model 4 contains the quadratic curvature $\beta_2$ and strategic interaction $\beta_5$ required to formalize diminishing returns and evaluate the contraction mapping condition ($\rho < 1$). Models 0–1 lack these structural terms and cannot be used for game-theoretic analysis.
2. **Predictive Context:** Model 0 outperforms all other models out-of-sample ($R^2_{\mathrm{OOS}} = 0.4768$ vs. $0.3965$). However, locking Model 0 would defeat the entire purpose of the potential game project. Among models containing peer packaging terms (M2–M4), Model 4 achieves the lowest validation error ($R^2_{\mathrm{OOS}} = 0.3965$ vs. $0.3836$ in M2 and $0.3881$ in M3).
3. **Specification Lock:** Model exploration was terminated; Model 4-Centered was frozen.

---

## 6. Internal Holdout Evaluation on Test Channels (Step 6)

### 6.1 Methodological Disclosure of Exploratory History
In adherence to scientific integrity, we explicitly disclose that all 809 observations were initially examined during early exploratory Phase 1 feasibility analyses before the channel-stratified train/validation/test protocol was finalized. Consequently, this Step 6 evaluation represents an **internal holdout validation** of the frozen Model 4-Centered specification rather than a pure pre-registered ex-ante clinical trial.

### 6.2 Test Evaluation Setup & Execution ($N = 132$, $G = 6$)
The locked Model 4-Centered specification was refitted on the combined **Training + Validation sample (32 channels, 677 videos)**. Test peer features were drawn strictly from the combined 32-channel pool.
- Combined sample centering: $\mu_x^{\mathrm{Train+Val}} = 0.393912, \mu_{\bar{x}}^{\mathrm{Train+Val}} = 0.393521, \bar{y}_{\mathrm{Train+Val}} = 14.045068$.
- Total test sum of squares: $\mathrm{SST}_{\mathrm{Test}} = 694.24$.

```
+--------------------------+-------------+-------------+-------------+-----------+-----------+
| Model Specification      |  SSE_Test   |  SST_Test   | R²_OOS_Test | RMSE_Test | MAE_Test  |
+--------------------------+-------------+-------------+-------------+-----------+-----------+
| Model 0 (Controls Base)  |   171.13    |   694.24    |   0.7535    |  1.1386   |  0.9170   |
| Model 1 (Linear Appeal)  |   175.42    |   694.24    |   0.7473    |  1.1528   |  0.9334   |
| Model 2 (Quadratic)      |   172.21    |   694.24    |   0.7519    |  1.1422   |  0.9254   |
| Model 3 (Peer Congestion)|   171.20    |   694.24    |   0.7534    |  1.1388   |  0.9193   |
| Model 4 [LOCKED SPEC]    |   171.02    |   694.24    | **0.7537**  |**1.1382** |**0.9190** |
| Model 5 (Full Tags)      |   169.50    |   694.24    |   0.7559    |  1.1332   |  0.9103   |
+--------------------------+-------------+-------------+-------------+-----------+-----------+
```

**Discussion:** While Model 4 yields $R^2_{\mathrm{OOS}} = 0.7537$, Model 0 (controls alone) yields $R^2_{\mathrm{OOS}} = 0.7535$. The incremental out-of-sample predictive contribution of the entire set of packaging variables ($x, x^2, \bar{x}, x \cdot \bar{x}$) is a negligible $\Delta R^2_{\mathrm{OOS}} = +0.0002$ ($0.02$ percentage points). The apparent high test $R^2$ is almost entirely driven by the creator scale control ($\ln(s_i)$).

### 6.3 Test Channel Error Dispersion
Channel-by-channel performance of the locked specification across the 6 held-out test channels:

```
+------------------+---------------------+---------------+---------+----------+----------+----------+
| Channel Name     | Subscriber Tier     | Subscribers   | Videos  |   MAE    |   RMSE   |   Bias   |
+------------------+---------------------+---------------+---------+----------+----------+----------+
| Airrack          | Tier 2 (5M–20M)     | 19,600,000    |   22    |  1.0671  |  1.1733  | +0.9836  |
| Foodie We        | Tier 1 (<5M)        |  2,510,000    |   24    |  0.5334  |  0.6046  | +0.2152  |
| How Ridiculous   | Tier 3 (>20M)       | 23,900,000    |   24    |  1.0823  |  1.2951  | -1.0750  |
| Joel Hansen      | Tier 1 (<5M)        |  1,170,000    |   24    |  1.0145  |  1.4218  | -0.3628  |
| Matt Stonie      | Tier 2 (5M–20M)     | 16,400,000    |   15    |  0.6911  |  0.7646  | -0.2466  |
| Stokes Twins     | Tier 3 (>20M)       | 150,000,000   |   23    |  1.0581  |  1.2233  | +0.9557  |
+------------------+---------------------+---------------+---------+----------+----------+----------+
```

Across all tiers, MAE remains bounded between $0.53$ and $1.08$ log-views, confirming that the model makes stable predictions across subscriber scales.

---

## 7. Post-Test Full-Sample Structural Calibration & Inference (Step 7)

Following holdout test evaluation, Model 4-Centered was estimated on all 38 channels ($N = 809$) to calibrate structural parameters for game-theoretic analysis.

### 7.1 Full-Sample Estimates ($N = 809, G = 38$)
Sample means: $\mu_x = 0.392395, \mu_{\bar{x}} = 0.391390$.

$$\ln(V_{it}) = \beta_0^* + \beta_1^* (x_{it} - \mu_x) + \beta_2^* (x_{it} - \mu_x)^2 + \beta_3^* (\bar{x}_{-i,t} - \mu_{\bar{x}}) + \beta_5^* (x_{it} - \mu_x)(\bar{x}_{-i,t} - \mu_{\bar{x}}) + \gamma_s \ln(s_i) + \gamma_d \ln(\text{dur}_{it}) + \gamma_a \text{age}_{it} + \varepsilon_{it}$$

```
+--------------------+------------+------------+-----------+-----------+---------------------+
| Regressor          | Coeff (b)  | CRSE (SE)  |  t-stat   | p (df=37) | 95% Conf. Interval  |
+--------------------+------------+------------+-----------+-----------+---------------------+
| Intercept (b0_c)   |   -2.7904  |   2.4168   |  -1.155   |  0.2557   | [ -7.687,  +2.106]  |
| Own Appeal (b1_c)  |   -0.3135  |   1.9191   |  -0.163   |  0.8711   | [ -4.202,  +3.575]  |
| Curvature (b2_c)   |   -8.1974  |   8.2994   |  -0.988   |  0.3297   | [-25.014,  +8.619]  |
| Peer Congest (b3_c)|   +2.6172  |   6.0386   |  +0.433   |  0.6672   | [ -9.618, +14.853]  |
| Interaction (b5_c) |  +20.7942  |  60.2260   |  +0.345   |  0.7318   | [-101.23, +142.82]  |
| Log Subs (gamma_s) |   +1.0417  |   0.1147   |  +9.079   |  0.0000   | [ +0.809,  +1.274]  |
| Log Dur (gamma_d)  |   +0.0454  |   0.2534   |  +0.179   |  0.8588   | [ -0.468,  +0.559]  |
| Age Days (gamma_a) |   +0.0003  |   0.0003   |  +1.013   |  0.3178   | [ -0.000,  +0.001]  |
+--------------------+------------+------------+-----------+-----------+---------------------+
```

### 7.2 Analytical Uncentering of Payoff Parameters (Paper Eq. 2)
The paper's structural game-theoretic payoff (Equation 2) is formulated for uncentered click appeal $x, \bar{x} \in [0, 1]$:
$$\pi_i(x_i, \bar{x}) = b_{0\_u} + b_{1\_u} x_i + b_{2\_u} x_i^2 + b_{3\_u} \bar{x} + b_{5\_u} x_i \bar{x} + \gamma_s \ln(s_i) + \gamma_d \ln(\text{dur}) + \gamma_a \text{age}$$

Expanding the centered regression equation reveals the exact analytical uncentering identities:
$$\begin{aligned}
b_{1\_u} &= b_{1c} - 2 b_2 \mu_x - b_5 \mu_{\bar{x}} = -0.3135 - 2(-8.1974)(0.392395) - 20.7942(0.391390) = \mathbf{-2.0190} \\
b_{3\_u} &= b_{3c} - b_5 \mu_x = +2.6172 - 20.7942(0.392395) = \mathbf{-5.5424} \\
b_{2\_u} &= b_2 = \mathbf{-8.1974} \\
b_{5\_u} &= b_5 = \mathbf{+20.7942} \\
b_{0\_u} &= b_{0c} - b_{1c}\mu_x + b_2 \mu_x^2 - b_{3c}\mu_{\bar{x}} + b_5 \mu_x \mu_{\bar{x}} = \mathbf{-1.7604}
\end{aligned}$$

Prediction equivalence between centered and uncentered models is verified to machine precision ($\max |\hat{y}_c - \hat{y}_u| = 5.33 \times 10^{-15} < 10^{-8}$).

```
+--------------------+------------+------------+-----------+-----------+---------------------+
| Regressor          | Coeff (b)  | Delta SE   |  t-stat   | p (df=37) | 95% Conf. Interval  |
+--------------------+------------+------------+-----------+-----------+---------------------+
| Intercept (b0_u)   |   -1.7604  |     —    |     —   |     —   |          —         |
| Own Appeal (b1_u)  |   -2.0190  |  20.9201   |  -0.097   |  0.9236   | [ -44.407, +40.369] |
| Curvature (b2_u)   |   -8.1974  |   8.2994   |  -0.988   |  0.3297   | [ -25.014,  +8.619] |
| Peer Congest (b3_u)|   -5.5424  |  22.8522   |  -0.243   |  0.8097   | [ -51.845, +40.760] |
| Interaction (b5_u) |  +20.7942  |  60.2260   |  +0.345   |  0.7318   | [-101.235,+142.824] |
+--------------------+------------+------------+-----------+-----------+---------------------+
```

### 7.3 Variance Decomposition & Weak Identification Diagnostics
1. **Variance Decomposition:**
   - Controls-only model: $R^2 = 0.6308$.
   - Full strategic model: $R^2 = 0.6353$.
   - Incremental $R^2$ from packaging: $\Delta R^2 = 0.0045$ ($0.45$ percentage points). Controls explain $99.3\%$ of the predictable variance in view counts.
2. **Joint Hypothesis Test on Packaging:**
   - $H_0: \beta_1 = \beta_2 = \beta_3 = \beta_5 = 0$.
   - $F(4, 37) = 0.4739$, $p = \mathbf{0.7546}$. The hypothesis that video packaging terms exert zero joint influence on views cannot be rejected.
3. **Severe Attenuation of Peer Variation:**
   - Standard deviation of own click appeal: $\mathrm{SD}(x) = 0.1037$.
   - Standard deviation of peer click appeal: $\mathrm{SD}(\bar{x}) = 0.0074$ (14 times smaller).
   - Because each video's peer average is calculated across the ~20–30 active competing channels, idiosyncratic packaging variations wash out, creating weak empirical identification for peer congestion ($\beta_3$) and interaction ($\beta_5$).
4. **Instability Across Partitions:**
   - In the training cohort ($N = 541, G = 26$): $\hat{\beta}_2 = +1.5578, \hat{\beta}_5 = -44.8756$ (apparent convexity and strategic substitutability).
   - In the full sample ($N = 809, G = 38$): $\hat{\beta}_2 = -8.1974, \hat{\beta}_5 = +20.7942$ (concavity and strategic complementarity).
   - This sign instability illustrates the large sampling variance documented by the standard errors ($\mathrm{SE}(\beta_5) = 60.23$).

### 7.4 Corrected Bootstrap Inference ($B = 2{,}000$ Cluster Pairs)
Using a cluster-pairs bootstrap ($B = 2{,}000$ draws, resampling complete channel clusters with replacement, seed 42) without artificial within-cluster demeaning, bootstrap standard errors match CRSE within $6\%$ across all regressors:

```
+--------------------+------------+------------+------------+-------------+----------------------+
| Parameter          | Point Est. |  Boot SE   |  CRSE SE   | Ratio B/C   | 95% Percentile CI    |
+--------------------+------------+------------+------------+-------------+----------------------+
| Intercept (b0_c)   |   -2.7904  |   2.4993   |   2.4168   |    1.034    | [ -7.735,   +1.986]  |
| Own Appeal (b1_c)  |   -0.3135  |   1.8144   |   1.9191   |    0.945    | [ -3.716,   +3.243]  |
| Curvature (b2_c)   |   -8.1974  |   8.0165   |   8.2994   |    0.966    | [-23.303,   +6.516]  |
| Peer Congest (b3_c)|   +2.6172  |   6.1283   |   6.0386   |    1.015    | [-10.147,  +13.357]  |
| Interaction (b5_c) |  +20.7942  |  59.7562   |  60.2260   |    0.992    | [-115.043, +118.044] |
| Log Subs (gamma_s) |   +1.0417  |   0.1220   |   0.1147   |    1.064    | [ +0.789,   +1.279]  |
| Own Uncentered b1_u|   -2.0190  |  21.2315   |  20.9201   |    1.015    | [-36.847,  +46.533]  |
| Peer Uncentered b3_u|  -5.5424  |  22.5215   |  22.8522   |    0.986    | [-43.503,  +45.488]  |
+--------------------+------------+------------+------------+-------------+----------------------+
```

**Bootstrap Hypothesis Distribution:**
- **Concavity ($b_2 < 0$):** $81.40\%$ of draws ($1{,}628 / 2{,}000$) are strictly concave. The earlier report claim of "100% concave" has been corrected.
- **Both Concave and Contraction Satisfied ($b_2 < 0$ and $\rho < 1$):** Exactly $18.65\%$ of draws ($373 / 2{,}000$).
- **Contraction Ratio ($\rho = |b_5| / (2|b_2|)$ among concave draws):**
  - Point estimate: $\rho = 1.2683$
  - 95% Percentile Confidence Interval: $[0.1140, 35.7036]$
  - Quantiles: $p_{10} = 0.4200, p_{25} = 1.1053, p_{50} = 2.1528, p_{75} = 4.0546, p_{90} = 9.4619$.

---

### 7.5 State-Dependent Peer Congestion Function $E(x) = b_{3\_u} + b_{5\_u} x$
Because of the strategic interaction term $b_5 x \bar{x}$, peer attention congestion cannot be evaluated on $b_3$ alone; it is state-dependent and must be evaluated through the marginal effect $E(x) = \frac{\partial \ln(V)}{\partial \bar{x}} = b_{3\_u} + b_{5\_u} x$. At the sample mean $x = \mu_x = 0.3924$, $E(\mu_x) = b_{3c} = +2.6172$ ($\mathrm{SE}_{\Delta} = 6.0386, p = 0.6672$), whereas at zero own appeal $x = 0$, $E(0) = b_{3\_u} = -5.5424$ ($\mathrm{SE}_{\Delta} = 22.8522, p = 0.8097$). Evaluated across the entire observed data support $x \in [0.0054, 0.6843]$, both the delta-method 95% confidence intervals and cluster-pairs bootstrap 95% percentile intervals cover zero everywhere, with $31.25\%$ of bootstrap draws negative at the mean ($52.90\%$ at $x = 0$). Consequently, the empirical data support neither positive spillovers nor negative congestion at any conventional significance level.

---

## 8. Output for Downstream Use (Game-Theoretic Parameters)

This section provides the calibrated empirical structural parameters and covariance structure for downstream continuous Aggregative Exact Potential Game modeling.

### 8.1 Primary Structural Parameters (Paper Eq. 2)
The uncentered empirical payoff function parameters corresponding to paper Equation (2) are:
$$\pi_i(x_i, \bar{x}_{-i}) = b_{0\_u} + b_{1\_u} x_i + b_{2\_u} x_i^2 + b_{3\_u} \bar{x}_{-i} + b_{5\_u} x_i \bar{x}_{-i} + \gamma_s \ln(s_i) + \gamma_d \ln(\text{dur}_{it}) + \gamma_a \text{age}_{it}$$

- **Uncentered Point Estimates:**
  $$b_{1\_u} = -2.0190, \quad b_{2\_u} = -8.1974, \quad b_{3\_u} = -5.5424, \quad b_{5\_u} = +20.7942$$
  $$b_{0\_u} = -1.7604, \quad \gamma_s = 1.0417, \quad \gamma_d = 0.0454, \quad \gamma_a = 0.0003$$
- **Centering Sample Means:**
  $$\mu_x = 0.392395, \quad \mu_{\bar{x}} = 0.391390$$
- **Delta-Method Standard Errors & Student-$t$ $p$-values ($df = 37$):**
  - $\mathrm{SE}(b_{1\_u}) = 20.9201 \implies t = -0.0965, p = 0.9236$
  - $\mathrm{SE}(b_{2\_u}) = 8.2994 \implies t = -0.9877, p = 0.3297$
  - $\mathrm{SE}(b_{3\_u}) = 22.8522 \implies t = -0.2425, p = 0.8097$
  - $\mathrm{SE}(b_{5\_u}) = 60.2260 \implies t = +0.3453, p = 0.7318$
- **Centered Estimation Coefficients (Model 4-Centered):**
  $$b_{1c} = -0.3135 \; (\mathrm{SE} = 1.9191), \quad b_{2c} = -8.1974 \; (\mathrm{SE} = 8.2994)$$
  $$b_{3c} = +2.6172 \; (\mathrm{SE} = 6.0386), \quad b_{5c} = +20.7942 \; (\mathrm{SE} = 60.2260)$$

### 8.2 Empirical Covariance Matrix for Downstream Error Propagation
The analytical delta-method covariance matrix $\mathbf{\Sigma}_u = \mathbf{J} \mathbf{\Sigma}_c \mathbf{J}^T$ for the vector $(b_{1\_u}, b_{2\_u}, b_{3\_u}, b_{5\_u})$ is:

$$\mathbf{\Sigma}_u = \begin{pmatrix}
437.6519 & 81.7337 & 456.3322 & -1233.9756 \\
81.7337 & 68.8802 & 108.1554 & -316.4174 \\
456.3322 & 108.1554 & 522.2211 & -1330.6057 \\
-1233.9756 & -316.4174 & -1330.6057 & 3627.1741
\end{pmatrix}$$

Centered covariance matrix $\mathbf{\Sigma}_c$ (order: $x_c, x_c^2, \bar{x}_c, x_c \cdot \bar{x}_c$):
$$\mathbf{\Sigma}_c = \begin{pmatrix}
3.6829 & 11.9476 & -4.1607 & -62.6574 \\
11.9476 & 68.8802 & -16.0052 & -316.4174 \\
-4.1607 & -16.0052 & 36.4649 & 92.6782 \\
-62.6574 & -316.4174 & 92.6782 & 3627.1741
\end{pmatrix}$$

### 8.3 Factual Mathematical Properties
Factually, at the empirical point estimates:
- $b_{5\_u} > 2|b_{2\_u}|$ ($20.7942 > 16.3948$) and $b_{2\_u} + b_{5\_u} > 0$ ($-8.1974 + 20.7942 = +12.5968 > 0$).
- Across $2{,}000$ cluster-pairs bootstrap draws, only $18.65\%$ of draws satisfy $|b_5| < 2|b_2|$.

---

## 9. Publication Figures (Step 8)

Seven high-resolution figures (300 DPI) are maintained in `payoff_calculation/plots/`:
1. **`01_model_progression_fit.png`:** In-sample $R^2$, 26-fold LOCO-CV $R^2_{\mathrm{OOS}}$, and Validation $R^2_{\mathrm{OOS}}$ alongside AIC and BIC curves across models M0–M5.
2. **`02_actual_vs_predicted.png`:** Actual vs. predicted log views across Training ($R^2 = 0.638$), Validation ($R^2_{\mathrm{OOS}} = 0.396$), and Test ($R^2_{\mathrm{OOS}} = 0.754$) cohorts, color-coded by subscriber tier.
3. **`03_residual_audit.png`:** Four-panel diagnostic grid (residuals vs. fitted values, normal Q-Q plot, residuals vs. own click appeal, and residuals vs. creator scale).
4. **`04_payoff_response_curves.png`:** Gross view response curves across low ($p_{10}$), median ($p_{50}$), and high ($p_{90}$) peer competition, contrasted with net payoffs under linear vs. convex production costs.
5. **`05_peer_interaction_heatmap.png`:** Strategic contour surface $\ln V(x_i, \bar{x}_{-i})$ with the best-response reaction path overlaid.
6. **`06_channel_error_dispersion.png`:** Channel-by-channel error dispersion (MAE and bias) across all 38 channels, color-coded by sample partition.
7. **`07_congestion_test.png`:** State-dependent peer congestion function $E(x) = b_{3\_u} + b_{5\_u} x$ across the observed support $x \in [0.00, 0.68]$ with delta-method 95% CI band, bootstrap percentile 95% CI band, and bootstrap probability $\Pr(E < 0)$.

---

## 10. Methodological Limitations & Boundary Conditions

1. **Weak Identification of Peer Effects by Design:** The peer packaging variable is computed as a mean of channel-level means across the ~20–30 active competing channels within a rolling 60-day window. This compresses cross-sectional peer variation ($\mathrm{SD}(\bar{x}) = 0.0074$ vs. $\mathrm{SD}(x) = 0.1037$). This low variation is inherent to the aggregative mean-of-means definition by design, not merely an artifact of small sample size, preventing statistically precise identification of $\beta_3$ and $\beta_5$.
2. **Confounding of Own Appeal with Unobserved Channel Quality:** Own click appeal varies primarily between channels rather than within individual channels over time. Consequently, the coefficients $b_1$ and $b_2$ may be partially confounded with unobserved channel quality, baseline production value, or creator celebrity status, rather than representing an exogenous treatment effect of thumbnail/title packaging alone.
3. **Sample Scope Divergence from Initial Proposal:** The empirical sample comprises $N = 809$ videos across 38 channels in **global English-language digital entertainment** (Gaming, Comedy, General Entertainment, Food/Tech Lifestyle), rather than the Indian tech-review niche originally outlined in the project proposal. While this provides a substantial and realistic competitive market across creators from $118\text{K}$ to $526\text{M}$ subscribers, findings reflect global entertainment attention dynamics rather than regional tech niche competition.
4. **Retrospective Subscriber Snapshot:** Subscriber counts represent a late-2026 snapshot rather than historical time-varying subscriber counts at the exact video upload date. In our models, $\ln(s_i)$ acts as a channel-level fixed scale proxy.
5. **Observational vs. Causal Interpretation:** Packaging metrics ($x_{it}$) are observational rather than experimentally manipulated. While we control for duration, age, and creator scale, point estimates represent predictive associations and structural calibration inputs, not causal treatments.
6. **Attention vs. Revenue Payoff:** View counts proxy viewer attention and algorithmic impressions. True creator profit depends on monetization rates (RPM), sponsorship integration, and physical production overhead.
7. **Extreme-Duration Outlier Not Tested:** The 16.46-hour charity livestream (59,251 seconds) is retained in the sample; no robustness check excluding it or other extreme-duration videos was run, so its influence on the duration coefficient and fit is unknown.

---

## 11. Complete Artifact Catalog & Reproduction Pipeline

All primary scripts, datasets, and generated estimation artifacts are maintained within `payoff_calculation/`:

```text
payoff_calculation/
|--  data/
|   |--  processed_payoff_data.csv             # Stage 1: Ingested & validated dataset (N=809)
|   |--  data_dictionary.csv                   # Stage 1: Complete variable definitions
|   |--  channel_splits.json                   # Stage 2: Channel-level stratification partition map
|   \--  partitioned_payoff_data.csv           # Stage 2: Full dataset labeled by partition
|--  scripts/
|   |--  01_prepare_payoff_data.py             # Stage 1: Data preparation & audit
|   |--  02_create_stratified_split.py         # Stage 2: Stratified split (26/6/6)
|   |--  03_train_models.py                    # Stage 3: Primary training regressions
|   |--  04_evaluate_loco_cv.py                # Stage 4: 26-fold LOCO-CV on training cohort
|   |--  05_validate_and_freeze.py             # Stage 5: Validation evaluation & model locking
|   |--  06_evaluate_test_holdout.py           # Stage 6: Holdout test evaluation
|   |--  07_post_test_structural_calibration.py# Stage 7: Full-sample estimation, uncentering & bootstrap
|   |--  08_generate_plots.py                  # Stage 8: Publication figures generator
|   |--  09_backward_looking_robustness.py     # Stage 9: Backward-looking peer window check
|   \--  10_congestion_test.py                 # Stage 10: State-dependent peer congestion test E(x)
|--  results/
|   |--  game_theory_coefficients.json         # Complete structural payload for downstream game theory
|   |--  game_theory_coefficients.csv          # Flat parameter summary table
|   |--  post_test_structural_regression_results.csv # Full-sample regression table
|   |--  bootstrap_inference_results.csv       # Bootstrap CIs & contraction stats
|   |--  bootstrap_draws.npz                   # 2,000 cluster-pairs bootstrap parameter draws
|   |--  congestion_test.csv                   # Stage 10: Congestion function E(x) grid & bootstrap results
|   |--  training_regression_results.csv       # Training model estimates
|   |--  training_model_comparison.csv         # Training AIC, BIC, R²
|   |--  training_multicollinearity_diagnostics.csv # Centered vs uncentered VIFs
|   |--  subscriber_robustness_table.csv       # Scale and FE robustness specifications
|   |--  loco_cv_aggregate_summary.csv         # Stage 4 LOCO-CV summary
|   |--  loco_cv_fold_results.csv              # Stage 4 fold-by-fold results
|   |--  validation_evaluation.csv             # Stage 5 validation evaluation
|   |--  validation_channel_dispersion.csv     # Stage 5 validation channel breakdown
|   |--  test_set_evaluation.csv               # Stage 6 test evaluation
|   |--  test_channel_dispersion.csv           # Stage 6 test channel breakdown
|   \--  backward_looking_peer_robustness.csv  # Contemporaneous vs backward peer table
|--  plots/
|   |--  01_model_progression_fit.png          # Figure 1: In-sample vs OOS R² and AIC/BIC
|   |--  02_actual_vs_predicted.png            # Figure 2: Actual vs predicted log views
|   |--  03_residual_audit.png                 # Figure 3: Four-panel residual diagnostics
|   |--  04_payoff_response_curves.png         # Figure 4: Payoff response curves under competition
|   |--  05_peer_interaction_heatmap.png       # Figure 5: Strategic contour surface ln V(x, x_bar)
|   |--  06_channel_error_dispersion.png       # Figure 6: Channel-level error dispersion
|   \--  07_congestion_test.png                # Figure 7: State-dependent peer congestion function E(x)
|--  tests/
|   |--  test_bootstrap_verification.py        # Automated test for cluster-pairs bootstrap
|   |--  test_report_consistency.py           # Automated test verifying report numerical accuracy
|   |--  test_equilibrium_consistency.py       # Structural transformation & parameter consistency
|   \--  test_encoding.py                      # UTF-8 and mojibake audit test
|--  _extras/                                  # Archived downstream equilibrium scripts, sweeps & legacy files
|--  payoff_estimation.py                     # Master runner executing stages 01 through 10
|--  FINAL_PAYOFF_REPORT.md                     # Comprehensive master research report
\--  README.md                                  # Repository overview and quick-start guide
```

### End-to-End Reproduction Command
To execute all 10 empirical estimation stages sequentially and log execution to `logs/master_pipeline.log`:
```bash
python payoff_estimation.py
```
*Total pipeline execution duration: ~45 seconds on standard hardware.*
