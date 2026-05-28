# Alternative Models for Kalman Filter under Non-Normality \& Orthogonality Violation

> The standard Kalman Filter (KF) requires two critical assumptions: (1) Gaussian noise and (2) orthogonality (uncorrelated process \& measurement noise, and estimation error orthogonal to observations). When \*\*both\*\* are violated, this repo catalogs \*\*20+ alternative models\*\* across \*\*14 categories\*\* with formulas, pros/cons, financial industry pitfalls, and academic/industry status.
> This file provides a full quick overview of all potential alternatives for linear KF.

\---

## 📑 Table of Contents

* [Background](#-background)
* [Quick Comparison Matrix](#-quick-comparison-matrix)
* [Category 1: Sequential Monte Carlo Methods](#category-1-sequential-monte-carlo-methods)

  * [1.1 Particle Filter (PF / SMC)](#11-particle-filter-pf--smc)
  * [1.2 Particle MCMC (PMCMC)](#12-particle-mcmc-pmcmc)
* [Category 2: Non-Gaussian Parametric Distribution Models](#category-2-non-gaussian-parametric-distribution-models)

  * [2.1 Student-t Kalman Filter](#21-student-t-kalman-filter)
  * [2.2 Skew-Normal / Skew-t Kalman Filter](#22-skew-normal--skew-t-kalman-filter)
* [Category 3: Gaussian Mixture Approaches](#category-3-gaussian-mixture-approaches)

  * [3.1 Gaussian Sum Filter (GSF)](#31-gaussian-sum-filter-gsf)
* [Category 4: Information-Theoretic Approaches](#category-4-information-theoretic-approaches)

  * [4.1 Maximum Correntropy Criterion Filter (MCCF)](#41-maximum-correntropy-criterion-filter-mccf)
  * [4.2 Wasserstein Filter](#42-wasserstein-filter)
  * [4.3 Minimum Error Entropy Filter (MEE)](#43-minimum-error-entropy-filter-mee)
* [Category 5: Robust / Minimax Approaches](#category-5-robust--minimax-approaches)

  * [5.1 H∞ Filter](#51-h-filter)
  * [5.2 Huber-Based Robust Kalman Filter](#52-huber-based-robust-kalman-filter)
* [Category 6: Information Geometry / Projection Filters](#category-6-information-geometry--projection-filters)

  * [6.1 Projection Filter](#61-projection-filter)
* [Category 7: Variational Bayesian Methods](#category-7-variational-bayesian-methods)

  * [7.1 Variational Bayesian Filter (VBF)](#71-variational-bayesian-filter-vbf)
* [Category 8: Score-Driven / Observation-Driven Models](#category-8-score-driven--observation-driven-models)

  * [8.1 GAS Filter (Generalized Autoregressive Score)](#81-gas-filter-generalized-autoregressive-score)
* [Category 9: Moving Horizon Estimation](#category-9-moving-horizon-estimation)

  * [9.1 Moving Horizon Estimator (MHE)](#91-moving-horizon-estimator-mhe)
* [Category 10: Copula-Based State Space Models](#category-10-copula-based-state-space-models)

  * [10.1 Copula State Space Filter](#101-copula-state-space-filter)
* [Category 11: Koopman Operator-Based Methods](#category-11-koopman-operator-based-methods)

  * [11.1 Koopman Kalman Filter](#111-koopman-kalman-filter)
* [Category 12: Deep Learning / Neural Network Methods](#category-12-deep-learning--neural-network-methods)

  * [12.1 Neural Kalman Filter / Flow-Based Bayesian Filter](#121-neural-kalman-filter--flow-based-bayesian-filter)
* [Category 13: Ensemble Methods](#category-13-ensemble-methods)

  * [13.1 Ensemble Kalman Filter \& Langevinized EnKF](#131-ensemble-kalman-filter--langevinized-enkf)
* [Category 14: Physics-Inspired / Unconventional Methods](#category-14-physics-inspired--unconventional-methods)

  * [14.1 Feynman-Kac / Path Integral Filter](#141-feynman-kac--path-integral-filter)
  * [14.2 Diffusion Maps Kalman Filter](#142-diffusion-maps-kalman-filter)
* [Recommended Strategy](#-recommended-strategy)
* [References \& Further Reading](#-references--further-reading)

\---

## 🧠 Background

The standard Kalman Filter is the optimal linear estimator under two conditions:

|Assumption|Mathematical Statement|What It Implies|
|-|-|-|
|**Gaussianity**|$w\_t \\sim \\mathcal{N}(0, Q\_t)$, $v\_t \\sim \\mathcal{N}(0, R\_t)$|Posterior is Gaussian; mean = MMSE estimate|
|**Orthogonality**|$E\[w\_t v\_s^T] = 0 ;\\forall t,s$ and $E\[\\varepsilon\_t y\_s^T] = 0$|Process \& measurement noise uncorrelated; estimation error orthogonal to observations|

When these **both fail** — common in financial time series with heavy-tailed returns, skewed distributions, and correlated state-observation noise — the KF is no longer optimal and can produce severely biased or overconfident estimates. This repo documents every viable alternative.

\---

## 📊 Quick Comparison Matrix

|#|Model|Non-Gaussian|Orthogonality Violation|Compute Cost|Academia ⭐|Finance Industry ⭐|
|-|-|-|-|-|-|-|
|1|Particle Filter|✅ Full|✅ Full|High|★★★★★|★★☆|
|2|PMCMC|✅ Full|✅ Full|Very High|★★★★☆|★★☆|
|3|Student-t KF|✅ Heavy tails|⚠️ Partial|Low|★★★★☆|★★★★|
|4|Skew-Normal/t KF|✅ Skew+tails|✅ CSN handles|Medium|★★★☆☆|★★☆|
|5|Gaussian Sum Filter|✅ Approximate|⚠️ Augmented|Medium|★★★☆☆|★★★☆|
|6|MCC Filter|✅ Impulsive|❌ Not directly|Medium|★★★☆☆|★☆|
|7|Wasserstein Filter|✅ Full|✅ Full|High|★★★☆☆|★☆|
|8|MEE Filter|✅ Heavy tails|❌ Not directly|High|★★☆☆☆|★☆|
|9|H∞ Filter|✅ Any|✅ Any|Low|★★★☆☆|★★☆|
|10|Huber-KF|⚠️ Outlier-only|❌ Not directly|Low|★★☆☆☆|★★★|
|11|Projection Filter|✅ Exp. family|❌ Not directly|High|★★☆☆☆|★☆|
|12|Variational Bayes|✅ Flexible Q|⚠️ Augmented|Medium|★★★★☆|★★★|
|13|GAS/Score-Driven|✅ Full|❌ No state noise|Low|★★★★★|★★★★|
|14|Moving Horizon Est.|✅ Via cost|✅ Via constraints|High|★★★☆☆|★★☆|
|15|Copula SSM|✅ Full|✅ Via copula|Very High|★★★☆☆|★★★|
|16|Koopman KF|⚠️ Approximate|⚠️ In lifted space|Medium|★★★★★|★★☆|
|17|Neural/Flow Filter|✅ Full|✅ Full (learned)|High|★★★★★|★★★|
|18|EnKF + Langevin|✅ Heavy tails|⚠️ Approximate|Medium|★★★★☆|★☆|
|19|Feynman-Kac|✅ Full|✅ Full|Intractable|★☆☆☆☆|★☆|
|20|Diffusion Maps KF|⚠️ Manifold|❌ Not directly|High|★★☆☆☆|★☆|

\---

## Category 1: Sequential Monte Carlo Methods

### 1.1 Particle Filter (PF / SMC)

**Core Idea:** Approximate the posterior $p(x\_t | y\_{1:t})$ by a set of weighted random samples (particles) ${x\_t^{(i)}, w\_t^{(i)}}\_{i=1}^N$, avoiding any Gaussian assumption entirely.

**Key Formula:**

$$p(x\_t | y\_{1:t}) \\approx \\sum\_{i=1}^N w\_t^{(i)} \\delta\_{x\_t^{(i)}}(x\_t)$$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$\\delta\_{x\_t^{(i)}}(\\cdot)$|Dirac delta at particle $x\_t^{(i)}$|Point-mass approximation of posterior|Sampling/resampling scheme|
|$w\_t^{(i)}$|Normalized importance weight|Quantifies particle likelihood given obs|$w\_t^{(i)} \\propto w\_{t-1}^{(i)} \\cdot \\frac{p(y\_t \| x\_t^{(i)}) , p(x\_t^{(i)} \| x\_{t-1}^{(i)})}{q(x\_t^{(i)} \| x\_{t-1}^{(i)}, y\_t)}$|
|$N$|Number of particles|Accuracy vs. speed trade-off|Typically $10^3$–$10^5$|
|$q(\\cdot)$|Proposal distribution|Guides particle generation|Design choice; optimal: $q \\propto p(y\_t\|x\_t)p(x\_t\|x\_{t-1})$|

**✅ Pros:**

* Theoretically converges to true posterior as $N \\to \\infty$
* Handles **any** non-Gaussian distribution and arbitrary nonlinearity
* Naturally accommodates correlated/dependent noise (no orthogonality needed)
* Can represent multimodal posteriors

**❌ Cons:**

* Particle degeneracy: most weights → 0 after many steps
* Curse of dimensionality: exponentially more particles in high-D
* Computational cost: $O(N \\cdot d\_x)$ per step
* Resampling introduces sample impoverishment

**⚠️ Financial Industry Pitfalls:**

* $d\_x > 50$ (e.g., HFT) → computationally infeasible
* Sample impoverishment can miss rare regime transitions
* Poor proposal → catastrophic failure in fat-tailed returns

**🔧 Optimization Directions:**

* Rao-Blackwellized PF (marginalize out linear/Gaussian components)
* Adaptive proposals (auxiliary PF, likelihood PF)
* GPU-accelerated parallel resampling

**📚 Related:** Differentiable PF (Corenflos et al., 2021), NF-based Differentiable PF (2024)

||Academia|Financial Industry|
|-|-|-|
|Active research?|✅ Very active|⚠️ Limited|
|Attention level|★★★★★|★★☆☆☆|

\---

### 1.2 Particle MCMC (PMCMC)

**Core Idea:** Combine SMC for state inference with MCMC for parameter inference — joint Bayesian estimation in non-linear/non-Gaussian state-space models.

**Key Formula (PMMH):**

$$\\alpha(\\theta' | \\theta) = \\min\\left(1, \\frac{\\hat{p}(y\_{1:T}|\\theta') \\cdot \\pi(\\theta')}{\\hat{p}(y\_{1:T}|\\theta) \\cdot \\pi(\\theta)} \\cdot \\frac{q(\\theta|\\theta')}{q(\\theta'|\\theta)}\\right)$$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$\\hat{p}(y\_{1:T}\|\\theta)$|Particle likelihood estimate|Unbiased substitute for intractable likelihood|Run PF with $N$ particles under $\\theta$|
|$\\pi(\\theta)$|Prior on parameters|Encodes domain knowledge|Domain expertise|
|$q(\\theta'\|\\theta)$|Proposal kernel|Parameter transition proposal|RW-MH, MALA, or HMC|

**✅ Pros:**

* Exact-approximate Bayesian inference (targets true posterior)
* Handles any non-Gaussian, non-orthogonal noise
* Joint state + parameter estimation

**❌ Cons:**

* Extremely expensive (PF inside each MCMC iteration)
* Slow mixing in high-D parameter spaces
* Requires careful proposal tuning

**⚠️ Financial Pitfalls:** Real-time use impossible; false convergence in risk calibration

||Academia|Financial Industry|
|-|-|-|
|Active research?|✅ Active|⚠️ Niche|
|Attention level|★★★★☆|★★☆☆☆|

\---

## Category 2: Non-Gaussian Parametric Distribution Models

### 2.1 Student-t Kalman Filter

**Core Idea:** Replace Gaussian distributions with multivariate Student-t — heavier tails for robustness to outliers.

**Key Formula:**

$$p(x\_t | y\_{1:t-1}) = \\text{St}(x\_t; \\hat{x}*{t|t-1}, P*{t|t-1}, \\nu\_t)$$

$$\\text{St}(x;\\mu,\\Sigma,\\nu) = \\frac{\\Gamma\[(\\nu+d)/2]}{\\Gamma(\\nu/2),\\nu^{d/2},\\pi^{d/2},|\\Sigma|^{1/2}} \\left\[1 + \\frac{(x-\\mu)^T\\Sigma^{-1}(x-\\mu)}{\\nu}\\right]^{-(\\nu+d)/2}$$

**VB-Adaptive Update:**

$$K\_t = P\_{t|t-1} H\_t^T \\left( H\_t P\_{t|t-1} H\_t^T + R\_t / \\lambda\_t \\right)^{-1}$$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$\\nu\_t$|Degrees of freedom|Tail heaviness ($\\nu\\to\\infty$ → Gaussian)|ML or VB; typical $\\nu \\in \[3,30]$|
|$\\lambda\_t$|Adaptive scaling|Downweights outlier observations|$\\lambda\_t = \\frac{\\nu\_{y,t}+d\_y}{\\nu\_{y,t}+d\_y+\\Delta\_t}$ where $\\Delta\_t$ is Mahalanobis distance|
|$P\_{t\|t-1}$|Predicted scale matrix|Spread of predictive t-distribution|$F\_t P\_{t-1\|t-1} F\_t^T + Q\_t$|

**✅ Pros:** Closed-form approximate updates; robust to heavy tails; cheap vs PF
**❌ Cons:** Symmetric only (no skewness); VB introduces bias; conjugate structure needed
**⚠️ Financial Pitfalls:** Returns are often skewed; $\\nu$ unstable in small samples; bimodal → misleading CIs

||Academia|Financial Industry|
|-|-|-|
|Active research?|✅ Active|✅ Widely used|
|Attention level|★★★★☆|★★★★☆|

\---

### 2.2 Skew-Normal / Skew-t Kalman Filter

**Core Idea:** Use skew-normal or skew-t distributions to capture both asymmetry and heavy tails.

**Key Formula (Skew-Normal):**

$$f(x) = 2,\\phi\_d(x; \\mu, \\Sigma),\\Phi(\\alpha^T \\omega^{-1}(x - \\mu))$$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$\\alpha$|Shape (skewness) vector|Direction \& degree of asymmetry; $\\alpha=0$ → Gaussian|ML estimation|
|$\\omega$|$\\text{diag}(\\Sigma)^{1/2}$|Standardizes before applying skewness|Derived from $\\Sigma$|
|$\\phi\_d(\\cdot)$|$d$-variate normal PDF|Base symmetric density|—|
|$\\Phi(\\cdot)$|Standard normal CDF|Introduces asymmetry|—|

**✅ Pros:** Captures asymmetry (leverage effects); nests Gaussian; CSN handles $E\[w\_t v\_t^T] \\neq 0$
**❌ Cons:** Computationally complex; parameter proliferation; requires mode-matching/VB approximations
**⚠️ Financial Pitfalls:** Overfitting; interpretability of $\\alpha$; biased estimates in risk scenarios

||Academia|Financial Industry|
|-|-|-|
|Active research?|✅ Active (CSN, 2025)|⚠️ Emerging|
|Attention level|★★★☆☆|★★☆☆☆|

\---

## Category 3: Gaussian Mixture Approaches

### 3.1 Gaussian Sum Filter (GSF)

**Core Idea:** Approximate any non-Gaussian distribution as a weighted sum of Gaussians; each component runs its own KF.

$$p(x\_t | y\_{1:t}) \\approx \\sum\_{j=1}^J w\_t^{(j)} \\cdot \\mathcal{N}(x\_t; \\mu\_t^{(j)}, P\_t^{(j)})$$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$J$|Number of components|Approximation fidelity|Budget; typical 3–20|
|$w\_t^{(j)}$|Mixture weight|Probability mass for component $j$|$w\_t^{(j)} \\propto w\_{t-1}^{(j)} \\cdot \\mathcal{N}(y\_t; H\_t\\mu\_{t\|t-1}^{(j)}, S\_t^{(j)})$|
|$\\mu\_t^{(j)}, P\_t^{(j)}$|Component mean \& covariance|Each component's KF state|Separate KF per component|

**✅ Pros:** Approximates any distribution ($J\\to\\infty$); efficient per component; captures multimodality
**❌ Cons:** Exponential component growth; merging loses info; augmented state needed for correlated noise
**⚠️ Financial Pitfalls:** Small $J$ → poor tail approximation; merging erases tail risk

||Academia|Financial Industry|
|-|-|-|
|Active research?|✅ Moderate (GMM-VB)|✅ Used in tracking, risk|
|Attention level|★★★☆☆|★★★☆☆|

\---

## Category 4: Information-Theoretic Approaches

### 4.1 Maximum Correntropy Criterion Filter (MCCF)

**Core Idea:** Replace MSE with maximum correntropy — a bounded, local similarity measure robust to impulsive noise.

$$V\_\\sigma(a, b) = E\\left\[\\exp\\left(-\\frac{(a-b)^2}{2\\sigma^2}\\right)\\right]$$

$$K\_t^{MCC} = P\_{t|t-1} H\_t^T \\left( H\_t P\_{t|t-1} H\_t^T + R\_t \\oslash C\_t \\right)^{-1}$$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$\\sigma$|Kernel bandwidth|Receptive field size; large→MSE-like|Silverman's rule; typical 0.5×–5× noise std|
|$C\_t$|Correntropy weight matrix|Downweights large innovations|$\[C\_t]*{ij} = \\exp(-e*{t,i}^2/2\\sigma^2)$|
|$\\oslash$|Element-wise division|Modifies KF gain|—|

**✅ Pros:** Robust to impulsive noise; bounded cost; KF-like structure; no distribution knowledge needed
**❌ Cons:** $\\sigma$ hard to choose; only observation noise addressed; stability proofs weaker
**⚠️ Financial Pitfalls:** May confuse crashes with outliers; not widely adopted

||Academia|Financial Industry|
|-|-|-|
|Active research?|✅ Active|❌ Rarely used|
|Attention level|★★★☆☆|★☆☆☆☆|

\---

### 4.2 Wasserstein Filter

**Core Idea (Prabhat \& Bhattacharya, 2024):** Formulate state estimation as optimal transport — minimize Wasserstein distance from posterior error to Dirac delta (perfect knowledge). **Distribution-agnostic.**

$$\hat{T}^* = \arg\min_{T} W_p\left(T_{\sharp} \gamma(x_0, z), \delta_0\right)$$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$T$|Transport map|How prior + measurement are fused|Optimization|
|$\\gamma(x\_0, z)$|Joint state-measurement distribution|Encodes coupling (handles correlated noise)|Model $p(x\_t, y\_t)$|
|$W\_p$|$p$-Wasserstein distance|Cost of transporting one distribution to another|Typically $p=2$|
|$\\delta\_0$|Dirac delta at 0|Target: zero estimation error|—|

**Key result:** Recovers KF exactly for Gaussian; recovers GSF for Gaussian mixtures (and reveals its suboptimality).

**✅ Pros:** Fully distribution-agnostic; no particles needed; handles correlated noise via $\\gamma$; principled generalization of KF
**❌ Cons:** Very new (2024); OT computation can be expensive; no standard software; unproven scalability
**⚠️ Financial Pitfalls:** Too early for production; real-time OT may be prohibitive

||Academia|Financial Industry|
|-|-|-|
|Active research?|✅ Emerging|❌ Not yet adopted|
|Attention level|★★★☆☆|★☆☆☆☆|

\---

### 4.3 Minimum Error Entropy Filter (MEE)

$$\\hat{x}*{t|t} = \\arg\\min*{\\hat{x}} H\_\\alpha(e\_t) = \\arg\\min\_{\\hat{x}} \\frac{1}{1-\\alpha}\\log \\int p^{\\alpha}(e\_t) , de\_t$$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$e\_t$|Estimation error $x\_t - \\hat{x}\_t$|Distribution to concentrate|Filter quality|
|$H\_\\alpha$|Rényi entropy of order $\\alpha$|Measures uncertainty; $\\alpha\\to1$ → Shannon|Design; common $\\alpha=2$|
|$p(e\_t)$|Error PDF|Target for minimization|Estimated online|

**✅ Pros:** Captures higher-order statistics; robust to impulsive noise
**❌ Cons:** Requires online density estimation; kernel approximations introduce bias; stability hard to guarantee

||Academia|Financial Industry|
|-|-|-|
|Active research?|⚠️ Moderate|❌ Not adopted|
|Attention level|★★☆☆☆|★☆☆☆☆|

\---

## Category 5: Robust / Minimax Approaches

### 5.1 H∞ Filter

**Core Idea:** Minimize worst-case estimation error gain — **no distributional assumption at all.**

$$R\_{e,t} = \\begin{bmatrix} I + H\_t P\_{t-1|t-1} H\_t^T \& H\_t P\_{t-1|t-1} L\_t^T \\ L\_t P\_{t-1|t-1} H\_t^T \& -\\gamma^2 I + L\_t P\_{t-1|t-1} L\_t^T \\end{bmatrix}$$

Guarantees: $\\frac{|x\_t - \\hat{x}\_t|\_2^2}{|w|\_2^2 + |v|\_2^2 + |x\_0 - \\hat{x}*0|*{P\_0^{-1}}^2} < \\gamma^2$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$\\gamma$|Performance bound|Worst-case gain limit|Smallest feasible via $\\gamma$-iteration|
|$L\_t$|Output matrix|Which states to estimate|Application; often $L\_t=I$|
|$Q\_t, R\_t$|Design parameters (not statistical covariances)|Tune filter behavior|Tuning|

**✅ Pros:** No distribution assumption; worst-case guarantees; handles correlated noise
**❌ Cons:** Conservative; no probabilistic interpretation; $\\gamma$ tuning non-trivial
**⚠️ Financial Pitfalls:** Over-conservative capital allocation; no confidence intervals for Basel III/IV

||Academia|Financial Industry|
|-|-|-|
|Active research?|✅ Moderate|⚠️ Niche|
|Attention level|★★★☆☆|★★☆☆☆|

\---

### 5.2 Huber-Based Robust Kalman Filter

**Core Idea:** Replace quadratic loss with Huber loss — quadratic for small residuals, linear for large ones.

$$\\rho\_\\gamma(e) = \\begin{cases} \\frac{1}{2}e^2 \& |e| \\leq \\gamma \\ \\gamma|e| - \\frac{1}{2}\\gamma^2 \& |e| > \\gamma \\end{cases}$$

$$\\hat{x}*{t|t} = \\hat{x}*{t|t-1} + K\_t^{Huber} \\psi\_\\gamma(y\_t - H\_t \\hat{x}\_{t|t-1})$$

where $\\psi\_\\gamma(e) = \\min(\\max(e, -\\gamma), \\gamma)$ (Winsorized residual).

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$\\gamma$|Threshold|Separates normal from outlier obs|$\\gamma = 1.345\\hat{\\sigma}$ (95% Gaussian efficiency)|

**✅ Pros:** Simple KF modification; robust to outliers; near-optimal under Gaussianity
**❌ Cons:** Only observation outliers; doesn't handle orthogonality violation; symmetric; $\\gamma$-sensitive
**⚠️ Financial Pitfalls:** $\\gamma$-tuning during regime shifts; may downweight legitimate tail events

||Academia|Financial Industry|
|-|-|-|
|Active research?|⚠️ Mature|✅ Used in navigation, risk|
|Attention level|★★☆☆☆|★★★☆☆|

\---

## Category 6: Information Geometry / Projection Filters

### 6.1 Projection Filter

**Core Idea (Brigo, Hanzon, Le Gland):** Project the true posterior onto a finite-dimensional exponential family manifold via Fisher information metric — the **information-geometrically optimal** approximation.

$$\\frac{d\\theta\_t}{dt} = \\Pi\_\\theta \\left\[ \\mathcal{L}^\* p\_t \\right]\_{\\theta=\\theta\_t}$$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$\\theta\_t$|Natural parameters|Parameterize projected posterior|Projection dynamics|
|$\\Pi\_\\theta$|Projection operator|Projects onto exponential family tangent space|$\\Pi\_\\theta\[f] = g^{ij}E\_\\theta\[\\phi\_j f]\\partial\_{\\theta\_i}$|
|$\\mathcal{L}^\*$|Fokker-Planck operator|True posterior evolution|State-space model|
|$g^{ij}$|Inverse Fisher metric|Statistical manifold metric|$g\_{ij} = E\_\\theta\[\\partial\_{\\theta\_i}\\log p \\cdot \\partial\_{\\theta\_j}\\log p]$|

**Key insight:** KF = projection filter onto Gaussian manifold.

**✅ Pros:** Principled geometric optimality; generalizes KF; quantifiable approximation error
**❌ Cons:** Expensive integrations; limited scalability; rare implementations
**⚠️ Financial Pitfalls:** Impractical for HFT; no software ecosystem

||Academia|Financial Industry|
|-|-|-|
|Active research?|⚠️ Niche|❌ Not adopted|
|Attention level|★★☆☆☆|★☆☆☆☆|

\---

## Category 7: Variational Bayesian Methods

### 7.1 Variational Bayesian Filter (VBF)

**Core Idea:** Approximate $p(x\_t | y\_{1:t})$ with tractable $q(x\_t)$ by minimizing KL divergence.

$$q^\*(x\_t) = \\arg\\min\_{q \\in \\mathcal{Q}} KL(q(x\_t) | p(x\_t | y\_{1:t}))$$

**VB-KF Update:**

$$\\hat{x}*{t|t}^{(i+1)} = \\hat{x}*{t|t-1} + P\_{t|t-1} H\_t^T \\left(H\_t P\_{t|t-1} H\_t^T + \\tilde{R}*t^{(i)}\\right)^{-1}(y\_t - H\_t \\hat{x}*{t|t-1})$$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$\\mathcal{Q}$|Variational family|Tractable approximation space|Design: Gaussian, GMM, NF|
|$\\tilde{R}\_t^{(i)}$|VB-estimated noise covariance|Adaptive effective measurement noise|$\\tilde{R}\_t = \\text{diag}(R\_t)/\\lambda\_t^{(i)}$|

**✅ Pros:** Flexible variational families; framework for correlated noise; adaptive online
**❌ Cons:** KL mode-seeking → variance underestimation; inductive bias from $\\mathcal{Q}$; convergence not guaranteed
**⚠️ Financial Pitfalls:** Variance underestimation dangerous for risk; GMM-VB adds complexity

||Academia|Financial Industry|
|-|-|-|
|Active research?|✅ Very active|⚠️ Emerging|
|Attention level|★★★★☆|★★★☆☆|

\---

## Category 8: Score-Driven / Observation-Driven Models

### 8.1 GAS Filter (Generalized Autoregressive Score)

**Core Idea:** Update time-varying parameters using the score (gradient) of the log-likelihood — **automatically adapts** to the assumed distribution.

$$f\_{t+1} = \\omega + A s\_t + B f\_t, \\quad s\_t = \\mathcal{S}*t \\cdot \\nabla*{f\_t} \\log p(y\_t | f\_t, \\mathcal{F}\_{t-1})$$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$f\_t$|Time-varying parameter|Latent state (volatility, location, etc.)|Recursive score update|
|$s\_t$|Scaled score|Optimal update direction for assumed distribution|$\\nabla\_{f\_t}\\log p \\times \\mathcal{S}\_t$|
|$\\mathcal{S}\_t$|Scaling matrix|Normalizes score|Identity, $\\mathcal{I}(f\_t)^{-1}$, or $\\mathcal{I}^{-1/2}$|
|$\\omega, A, B$|Coefficient matrices|Long-run level, adaptation speed, persistence|MLE|

**Key insight:** If $p(y\_t|f\_t)$ is Student-t → score auto-downweights outliers. If Poisson → adapts to counts.

**✅ Pros:** Distribution-adaptive by construction; handles skewness, heavy tails, discrete obs; easy MLE; no orthogonality needed
**❌ Cons:** Not a full state-space model; no separate state noise; cannot handle $E\[w\_t v\_t^T] \\neq 0$
**⚠️ Financial Pitfalls:** No "true" latent process dynamics; may underreact to structural breaks

||Academia|Financial Industry|
|-|-|-|
|Active research?|✅ Very active|✅ Widely used|
|Attention level|★★★★★|★★★★☆|

\---

## Category 9: Moving Horizon Estimation

### 9.1 Moving Horizon Estimator (MHE)

**Core Idea:** Solve constrained optimization over a sliding window — cost function can encode **any** distributional assumption.

$$\\hat{x}*{t-N:t}^\* = \\arg\\min*{\\hat{x}*{t-N:t}} \\sum*{k=t-N}^{t} \\ell\_k(\\hat{x}*k, y\_k) + \\Gamma*{t-N}(\\hat{x}\_{t-N})$$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$N$|Horizon length|Past information window|Budget vs. accuracy; typical 10–100|
|$\\ell\_k$|Stage cost|Replaces Gaussian log-likelihood; can be Huber, correntropy, $\\ell\_1$, quantile|Noise assumption / robustness|
|$\\Gamma\_{t-N}$|Arrival cost|Summarizes pre-window info|KF-like prior or sampling|

**✅ Pros:** Any distribution via $\\ell\_k$; correlated noise directly; hard constraints; nonlinear systems; unifies many robust filters
**❌ Cons:** Optimization per step; arrival cost approximation critical; no closed-form; limited scalability
**⚠️ Financial Pitfalls:** Solve time vs. real-time trading; uncertain constraint specification

||Academia|Financial Industry|
|-|-|-|
|Active research?|✅ Active|⚠️ Niche|
|Attention level|★★★☆☆|★★☆☆☆|

\---

## Category 10: Copula-Based State Space Models

### 10.1 Copula State Space Filter

**Core Idea:** Decouple marginals from dependence via Sklar's theorem. Any non-Gaussian marginal + any (non-orthogonal) copula dependence.

$$F(x\_t, y\_t) = C(F\_{x\_t}(x\_t), F\_{y\_t}(y\_t); \\theta\_t)$$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$C(\\cdot,\\cdot;\\theta\_t)$|Copula function|Dependence structure independent of marginals|Family: Gaussian, Clayton, Frank, Gumbel, t-copula|
|$F\_{x\_t}, F\_{y\_t}$|Marginal CDFs|Can be **any** distribution|Marginal model|
|$\\theta\_t$|Copula parameter|Strength \& type of dependence|ML or Bayesian|
|$c(u,v;\\theta\_t)$|Copula density|Enters likelihood for filtering|$\\partial^2 C / \\partial u \\partial v$|

**✅ Pros:** Most flexible for non-Gaussian + non-orthogonal; asymmetric dependence; directly handles $E\[w\_t v\_t^T] \\neq 0$
**❌ Cons:** Expensive inference (MCMC/particle); copula selection non-trivial; high-D curse
**⚠️ Financial Pitfalls:** Li's Gaussian copula \& CDO crisis; tail dependence mis-specification → catastrophic

||Academia|Financial Industry|
|-|-|-|
|Active research?|✅ Active|⚠️ Controversial|
|Attention level|★★★☆☆|★★★☆☆|

\---

## Category 11: Koopman Operator-Based Methods

### 11.1 Koopman Kalman Filter

**Core Idea:** Lift nonlinear/non-Gaussian dynamics to higher-D space where Koopman operator is linear → apply KF there.

$$z\_{t+1} = K z\_t + \\eta\_t, \\quad y\_t = H\_z z\_t + \\nu\_t, \\quad z\_t = \\psi(x\_t)$$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$\\psi(\\cdot)$|Lifting / observable functions|Nonlinear → linear coordinates|Basis dictionary or Deep Koopman|
|$K$|Koopman matrix|Linear operator in lifted space|EDMD or deep learning|
|$\\eta\_t, \\nu\_t$|Lifted-space noise|Lifting error + actual noise|Residual after fitting|

**✅ Pros:** Nonlinear → linear KF; Deep Koopman learns from data; cheap after lifting
**❌ Cons:** Infinite-D operator → finite approximation error; lifted noise often non-Gaussian; interpretability loss
**⚠️ Financial Pitfalls:** Out-of-sample degradation; lifted states not economically meaningful

||Academia|Financial Industry|
|-|-|-|
|Active research?|✅ Very active|⚠️ Emerging|
|Attention level|★★★★★|★★☆☆☆|

\---

## Category 12: Deep Learning / Neural Network Methods

### 12.1 Neural Kalman Filter / Flow-Based Bayesian Filter

**Core Idea:** Learn the filtering distribution $p(x\_t | y\_{1:t})$ from data via neural networks, bypassing all distributional assumptions.

$$\\log q\_\\phi(x\_t | y\_{1:t}) = \\log p\_Z(z\_0) - \\sum\_{k=1}^K \\log \\left| \\det \\frac{\\partial f\_k}{\\partial z\_{k-1}} \\right|$$

where $x\_t = f\_K \\circ \\cdots \\circ f\_1(z\_0)$ and $z\_0 \\sim p\_Z$.

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$\\phi$|NN parameters|Parameterize the normalizing flow|Training: MLE or VI|
|$f\_k$|Flow transformation|Transforms base density → complex posterior|Architecture: RealNVP, MAF|
|$K$|Number of flow steps|Expressiveness|4–16 typical|
|$p\_Z$|Base distribution|Simple starting density|Standard normal|

**✅ Pros:** Universal approximator; handles any distribution, nonlinearity, correlated noise; end-to-end trainable
**❌ Cons:** Large data needed; black-box; no theoretical guarantees; OOD unreliable; overfitting
**⚠️ Financial Pitfalls:** Regulatory nightmare; overfitting to historical regimes; adversarial vulnerability

||Academia|Financial Industry|
|-|-|-|
|Active research?|✅ Very active|⚠️ Experimental|
|Attention level|★★★★★|★★★☆☆|

\---

## Category 13: Ensemble Methods

### 13.1 Ensemble Kalman Filter \& Langevinized EnKF

**Core Idea:** Use ensemble of state realizations to estimate covariance; Langevin step extends to non-Gaussian.

$$K\_t^{ens} = P\_t^{ens} H\_t^T (H\_t P\_t^{ens} H\_t^T + R\_t)^{-1}, \\quad P\_t^{ens} = \\frac{1}{N\_e - 1} \\sum\_{i=1}^{N\_e} (x\_t^{f,(i)} - \\bar{x}\_t^f)(\\cdot)^T$$

**Langevinized EnKF (2024):**

$$x\_t^{(i)} \\leftarrow x\_t^{(i)} + \\frac{\\epsilon}{2}\\nabla\_{x\_t}\\log p(x\_t | y\_{1:t}) + \\sqrt{\\epsilon},\\xi\_t$$

|Symbol|Meaning|Role|How Determined|
|-|-|-|-|
|$N\_e$|Ensemble size|Parallel trajectories|Budget; typical 50–1000|
|$\\epsilon$|Langevin step size|Adjustment strength|Tuning|

**✅ Pros:** Scalable to $10^6$+ dimensions; parallelizable; Langevinized handles non-Gaussian
**❌ Cons:** Small ensembles → spurious correlations; KF-like gain assumes Gaussian; localization ad-hoc
**⚠️ Financial Pitfalls:** Small ensembles in finance → poor covariance; spurious risk factor correlations

||Academia|Financial Industry|
|-|-|-|
|Active research?|✅ Active|⚠️ Very niche|
|Attention level|★★★★☆|★☆☆☆☆|

\---

## Category 14: Physics-Inspired / Unconventional Methods

### 14.1 Feynman-Kac / Path Integral Filter

**Core Idea:** Filtering distribution as a path integral via the Feynman-Kac formula from quantum mechanics.

$$p(x\_{0:t} | y\_{1:t}) \\propto \\exp\\left(-S\[x\_{0:t}]\\right) \\cdot p\_0(x\_0) \\prod\_{k=1}^t p(x\_k | x\_{k-1})$$

$S\[x\_{0:t}]$: "action" functional = accumulated observation likelihood over path.

**✅ Pros:** Extremely general; no assumptions; subsumes PF
**❌ Cons:** Intractable computation; requires Monte Carlo; purely theoretical

||Academia|Financial Industry|
|-|-|-|
|Active research?|⚠️ Very niche|❌ Not adopted|
|Attention level|★☆☆☆☆|★☆☆☆☆|

\---

### 14.2 Diffusion Maps Kalman Filter

**Core Idea:** Use diffusion maps (manifold learning) to discover intrinsic geometry, then apply KF in diffusion coordinates.

$$\\Psi(x) = \[\\psi\_1(x), \\ldots, \\psi\_d(x)]^T, \\quad z\_{t+1} = A\_d z\_t + w\_t^d$$

**✅ Pros:** Non-parametric geometry discovery; efficient KF in diffusion coordinates
**❌ Cons:** Offline computation; out-of-sample approximate; $\\epsilon$-sensitive; not designed for non-Gaussian noise

||Academia|Financial Industry|
|-|-|-|
|Active research?|⚠️ Niche|❌ Not adopted|
|Attention level|★★☆☆☆|★☆☆☆☆|

\---

## 🎯 Recommended Strategy

When **both** non-normality and orthogonality violation are present:

|Priority|Model|Rationale|
|-|-|-|
|🥇|**Particle Filter**|Any distribution + any dependence; gold standard|
|🥈|**Copula SSM**|Explicit non-Gaussian marginals + non-orthogonal copula dependence|
|🥉|**Wasserstein Filter**|Distribution-agnostic OT; handles joint $\\gamma(x,z)$; emerging|
|4th|**GAS + auxiliary state**|Score-driven observations + separate state evolution|
|5th|**Student-t VB-KF + augmented state**|Practical, fast, heavy tails + correlated noise|
|6th|**Neural/Flow Filter**|Max flexibility but black-box|


## 📚 References \& Further Reading

|Topic|Key References|
|-|-|
|Particle Filtering|Doucet et al. (2001); Arulampalam et al. (2002); Corenflos et al. (2021)|
|PMCMC|Andrieu et al. (2010); Lindsten et al. (2014); Chopin et al. (2013)|
|Student-t KF|Roth et al. (2017); Huang et al. (2022); Zhu et al. (2025)|
|Skew-Normal KF|Naveau et al. (2005); Genton (2004); Mode-matching CSN KF (2025)|
|Gaussian Sum Filter|Alspach \& Sorenson (1972); CUGSF (2016); GMM-VB (2023)|
|MCC Filter|Liu et al. (2017); DCM\_MCCKF (2024); Student-t kernel MCC (2022)|
|Wasserstein Filter|Prabhat \& Bhattacharya (2024)|
|H∞ Filter|Simon (2006); Shen \& Deng (1997); Ensemble H∞ (2025)|
|Huber-KF|Rist \& Moush (1985); Robust derivative-free Huber-KF (2013)|
|Projection Filter|Brigo et al. (1998, 2022); Optimal projection with info geometry (2023)|
|Variational Bayes|Särkkä \& Nummenmaa (2009); GMM-VB filter (2023)|
|GAS/Score-Driven|Creal, Koopman \& Lucas (2013); Harvey (2022)|
|Moving Horizon Est.|Rao et al. (2003); Robust MHE (2021); Koopman-MHE (2023)|
|Copula SSM|Kreuzer et al. (2023); Loaiza-Maya et al. (2023)|
|Koopman KF|Brunton \& Kutz (2019); Deep Koopman KF (2025)|
|Neural/Flow Filter|Corenflos et al. (2021); NF-based Differentiable PF (2024); Neural KF (2025)|
|Langevinized EnKF|Zhang et al. (2024)|
|Feynman-Kac|Kunita (1990); Euclidean QM \& Universal NL Filtering (2009)|
|Diffusion Maps KF|Frei \& Kase (2017); Diffusion Maps PF (2019)|

\---

> \*\*Citation:\*\* 
>
> Baofeng (2026). \*Alternative Models for Kalman Filter under Non-Normality \& Orthogonality Violation.\* GitHub Repository.

