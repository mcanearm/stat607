# Simulation Study

We'll be looking specifically at the following paper:

Greenland, Sander. “Methods for Epidemiologic Analyses of Multiple Exposures: A Review and Comparative Study of Maximum-Likelihood, Preliminary-Testing, and Empirical-Bayes Regression.” Statistics in Medicine 12, no. 8 (1993): 717–36. https://doi.org/10.1002/sim.4780120802.


## Template file structure

```
simulation-study/
├── data/
│ └── simulated/ # cache simulation replicates if needed
├── src/
│ ├── dgps.py # data-generating functions
│ ├── methods.py # statistical methods being evaluated
│ ├── metrics.py # performance measure calculations
│ └── simulation.py # main simulation orchestration
├── results/
│ ├── figures/ # visualizations
│ └── raw/ # raw simulation output (*.csv, *.pkl)
├── tests/
├── Makefile # automated workflow
├── ADEMP.md # simulation design document
├── README.md
├── requirements.txt
└── .gitignore
```

## Aims

Identify the accuracy gains in modelling from hierarchical Bayesian methods, specifically empirical Bayes regression and "semi-Bayes" regression vs. full-model maximum likelihood and reduction via preliminary testing.

Colloquially, if you fit a normal logistic regression model, you tend to underpower your regression by not detecting true effects.

## Data-Generating Mechanism

$$
\begin{align}Pr(y = 1 | \textbf{x}) &= \frac{\omega(\textbf{x})}{1 + \omega(\textbf{x})} \tag{8}\\
g(E[y | \textbf{x}, \textbf{w}]) &= \alpha + \textbf{x}\beta + \textbf{w}\gamma
\end{align}
$$

Here, $g$ is a known, increasing link function (as in logistic regression).

> Pre-adjustment of the data for nuisance variables $\textbf{w}$ involves addition of an 'offset' variable (with a coefficient fixed at 1) to the linear predictor $\alpha + \textbf{x}\beta$.

$$
\begin{align*}
\delta_i & \sim N(0, \tau_0^2) \\
\mu_i & \sim Exp(\tau_1) \\
z_i & \sim Bern(0.2) \\
\beta_i &= z_i \pi_i + \delta_i \tag{9} \\
\beta &= Z\pi + \delta = \mu + \delta \tag{3}
\end{align*}
$$

For each individual $i$, simulate the effect of exposure (each $\beta_i$).

$$
Var(\beta_i) = 0.8(1-0.8)\tau_1^2 + \tau_0^2
$$

> Of primary interest here is the performance of the simple Gaussian-approximation EB and SB  methods relative to ordinary ML and PT. Thus, the fitted prior assumed pi = ,u + 6i, with ,u an  unknown fixed constant and the 6i independent Gaussian variates with zero means and common variance, so that the fitted stage II distribution was misspecified.

The design matrix for each simulation run is generated via

$$
\begin{align*}
C & = \begin{bmatrix}
1 &  \cdots & r \\
\vdots & \ddots & \vdots \\
r &  \cdots & 1
\end{bmatrix}_{n \times n} \\
Z_i \in \mathbb{R}^n & \sim N(\textbf{0}, C) \\
c_j & \sim \text{Unif}(-0.25, 0.25) \\
X_{ij} &= \begin{cases}
1 & \text{if } Z_{ij} > c_j \\
0 & \text{otherwise}
\end{cases}
\end{align*}
$$

Somewhat confusingly, $n$ is the number of covariates, whereas $N$ is the number of observations.

For row $k$ (individual level),

$$
\begin{align*}
Pr(y_k = 1) & = \omega_k/(1+\omega_k) \\
\omega_k &= \exp\{\alpha + \textbf{x}_k\beta + \epsilon_k\} \\
\epsilon_k &\sim N(0, \sigma^2),
\end{align*}
$$

where $\textbf{x}_k$ represents the exposures for a particular individual and $\epsilon_k$ represents an individual level imprecision. In each trial, $\alpha = \frac{1}{n}\sum_{k=1}^n \textbf{x}_k \beta + \epsilon_k$, which gives a roughly 50/50 division of positive to negative cases, because this "produced efficient simulations."

## Estimands/Targets

### Estimands

$\beta$ - the true parameters. About 80% should be 0 in any simulation, with the remaining 20% having various effect sizes from an exponential distribution, plus some residual $\delta_i$.

## Methods

We need closed form solutions for each of these. Logistic regression is common practice, but for the shrinkage we need to use method of moment estimators in order to have a faster simulation; otherwise, we're doing MCMC sampling or something on each run, which takes too long.

### Maximum Likelihood (ML)

The random effect $\hat{\beta}$ were computed from each sample $(y, X)$ under model 8, the normal logistic regression model. This is mis-specified because there is no assumed error term for each individual under the normal model, $\epsilon_k$.

### Parametric Empirical Bayes (EB)

Method of moment estimators - there are a lot, but we'll try to organize them.

> Suppose the likelihood for $\beta$ is well approximated by an MVN, i.e. $\beta \sim N(\hat{\beta}, \hat{V})$.

$\hat{\beta}$ is the MLE estimate^[Found in the normal way?], $\hat{V}$ is the inverse of the observed information matrix evaluated at $\beta$.

MOM Estimators:

$$
\begin{align*}
W^* &= (\hat{V} + \tilde{\tau}^2I)^{-1} \\
\bar{V}^* &= \frac{W^*\hat{V}}{\sum_{ij}W^*_{ij}} \\
\tilde{\tau}^2 &= \frac{nR}{n-p} - \bar{V}^* \\
B^* &= (\hat{V} + \tilde{\tau}^2I)^{-1} \\
\pi^* &= (Z^\prime W^* Z)^{-1} Z^{\prime}W^*\hat{\beta} \\
\mu^* &= Z\pi^* \\
\beta^* &= B^* \mu^* + (I-B^*)\hat{\beta} \\
e &= \hat{\beta} - \mu^* \\
R &= \frac{e^\prime W^*e}{\sum_{ij}W_{ij}^*} \\
A &= \frac{2B^*e(B^*e)^T}{n-p} \\
C^* &= \hat{V}[I - (n-\beta)B^*/n] + A
\end{align*}
$$

### Semi-Bayes (SB)

Similar, but smaller set. $\tilde{C}$ is the posterior covariance of the MVN distribution for $\beta$.

$$
\begin{align*}
\tilde{C} &= \hat{V} \left[I - \frac{(n-p)B}{n}\right] \\
\tilde\pi &= (Z^TWZ)^{-1}Z^TW\hat\beta \\
\tilde\mu &= Z \tilde\pi \\
\tilde{\beta} &= B\tilde\mu + (I-B)\hat\beta \\
W &= (\hat{V} + \tau^2I)^{-1}
\end{align*}
$$

## Performance 

- Mean coverage rates of 95% interval for $\hat\beta$
- Mean lengths of simulated 95% confidence intervals
- RMSE of point estimators as percent of error of ML estimator