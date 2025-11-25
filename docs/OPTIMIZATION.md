# Optimizations

## Chatty Printing

The `numpy` logging module was issuing a ton of print statements thanks to the conversion of the numpy arrays into strings. Removing this improved our overall performance by about 20 seconds overall, as between the first profiling and the second, printing is not anywhere near the top of the list. 

The code change was simple as well the original line was 

```python
logger.debug(data_generation.__dict__)  # original code
```

and I instead changed it to explicity pass in values. This prevented passing in the covariance matrix, which dramatically reduced the overhead from printing, apparently. 
```python
logger.debug(  # better code
    "sim complete — n=%d rho=%.3f tau0=%.3f tau1=%.3f sigma2=%.3f",
    data_generation_fn.n,
    data_generation_fn.rho,
    data_generation_fn.tau_0,
    data_generation_fn.tau_1,
    data_generation_fn.sigma2,
)
```

## Fit Parametric EB

I identified this as the largest estimator bottleneck because it is an iterative method and I was doing a lot of matrix inversions within the loop. I spent a lot of time on this, converting inverts to solves. Ultimately, I used ChatGPT to get the idea of using an eigendecomposition to dramatically reduce the number of operations. For example, the original iterative 
loop is shown below, and closely mirrors the stated algorithm in the original paper.

```python
beta_hat, V_hat = __get_mle_vhat(model)
n = len(beta_hat)
p = 1  # intercept-only prior mean
Z = np.ones((n, p))

# Initialize
tau_tilde2 = 1e-3
W_star = np.linalg.solve(V_hat + tau_tilde2 * np.eye(n), np.eye(n))
e = beta_hat - np.zeros(n)
for _ in range(max_iter):
    # Prior mean
    A_t = np.linalg.solve(Z.T @ W_star @ Z, np.eye(p))
    pi_star = A_t @ (Z.T @ W_star @ beta_hat)
    mu_star = Z @ pi_star

    # Residuals
    e = beta_hat - mu_star

    # Update R
    R = (e.T @ W_star @ e) / np.trace(W_star)

    # Update tau^2
    # TODO: confirm V_bar_star calculation - I think it's wrong.
    V_bar_star = np.trace(W_star @ V_hat) / np.trace(W_star)
    tau_new = max(n * R / (n - p) - V_bar_star, 1e-8)  # avoid negative

    # Update weights
    W_star = np.linalg.solve(V_hat + tau_new * np.eye(n), np.eye(n))
    B_star = (n - p - 2) / (n - p) * W_star @ V_hat
    beta_star = B_star @ mu_star + (np.eye(n) - B_star) @ beta_hat

    # Check convergence
    logger.debug(f"Iter {_}: tau^2 = {tau_new}")
    if np.abs(tau_new - tau_tilde2) < tol:
        tau_tilde2 = tau_new
        logger.debug(f"Converged after {_} iterations.")
        break
    tau_tilde2 = tau_new
```

And the eigendeomposition shortens this fairly dramatically, avoiding the matrix inversions entirely, and relying on a single one outside the loop.

```python
beta_hat, V_hat = __get_mle_vhat(model)
n = len(beta_hat)
Z = np.ones(n)

# Initialize
tau_tilde2 = 1e-3

lam, Q = np.linalg.eigh(V_hat)

u1 = Q.T @ Z
u_beta = Q.T @ beta_hat

for _ in range(max_iter):
    inv_eigh_tau = 1 / (lam + tau_tilde2)
    ZtWZ = np.sum(u1 * u1 * inv_eigh_tau)
    ZtWbeta = np.sum(u1 * u_beta * inv_eigh_tau)
    # Prior mean
    pi_star = ZtWbeta / ZtWZ

    # Residuals in eigenbasis (?)
    e = u_beta - pi_star * u1

    # Update R
    trW = np.sum(inv_eigh_tau)
    eWe = np.sum(e * e * inv_eigh_tau)
    R = eWe / trW

    # Update tau^2
    # TODO: confirm V_bar_star calculation - I think it's wrong.
    trWV = np.sum(lam * inv_eigh_tau)
    V_bar_star = trWV / trW
    tau_new = max(n * R / (n - 1) - V_bar_star, 1e-8)  # avoid negative

    # Check convergence
    logger.debug(f"Iter {_}: tau^2 = {tau_new}")
    if np.abs(tau_new - tau_tilde2) < tol:
        tau_tilde2 = tau_new
        logger.debug(f"Converged after {_} iterations.")
        break
    tau_tilde2 = tau_new
```

This only reduces my total runtime by about 2 seconds however, and to be honest, I don't think it was a particularly useful way to spend my time. I understand my code a little less in exchange for a pretty small incresae in speed.

Small Results are par for the course after that - in fact, I mistakenly thought that several lines in the baseline were due to the MLE method, and so sought to shorten those. However, my MLE code actually end up running slightly slower. All told, a comparison of the profiling before and after yields a table that I think is fairly compelling. These results are generating in `docs/timings.ipynb`.

|    | func                   |   tottime_v1 |   tottime_v2 |   tottime_diff |
|---:|:-----------------------|-------------:|-------------:|---------------:|
|  0 | fit_mle                |     0.160261 |   1.82315    |       1.66289  |
|  4 | fit_semiBayes          |     4.24764  |   4.40158    |       0.153941 |
|  5 | generate_design_matrix |     6.19043  |   5.73713    |      -0.453293 |
|  2 | fit_parametricEB       |     7.68701  |   5.22712    |      -2.45989  |
|  1 | arrayprint:recurser    |     7.73412  |   2.2962e-05 |      -7.7341   |
|  3 | _path_join             |     7.87764  |   0.00472487 |      -7.87292  |`

So the largest decrease in my time was from chatty printing (it's missing now), followed by parametricEB. My changes
to the MLE actually made my code slower, though not enough to offset the other changes. Across all changes, we see the following
relationship between the two versions:
![timing comparison](log-log-timings.png)

The best return on investment turned out to be a data conversion issue (numpy array to string) that I was completely overlooking in my previous code for the sake of transparency and reproducibility. Once I fixed that, I saw my largest gains. In fact, my relatively naive implementation, because it was already vectorized and utilized optimized packages, saw only minor gains, and I actually made my MLE method worse. But truly, the biggest surprise was the dramatic speedup from removing the long chatty debug statement, especially since by default it should not have even been printing. It appears that in Python logging, the message is always created, but not necessarily shown, depending on the logging level.

In earlier runs, I attempted to use JAX to do faster coding, but that proved a bit fruitless as well. I spent more time
compiling than I did running my code quickly, and it required a pretty hefty refactor that I never got around to finishing. I still think it might speed things up though, since now my timings are spread around a lot evenly throughout all my Python calls. Limiting the amount of time I spend in Python seems to be the next logical step.

