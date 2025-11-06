import numpy as np
import logging
from collections import namedtuple
from jax import numpy as jnp
import jax
import jax.lax as lax

logger = logging.getLogger(__name__)

mle_results = namedtuple("mle_results", ["beta_hat", "V_hat"])


def __get_mle_vhat(model):
    """
    Extract MLE estimates and covariance from a fitted statsmodels model.
    Simple helper function to avoid boilerplate code in EB and SB methods.

    Deprecated: this is leftover from utilizing statsmodels; since we are
    using scikit-learn now, just use a model of the beta hat estimates and the
    calculated variance directly.
    """

    # ignore the intercept
    return model


def _add_intercept(X):
    X = jnp.asarray(X, dtype=jnp.float64)
    return jnp.concatenate([jnp.ones((X.shape[0], 1), X.dtype), X], axis=1)


@jax.jit
def fit_mle(
    X,
    y,
    se_threshold=jnp.sqrt(10.0),
    max_iter: int = 100,
    tol: float = 1e-6,
    ridge_eps: float = 1e-8,  # tiny damping for numerical stability
):
    """
    ChatGPT Generated JAX implementation of binary logistic regression
    Logistic regression MLE via IRLS/Newton:
      beta_{t+1} = beta_t + (X^T W X + eps I)^{-1} X^T (y - p_t)
    Covariance uses the observed info inverse at the solution.
    Returns mle_results(beta_hat (p,), V_hat (p,p)) with intercept removed.
    """
    X = _add_intercept(X)
    y = jnp.asarray(y, dtype=jnp.float64).reshape((-1,))
    n, p_tot = X.shape

    def step(beta):
        eta = X @ beta
        p = jax.nn.sigmoid(eta)
        w = p * (1.0 - p)  # shape (n,)
        XtWX = X.T @ (w[:, None] * X)  # (p_tot, p_tot)
        g = X.T @ (y - p)  # score
        H = XtWX + ridge_eps * jnp.eye(p_tot)  # damped Hessian (PD)
        # Newton/IRLS increment
        delta = jnp.linalg.solve(H, g)
        return beta + delta, delta, H

    # while-loop state: (it, beta, step_norm, H_last)
    beta0 = jnp.zeros((p_tot,), dtype=X.dtype)
    it0 = jnp.array(0, dtype=jnp.int32)
    step_norm0 = jnp.array(jnp.inf, dtype=X.dtype)
    H_init = jnp.eye(p_tot, dtype=X.dtype)

    def cond_fun(state):
        it, beta, step_norm, _H = state
        return jnp.logical_and(it < max_iter, step_norm > tol)

    def body_fun(state):
        it, beta, _step_norm, _H = state
        beta_new, delta, H_new = step(beta)
        step_norm = jnp.linalg.norm(delta, ord=jnp.inf)
        return (it + 1, beta_new, step_norm, H_new)

    it_f, beta_f, _step_norm_f, H_f = lax.while_loop(
        cond_fun, body_fun, (it0, beta0, step_norm0, H_init)
    )

    # Observed-info covariance at the solution (use undamped XtWX if you prefer)
    eta_f = X @ beta_f
    p_f = jax.nn.sigmoid(eta_f)
    w_f = p_f * (1.0 - p_f)
    XtWX_f = X.T @ (w_f[:, None] * X)
    # pinv guards against borderline designs
    V_full = jnp.linalg.pinv(XtWX_f + ridge_eps * jnp.eye(p_tot))

    # strip intercept
    beta_hat = beta_f[1:]
    V_hat = V_full[1:, 1:]

    # simple guardrail: treat “wild” fits as invalid (stay pure: no raise)
    bad = jnp.logical_or(
        jnp.any(jnp.diag(V_hat) >= se_threshold**2),
        jnp.logical_not(jnp.isfinite(V_hat).all()),
    )
    beta_hat = jnp.where(bad, jnp.nan, beta_hat)
    V_hat = jnp.where(bad, jnp.nan, V_hat)

    return mle_results(beta_hat=beta_hat, V_hat=V_hat)


parametricEBResults = namedtuple(
    "parametricEBResults", ["beta_star", "C_star", "tau_tilde2"]
)


def fit_parametricEB(model, max_iter=250, tol=1e-6):
    """
    Parametric Empirical Bayes for first-stage MLEs (model.params)
    Assumes simple prior: beta_i ~ N(mu, tau^2), Z = 1.

    Parameters
    ----------
    model : statsmodels.discrete.discrete_model.BinaryResults
        Fitted model from statsmodels (MLEs + cov)
    max_iter : int
        Maximum iterations for fixed-point
    tol : float
        Convergence tolerance

    Returns
    -------
    beta_star : np.ndarray
        Empirical-Bayes shrunk estimates
    C_star : np.ndarray
        Approximate posterior covariance
    tau_tilde2 : float
        Estimated prior variance

    First attempt by myself following the notation in the ADEMP doc, but final
    imeplementation was mostly done by ChatGPT to adjust some mistakes I was making.
    """
    beta_hat, V_hat = __get_mle_vhat(model)
    n = len(beta_hat)
    p = 1  # intercept-only prior mean
    Z = jnp.ones((n, p))

    # Initialize
    tau_tilde2 = 1e-3
    W_star = jnp.linalg.solve(V_hat + tau_tilde2 * np.eye(n), np.eye(n))
    e = beta_hat - np.zeros(n)
    for _ in range(max_iter):
        # Prior mean
        A_t = jnp.linalg.solve(Z.T @ W_star @ Z, np.eye(p))
        pi_star = A_t @ (Z.T @ W_star @ beta_hat)
        mu_star = Z @ pi_star

        # Residuals
        e = beta_hat - mu_star

        # Update R
        R = (e.T @ W_star @ e) / jnp.trace(W_star)

        # Update tau^2
        # TODO: confirm V_bar_star calculation - I think it's wrong.
        V_bar_star = np.trace(W_star @ V_hat) / jnp.trace(W_star)
        tau_new = max(n * R / (n - p) - V_bar_star, 1e-8)  # avoid negative

        # Update weights
        W_star = jnp.linalg.solve(V_hat + tau_new * jnp.eye(n), jnp.eye(n))
        B_star = (n - p - 2) / (n - p) * W_star @ V_hat
        beta_star = B_star @ mu_star + (jnp.eye(n) - B_star) @ beta_hat

        # Check convergence
        logger.debug(f"Iter {_}: tau^2 = {tau_new}")
        if jnp.abs(tau_new - tau_tilde2) < tol:
            tau_tilde2 = tau_new
            logger.debug(f"Converged after {_} iterations.")
            break
        tau_tilde2 = tau_new

    # post convergence estimates
    # inside fit_parametricEB after convergence
    W_star = jnp.linalg.solve(V_hat + tau_tilde2 * jnp.eye(n), jnp.eye(n))

    # B*: use W*V with the small-sample factor
    B_star = ((n - p - 2) / (n - p)) * (W_star @ V_hat)

    # posterior mean
    beta_star = B_star @ mu_star + (jnp.eye(n) - B_star) @ beta_hat

    # A term
    e = beta_hat - mu_star
    be = B_star @ e
    A = 2 * jnp.outer(be, be) / (n - p)

    # base covariance (eq. (4))
    C_star = V_hat @ (jnp.eye(n) - (n - p) * B_star / n) + A

    # componentwise variance (eq. (12))
    A_t = jnp.linalg.solve(Z.T @ W_star @ Z, jnp.eye(p))
    H_star = Z @ A_t @ Z.T @ W_star
    v_star = jnp.trace(W_star @ V_hat) / jnp.trace(W_star)  # v*
    VBs = V_hat @ B_star
    WA = W_star @ A  # *** matrix product ***

    adj_vars = (
        jnp.diag(V_hat)
        - (1.0 - jnp.diag(H_star)) * jnp.diag(VBs)
        + (v_star + tau_tilde2) * jnp.diag(WA)
    )
    C_star = jnp.fill_diagonal(C_star, adj_vars, inplace=False)

    if jnp.any(jnp.diagonal(C_star) <= 0):
        raise RuntimeError("Parametric EB covariance has non-positive variances.")
    elif jnp.any(jnp.diagonal(C_star) >= 10):
        raise RuntimeError("Ill formed Parametric EB covariance matrix")

    return parametricEBResults(beta_star, C_star, tau_tilde2)


semiBayesResults = namedtuple("semiBayesResults", ["beta_tilde", "C_tilde"])


def fit_semiBayes(model, tau2=1.0):
    """
    Semi-Bayes estimator for first-stage MLEs (model.params)
    Assumes simple prior: beta_i ~ N(mu, tau^2), Z = 1
    Because we assumed the variance around the Beta parameters, this is a closed
    form solution and requires no iterations.
    Parameters
    ----------
    model : statsmodels.discrete.discrete_model.BinaryResults
        Fitted model from statsmodels (MLEs + cov)
    tau2 : float
        Prior variance
    """
    beta_hat, V_hat = __get_mle_vhat(model)

    n = len(beta_hat)
    p = 1
    Z = jnp.ones((n, 1))

    W = jnp.linalg.solve(V_hat + tau2 * jnp.eye(n), jnp.eye(n))
    B = W @ V_hat

    A_t = jnp.linalg.solve(Z.T @ W @ Z, jnp.eye(p))
    pi_tilde = A_t @ (Z.T @ W @ beta_hat)
    mu_tilde = Z @ pi_tilde  # since Z=1
    C_tilde = V_hat @ (jnp.eye(n) - (n - p) * B / n)  # see ADEMP doc for A def

    beta_tilde = B @ mu_tilde + (jnp.eye(n) - B) @ beta_hat

    # update variances of C_tilde
    H_tilde = Z @ A_t @ Z.T @ W
    var_adjusted = jnp.diagonal(V_hat) - (1 - jnp.diagonal(H_tilde)) * jnp.diagonal(
        V_hat @ B
    )
    C_tilde = jnp.fill_diagonal(C_tilde, var_adjusted, inplace=False)

    return semiBayesResults(beta_tilde, C_tilde)
