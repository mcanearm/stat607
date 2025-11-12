import statsmodels.api as sm
import numpy as np
import logging
from collections import namedtuple

logger = logging.getLogger(__name__)


def __get_mle_vhat(model):
    """
    Extract MLE estimates and covariance from a fitted statsmodels model.
    Simple helper function to avoid boilerplate code in EB and SB methods.
    """

    # ignore the intercept
    beta_hat = (
        model.params.values[1:] if hasattr(model.params, "values") else model.params[1:]
    )
    V_hat = (
        model.cov_params().values
        if hasattr(model.cov_params(), "values")
        else model.cov_params()
    )
    V_hat = V_hat[1:, 1:]  # drop intercept row/col
    return beta_hat, V_hat


def fit_mle(X, y, **fit_params):
    """
    Compute the MLE for logistic regression; servces as a
    baseline method for our simulations.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (N, n).
    y : np.ndarray
        Binary response vector of shape (N,).

    Returns
    -------
    model: statsmodels.discrete.discrete_model.BinaryResults
    """

    fit_params = fit_params or {"disp": False, "maxiter": 100}
    X = sm.add_constant(X.astype(int))
    model = sm.Logit(y, X).fit(**fit_params)

    if not model.mle_retvals["converged"]:
        raise RuntimeError("MLE fitting did not converge.")
    else:
        return model


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

    # post convergence estimates
    # inside fit_parametricEB after convergence
    tau = tau_tilde2
    d = lam + tau
    inv_eigh_tau = 1 / d
    W_star = (
        Q * inv_eigh_tau
    ) @ Q.T  # equivalent to matrix mult on diag thanks to broadcasting

    # B*: use W*V with the small-sample factor
    c = (n - 1 - 2) / (n - 1)
    B_diag = c * (lam / (lam + tau))
    B_star = (Q * B_diag) @ Q.T

    ZtWZ = np.sum(u1 * u1 * inv_eigh_tau)
    ZtWbeta = np.sum(u1 * u_beta * inv_eigh_tau)

    pi = ZtWbeta / ZtWZ
    mu_star = pi * Z

    # posterior mean
    beta_star = B_star @ mu_star + (np.eye(n) - B_star) @ beta_hat

    # A term
    e = beta_hat - mu_star
    be = B_star @ e
    A = 2 * np.outer(be, be) / (n - 1)

    # base covariance (eq. (4))
    C_star = V_hat @ (np.eye(n) - (n - 1) * B_star / n) + A

    # componentwise variance (eq. (12))
    Z = np.ones((n, 1))
    A_t = 1 / ZtWZ
    W1 = W_star @ Z
    diag_H = A_t * W1
    v_star = np.trace(W_star @ V_hat) / np.trace(W_star)  # v*
    VBs = V_hat @ B_star
    WA = W_star @ A  # *** matrix product ***

    adj_vars = (
        np.diag(V_hat)
        - (1.0 - diag_H) * np.diag(VBs)
        + (v_star + tau_tilde2) * np.diag(WA)
    )
    np.fill_diagonal(C_star, adj_vars)

    if np.any(np.diagonal(C_star) <= 0):
        raise RuntimeError("Parametric EB covariance has non-positive variances.")
    elif np.any(np.diagonal(C_star) >= 10):
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
    Z = np.ones((n, 1))

    W = np.linalg.solve(V_hat + tau2 * np.eye(n), np.eye(n))
    B = W @ V_hat

    A_t = np.linalg.solve(Z.T @ W @ Z, np.eye(p))
    pi_tilde = A_t @ (Z.T @ W @ beta_hat)
    mu_tilde = Z @ pi_tilde  # since Z=1
    C_tilde = V_hat @ (np.eye(n) - (n - p) * B / n)  # see ADEMP doc for A def

    beta_tilde = B @ mu_tilde + (np.eye(n) - B) @ beta_hat

    # update variances of C_tilde
    H_tilde = Z @ A_t @ Z.T @ W
    var_adjusted = np.diagonal(V_hat) - (1 - np.diagonal(H_tilde)) * np.diagonal(
        V_hat @ B
    )
    np.fill_diagonal(C_tilde, var_adjusted)

    return semiBayesResults(beta_tilde, C_tilde)
