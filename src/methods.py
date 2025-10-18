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
    beta_hat = model.params.values if hasattr(model.params, "values") else model.params
    V_hat = (
        model.cov_params().values
        if hasattr(model.cov_params(), "values")
        else model.cov_params()
    )
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

    fit_params = fit_params or {"disp": False, "maxiter": 1000}

    model = sm.Logit(y, X).fit(**fit_params)
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
    p = 1  # intercept-only prior mean
    Z = np.ones((n, p))

    # Initialize
    tau_tilde2 = 1e-3
    W_star = np.linalg.inv(V_hat + tau_tilde2 * np.eye(n))

    for _ in range(max_iter):
        # Prior mean
        pi_star = np.linalg.inv(Z.T @ W_star @ Z) @ (Z.T @ W_star @ beta_hat)
        mu_star = Z @ pi_star

        # Residuals
        e = beta_hat - mu_star

        # Update R
        R = (e.T @ W_star @ e) / np.trace(W_star)

        # Update tau^2
        V_bar_star = np.trace(W_star @ V_hat) / np.trace(W_star)
        tau_new = max(n * R / (n - p) - V_bar_star, 1e-8)  # avoid negative

        # Update weights
        W_star = np.linalg.inv(V_hat + tau_new * np.eye(n))

        # Check convergence
        logger.debug(f"Iter {_}: tau^2 = {tau_new}")
        if np.abs(tau_new - tau_tilde2) < tol:
            tau_tilde2 = tau_new
            logger.debug(f"Converged after {_} iterations.")
            break
        tau_tilde2 = tau_new

    # Final B_star
    B_star = tau_tilde2 * np.linalg.inv(V_hat + tau_tilde2 * np.eye(n))
    beta_star = B_star @ mu_star + (np.eye(n) - B_star) @ beta_hat

    # Covariance adjustment
    be = B_star @ e
    A = 2 * np.outer(be, be) / (n - p)
    C_star = V_hat @ (np.eye(n) - (n - p) * B_star / n) + A

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
    B = np.linalg.inv((V_hat + tau2 * np.eye(len(beta_hat))))
    pi_tilde = B @ beta_hat
    mu_tilde = pi_tilde  # since Z=1
    C_tilde = V_hat @ (np.eye(n) - (n - p) * B / n)  # see ADEMP doc for A def

    beta_tilde = B @ mu_tilde + (np.eye(n) - B) @ beta_hat
    return semiBayesResults(beta_tilde, C_tilde)
