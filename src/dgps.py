import numpy as np


def generate_beta(n, tau_0, tau_1, rng=None):
    if rng is None:
        rng = np.random.default_rng(rng)

    z_i = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    pi_i = rng.exponential(tau_1, size=n)

    delta_i = rng.normal(0, tau_0, size=n)
    beta = z_i * pi_i + delta_i
    return beta


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def generate_design_matrix(N, cov_mat, sigma2, beta, rng=None):
    if rng is None:
        rng = np.random.default_rng(rng)

    Z = rng.multivariate_normal(
        mean=np.zeros(cov_mat.shape[0]),
        cov=cov_mat,
        size=N,
        check_valid="warn",
        tol=1e-8,
    )
    c_j = rng.uniform(-0.25, 0.25, size=cov_mat.shape[0])

    eps_k = rng.normal(0, sigma2, size=N)
    X = Z > c_j

    # add intercept that centers the logits
    alpha = -np.mean(X @ beta + eps_k)
    logits = alpha + X @ beta + eps_k
    p_i = sigmoid(logits)
    y = rng.binomial(1, p_i)

    return X, y


def create_covariance_matrix(n, rho=0.0):
    """
    Create an n x n covariance matrix with 1s on the diagonal and rho elsewhere.
    """
    C = np.ones((n, n)) * rho
    # huh, so this modifies in place for some reason
    np.fill_diagonal(C, 1)
    return C


class DataGenerator(object):
    def __init__(self, n, rho=0.0, tau_0=1.0, tau_1=1.0, sigma2=1.0, rng=None):
        if rng is None:
            rng = np.random.default_rng(rng)
        self.n = n
        self.rho = rho
        self.tau_0 = tau_0
        self.tau_1 = tau_1
        self.sigma2 = sigma2
        self.rng = rng

        self.cov_mat = create_covariance_matrix(n, rho)
        self.beta = generate_beta(n, tau_0, tau_1, rng)

    def generate(self, N):
        return generate_design_matrix(
            N, self.cov_mat, self.sigma2, self.beta, rng=self.rng
        )