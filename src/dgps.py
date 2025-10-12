import numpy as np
import logging


logger = logging.getLogger(__name__)


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def create_covariance_matrix(n, rho=0.0):
    """
    Create an n x n covariance matrix with 1s on the diagonal and rho elsewhere.
    """
    C = np.ones((n, n)) * rho
    # huh, so this modifies in place for some reason
    np.fill_diagonal(C, 1)
    return C


class DatasetGenerator(object):
    def __init__(self, n=5, rho=0.0, tau_0=1.0, tau_1=1.0, sigma2=1.0, rng=None):
        if rng is None:
            rng = np.random.default_rng(rng)
        self.n = n
        self.rho = rho
        self.tau_0 = tau_0
        self.tau_1 = tau_1
        self.sigma2 = sigma2
        self.rng = rng
        self.cov_mat = create_covariance_matrix(n, rho)

    def __call__(self, N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        true_beta = self.generate_beta()
        X, y = self.generate_design_matrix(N, true_beta)
        return X, y, true_beta

    def generate_design_matrix(self, N, true_beta):
        logger.debug(f"Using RNG: {self.rng}")

        Z = self.rng.multivariate_normal(
            mean=np.zeros(self.cov_mat.shape[0]),
            cov=self.cov_mat,
            size=N,
            check_valid="warn",
            tol=1e-8,
        )
        c_j = self.rng.uniform(-0.25, 0.25, size=self.cov_mat.shape[0])
        logging.debug(f"c_j values: {', '.join(f'{val:.4f}' for val in c_j)}")

        eps_k = self.rng.normal(0, self.sigma2, size=N)
        X = Z > c_j

        # add intercept that centers the logits
        alpha = -np.mean(X @ true_beta + eps_k)
        logging.debug(f"Alpha (intercept) value: {alpha:.4f}")

        logits = alpha + X @ true_beta + eps_k
        p_i = sigmoid(logits)
        y = self.rng.binomial(1, p_i)
        logging.debug(f"{np.sum(y)}/{N} positive responses {np.mean(y):0.3f}.")

        return X, y

    def generate_beta(self, rng=None):
        if rng is None:
            rng = np.random.default_rng(rng)

        z_i = rng.choice([0, 1], size=self.n, p=[0.8, 0.2])
        pi_i = rng.exponential(self.tau_1, size=self.n)

        delta_i = rng.normal(0, self.tau_0, size=self.n)
        beta = z_i * pi_i + delta_i
        return beta
