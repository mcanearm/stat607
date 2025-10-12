import numpy as np
import logging
from dataclasses import dataclass
import json
import pickle as pkl
from pathlib import Path
from typing import Union


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


@dataclass
class SimulatedData(object):
    X: np.ndarray
    y: np.ndarray
    true_beta: np.ndarray
    n: int
    rho: float
    tau_0: float
    tau_1: float
    sigma2: float
    rng: np.random.Generator

    def __repr__(self):
        return (
            f"SimulatedData(n={self.n}, rho={self.rho}, tau_0={self.tau_0}, "
            f"tau_1={self.tau_1}, sigma2={self.sigma2}, "
            f"X_shape={self.X.shape}, y_shape={self.y.shape})"
        )

    def __iter__(self):
        yield from (self.X, self.y, self.true_beta)

    def __str__(self) -> str:
        return json.dumps(self.__dict__)

    def __create_filename(self):
        return Path(
            f"simdata_n{self.n}_rho={self.rho}_tau0={self.tau_0}_"
            f"tau1={self.tau_1}_sigma2={self.sigma2}.pkl"
        )

    def save(self, output_dir: Union[Path, str]):
        output_file = Path(output_dir) / self.__create_filename()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Saving simulated data to {output_file}")
        with open(output_file, "wb") as f:
            pkl.dump(self, f)
        logging.debug(f"Simulated data saved to {output_file}")
        return output_file

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            data = pkl.load(f)
        assert isinstance(data, cls), f"Loaded data is not of type {cls.__name__}"
        return data


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

    def __call__(self, N) -> SimulatedData:
        true_beta = self.generate_beta()
        X, y = self.generate_design_matrix(N, true_beta)
        return SimulatedData(
            X,
            y,
            true_beta,
            self.n,
            self.rho,
            self.tau_0,
            self.tau_1,
            self.sigma2,
            self.rng,
        )

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

    def generate_beta(self):
        z_i = self.rng.choice([0, 1], size=self.n, p=[0.8, 0.2])
        pi_i = self.rng.exponential(self.tau_1, size=self.n)

        delta_i = self.rng.normal(0, self.tau_0, size=self.n)
        beta = z_i * pi_i + delta_i
        return beta
