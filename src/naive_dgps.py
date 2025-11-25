import numpy as np
import logging
from dataclasses import dataclass
import json
import pickle as pkl
from pathlib import Path
from typing import Union, Mapping


logger = logging.getLogger(__name__)


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def create_covariance_matrix(n, rho=0.0, tol=1e-6):
    """
    Create an n x n covariance matrix with 1s on the diagonal and rho elsewhere.
    """
    C = np.ones((n, n)) * rho
    # This modifies in place for some reason
    np.fill_diagonal(C, 1)
    return C


@dataclass
class SimulatedData(object):
    """
    Container for simulated data and parameters.
    Also includes methods for saving and loading from disk.

    Attributes
    ----------
    X : np.ndarray
        Design matrix of shape (N, n).
    y : np.ndarray
        Binary response vector of shape (N,).
    true_beta : np.ndarray
        True regression coefficients of shape (n,).
    n : int
        See DatasetGenerator.
    rho : float
        See DatasetGenerator.
    tau_0 : float
        See DatasetGenerator.
    tau_1 : float
        See DatasetGenerator.
    sigma2 : float
        See DatasetGenerator.
    rng : np.random.Generator
        Random number generator used for reproducibility.
    """

    X: np.ndarray
    y: np.ndarray
    true_beta: np.ndarray
    n: int
    rho: float
    tau_0: float
    tau_1: float
    sigma2: float
    rng: np.random.Generator
    generator_state: Mapping

    def __post_init__(self):
        self.N = self.X.shape[0]

    def __repr__(self):
        """
        More detailed representation for debugging.
        """
        return (
            f"<SimulatedData(N={self.N}, n={self.n}, rho={self.rho}, tau_0={self.tau_0}, "
            f"tau_1={self.tau_1}, sigma2={self.sigma2}, "
            f"X_shape={self.X.shape}, y_shape={self.y.shape})>"
        )

    def __iter__(self):
        """
        Allow unpacking like a tuple: X, y, true_beta = simulated_data
        """
        yield from (self.X, self.y, self.true_beta)

    def __str__(self) -> str:
        return json.dumps(self.__dict__)

    def __create_filename(self):
        return Path(
            f"simdata_N{self.N}n{self.n}_rho={self.rho}_tau0={self.tau_0}_"
            f"tau1={self.tau_1}_sigma2={self.sigma2}.pkl"
        )

    def save(self, output_dir: Union[Path, str]):
        """
        Save the simulated data for later, along with RNG state for
        reproducibility.

        Parameters
        ----------
        output_dir : Union[Path, str]
            Directory to save the data file in. Note that you provide a
            directory because the filename is decided based on the parameters
            used in the generation of the data.
        """
        output_file = Path(output_dir) / self.__create_filename()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Saving simulated data to {output_file}")
        with open(output_file, "wb") as f:
            pkl.dump(self, f)
        logging.debug(f"Simulated data saved to {output_file}")
        return output_file

    @classmethod
    def load(cls, filepath):
        """
        Load simulated data from disk. Note that this can be used to load
        any pickled object, so we check that the loaded object is of the
        correct type.

        Parameters
        ----------
        filepath : Union[Path, str]
            Path to the saved data file.

        Returns
        -------
        SimulatedData
        """
        with open(filepath, "rb") as f:
            data = pkl.load(f)
        assert isinstance(data, cls), f"Loaded data is not of type {cls.__name__}"
        return data


class DatasetGenerator(object):
    def __init__(self, n=5, rho=0.0, tau_0=1.0, tau_1=1.0, sigma2=1.0, rng=None):
        """
        Constructor class for generating simulated datasets with the given
        parameters. This class utilizes a callable interface, so when generating
        from this class, you use it like a function (e.g. `generate_data(N)`),
        where N is the number of samples to generate.

        Parameters
        ----------
        n : int
            Number of predictors.
        rho : float
            Correlation parameter used in covariance matrix generation of Z
            values. Combined with draws from a uniform (-0.25, 0.25) uniform
            distribution to create correlated binary predictors in X,
            representing exposures.
        tau_0 : float
            Standard deviation for noise (δ_i) in true beta generation.
        tau_1 : float
            Scale parameter for the exponential distribution in beta generation,
            the actual beta effect. For any given beta, there is an
            80% change that this simulated value is ignored and assigned a 0.
        sigma2 : float
            Standard deviation of the noise (ϵ_k) added to the logits for
            individual rows, i.e. additional noise in the log-odds not accounted
            for by the binomial distribution assumed on Y conditional on X.
        rng : np.random.Generator
            Random number generator used for reproducibility.
        """
        if rng is None:
            rng = np.random.default_rng(rng)
        self.n = n
        self.rho = rho
        self.tau_0 = tau_0
        self.tau_1 = tau_1
        self.sigma2 = sigma2
        self.rng = rng
        self.cov_mat = create_covariance_matrix(n, rho)

    def __call__(self, N=100, state: Union[Mapping, None] = None) -> SimulatedData:
        if state is not None:
            self.rng.bit_generator.state = state
        generator_state = self.rng.bit_generator.state
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
            generator_state,
        )

    def generate_design_matrix(self, N, true_beta):
        """
        Generate the design matrix X and binary response vector y as outlined
        in Greenland (1993). The design matrix X is a binary matrix of u
        exposures, generated by thresholding correlated normal variables Z.
        Additional noise is added over the standard binomial variance for
        y | X. Finally, the alpha intercept is chosen to center the logits
        and ensure that roughly 50% of the responses are 1s.
        """
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
        """
        Generate true regression coefficients for simulation. Interestingly,
        the beta draws are independent, despite us assuming correlation
        when estimated a parameter shrinkage term.
        """
        z_i = self.rng.choice([0, 1], size=self.n, p=[0.8, 0.2])
        pi_i = self.rng.exponential(self.tau_1, size=self.n)

        delta_i = self.rng.normal(0, self.tau_0, size=self.n)
        beta = z_i * pi_i + delta_i
        return beta
