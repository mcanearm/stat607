import pytest
from src.dgps import DatasetGenerator, create_covariance_matrix 
import numpy as np


@pytest.mark.parametrize("n", [2, 5, 10], ids=lambda n: f"n={n}")
@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.5], ids=lambda r: f"rho={r}")
def test_covar_generation(n, rho):
    C = create_covariance_matrix(n, rho)
    assert C.shape == (n, n)
    assert np.all(np.diag(C) == 1)
    assert np.all(C >= -1) and np.all(C <= 1)


@pytest.mark.parametrize("N", [10, 100], ids=lambda N: f"N={N}")
@pytest.mark.parametrize("n", [1, 2, 5, 10], ids=lambda n: f"n={n}")
@pytest.mark.parametrize("rho", [0, 0.5, -0.5], ids=lambda rho: f"rho={rho}")
def test_data_simulation(N, n, rho):

    rng = np.random.default_rng(42)
    generate_data = DatasetGenerator(n=n, rho=rho, tau_0=1.0, tau_1=1.0, sigma2=1.0, rng=rng)

    assert generate_data.n == n

    X, y, true_beta = generate_data(N)

    assert X.shape == (N, n)
    assert np.all(np.isin(X, [0, 1]))
    assert y.shape == (N,)
    assert np.isin(y, [0, 1]).all()
    assert true_beta.shape == (n,)

    # make sure we can't rule out 50% for the mean of Y with a silly CI test
    # note that this still may fail sometimes due to randomness
    assert (
        np.mean(y) - 1.96 * np.std(y) / np.sqrt(N)
        < 0.5
        < np.mean(y) + 1.96 * np.std(y) / np.sqrt(N)
    )
