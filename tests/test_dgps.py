import pytest
from src.dgps import DatasetGenerator, create_covariance_matrix, SimulatedData
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
    generate_data = DatasetGenerator(
        n=n, rho=rho, tau_0=1.0, tau_1=1.0, sigma2=1.0, rng=rng
    )

    assert generate_data.n == n

    X, y, true_beta = (simulated_data := generate_data(N))

    assert isinstance(simulated_data, SimulatedData)

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


# tmp_path is a fixture that is always available in pytest
def test_saving_loading(tmpdir):
    rng = np.random.default_rng(42)
    generate_data = DatasetGenerator(
        n=5, rho=0.0, tau_0=1.0, tau_1=1.0, sigma2=1.0, rng=rng
    )
    simulated_data = generate_data(100)

    output_file = simulated_data.save(tmpdir)
    loaded_data = SimulatedData.load(output_file)

    assert np.array_equal(simulated_data.X, loaded_data.X)
    assert np.array_equal(simulated_data.y, loaded_data.y)
    assert np.array_equal(simulated_data.true_beta, loaded_data.true_beta)
    assert simulated_data.n == loaded_data.n
    assert simulated_data.rho == loaded_data.rho
    assert simulated_data.tau_0 == loaded_data.tau_0
    assert simulated_data.tau_1 == loaded_data.tau_1
    assert simulated_data.sigma2 == loaded_data.sigma2

    # ensure rng state is the same between loaded and generated
    assert np.isclose(loaded_data.rng.normal(), simulated_data.rng.normal())
