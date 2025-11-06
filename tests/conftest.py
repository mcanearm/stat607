import pytest
from src.dgps import DatasetGenerator
import jax


@pytest.fixture(scope="session")
def prng_key():
    return jax.random.PRNGKey(0)


@pytest.fixture(scope="session")
def generate_data():
    return DatasetGenerator(n=10, rho=0.0, tau_0=1.0, tau_1=1.0, sigma2=1.0)
