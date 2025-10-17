import pytest
import numpy as np
from src.dgps import DatasetGenerator


@pytest.fixture()
def generate_data():
    rng = np.random.default_rng(42)
    return DatasetGenerator(n=10, rho=0.0, tau_0=1.0, tau_1=1.0, sigma2=1.0, rng=rng)
