import numpy as np
import pytest

from src.dgps import DatasetGenerator
from src.methods import mle, parametricEB


@pytest.fixture
def generate_data():
    rng = np.random.default_rng(42)
    return DatasetGenerator(n=10, rho=0.0, tau_0=1.0, tau_1=1.0, sigma2=1.0, rng=rng)


@pytest.fixture
def generated_data(generate_data):
    return generate_data(100)


@pytest.fixture
def mle_model(generated_data):
    X, y, _ = generated_data
    model = mle(X, y)
    return model


def test_mle(generated_data):
    X, y, true_beta = generated_data
    model = mle(X, y)

    assert model
    assert len(model.params) == X.shape[1]


def test_parametricEB(mle_model):
    beta_hat, cov, tau_2 = parametricEB(mle_model, max_iter=1000)
    assert beta_hat is not None
    assert len(beta_hat) == len(mle_model.params)  # should return 5 items


def test_semi_bayes(mle_model):
    pass
