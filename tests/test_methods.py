import numpy as np
import pytest

from src.dgps import DatasetGenerator
from src.methods import mle, parametricEB, semi_bayes


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


@pytest.mark.parametrize("tau2", [1.0, 10.0, 1000], ids=lambda t: f"tau2={t}")
def test_semi_bayes(mle_model, tau2):
    beta_star, _ = semi_bayes(mle_model, tau2)

    # assert shrinkage occurs, at least when tau2 > 1
    assert np.linalg.norm(beta_star) <= np.linalg.norm(mle_model.params)


