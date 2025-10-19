import numpy as np
import pytest

from src.methods import fit_mle, fit_parametricEB, fit_semiBayes, __get_mle_vhat


@pytest.fixture
def generated_data(generate_data):
    return generate_data(100)


@pytest.fixture
def mle_model(generated_data):
    X, y, _ = generated_data
    model = fit_mle(X, y)
    return model


def test_mle(generated_data):
    X, y, true_beta = generated_data
    model = fit_mle(X, y)
    beta_hat, V_hat = __get_mle_vhat(model)

    assert model
    assert len(beta_hat) == X.shape[1]


def test_parametricEB(mle_model):
    beta_hat, cov, tau_2 = fit_parametricEB(mle_model, max_iter=1000)
    mle_model_params = __get_mle_vhat(mle_model)[0]
    assert beta_hat is not None
    assert len(beta_hat) == len(mle_model_params)


@pytest.mark.parametrize("tau2", [0.5, 1.0, 10.0, 1000], ids=lambda t: f"tau2={t}")
def test_semi_bayes(mle_model, tau2):
    beta_star, _ = fit_semiBayes(mle_model, tau2)

    # assert shrinkage occurs, at least when tau2 is not TOO small
    assert np.linalg.norm(beta_star) <= np.linalg.norm(mle_model.params)
