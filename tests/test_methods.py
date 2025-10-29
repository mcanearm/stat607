import numpy as np
import pytest

from src.methods import fit_mle, fit_parametricEB, fit_semiBayes, __get_mle_vhat
from src.dgps import DatasetGenerator


@pytest.fixture
def generated_data(generate_data):
    return generate_data(100)


@pytest.fixture
def mle_model(generated_data):
    X, y, _ = generated_data
    model = fit_mle(X, y)
    return model


def test_mle_estimation(generated_data):
    X, y, true_beta = generated_data
    mle_model = fit_mle(X, y)
    beta_hat, V_hat = __get_mle_vhat(mle_model)
    low = beta_hat - 1.96 * np.sqrt(np.diagonal(V_hat))
    high = beta_hat + 1.96 * np.sqrt(np.diagonal(V_hat))

    assert np.mean((true_beta >= low) & (true_beta <= high)) >= 0.8


def test_coverage(generated_data):
    X, y, true_beta = generated_data
    model = fit_mle(X, y)
    beta_hat, V_hat = __get_mle_vhat(model)

    se_beta = np.sqrt(np.diagonal(V_hat))
    beta_hat_low = beta_hat - 1.96 * se_beta
    beta_hat_high = beta_hat + 1.96 * se_beta

    assert model
    assert (
        np.mean((beta_hat_low < true_beta) & (true_beta < beta_hat_high)) > 0.8
    )  # at least 80% coverage


def test_parametricEB(mle_model):
    beta_hat, cov, tau_2 = fit_parametricEB(mle_model, max_iter=1000)
    mle_model_params = __get_mle_vhat(mle_model)[0]
    assert beta_hat is not None
    assert len(beta_hat) == len(mle_model_params)


@pytest.mark.parametrize("tau2", [0.5, 1.0, 10.0, 1000], ids=lambda t: f"tau2={t}")
def test_semi_bayes(mle_model, tau2):
    beta_star, _ = fit_semiBayes(mle_model, tau2)

    # assert shrinkage occurs, at least when tau2 is not TOO small
    assert np.linalg.norm(beta_star) <= np.linalg.norm(mle_model[0])


def test_numerical_stability_catch():
    generate_data = DatasetGenerator(n=10, rho=0.0, tau_0=1.0, tau_1=1.0, sigma2=1.0)
    with pytest.raises(RuntimeError):
        for _ in range(1000):
            N = 40
            X, y, true_beta = generate_data(N)
            mle_model = fit_mle(X, y)
            assert mle_model
