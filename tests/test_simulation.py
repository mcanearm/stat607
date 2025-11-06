import pytest
from src.simulation import run_simulation, save_simulation_output
from src.dgps import DatasetGenerator
import numpy as np
import pickle as pkl


def test_simulation(prng_key, generate_data):
    out = run_simulation(prng_key, 100, generate_data)

    assert not set(out["beta_hat"].dims).difference(
        {"simulation", "estimator", "param", "var"}
    )
    assert set(out.coords["estimator"].values) == {
        "mle",
        "parametric_eb",
        "semi_bayes_0.5",
        "semi_bayes_1.0",
        "semi_bayes_2.0",
    }


def test_simulation_output(prng_key, generate_data):
    result = run_simulation(prng_key, 50, generate_data)
    assert result.attrs["N_sim"] == 50
    assert not set(generate_data.__dict__.keys()).difference(result.attrs.keys())


@pytest.fixture()
def simulation_run(prng_key):
    generate_data = DatasetGenerator(n=5, rho=0.0, tau_0=1.0, tau_1=1.0, sigma2=1.0)
    sim_results = run_simulation(
        prng_key,
        N_sim=10,
        data_generation_fn=generate_data,
        N=100,
    )
    return sim_results


def test_save_simulation(simulation_run, tmpdir):
    save_simulation_output(simulation_run, tmpdir)
    # Check that a file was created
    files = tmpdir.listdir()
    with open(files[0], "rb") as f:
        loaded_sim_run = pkl.load(f)

    assert np.allclose(
        loaded_sim_run["beta_hat"].values, simulation_run["beta_hat"].values
    ), "Saved and loaded simulation results do not match"

    rerun = run_simulation(
        loaded_sim_run.attrs["key"],
        N_sim=loaded_sim_run.attrs["N_sim"],
        data_generation_fn=loaded_sim_run.attrs["data_generation_fn"],
        N=loaded_sim_run.attrs["N"],
    )

    assert np.allclose(rerun["beta_hat"].values, loaded_sim_run["beta_hat"].values), (
        "Cannot reproduce saved simulation results"
    )


def test_coverage_comparison(prng_key):
    # All simulations in the paper had over 95% coverage, or maybe some variant
    # of 94%. If it's less than that, we're doing something wrong in the fit methods.
    generate_data = DatasetGenerator(n=4, rho=0.5, tau_0=0.2, tau_1=0.2, sigma2=0.5)

    sim_results = run_simulation(
        prng_key,
        N_sim=1000,
        data_generation_fn=generate_data,
        N=100,
        sbParams={"tau2": 1.0},
    )

    true_beta = sim_results["true_beta"]
    beta_hat = sim_results["beta_hat"][:, 0, :, :]
    se = sim_results["beta_hat"][:, 1, :, :]

    lower_ci = beta_hat - 1.96 * se
    upper_ci = beta_hat + 1.96 * se

    coverage = ((true_beta >= lower_ci) & (true_beta <= upper_ci)).mean(
        dim="simulation"
    )

    assert np.all(coverage >= 0.94)
