import pytest
from src.simulation import run_simulation, save_simulation_output
from src.dgps import DatasetGenerator
import numpy as np
import pickle as pkl


def test_simulation(generate_data):
    out = run_simulation(100, generate_data)

    assert not set(out["beta_hat"].dims).difference(
        {"simulation", "method", "param", "var"}
    )
    assert set(out.coords["method"].values) == {"mle", "parametric_eb", "semi_bayes"}


def test_simulation_output(generate_data):
    result = run_simulation(50, generate_data)
    assert result.attrs["N"] == 50
    assert not set(generate_data.__dict__.keys()).difference(result.attrs.keys())


@pytest.fixture()
def simulation_run():
    rng = np.random.default_rng(42)
    generate_data = DatasetGenerator(
        n=5, rho=0.0, tau_0=1.0, tau_1=1.0, sigma2=1.0, rng=rng
    )

    sim_results = run_simulation(
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

    assert np.all(
        simulation_run["beta_hat"].values == loaded_sim_run["beta_hat"].values
    )
    assert loaded_sim_run.rng.normal() == simulation_run.rng.normal()
