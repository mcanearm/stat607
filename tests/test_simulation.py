from src.simulation import run_simulation


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
