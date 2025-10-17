from src.simulation import run_simulation


def test_simulation(generate_data):
    out = run_simulation(generate_data)
    assert out.dims == ("method", "param", "var")
    assert set(out.coords["method"].values) == {"mle", "parametric_eb", "semi_bayes"}
    assert out.shape[0] == 3  # methods


def test_simulation_output(generate_data):
    result = run_simulation(generate_data, N=50)
    assert result.attrs["N"] == 50
    assert not set(generate_data.__dict__.keys()).difference(result.attrs.keys())
