import pytest
import numpy as np
from src.simulation import run_simulation
from src.analysis import (
    get_coverage,
    rmse,
    mean_interval_length,
    summarize_results,
    concat_results,
)


@pytest.fixture
def sample_output(generate_data):
    results = run_simulation(
        N_sim=100,
        data_generation_fn=generate_data,
        N=100,
        sbParams={"tau2": 1.0},
        base_rng=np.random.default_rng(12345),
    )
    return results


def test_get_coverage(sample_output):
    coverage = get_coverage(sample_output)
    assert np.all((coverage >= 0) & (coverage <= 1)), (
        "Coverage rates must be between 0 and 1"
    )


def test_rmse(sample_output):
    rmse_values = rmse(sample_output)
    assert rmse_values.sizes["estimator"] == 5, "RMSE shape mismatch"
    assert np.all(rmse_values >= 0), "RMSE values must be non-negative"


def test_mean_interval_length(sample_output):
    mil = mean_interval_length(sample_output)
    assert mil.sizes["estimator"] == 5, "Mean interval length shape mismatch"
    assert np.all(mil > 0), "Mean interval lengths must be positive"

    mil_normalized = mean_interval_length(sample_output, normalize_by_mle=True)
    assert mil_normalized.sizes["estimator"] == 5, (
        "Normalized mean interval length shape mismatch"
    )
    assert np.all(mil_normalized >= 0), (
        "Normalized mean interval lengths must be non-negative"
    )
    assert np.all(mil_normalized.sel(estimator="mle") == 1.0), (
        "MLE normalized lengths must be 1.0"
    )


def test_summary_results(sample_output):
    summary = summarize_results(sample_output)
    expected_methods = [
        "mle",
        "parametric_eb",
        "semi_bayes_0.5",
        "semi_bayes_1.0",
        "semi_bayes_2.0",
    ]
    expected_metrics = [
        "coverage",
        "rmse",
        "mean_interval_length",
        "normalized_mean_interval_length",
    ]

    assert set(summary.coords["estimator"].values) == set(expected_methods), (
        "Methods in summary results mismatch"
    )
    assert set(summary.coords["metric"].values) == set(expected_metrics), (
        "Metrics in summary results mismatch"
    )
    assert summary.shape == (
        len(expected_metrics),
        len(expected_methods),
        sample_output.sizes["param"],
    ), "Summary results shape mismatch"


def test_concat_results(sample_output):
    # Create a second sample output with different parameters
    sample_out1 = sample_output.copy()
    sample_out2 = sample_output.copy()

    concatenated = concat_results([sample_out1, sample_out2])
    assert len(concatenated.attrs["n"]) == 2, (
        "Concatenated results should store each set of attr values as a list"
    )
    assert "scenario" in concatenated.dims, (
        "Concatenated results should have a new 'scenario' dimension"
    )
