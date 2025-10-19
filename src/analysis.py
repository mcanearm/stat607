import xarray as xr
import numpy as np


def get_coverage(sim_results: xr.Dataset) -> xr.DataArray:
    beta_hat = sim_results["beta_hat"].sel(
        var="estimate"
    )  # (simulation, method, param)

    se_hat = sim_results["beta_hat"].sel(var="std_error")  # same dims
    true_beta = sim_results["true_beta"]  # (simulation, param)

    # Expand true_beta along method dimension so shapes align
    true_beta_expanded = true_beta.expand_dims(
        method=sim_results.coords["method"]
    ).transpose("simulation", "method", "param")

    lower = beta_hat - 1.96 * se_hat
    upper = beta_hat + 1.96 * se_hat

    covered = (true_beta_expanded >= lower) & (true_beta_expanded <= upper)

    coverage_rate = covered.mean(dim="simulation")
    return coverage_rate


def rmse(sim_results: xr.Dataset) -> xr.DataArray | np.ndarray:
    beta_hat = (
        sim_results["beta_hat"].sel(var="estimate").drop("var")
    )  # (simulation, method, param)
    true_beta = sim_results["true_beta"]  # (simulation, param)

    # Expand true_beta along method dimension so shapes align
    true_beta_expanded = true_beta.expand_dims(
        method=sim_results.coords["method"]
    ).transpose("simulation", "method", "param")

    mse = ((beta_hat - true_beta_expanded) ** 2).mean(dim="simulation")
    rmse = np.sqrt(mse)
    return rmse


def mean_interval_length(
    sim_results: xr.Dataset, normalize_by_mle=False
) -> xr.DataArray:
    se_hat = (
        sim_results["beta_hat"].sel(var="std_error").drop("var")
    )  # (simulation, method, param)

    interval_length = 2 * 1.96 * se_hat
    mean_length = interval_length.mean(dim="simulation")
    if normalize_by_mle:
        # mle_length = mean_length.sel(method="mle")  # this should have worked, but didn't. Debug later.
        normalized_length = (
            mean_length / mean_length[0, :]
        )  # gross, but assume MLE is first - it SHOULD be.
        return normalized_length
    else:
        return mean_length


def summarize_results(sim_results: xr.Dataset) -> xr.DataArray:
    coverage = get_coverage(sim_results)
    rmse_values = rmse(sim_results)
    mil = mean_interval_length(sim_results)
    mil_normalized = mean_interval_length(sim_results, normalize_by_mle=True)

    summary_results = xr.DataArray(
        [coverage, rmse_values, mil, mil_normalized],
        dims=["metric", "method", "param"],
        coords={
            "metric": [
                "coverage",
                "rmse",
                "mean_interval_length",
                "normalized_mean_interval_length",
            ],
            "method": sim_results.coords["method"].values,
            "param": sim_results.coords["param"].values,
        },
    )
    return summary_results
