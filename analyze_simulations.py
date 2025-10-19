from pathlib import Path
from src.simulation import load_simulation_output
from src.analysis import summarize_results
import xarray as xr
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

scenarios = list(Path("./results/raw/").glob("*.pkl"))

sim_results = [load_simulation_output(scenario) for scenario in scenarios]

metadata_columns = ["N", "sb_tau2", "n"]

metadata = pd.DataFrame(
    [],
    columns=metadata_columns,
)

sim_summary = [
    (
        summarize_results(sim_result),
        [float(sim_result.attrs[k]) for k in metadata_columns],
    )
    for sim_result in sim_results
]

single_result = sim_results[0]
sim_rmse = np.sqrt(
    (
        (single_result["beta_hat"].sel(var="estimate") - single_result["true_beta"])
        ** 2
    ).mean(dim="param")
)

plot_vals = np.log(sim_rmse)

boxplot_data = {
    str(est): plot_vals.sel(estimator=est).values
    for est in plot_vals.coords["estimator"].values
}
fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(boxplot_data.values())
ax.set_xticklabels(boxplot_data.keys())
plt.show()


mean_beta_coverage = [
    summary[0].sel(metric="coverage").mean(dim="param") for summary in sim_summary
]

beta_line_plot = xr.concat(mean_beta_coverage, dim="scenario")
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(beta_line_plot[0, :])
ax.plot(beta_line_plot[1, :])
ax.plot(beta_line_plot[2, :])
ax.legend()
plt.show()
