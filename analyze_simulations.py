from pathlib import Path
from src.simulation import load_simulation_output

scenarios = list(Path("./results/raw/").glob("*.pkl"))

sim_results = [load_simulation_output(scenario) for scenario in scenarios]


ds = sim_results[0]  # your dataset with coords: simulation, var, method, param


beta_hat = ds["beta_hat"].sel(var="estimate")  # (simulation, method, param)

se_hat = ds["beta_hat"].sel(var="std_error")  # same dims
true_beta = ds["true_beta"]  # (simulation, param)

# Expand true_beta along method dimension so shapes align
true_beta_expanded = true_beta.expand_dims(method=ds.coords["method"]).transpose(
    "simulation", "method", "param"
)

lower = beta_hat - 1.96 * se_hat
upper = beta_hat + 1.96 * se_hat

covered = (true_beta_expanded >= lower) & (true_beta_expanded <= upper)

coverage_rate = covered.mean(dim="simulation")
coverage_rate
