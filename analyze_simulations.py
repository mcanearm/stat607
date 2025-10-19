from pathlib import Path
from src.simulation import load_simulation_output

scenarios = list(Path("./results/raw/").glob("*.pkl"))

sim_results = [load_simulation_output(scenario) for scenario in scenarios]

sim_results[0]

sim_results[0]["beta_hat"][:, 0, :, :]
