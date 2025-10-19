from pathlib import Path
from src.simulation import load_simulation_output
from src.analysis import summarize_results

scenarios = list(Path("./results/raw/").glob("*.pkl"))

sim_results = [load_simulation_output(scenario) for scenario in scenarios]

summarize_results(sim_results[0])
