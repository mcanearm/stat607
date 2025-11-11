from scripts.run_simulations import scenarios, run_scenario
import logging
import os
import pandas as pd
from src.utils import time_fn, get_git_hash
from pathlib import Path


logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    n_sims = [100, 500, 1000, 2000]

    timings = [
        (
            scenario[0],
            scenario[1],
            n_sim,
            time_fn(run_scenario)(scenario, N_sim=n_sim)[1],
        )
        for scenario in scenarios
        for n_sim in n_sims
    ]

    timing_df = pd.DataFrame(
        timings, columns=["n", "N", "n_sim", "time_seconds"]
    ).set_index(["n", "N", "n_sim"])

    git_hash = get_git_hash()

    (output_dir := Path("./results/timings/")).mkdir(parents=True, exist_ok=True)
    timing_df.to_csv(output_dir / f"simulation_timings_{git_hash}.csv")
