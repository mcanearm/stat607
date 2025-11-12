from scripts.run_simulations import scenarios, run_scenario
import logging
import os
import pandas as pd
from src.utils import time_fn, get_git_hash
from pathlib import Path
import argparse
import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/timings/",
        help="Directory to save timing results",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=1,
        help="Number of CPU cores to use in each scenario (within job parallelism)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag for the output filename. Defaults to the Git hash of the current code version.",
    )
    args = parser.parse_args()

    n_sims = [100, 500, 1000, 2000, 4000, 8000]
    file_tag = args.tag or get_git_hash()

    seed = 20250607
    rng = np.random.default_rng(seed)
    new_rngs = rng.spawn(len(scenarios) * len(n_sims))

    scenarios = [
        (
            scenario[0],
            scenario[1],
            n_sim,
        )
        for scenario in scenarios
        for n_sim in n_sims
    ]

    timings = [
        (*scenario, time_fn(run_scenario)((new_rngs[i], args.num_cores, *scenario))[1])
        for i, scenario in enumerate(scenarios)
    ]

    # time_fn(run_scenario)(scenario)[1],
    timing_df = pd.DataFrame(
        timings, columns=["n", "N", "n_sim", "time_seconds"]
    ).set_index(["n", "N", "n_sim"])

    (output_dir := Path("./results/timings/")).mkdir(parents=True, exist_ok=True)
    timing_df.to_csv(output_dir / f"simulation_timings_{file_tag}.csv")
