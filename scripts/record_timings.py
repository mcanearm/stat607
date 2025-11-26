"""
This script acts a wrapper around the "run_simulations" script that bridges between two versions; the v2 version
present at hash b44412815b86fc8d25cf192c7857a43b20c1e0c1 and the newer, updated version that exists on the dev
branch, which will be tagged later.

We use the script to compare top level timings for all changes. A feature by feature comparison is made
using the output of profiling the scripts/run_simulations.py script and scripts/naive_run_simulations.py scripts
directly.
"""

import logging
import os
import pandas as pd
from src.utils import time_fn, get_git_hash
from pathlib import Path
import argparse
import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "ERROR"))


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
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        help="Version of the simulation scenarios to run (v1 or v2).",
        dest="version",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20250607,
        help="Random seed for reproducibility",
        dest="seed",
    )
    args = parser.parse_args()

    seed = args.seed
    rng = np.random.default_rng(seed)
    n_sims = [100, 500, 1000, 2000, 4000, 8000]

    # this is complicated, but it's due to supporting two slightly different
    # interfaces in the scenarios between earlier and later versions of the
    # code. This could be refactored. For now though,
    # each scenario is n, N. We add n_sim and the other parameters that our
    # function is expecting.
    if args.version == "v1":
        from scripts.naive_run_simulations import (
            run_scenario,
            scenarios,
            SimulationScenario,
        )

        new_rngs = rng.spawn(len(scenarios) * len(n_sims))
        scenarios = [
            SimulationScenario(new_rngs[i], 1, scenario[0], scenario[1], nsim)
            for i, scenario in enumerate(scenarios)
            for nsim in n_sims
        ]
    elif args.version == "v2":
        from scripts.run_simulations import run_scenario, scenarios, SimulationScenario

        new_rngs = rng.spawn(len(scenarios) * len(n_sims))
        scenarios = [
            SimulationScenario(
                new_rngs[i],
                1,
                scenario[0],
                scenario[1],
                nsim,
                0,
            )  # add tqdm position bar for parallelism, but set to 0 for flattening.
            for i, scenario in enumerate(scenarios)
            for nsim in n_sims
        ]
    else:
        raise ValueError("Invalid version specified. Use 'v1' or 'v2'.")

    file_tag = args.tag or get_git_hash()

    timings = [
        (
            *(scenario.n, scenario.N, scenario.n_sim),
            time_fn(run_scenario)(scenario)[1],
        )
        for scenario in scenarios
    ]

    # time_fn(run_scenario)(scenario)[1],
    timing_df = pd.DataFrame(
        timings, columns=["n", "N", "n_sim", "time_seconds"]
    ).set_index(["n", "N", "n_sim"])

    (output_dir := Path("./results/timings/")).mkdir(parents=True, exist_ok=True)
    timing_df.to_csv(output_dir / f"simulation_timings_{file_tag}.csv")
