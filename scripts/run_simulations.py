import argparse
import logging
import os
import sys
import warnings
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning

from src.dgps import DatasetGenerator
from src.simulation import run_simulation, save_simulation_output, true_tau

# Constants used in the paper by Greenland
RHO = 0.5
TAU_0 = TAU_1 = 0.2
TRUE_TAU = true_tau(TAU_0, TAU_1)
SIGMA2 = 1.0
TARGET_SIM_SIZE = 8000  # target number of simulations per scenario
scenarios1 = product(
    [4, 10],  # n
    [40, 100, 500],  # N
)
scenarios2 = product([20], [100, 500, 2000])  # n, N, true_tau
scenarios = list(scenarios1) + list(scenarios2)


loglevel = os.environ.get("LOGLEVEL", "INFO")
logging.basicConfig(level=loglevel)
logger = logging.getLogger(__name__)


# CONSTANTS
OUTPUT_DIR = Path("./results/raw/")
FILTER_WARNINGS = True


def run_scenario(scenario):
    n, N = scenario
    rng = np.random.default_rng()
    data_gen = DatasetGenerator(
        n=n, tau_0=TAU_0, tau_1=TAU_1, sigma2=SIGMA2, rng=rng, rho=RHO
    )
    scenario_msg = f"n={n}, N={N}, true_tau={TRUE_TAU:0.3f}"

    logger.info(f"Running scenario: n={n}, N={N}, true_tau={TAU_0:0.3f}")

    with warnings.catch_warnings():
        if FILTER_WARNINGS:
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", PerfectSeparationWarning)
        results = run_simulation(
            N_sim=8000,
            data_generation_fn=data_gen,
            N=N,
            mleParams={"disp": False, "maxiter": 500},
        )
    save_simulation_output(results, OUTPUT_DIR)
    logger.info(
        f"Scenario complete: {scenario_msg}, success_rate={results.attrs['success_rate']:0.3f}"
    )

    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run simulations for various scenarios."
    )
    parser.add_argument("--num-cores", type=int, default=1, dest="cores")
    args = parser.parse_args()
    core_count = args.cores

    with Pool(processes=core_count) as p:
        list(p.imap_unordered(run_scenario, scenarios))
