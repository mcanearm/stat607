import argparse
import os
import logging
from src.simulation import run_simulation, save_simulation_output
from pathlib import Path
import numpy as np
from src.dgps import DatasetGenerator
from itertools import product


from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning
import warnings


loglevel = os.environ.get("LOGLEVEL", "ERROR")
logging.basicConfig(level=loglevel)
logger = logging.getLogger(__name__)

output_dir = Path("./results/raw/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="./results/raw/",
        help="Directory to save simulation results",
    )
    parser.add_argument(
        "--filter-warnings",
        action="store_true",
        help="Filter warnings during simulation runs",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios1 = product(
        [4, 10],  # n
        [40, 100],  # N
        [1],  # true_tau
        [0.5, 1, 2],  # prior_tau
    )
    scenarios2 = product([20], [100, 500, 2000], [1], [0.5, 1, 2])
    scenarios = list(scenarios1) + list(scenarios2)

    for scenario in scenarios:
        n, N, true_tau, prior_tau = scenario
        rng = np.random.default_rng()
        data_gen = DatasetGenerator(
            n=n,
            tau_0=true_tau,
            tau_1=true_tau,
            rng=rng,
        )

        logger.info(
            "Running scenario: n={}, N={}, true_tau={}, prior_tau={}".format(
                n, N, true_tau, prior_tau
            )
        )

        with warnings.catch_warnings():
            if parser.parse_args().filter_warnings:
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", PerfectSeparationWarning)
            results = run_simulation(
                N_sim=8000,
                data_generation_fn=data_gen,
                N=N,
                sbParams={"tau2": prior_tau},
                mleParams={"disp": False, "maxiter": 500},
            )
        save_simulation_output(results, output_dir)
        logger.info(
            "Scenario complete: n={}, N={}, true_tau={}, prior_tau={}".format(
                n, N, true_tau, prior_tau
            )
        )
