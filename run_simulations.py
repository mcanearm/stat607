import os
import logging
from src.simulation import run_simulation, save_simulation_output, true_t
from pathlib import Path
import numpy as np
from src.dgps import DatasetGenerator
from itertools import product
from multiprocessing import Pool
import sys


from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning
import warnings


loglevel = os.environ.get("LOGLEVEL", "ERROR")
logging.basicConfig(level=loglevel)
logger = logging.getLogger(__name__)


# CONSTANTS
OUTPUT_DIR = Path("./results/raw/")
FILTER_WARNINGS = True


def run_scenario(scenario):
    n, N, true_tau, prior_tau = scenario
    rng = np.random.default_rng()
    data_gen = DatasetGenerator(
        n=n, tau_0=true_tau, tau_1=true_tau, sigma2=0.5, rng=rng, rho=0.5
    )

    logger.info(
        f"Running scenario: n={n}, N={N}, true_tau={true_tau:0.3f}, prior_tau={prior_tau:0.3f}"
    )

    with warnings.catch_warnings():
        if FILTER_WARNINGS:
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
    save_simulation_output(results, OUTPUT_DIR)
    logger.info(
        f"Scenario complete: n={n}, N={N}, true_tau={true_tau:0.3f}, prior_tau={prior_tau:0.3f}, success_rate={results.attrs['success_rate']:0.3f}"
    )

    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # a bit silly, but in the simulations scenarios true tau is always
    # set to 0.2 for tau0 and tau1, so we compute true_t and pass in the
    # prior taus as multiples of that value
    true_tau = true_t(0.2, 0.2)
    prior_taus = [i * true_tau for i in [0.5, 1, 2]]

    scenarios1 = product(
        [4, 10],  # n
        [40, 100],  # N
        [0.2],  # true_tau
        prior_taus,  # prior_tau
    )
    scenarios2 = product([20], [100, 500, 2000], [0.2], prior_taus)
    scenarios = list(scenarios1) + list(scenarios2)

    with Pool(processes=1) as p:  # for now, more processes seems slower, so skip
        p.map(run_scenario, scenarios)
