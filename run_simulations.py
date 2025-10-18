from src.simulation import run_simulation, save_simulation_output
from pathlib import Path
import numpy as np
from src.dgps import DatasetGenerator
from itertools import product


if __name__ == "__main__":
    rng = np.random.default_rng(8190)

    scenarios = product(
        [5, 10, 50],  # n
        [0.0, 0.2, 0.5, 0.8],  # rho
        [0.5, 1.0, 2.0],  # tau_0
        [0.5, 1.0, 2.0],  # tau_1
        [0.5, 1.0, 2.0],  # sigma2
    )

    for scenario in scenarios:
        n, rho, tau_0, tau_1, sigma2 = scenario
        data_gen = DatasetGenerator(
            n=n, rho=rho, tau_0=tau_0, tau_1=tau_1, sigma2=sigma2, rng=rng
        )
        data_gen.n = n
        data_gen.rho = rho
        data_gen.tau_0 = tau_0
        data_gen.tau_1 = tau_1
        data_gen.sigma2 = sigma2

        results = run_simulation(
            N_sim=500,
            data_generation_fn=data_gen,
            N=100,
        )
        output_dir = Path("./results/")
        save_simulation_output(results, output_dir)
