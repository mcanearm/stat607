import logging
import xarray as xr
import numpy as np
import inspect
from pathlib import Path

import pickle as pkl
from src.dgps import DatasetGenerator
from src.methods import fit_mle, fit_parametricEB, fit_semiBayes


from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning
import warnings


logger = logging.getLogger(__name__)


def get_sim_args(func, sim_args: dict, prepend=None):
    signature = inspect.signature(func)
    prepend = prepend + "_" if prepend else ""
    default_args = {
        f"{prepend + k}": v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect._empty
    }

    # override any default arguments with provided arguments
    sim_args_complete = {**default_args, **sim_args}
    return sim_args_complete


def run_simulation(
    N_sim,
    data_generation_fn: DatasetGenerator,
    sbParams=None,
    ebParams=None,
    mleParams=None,
    **generation_kwargs,
):
    # get default from fitting functions if this is an empty dict
    sbParams = get_sim_args(fit_semiBayes, sbParams or {})
    ebParams = get_sim_args(fit_parametricEB, ebParams or {})
    mleParams = get_sim_args(fit_mle, mleParams or {})

    def _run_simulation():
        # get passed in dict values or initialize empty dicts
        X, y, beta = data_generation_fn(**generation_kwargs)

        mle = fit_mle(X, y)
        parametric_eb = fit_parametricEB(mle, **ebParams)
        semi_bayes = fit_semiBayes(mle, **sbParams)

        mle_beta, mle_cov = mle.params, np.diagonal(mle.cov_params())
        pb_beta, pb_cov, _ = parametric_eb
        sb_beta, sb_cov = semi_bayes

        beta_estimates = np.stack(
            [
                np.stack([mle_beta, pb_beta, sb_beta]),
                np.stack(
                    [
                        np.sqrt(mle_cov),
                        np.sqrt(np.diagonal(pb_cov)),
                        np.sqrt(np.diagonal(sb_cov)),
                    ]
                ),
            ]
        )

        sim_estimates = xr.Dataset(
            {
                "beta_hat": (("var", "method", "param"), beta_estimates),
                "true_beta": (("param",), beta),
            },
            coords={
                "var": ["estimate", "std_error"],
                "method": ["mle", "parametric_eb", "semi_bayes"],
                "param": [f"beta{i + 1}" for i in range(X.shape[1])],
            },
        )

        logger.debug(f"simulation complete -- {data_generation_fn.__dict__}")
        return sim_estimates

    sim_results = []
    i = 0
    while len(sim_results) < N_sim:
        i += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", PerfectSeparationWarning)
                sim_output = _run_simulation()
        except Exception as e:
            logger.debug(
                f"Simulation iteration {i} failed: {e} -- success_rate = {(len(sim_results) / i):0.3f}"
            )
            continue
        else:
            sim_results.append(sim_output)
        logger.debug(
            f"Successes {len(sim_results)}/{i} = {(len(sim_results) / i):0.3f}"
        )
    log_dict = {k: v for k, v in data_generation_fn.__dict__.items() if k != "cov_mat"}
    logger.info(
        f"{N_sim} simulations completed, success rate = {(len(sim_results) / i):0.3f}, parameters: {log_dict}"
    )

    sim_output = xr.concat(sim_results, dim="simulation")
    sim_output.attrs = {
        "N": N_sim,
        **{f"sb_{k}": v for k, v in sbParams.items()},
        **{f"eb_{k}": v for k, v in ebParams.items()},
        **generation_kwargs,
        **data_generation_fn.__dict__,
    }
    return sim_output


def construct_fp(sim_results):
    filename = (
        f"sim_N={sim_results.attrs['N']}_n={sim_results.attrs['n']}"
        f"_rho={sim_results.attrs['rho']}_tau0={sim_results.attrs['tau_0']}"
        f"_tau1={sim_results.attrs['tau_1']}_sigma2={sim_results.attrs['sigma2']}.pkl"
    )
    return Path(filename)


def save_simulation_output(sim_data: xr.Dataset, output_dir: str | Path):
    filename = construct_fp(sim_data)
    output_path = Path(output_dir) / filename
    with open(output_path, "wb") as f:
        pkl.dump(sim_data, f)
    logging.info(f"Simulation results saved to {output_path}")


def load_simulation_output(file_path: str) -> xr.Dataset:
    with open(file_path, "rb") as f:
        sim_data = pkl.load(f)
    logging.info(f"Simulation results loaded from {file_path}")
    return sim_data
