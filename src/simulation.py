import logging
import xarray as xr
import numpy as np
import inspect
from pathlib import Path

import pickle as pkl
from src.dgps import DatasetGenerator
from src.methods import fit_mle, fit_parametricEB, fit_semiBayes, __get_mle_vhat

from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning

logger = logging.getLogger(__name__)


def true_tau(tau0, tau1, p=0.2):
    """
    function to compute the true variance based on the DGP parameters
    """
    return np.sqrt(tau0**2 + p * (1 - p) * tau1**2)


def get_sim_args(func, sim_args: dict, prepend=None):
    """
    Get default arguments from a function signature and
    override with any provided arguments to a simulation.

    Parameters
    ----------
    func : function
        Function to inspect for default arguments.
    sim_args : dict
        Dictionary of arguments to override defaults.
    prepend : str, optional
        String to prepend to argument names when looking up defaults; prevents
        name collisions when multiple functions have arguments with the same name;
        and this is necessary since you'll probably be stuffing all of these
        arguments into a single dictionary.

    Returns
    -------
    sim_args_complete : dict
    """
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
    """
    Run a simulation scenario comparing MLE, Parametric EB, and Semi-Bayes methods.
    Parameters
    ----------
    N_sim : int
        Number of simulation replications.
    data_generation_fn : function
        Data generation function that returns X, y, beta_true. Since each simulation needs its own
        dataset, this should be a callable that generates a new dataset each time it is called.
    sbParams : dict, optional
        Parameters to pass to fit_semiBayes.
    ebParams : dict, optional
        Parameters to pass to fit_parametricEB.
    mleParams : dict, optional
        Parameters to pass to fit_mle.
    **generation_kwargs : dict
        Additional keyword arguments to pass to data_generation_fn.
    Returns
    -------
    sim_output : xarray.Dataset
        Dataset containing simulation results.
    """
    # get default from fitting functions if this is an empty dict
    sbParams = get_sim_args(fit_semiBayes, sbParams or {})
    ebParams = get_sim_args(fit_parametricEB, ebParams or {})
    mleParams = get_sim_args(fit_mle, mleParams or {})

    def _run_simulation():
        # get passed in dict values or initialize empty dicts
        X, y, beta = data_generation_fn(**generation_kwargs)

        if np.linalg.matrix_rank(X) != data_generation_fn.n:
            raise np.linalg.LinAlgError("Design matrix X is rank deficient.")

        mle = fit_mle(X, y)
        parametric_eb = fit_parametricEB(mle, **ebParams)

        semi_bayes_results = [
            fit_semiBayes(mle, **{**sbParams, "tau2": ratio})
            for ratio in [0.5, 1.0, 2.0]
        ]

        mle_beta, mle_cov = __get_mle_vhat(mle)
        mle_cov = np.diagonal(mle_cov)

        # check for bad values or estimates as evidence of poor convergence
        if np.any(np.abs((mle_beta - beta)) > 1e3):
            raise RuntimeError("MLE estimates are unreasonably large.")

        pb_beta, pb_cov, _ = parametric_eb
        sb_beta_0, sb_cov_0 = semi_bayes_results[0]
        sb_beta_1, sb_cov_1 = semi_bayes_results[1]
        sb_beta_2, sb_cov_2 = semi_bayes_results[2]

        beta_estimates = np.stack(
            [
                np.stack([mle_beta, pb_beta, sb_beta_0, sb_beta_1, sb_beta_2]),
                np.stack(
                    [
                        np.sqrt(mle_cov),
                        np.sqrt(np.diagonal(pb_cov)),
                        np.sqrt(np.diagonal(sb_cov_0)),
                        np.sqrt(np.diagonal(sb_cov_1)),
                        np.sqrt(np.diagonal(sb_cov_2)),
                    ]
                ),
            ]
        )

        sim_estimates = xr.Dataset(
            {
                "beta_hat": (("var", "estimator", "param"), beta_estimates),
                "true_beta": (("param",), beta),
            },
            coords={
                "var": ["estimate", "std_error"],
                "estimator": [
                    "mle",
                    "parametric_eb",
                    "semi_bayes_0.5",
                    "semi_bayes_1.0",
                    "semi_bayes_2.0",
                ],
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
            sim_output = _run_simulation()
        except (
            Exception,
            ConvergenceWarning,
            PerfectSeparationWarning,
            RuntimeWarning,
        ) as e:
            # catch all reasonable MLE exceptions and discard the simulation run on those
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
        f"{N_sim} simulations completed, success rate = {(len(sim_results) / i):0.3f}, data_parameters: {log_dict}"
    )

    sim_output = xr.concat(
        sim_results,
        dim="simulation",
    )
    sim_output = sim_output.assign_coords(simulation=np.arange(N_sim))

    sim_output.attrs = {
        "N": N_sim,
        **{
            **{f"sb_{k}": v for k, v in sbParams.items()},
            "prior_tau_ratio": [0.5, 1.0, 2.0],
        },
        **{f"eb_{k}": v for k, v in ebParams.items()},
        **generation_kwargs,
        **data_generation_fn.__dict__,
        "total_attempts": i,
        "successful_simulations": len(sim_results),
        "success_rate": len(sim_results) / i,
    }
    return sim_output


def construct_fp(sim_results: xr.Dataset) -> Path:
    """
    Construct a filename for saving simulation results based on attributes.
    Parameters
    ----------
    sim_results : xarray.Dataset
        Simulation results dataset with attributes; the result of the run_simulation function.
    Returns
    -------
    filename : Path
        Constructed filename.
    """
    filename = (
        f"sim_N={sim_results.attrs['N']}_n={sim_results.attrs['n']}"
        f"_rho={sim_results.attrs['rho']}_tau0={sim_results.attrs['tau_0']}"
        f"_tau1={sim_results.attrs['tau_1']}_tau2={sim_results.attrs['sb_tau2']:0.3f}"
        f"_sigma2={sim_results.attrs['sigma2']}.pkl"
    )
    return Path(filename)


def save_simulation_output(sim_data: xr.Dataset, output_dir: str | Path):
    """
    Save simulation results to a specified directory.
    Parameters
    ----------
    sim_data : xarray.Dataset
        Simulation results dataset.
    output_dir : str or Path
        Directory to save the results.
    Returns
    -------
    None
    """
    filename = construct_fp(sim_data)
    output_path = Path(output_dir) / filename
    with open(output_path, "wb") as f:
        pkl.dump(sim_data, f)
    logger.info(f"Simulation results saved to {output_path}")


def load_simulation_output(file_path: str | Path) -> xr.Dataset:
    """
    Load simulation results from a specified file.
    Parameters
    ----------
    file_path : Path|str
        Path to the file containing simulation results.
    Returns
    -------
    sim_data : xarray.Dataset
        Loaded simulation results dataset.
    """
    with open(file_path, "rb") as f:
        sim_data = pkl.load(f)
    logger.info(f"Simulation results loaded from {file_path}")
    return sim_data
