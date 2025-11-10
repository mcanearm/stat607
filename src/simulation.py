import logging
import xarray as xr
import numpy as np
import inspect
from pathlib import Path

import pickle as pkl
from src.dgps import DatasetGenerator
from src.methods import fit_mle, fit_parametricEB, fit_semiBayes, __get_mle_vhat

import tqdm

logger = logging.getLogger(__name__)


__all__ = [
    "run_simulation",
    "save_simulation_output",
    "load_simulation_output",
    "construct_fp",
    "true_tau",
    "get_sim_args",
]


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


def _run_simulation(
    data_generation_fn: DatasetGenerator,
    sbParams: dict,
    ebParams: dict,
    mleParams: dict,
    **generation_kwargs,
):
    """
    Run a single simulation attempt.
    Parameters
    ----------
    data_generation_fn : DatasetGenerator
        Data generation function.
    sbParams : dict
        Parameters for Semi-Bayes fitting.
    ebParams : dict
        Parameters for Empirical Bayes fitting.
    mleParams : dict
        Parameters for MLE fitting.
    generation_kwargs : dict
        Additional keyword arguments for data generation function.
    Returns
    -------
    beta_estimates : np.ndarray
        Estimated beta coefficients and their standard errors from different methods.
    beta : np.ndarray
        True beta coefficients used in data generation.
    """
    # get passed in dict values or initialize empty dicts
    X, y, beta = data_generation_fn(**generation_kwargs)

    if np.linalg.matrix_rank(X) != data_generation_fn.n:
        raise np.linalg.LinAlgError("Design matrix X is rank deficient.")

    mle = fit_mle(X, y, **mleParams)
    parametric_eb = fit_parametricEB(mle, **ebParams)

    semi_bayes_results = [
        fit_semiBayes(mle, **{**sbParams, "tau2": ratio}) for ratio in [0.5, 1.0, 2.0]
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

    logger.debug(
        "sim complete — n=%d rho=%.3f tau0=%.3f tau1=%.3f sigma2=%.3f",
        data_generation_fn.n,
        data_generation_fn.rho,
        data_generation_fn.tau_0,
        data_generation_fn.tau_1,
        data_generation_fn.sigma2,
    )
    return beta_estimates, beta


def _simulate_once_worker(
    attempt_id: int,
    rng: np.random.Generator,
    mleParams,
    ebParams,
    sbParams,
    data_generation_fn: DatasetGenerator,
    generation_kwargs: dict,
):
    """
    Worker function to run a single simulation attempt; main purpose is to swallow errors and exceptions, though these
    are still viewable if LOG_LEVEL is set to DEBUG in the main logger in `run_simulation.py`.

    TODO: Figure out how to enable logging messages to not get swallowed by TQDM progress bar.
    attempt_id : int
        Identifier for the simulation attempt.
    rng : np.random.Generator
        Random number generator for reproducibility.
    mleParams : dict
        Parameters for MLE fitting.
    ebParams : dict
        Parameters for Empirical Bayes fitting.
    sbParams : dict
        Parameters for Semi-Bayes fitting.
    data_generation_fn : DatasetGenerator
        Data generation function.
    generation_kwargs : dict
        Additional keyword arguments for data generation function.
    """

    # if RNG is passed to the data generation function, we want to override
    # it to ensure reproducibility with the given RNG state.
    gen_kwargs = dict(generation_kwargs)
    gen_kwargs["rng"] = rng

    try:
        out = _run_simulation(
            data_generation_fn=data_generation_fn,
            mleParams=mleParams,
            ebParams=ebParams,
            sbParams=sbParams,
            **gen_kwargs,
        )
        return True, attempt_id, out
    except Exception as e:
        # swallow failures; return a flag only
        logger.debug("Simulation attempt %d failed: %s", attempt_id, str(e))
        return False, attempt_id, None


def run_simulation(
    N_sim,
    data_generation_fn: DatasetGenerator,
    sbParams=None,
    ebParams=None,
    mleParams=None,
    max_workers: int = 1,
    base_rng: np.random.Generator | None = None,
    **generation_kwargs,
):
    """
    Run a simulation scenario comparing MLE, Parametric EB, and Semi-Bayes methods.
    Set parallel=True to collect N_sim successes using a rolling process pool.
    """
    # defaults from fitting functions
    sbParams = get_sim_args(fit_semiBayes, sbParams or {})
    ebParams = get_sim_args(fit_parametricEB, ebParams or {})
    mleParams = get_sim_args(fit_mle, mleParams or {})

    pbar = tqdm.tqdm(total=N_sim, desc="Running Simulations", unit="sims")
    sim_results = []
    attempts = 0
    successes = 0
    base_rng = np.random.default_rng() if not base_rng else base_rng

    if max_workers <= 1:
        # ------------ original, sequential ------------
        while len(sim_results) < N_sim:
            attempts += 1
            child_rng = base_rng.spawn(1)[0]

            ok, attempt_id, payload = _simulate_once_worker(
                attempt_id=attempts,
                rng=child_rng,
                mleParams=mleParams,
                ebParams=ebParams,
                sbParams=sbParams,
                data_generation_fn=data_generation_fn,
                generation_kwargs=generation_kwargs,
            )
            if not ok:
                logger.debug(
                    "Iteration %d failed: %s -- success_rate=%.3f",
                    attempts,
                    len(sim_results) / attempts,
                )
                continue
            else:
                sim_results.append((attempt_id, payload))
            pbar.update(1)
            pbar.set_postfix_str(
                f"Success %: {len(sim_results) / attempts:0.3f}, n: {data_generation_fn.n}, N: {generation_kwargs.get('N', 'NA')}"
            )
        pbar.close()

    else:
        # prebuild the args tuple once (picklable)
        # Issue - the RNG here is shard across processes so need to somehow change the RNG state on each worker...
        from concurrent.futures import ProcessPoolExecutor, as_completed

        inflight = max_workers * 3
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            # start some processes immediately
            for _ in range(inflight):
                attempts += 1
                child_rng = base_rng.spawn(1)[0]
                f = pool.submit(
                    _simulate_once_worker,
                    attempts,
                    child_rng,
                    mleParams,
                    ebParams,
                    sbParams,
                    data_generation_fn,
                    generation_kwargs,
                )
                futures[f] = attempts

            while len(sim_results) < N_sim:
                done = next(as_completed(futures))
                ok, _, payload = done.result()
                attempt_id = futures.pop(done)
                if ok:
                    successes += 1
                    sim_results.append((attempt_id, payload))
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"Success %: {len(sim_results) / attempts:0.3f}"
                    )
                if successes < N_sim:
                    attempts += 1
                    child_rng = base_rng.spawn(1)[0]
                    f = pool.submit(
                        _simulate_once_worker,
                        attempts,
                        child_rng,
                        mleParams,
                        ebParams,
                        sbParams,
                        data_generation_fn,
                        generation_kwargs,
                    )
                    futures[f] = attempts

            # empty out the future attempts to ensure that from run to run, we
            # get the exact same results
            for f in futures:
                f.cancel()

        pbar.close()

    # ---------- pack results ----------
    sim_results.sort(key=lambda x: x[0])

    beta_hat = np.stack([res[1][0] for res in sim_results], axis=0)
    true_beta = np.stack([res[1][1] for res in sim_results], axis=0)

    simulation_output = xr.Dataset(
        data_vars={
            "beta_hat": (("simulation", "var", "estimator", "param"), beta_hat),
            "true_beta": (("simulation", "param"), true_beta),
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
            "param": [f"beta{i + 1}" for i in range(beta_hat.shape[-1])],
            "simulation": np.arange(N_sim),
        },
    )

    simulation_output.attrs = {
        "N": N_sim,
        **{
            **{f"sb_{k}": v for k, v in sbParams.items()},
            "prior_tau_ratio": [0.5, 1.0, 2.0],
        },
        **{f"eb_{k}": v for k, v in ebParams.items()},
        **generation_kwargs,
        **{k: v for k, v in data_generation_fn.__dict__.items() if k != "cov_mat"},
        "total_attempts": attempts,
        "successful_simulations": len(sim_results),
        "success_rate": len(sim_results) / attempts,
    }
    return simulation_output


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
    return output_path


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
