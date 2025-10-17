import logging
import xarray as xr
import numpy as np

from src.dgps import DatasetGenerator
from src.methods import fit_mle, fit_parametricEB, fit_semiBayes

logger = logging.getLogger(__name__)


class SimulationResult(object):
    pass


def run_simulation(
    data_generation_fn: DatasetGenerator,
    sbParams=None,
    ebParams=None,
    **generation_kwargs,
):
    # get passed in dict values or initialize empty dicts
    sbParams = sbParams or {}
    ebParams = ebParams or {}

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

    sim_estimates = xr.DataArray(
        beta_estimates,
        dims=["var", "method", "param"],
        coords={
            "var": ["estimate", "std_error"],
            "method": ["mle", "parametric_eb", "semi_bayes"],
            "param": [f"beta{i + 1}" for i in range(X.shape[1])],
        },
    ).transpose("method", "param", "var")

    sim_estimates.attrs = {
        "true_beta": beta,
        **generation_kwargs,
        **data_generation_fn.__dict__,
        "sbParams": sbParams,
        "ebParams": ebParams,
    }
    logging.debug(f"simulation complete -- {sim_estimates.attrs}")

    return sim_estimates
