import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import summarize_results

# CONSTANTS
INPUT_DIR = Path("./results/raw/")  # should already exist...
(OUTPUT_DIR := Path("./results/processed/")).mkdir(parents=True, exist_ok=True)
TRIM_Q = 0.995  # set to None to disable trimming


def get_summary_df(result_summary):
    # Get summary statistics across all parameters; we're treating each beta_j
    # equally here.

    mean_params = (
        result_summary.mean(dim="param").to_dataframe(name="value").reset_index()
    )
    quantile_params = (
        result_summary.quantile([0.025, 0.5, 0.975], dim="param")
        .to_dataframe(name="value")
        .reset_index()
    )
    sd_params = result_summary.std(dim="param").to_dataframe(name="value").reset_index()

    metadata = {k: result_summary.attrs[k] for k in ["N", "n", "success_rate"]}

    full_summary = (
        pd.concat(
            {
                "mean": mean_params,
                "quantile": quantile_params,
                "std": sd_params,
            },
            axis=0,
            keys=["mean", "quantile", "std"],
            names=["summary_type"],
        )
        .reset_index(names=["summary_type"], level=0)
        .assign(**metadata)
    )

    full_summary["summary_type"] = np.where(
        (full_summary["quantile"].notna()),
        full_summary["summary_type"] + "_" + full_summary["quantile"].astype(str),
        full_summary["summary_type"],
    )
    full_summary = full_summary.drop(columns=["quantile"])

    return full_summary


if __name__ == "__main__":
    raw_simulation_outputs = INPUT_DIR.glob("*.pkl")
    res_files = []
    for fp in raw_simulation_outputs:
        with open(fp, "rb") as f:
            sim_data = pkl.load(f)
        res_files.append(sim_data)

    result_summaries = [summarize_results(dat) for dat in res_files]
    summary_df = pd.concat([get_summary_df(rs) for rs in result_summaries])

    outputFile = OUTPUT_DIR / "simulation_summaries.csv"
    summary_df.to_csv(outputFile, index=False)
