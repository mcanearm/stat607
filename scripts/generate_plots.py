from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from cycler import cycler
import pickle as pkl
from src.analysis import concat_results


# DEFAULT PLOTTING STYLE; whatever ChatGPT suggested for starters.
PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE"]  # Okabe–Ito (5)
plt.rcParams.update(
    {
        "axes.prop_cycle": cycler(color=PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "--",
        "lines.markersize": 5,
        "lines.linewidth": 1.8,
    }
)
SHOW = False
(FIGDIR := Path("./results/figures/")).mkdir(
    parents=True, exist_ok=True
)  # ensure output dir exists


# helper functions for later plotting
def pickle_load(fp: Path) -> xr.Dataset:
    with open(fp, "rb") as f:
        data = pkl.load(f)
    return data


def generate_plot_labels(plot_label):
    lookup = {
        "mle": "MLE",
        "parametric_eb": "PB",
        "semi_bayes_0.5": "SB$(0.5\\tau)$",
        "semi_bayes_1.0": "SB$(\\tau)$",
        "semi_bayes_2.0": "SB$(2\\tau)$",
    }
    return lookup.get(plot_label, plot_label)


def color_for(est_name: str) -> str:
    s = str(est_name).lower()
    MLE_COLOR = "#4477AA"
    PB_COLOR = "#EE6677"
    SB_COLOR = "#228833"
    if s.startswith("semi_bayes"):  # SB_0.5, SB_1.0, SB_2.0, etc.
        return SB_COLOR
    if s in ("parametric_eb", "parametric_bayes", "pb", "eb"):
        return PB_COLOR
    if s == "mle":
        return MLE_COLOR
    return "#666666"


sim_summaries = pd.read_csv("./results/processed/simulation_summaries.csv")
pairs = (
    sim_summaries[["N", "n"]].drop_duplicates().sort_values(["N", "n"]).values.tolist()
)

### PLOT 1: Mean RMSE Length by Estimator ###
"""
This is a diagnostic plot that I'm using to show how off the RMSE is. Hint; it's
pretty bad for small N, and improves as N increases.
"""
pairs = (
    sim_summaries[["N", "n"]].drop_duplicates().sort_values(["N", "n"]).values.tolist()
)
plot_data = sim_summaries.pivot_table(
    index=["N", "n", "estimator", "metric"], columns="summary_type", values="value"
).reset_index()

fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(12, 8), sharey=False, sharex=True)
ax = ax.flatten()
for j, (rmse_ns, ax_i) in enumerate(zip(pairs, ax)):
    sub_df = plot_data[
        (plot_data["N"] == rmse_ns[0])
        & (plot_data["n"] == rmse_ns[1])
        & (plot_data["metric"] == "rmse")
    ]  # pick one N for illustration
    x_labels = [generate_plot_labels(s) for s in sub_df["estimator"].values]
    ax_i.errorbar(
        sub_df["estimator"],
        sub_df["mean"],
        sub_df["std"],
        marker="o",
        label=f"n={rmse_ns[1]}",
    )
    ax_i.set_title(f"N={rmse_ns[0]}, n={rmse_ns[1]}")
    if j >= 6:
        # ax_i.set_xlabel("Estimator")
        ax_i.set_xticks(sub_df["estimator"])
        ax_i.set_xticklabels(x_labels)
    # ax_i.legend()
fig.supxlabel("Estimator")
fig.supylabel("RMSE")
fig.suptitle("RMSE by Estimator and Sample Size")
fig.tight_layout()
fig.savefig(FIGDIR / "rmse_by_estimator_and_sample_size.pdf", dpi=300)

if SHOW:
    plt.show()


### PLOT 2: Coverage Lineplots ###
# This is mostly me, with a little ChatGPT for improved visuals to make it look
# "publication ready".
little_n_pairs = {}
for pair in pairs:
    if pair[1] not in little_n_pairs:
        little_n_pairs[pair[1]] = []
    little_n_pairs[pair[1]].append(pair[0])

fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(12, 8), sharey=True, sharex=True)
ax = ax.flatten()

for j, ((little_n, N_list), ax_i) in enumerate(zip(little_n_pairs.items(), ax)):
    sub_df = plot_data[
        (plot_data["n"] == little_n) & (plot_data["metric"] == "coverage")
    ].copy()
    if sub_df.empty:
        ax_i.axis("off")
        continue

    # sort by N and keep numeric x for proper spacing
    sub_df = sub_df.sort_values("N")
    xticks = sorted(sub_df["N"].unique())
    ax_i.set_xticks(xticks)

    # plot each method series
    for method, group_df in sub_df.groupby("estimator"):
        g = group_df.sort_values("N")
        ax_i.plot(
            g["N"],  # numeric x
            g["mean"],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=generate_plot_labels(method),
        )

    # 95% reference line across the actual x-range
    if len(xticks) > 0:
        ax_i.axhline(0.95, color="gray", linestyle=(0, (4, 3)), linewidth=1.2, zorder=0)

    ax_i.set_title(f"n = {little_n}", fontsize=12)
    ax_i.grid(True, axis="y")
    ax_i.yaxis.set_major_formatter(PercentFormatter(5.0))
    ax_i.set_ylim(0.90, 1.02)  # tweak if your data needs more room

    if j == 0:
        handles, labels = ax_i.get_legend_handles_labels()
        if labels:
            ax_i.legend(handles, labels, loc="upper right", frameon=False)

# shared labels
ax[-1].set_xlabel("Sample size N", fontsize=12)
# for ax_i in ax:
fig.supylabel("Coverage Probability", fontsize=12)
fig.tight_layout()
fig.savefig(FIGDIR / "coverage_by_n_and_N.pdf", dpi=300)
if SHOW:
    plt.show()


### Plot 3: Violinplots of RMSE ###
raw_files = Path("./results/raw/").glob("*.pkl")
sim_results = [pickle_load(fp) for fp in raw_files]

all_simulations = concat_results(sim_results)

full_beta_hat = all_simulations["beta_hat"].sel(var="estimate").drop_vars("var")
true_beta = all_simulations["true_beta"]  # (simulation, param)

raw_mse = (
    ((full_beta_hat - true_beta) ** 2)
    .sum(dim="param")
    .to_dataframe(name="rmse")
    .reset_index()
)

# map each scenario -> its (N, n) from the list of datasets
metadata = pd.DataFrame(
    [
        {"scenario": i, "N": ds.attrs["N"], "n": ds.attrs["n"]}
        for i, ds in enumerate(sim_results)
    ]
)
raw_mse = raw_mse.merge(metadata, on="scenario", how="left")

# fixed estimator order for consistent positions/colors
est_order = [
    "mle",
    "parametric_eb",
    "semi_bayes_0.5",
    "semi_bayes_1.0",
    "semi_bayes_2.0",
]
x_labels = [generate_plot_labels(s) for s in est_order]

# ----- plot -----
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(12, 8), sharey=False, sharex=True)
ax = ax.flatten()

for j, (ns, ax_i) in enumerate(zip(pairs, ax)):  # pairs: list of (N, n) tuples
    N_val, n_val = ns
    sub_df = raw_mse[(raw_mse["N"] == N_val) & (raw_mse["n"] == n_val)].copy()
    if sub_df.empty:
        ax_i.axis("off")
        continue

    # trim extreme outliers PER ESTIMATOR within this panel (99th percentile)
    if "rmse" in sub_df:
        q99 = sub_df.groupby("estimator")["rmse"].transform(lambda s: s.quantile(0.99))
        sub_df.loc[sub_df["rmse"] > q99, "rmse"] = np.nan

    # build data in the fixed estimator order
    box_data = [
        sub_df.loc[sub_df["estimator"] == est, "rmse"].dropna().values
        for est in est_order
    ]
    # some estimators might be missing; keep empty arrays so positions align
    positions = np.arange(1, len(est_order) + 1)

    parts = ax_i.violinplot(
        dataset=box_data,
        positions=positions,
        widths=0.85,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # color each violin (match order)
    for body, color in zip(parts["bodies"], [color_for(est) for est in est_order]):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.8)
        body.set_linewidth(0.5)

    # add median points and IQR bars (boxplot feel)
    med = [np.median(d) if d.size else np.nan for d in box_data]
    q1 = [np.percentile(d, 25) if d.size else np.nan for d in box_data]
    q3 = [np.percentile(d, 75) if d.size else np.nan for d in box_data]

    # mask NaNs to avoid warnings
    ok = ~np.isnan(med)
    ax_i.scatter(positions[ok], np.array(med)[ok], color="black", s=18, zorder=3)
    ax_i.vlines(positions[ok], np.array(q1)[ok], np.array(q3)[ok], colors="black", lw=2)

    ax_i.set_xticks(positions)
    ax_i.set_xticklabels(x_labels, rotation=0)
    ax_i.set_title(f"N={N_val}, n={n_val}")
    ax_i.grid(True, axis="y", zorder=0)

# shared labels/titles
fig.supxlabel("Estimator")
fig.supylabel("RMSE (total)")
fig.suptitle("RMSE by Estimator and Sample Size")
fig.tight_layout()
fig.savefig("./results/figures/rmse_violinplots.pdf", dpi=300)

if SHOW:
    plt.show()
