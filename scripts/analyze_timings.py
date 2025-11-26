import os
import pstats
import re
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
import logging

# this assumes you're running at the package root.
FIGURES_DIR = Path("./results/figures")


logging.basicConfig(level=os.environ.get("LOGLEVEL", "ERROR"))


# ChatGPT generated code for combining dataframes together from the stats/cProf
# files. I've edited some parts to simplify.
# ---------- 1) Loader: .prof/.pstats -> tidy DataFrame ----------
def profile_to_df(src, strip_dirs=True):
    """
    src: path to .prof file OR an existing pstats.Stats object.
    Returns a tidy DataFrame with a stable 'func_key'.
    """
    st = pstats.Stats(src) if isinstance(src, (str, os.PathLike)) else src
    if strip_dirs:
        st.strip_dirs()

    def func_key(func_tuple, include_line=True):
        filename, lineno, funcname = func_tuple
        # normalize filename -> module-ish name
        if filename.startswith("{built-in"):
            base = "builtins"
        else:
            base = os.path.basename(filename) if strip_dirs else filename
            base = re.sub(r"\.pyc?$", "", base)
        return f"{base}:{funcname}:{lineno}" if include_line else f"{base}:{funcname}"

    rows = []
    for func, (cc, nc, tt, ct, callers) in st.stats.items():
        filename, lineno, funcname = func
        rows.append(
            {
                "file": os.path.basename(filename) if strip_dirs else filename,
                "line": lineno,
                "func": funcname,
                "func_key": func_key(
                    func, include_line=False
                ),  # stable join key (file:func:line)
                "ncalls": nc,  # total calls (incl. recursion)
                "ccalls": cc,  # primitive calls
                "tottime": tt,  # time in function body (excl. subcalls)
                "cumtime": ct,  # incl. subcalls
            }
        )

    # this part is NOT from GPT, and is a simplification of the generated code.
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # convenience cols
    df = df.groupby("func_key").sum()
    df["tottime_pct"] = df["tottime"] / df["tottime"].sum()
    df["cumtime_pct"] = df["cumtime"] / df["cumtime"].sum()
    return df.sort_values("tottime", ascending=False).reset_index(drop=False)


# Now generate the plot
naive = pd.read_csv("./results/timings/simulation_timings_naive.csv")
optimized = pd.read_csv("./results/timings/simulation_timings_optimized.csv")

output = (
    pd.concat([naive, optimized], keys=["naive", "optimized"])
    .reset_index(level=0)
    .rename(columns={"level_0": "implementation"})
)


# Next, generate a plot that shows the time complexity of both implementations.

# technically, you could change this around, but it's hard because there's 4 unique
# values of N but only 3 of n, so the 2x2 grid doesn't work as well in another format.
row_var = "N"  # facets
col_var = "n"  # linestyle
impl_var = "implementation"  # color+marker
x_max_pad = None  # e.g., 4000 to extend to the right; or None

row_vals = sorted(output[row_var].unique())  # N
col_vals = sorted(output[col_var].unique())  # n
impl_levels = sorted(
    output[impl_var].unique(), key=lambda s: str(s).lower()
)  # naive, optimized
col_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
for i, ax in enumerate(axes.flatten()):
    row_val = row_vals[i]
    sub_df = output[(output[row_var] == row_val)]
    for col_val in col_vals:
        for impl in impl_levels:
            plot_df = output[
                (output[row_var] == row_val)
                & (output[col_var] == col_val)
                & (output[impl_var] == impl)
            ]
            ax.plot(
                plot_df["n_sim"],
                plot_df["time_seconds"],
                linestyle="--" if impl == "naive" else "-",
                marker="o",
                color=col_colors[col_vals.index(col_val)],
                label=f"{col_var}={col_val}, {impl}",
                markersize=5,
            )

        ax.set_xscale("log", base=10)
        ax.set_yscale("log", base=10)

        if i in [0, 2]:
            ax.set_ylabel("Time (seconds)")

        if i in [1, 3]:
            ax.get_yaxis().set_ticks([])

        if i in [2, 3]:
            ax.set_xticks([100, 500, 1000, 2000, 4000, 8000])
            ax.set_xticklabels([100, 500, 1000, 2000, 4000, 8000])
            ax.set_xlabel("Number of Simulations")
        else:
            ax.get_xaxis().set_ticklabels([])
        ax.set_title(f"{row_var}={row_val}")

    if x_max_pad is not None:
        ax.set_xlim(right=x_max_pad)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
fig.suptitle("Simulation Timings by Implementation")
fig.tight_layout()

fig.savefig(FIGURES_DIR / "log-log-timings.pdf", bbox_inches="tight")

## Generate a table of the timings for each of the profiling steps; just
## look at the major functions that have been touched.
df1, df2 = (
    profile_to_df("./results/timings/profile_naive.prof"),
    profile_to_df("./results/timings/profile_optimized.prof"),
)

key_funcs = [
    "generate_design_matrix",
    "fit_mle",
    "fit_semiBayes",
    "fit_parametricEB",
    "recurser",
    "_path_join",
    "find_specified_spec",
    "inv",
    "solve",
]
comparison = pd.merge(
    df1[df1["func"].isin(key_funcs)][["func", "tottime", "cumtime"]],  # type: ignore
    df2[df2["func"].isin(key_funcs)][["func", "tottime", "cumtime"]],  # type: ignore
    on="func",
    suffixes=("_naive", "_optimized"),
    how="outer",
)

comparison["tottime_diff"] = (
    comparison["tottime_optimized"] - comparison["tottime_naive"]
)
print(
    comparison.sort_values("tottime_diff", ascending=True)[
        ["func", "tottime_naive", "tottime_optimized", "tottime_diff"]
    ]
    .reset_index(drop=True)
    .to_markdown()
)
