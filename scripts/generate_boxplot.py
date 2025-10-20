# plot_rmse_grid.py
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from src.simulation import load_simulation_output

# ---------- Config ----------
RESULTS_DIR = Path("./results/raw/")
TRIM_Q = 0.99  # set to None to disable trimming
FIGSIZE_UNIT_X = 4.0
FIGSIZE_UNIT_Y = 1.5
TITLE = "RMSE by estimator"
# ----------------------------

# --- load results ---
paths = sorted(RESULTS_DIR.glob("*.pkl"))
results = [load_simulation_output(p) for p in paths]
if not results:
    raise SystemExit(f"No results found in {RESULTS_DIR}")

# --- build grid (rows by N, cols by n) ---
Ns = sorted({int(r.attrs["N"]) for r in results})
ns = sorted({int(r.attrs["n"]) for r in results})

fig, axes = plt.subplots(
    nrows=len(Ns),
    ncols=len(ns),
    figsize=(FIGSIZE_UNIT_X * len(ns), FIGSIZE_UNIT_Y * len(Ns)),
    sharey=False,
    sharex=False,  # we'll control x ticks explicitly
)

# Normalize axes to 2D array
if len(Ns) == 1 and len(ns) == 1:
    axes = np.array([[axes]])
elif len(Ns) == 1:
    axes = axes[np.newaxis, :]
elif len(ns) == 1:
    axes = axes[:, np.newaxis]


# --- find estimator coord + fixed order ---
def get_est_coord(ds: xr.Dataset) -> str:
    for cand in ("estimator", "method"):
        if cand in ds.coords:
            return cand
    # fallback: pick a coord that matches #estimators (3 or 5)
    for c in ds.coords:
        if ds.coords[c].size in (3, 5):
            return c
    # last resort: look in dims
    for d in ds.dims:
        if ds.sizes[d] in (3, 5):
            return d
    raise KeyError("Could not find estimator/method coord/dim")


est_coord = get_est_coord(results[0])
est_order = list(results[0].coords[est_coord].values)


# --- label mapping ---
def abbr(name: str) -> str:
    s = str(name)
    low = s.lower()
    if low == "mle":
        return "MLE"
    if low.startswith("semi_bayes"):
        # e.g., semi_bayes_0.5, semi_bayes_1, semi_bayes_2
        parts = s.split("_", 2)
        suffix = parts[-1] if len(parts) >= 3 else ""
        return f"SB_{suffix}" if suffix else "SB"
    if low.startswith("parametric_") or low in ("pb", "eb"):
        return "PB"
    return s  # fallback


est_labels = [abbr(x) for x in est_order]


# --- rmse helpers ---
def rmse_df(ds: xr.Dataset) -> pd.DataFrame:
    """Per-simulation RMSE per estimator → DataFrame(cols = estimator)."""
    bh = ds["beta_hat"].sel(var="estimate")  # (simulation, estimator, param)
    tb = ds["true_beta"].expand_dims({est_coord: ds.coords[est_coord]})
    sim_rmse = np.sqrt(((bh - tb) ** 2).mean(dim="param"))  # (simulation, estimator)
    return pd.DataFrame(sim_rmse.values, columns=ds.coords[est_coord].values)


def trim_outliers_per_est(df: pd.DataFrame, q: float | None) -> pd.DataFrame:
    if q is None:
        return df
    out = {}
    for col in df.columns:
        thr = df[col].quantile(q)
        out[col] = df.loc[df[col] <= thr, col]
    return pd.DataFrame(out)


# --- pick one dataset per (N, n) (first file wins if multiple) ---
grid = {}
for ds in results:
    key = (int(ds.attrs["N"]), int(ds.attrs["n"]))
    grid.setdefault(key, ds)

# --- set column headers to n only (top row) ---
for c, n_val in enumerate(ns):
    axes[0, c].set_title(f"n = {n_val}", fontsize=11)

# --- plot panels ---
for r, N in enumerate(Ns):
    for c, n_val in enumerate(ns):
        ax = axes[r, c]
        ds = grid.get((N, n_val))
        if ds is None:
            ax.axis("off")
            continue

        df = rmse_df(ds)[est_order]
        df = trim_outliers_per_est(df, TRIM_Q)

        # one 1D array per estimator (fixed order), explicit positions
        data = [df[k].dropna().values for k in est_order]
        positions = np.arange(1, len(data) + 1)  # 1..#estimators

        ax.boxplot(data, positions=positions)
        ax.set_xticks(positions)
        ax.set_xticklabels(est_labels, rotation=0)

        # leftmost ylabel = N
        if c == 0:
            ax.set_ylabel(f"N = {N}")

# ensure bottom-left panel shows the last row label (e.g., N = 2000)
# Ensure each row has a visible left-side label (use the first visible axis in the row)
for r, N in enumerate(Ns):
    # find the first visible axis in this row
    target_ax = None
    for c in range(len(ns)):
        ax_candidate = axes[r, c]
        # an 'off' axis returns False for has_data(); also check visibility
        if ax_candidate.get_visible() and (
            ax_candidate.has_data() or ax_candidate.lines or ax_candidate.patches
        ):
            target_ax = ax_candidate
            break
    # if nothing has data, pick the first axis and turn it on so we can label the row
    if target_ax is None:
        target_ax = axes[r, 0]
        target_ax.axis("on")
    target_ax.set_ylabel(f"N = {N}")

# Give a bit more left margin so the bottom label (e.g., N=2000) is not clipped
fig.subplots_adjust(left=0.10)  # bump this if your labels are still tight
fig.suptitle(TITLE, y=0.995, fontsize=12)
fig.tight_layout()
plt.show()
