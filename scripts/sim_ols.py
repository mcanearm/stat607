import numpy as np
import time
import os
import numba as nb
import jax
import jax.numpy as jnp
from tqdm import tqdm
from functools import singledispatch
import sys


os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


@singledispatch
def simulate_data(rng, n, p, beta, sigma2, cov_mat=None) -> tuple:
    raise TypeError("Unsupported RNG type")


@simulate_data.register(np.random.Generator)
def _(rng, n, p, beta, sigma2, cov_mat=None):
    if not cov_mat:
        cov_mat = np.eye(p)
    X = rng.multivariate_normal(np.zeros(p), cov=cov_mat, size=n)
    eps = rng.normal(scale=np.sqrt(sigma2), size=n)
    y = X @ beta + eps
    return X, y


keyType = type(jax.random.PRNGKey(0))


@simulate_data.register(keyType)
def sim_jax(rng, n, p, beta, sigma2, cov_mat=None):
    if cov_mat is None:
        cov_mat = jnp.eye(p)
    X = jax.random.multivariate_normal(rng, jnp.zeros(p), cov_mat, shape=(n,))
    eps = jax.random.normal(rng, shape=(n,)) * jnp.sqrt(sigma2)
    y = jnp.dot(X, beta) + eps
    return X, y


def ols_np(X, y):
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    return beta


@nb.njit
def ols_nb(X, y):
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    return beta


@jax.jit
def jax_ols(X, y):
    XtX = jnp.dot(X.T, X)
    beta = jnp.linalg.solve(XtX, jnp.dot(X.T, y))
    return beta


p = 25
n_sims = int(5e5)
N = 1000
rng_np = np.random.default_rng()

rng_keys = jax.random.split(jax.random.PRNGKey(0), n_sims)
for fn in [ols_np, ols_nb]:
    X, y = simulate_data(rng_np, N, p, np.ones(p), 1.0)
    warmup = fn(X, y)
    with tqdm(range(n_sims), desc=f"Timing {fn.__name__}") as pbar:
        if fn == jax_ols:
            key = rng_keys[pbar.n - 1]
            true_beta = jax.random.normal(key, shape=(p,))
        else:
            key = rng_np
            true_beta = key.normal(size=p)
        start = time.time()
        X, y = simulate_data(key, N, p, true_beta, 1.0)
        for _ in pbar:
            void = fn(X, y)
        end = time.time()
    print(f"{fn.__name__} total time:", end - start)


def jax_simulate_and_ols(rng, n, p, beta, sigma2, cov_mat):
    X, y = sim_jax(rng, n, p, beta, sigma2, cov_mat)
    beta_hat = jax_ols(X, y)
    return jnp.square(jnp.linalg.norm(beta_hat - beta))


map_fn = jax.vmap(jax_simulate_and_ols, in_axes=(0, None, None, 0, None, None))

warmup = map_fn(rng_keys[0:5], N, p, jnp.ones((5, p)), 1.0, None)
true_betas = jax.random.normal(rng_keys[10], shape=(n_sims, p))


time_start = time.time()
fitted_betas = map_fn(rng_keys, N, p, true_betas, 1.0, None)
print(fitted_betas.shape)
time_end = time.time()

print("JAX total time:", time_end - time_start)

sys.exit(0)
