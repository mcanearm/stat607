# Stat 607

This was originally going to be my repo for all of the projects, but I'm adapting it to just be for units 2 and 3. 

## Unit 2 Project

I chose to recreate the analysis from Greenland, Sander. “Methods for Epidemiologic Analyses of Multiple Exposures: A Review and Comparative Study of Maximum-Likelihood, Preliminary-Testing, and Empirical-Bayes Regression.” Statistics in Medicine 12, no. 8 (1993): 717–36. https://doi.org/10.1002/sim.4780120802.

This repo is designed to be laid out as cleanly as possible, but to recreate my version of the analysis (which I do not claim perfectly reproduces Greenland's results), run the following command from the root directory:

```zsh
# available options are "all", "test", "analyze", "figures", and "simulate"
make all
```

The `all` option corresponds to `simulate` -> `analyze` -> `figures`.

## Unit 3 Project

"Optimizations" were added to unit 2. We saw some appreciable gains in performance for our `fit_parametricEB` function and some marginal
gains for changes in the data generating process, with some reductions in performance for the MLE. However, the most dramatic gains
came from removing print statements in the logging, which were slowing me down significantly. In terms of total time saved, I may have 
been much better off refactoring completely in JAX, though that was difficult enough that I abandoned the idea after a few hours
of work.

The `Makefile` has been improved significantly, and there are now environment variables for specifying certain behaviors. The environment
variables of note are:

1) LOGLEVEL: set this for the level of logging you want Python to output from various simulations
2) NSIM: The number of simulations to run.
3) NUM_CORES: The number of cores to utilize in parallism. This controls the jobs, but the within job parallelism is currently disabled when called from the `run_simulations.py` function.
4) VERSION: Several scripts invoke different versions of the functions - v1 corresponds to the "naive" implementation, while v2 corresponds to the "optimized" implementation.
5) PYTHONPATH: Defaults to the current directory for imports. Should work as long as make is run from the root directory.

### Packaging

I used the `uv` package to manage packaging and dependencies, but you should still be able to utilize the 
`requirements.txt` the normal way, i.e.:

```zsh
pip install -r requirements.txt
```

If you would like to try `uv`, then you can try running the following:

```zsh
uv sync
```

This should create the environment and get the required packages installed.
