NSIM ?= 8000 
NUM_CORES ?= 4

test:
	PYTHONPATH=. pytest tests/

simulate:
	PYTHONPATH=. python ./scripts/run_simulations.py --nsim $(NSIM) --num-cores $(NUM_CORES)

analyze:
	PYTHONPATH=. python ./scripts/generate_summaries.py

figures:
	PYTHONPATH=. python ./scripts/generate_plots.py

clean:
	rm -rf results/raw/* results/figures/* results/processed/*

all: simulate analyze figures