NSIM ?= 8000 
NUM_CORES ?= 4
TAG ?= $(shell git rev-parse --short HEAD)

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
	
profile: 
	PYTHONPATH=. python -m cProfile -o profile.out ./scripts/run_simulations.py --nsim $(NSIM) --num-cores $(NUM_CORES)

complexity:
	PYTHONPATH=. python ./scripts/record_timings.py --tag $(TAG)

all: simulate analyze figures