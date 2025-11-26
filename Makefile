# below are various arguments available for the makefile. Pass them in as environment
# variables when calling make, e.g. NSIM=1000 make simulate
NSIM ?= 8000 
NUM_CORES ?= 1
TAG ?= $(shell git rev-parse --short HEAD)
VERSION ?= "v2"
PYTHONPATH ?= $(CURDIR)
export PYTHONPATH

.PHONY: test simulate analyze figures clean profile complexity parallel stability-check 

test:
	pytest tests/

simulate:
	python ./scripts/run_simulations.py --nsim $(NSIM) --num-cores $(NUM_CORES)

analyze:
	python ./scripts/generate_summaries.py

figures:
	python ./scripts/generate_plots.py
	python ./scripts/analyze_timings.py

clean:
	rm -rf results/raw/* results/figures/* results/processed/*
	
profile: 
	python -m cProfile -o ./results/profiling/$(TAG).prof ./scripts/run_simulations.py --nsim 500 --num-cores 1

benchmark:
	mkdir -p ./results/timings
	if [ "$(VERSION)" = "v1" ]; then \
		python -m cProfile -o ./results/profiling/profile_$(TAG).prof ./scripts/naive_run_simulations.py --nsim $(NSIM) --num-cores $(NUM_CORES); \
	else \
		python -m cProfile -o ./results/profiling/profile_$(TAG).prof ./scripts/run_simulations.py --nsim $(NSIM) --num-cores $(NUM_CORES); \
	fi

complexity:
	python ./scripts/record_timings.py --tag $(TAG) --version $(VERSION)

parallel:
	NUM_CORES ?= 4
	NUM_CORES=$(NUM_CORES) make simulate
	
stability-check:
	mkdir -p ./results/profiling
	LOGLEVEL=WARNING make simulate 2> results/profiling/stability_check.log
	
all: simulate analyze figures
