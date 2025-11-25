NSIM ?= 8000 
NUM_CORES ?= 4
TAG ?= $(shell git rev-parse --short HEAD)
VERSION ?= "v2"

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
	if [ -f $(VERSION) = "v1" ];
	then 
		PYTHONPATH=. python -m cProfile -o profile_$(TAG).prof ./scripts/naive_run_simulations.py --nsim $(NSIM) --num-cores $(NUM_CORES)
	else
		PYTHONPATH=. python -m cProfile -o profile_$(TAG).prof ./scripts/run_simulations.py --nsim $(NSIM) --num-cores $(NUM_CORES)
	fi

complexity:
	PYTHONPATH=. python ./scripts/record_timings.py --tag $(TAG) --version $(VERSION)

all: simulate analyze figures