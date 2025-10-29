test:
	PYTHONPATH=. pytest tests/

simulate:
	PYTHONPATH=. python ./scripts/run_simulations.py --nsim 8000 --num-cores 8 

analyze:
	PYTHONPATH=. python ./scripts/generate_summaries.py

figures:
	PYTHONPATH=. python ./scripts/generate_plots.py

clean:
	rm -rf results/raw/* results/figures/* results/processed/*

all: simulate analyze figures