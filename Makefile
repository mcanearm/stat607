test:
	pytest tests/

simulate:
	python ./scripts/run_simulations.py

analyze:
	python ./scripts/generate_summaries.py

figures:
	python ./generate_plots.py

clean:
	rm -rf results/raw/* results/figures/* results/processed/*

all: simulate summarize generate_plots