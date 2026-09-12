.PHONY: verify figures surrogate-2016 surrogate-2021 smoke-2016 smoke-2021

verify:
	python3 reproduction/verify_results.py

figures:
	bash reproduction/reproduce_figures.sh

surrogate-2016:
	bash reproduction/reproduce_surrogate.sh 2016

surrogate-2021:
	bash reproduction/reproduce_surrogate.sh 2021

smoke-2016:
	bash reproduction/smoke_uced.sh 2016

smoke-2021:
	bash reproduction/smoke_uced.sh 2021
