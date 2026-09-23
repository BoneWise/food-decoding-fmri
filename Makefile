PYTHON ?= python

.PHONY: tables validate figures test reproduce
tables:
	$(PYTHON) scripts/reproduce_results.py
validate:
	$(PYTHON) scripts/validate_reference.py
figures:
	$(PYTHON) scripts/reproduce_figures.py
test:
	$(PYTHON) -m pytest -q
reproduce: tables validate figures
