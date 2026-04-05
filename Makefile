SHELL := /bin/bash

VENV ?= .venv
PY ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

ARTIFACTS ?= artifacts
CONFIG ?= configs/train_hybrid_pinn.yaml

.PHONY: setup lint train eval dev-run clean

setup:
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt

lint:
	ruff check .
	black --check .

train:
	$(PY) -m training.train_pinn_hybrid --config $(CONFIG)

eval:
	$(PY) -m eval.evaluate_model --config $(CONFIG)

dev-run:
	$(PY) -m training.train_pinn_hybrid --config $(CONFIG) --dev-run
	$(PY) -m eval.evaluate_model --config $(CONFIG)

clean:
	rm -rf $(ARTIFACTS)/metrics $(ARTIFACTS)/plots $(ARTIFACTS)/logs $(ARTIFACTS)/best_model.pt $(ARTIFACTS)/scalers.json $(ARTIFACTS)/teacher_cache
