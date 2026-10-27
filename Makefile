PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup label review train eval test lint format degrade splits features clean

setup:
	$(PIP) install -e ".[dev]"

label:
	$(PY) -m capturecheck.cli label data

degrade:
	$(PY) scripts/02_make_degraded_copies.py

features:
	$(PY) scripts/03_compute_features.py

splits:
	$(PY) scripts/08_make_splits.py

review:
	.venv/bin/streamlit run capturecheck/review/app.py

train:
	$(PY) -m capturecheck.cli train

eval:
	$(PY) -m capturecheck.cli evaluate

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check capturecheck tests scripts

format:
	.venv/bin/ruff format capturecheck tests scripts

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
