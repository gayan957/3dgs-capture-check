PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup demo label review serve ui ui-dev train eval export test test-all lint format degrade splits features check-degradations clean

setup:
	$(PIP) install -e ".[dev]"

demo:
	$(PY) scripts/00_make_demo_data.py --clean

label:
	$(PY) -m capturecheck.cli label data

degrade:
	$(PY) scripts/02_make_degraded_copies.py

features:
	$(PY) scripts/03_compute_features.py

splits:
	$(PY) scripts/08_make_splits.py

check-degradations:
	$(PY) scripts/12_check_degradations.py

review:
	.venv/bin/streamlit run capturecheck/review/app.py

# The web interface. `make ui` builds the React app once; `make serve` then
# serves it alongside the API on one port. For front-end work use `make ui-dev`
# in a second terminal: it reloads on save and proxies /api to `make serve`.
ui:
	npm --prefix frontend install
	npm --prefix frontend run build

ui-dev:
	npm --prefix frontend run dev

serve:
	$(PY) -m capturecheck.cli serve

train:
	$(PY) -m capturecheck.cli train

eval:
	$(PY) -m capturecheck.cli evaluate

export:
	$(PY) -m capturecheck.cli export

test:
	.venv/bin/pytest -q -m "not slow"

test-all:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check capturecheck tests scripts

format:
	.venv/bin/ruff format capturecheck tests scripts

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
