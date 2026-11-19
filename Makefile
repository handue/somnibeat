PY ?= .venv/bin/python
CONFIG ?= config/default.toml

.PHONY: reproduce setup download check features rpeaks evaluate web docs parity test clean-results

## One command: raw downloads -> results/*.json
reproduce: setup download check features rpeaks evaluate web docs parity test

setup:
	test -x $(PY) || python3 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt

download:
	$(PY) scripts/download.py --config $(CONFIG)

check:
	$(PY) scripts/check_data.py

features:
	$(PY) scripts/build_features.py --config $(CONFIG)

rpeaks:
	$(PY) scripts/validate_rpeaks.py

evaluate:
	$(PY) scripts/train_eval.py --config $(CONFIG) --tasks sleep_wake wake_nrem_rem four_class apnea

## Web demo data (web/data/*) from results/*.json and the trained models
web:
	$(PY) scripts/export_web.py

## README.md and docs/WRITEUP.md, every number filled from results/*.json
docs:
	$(PY) scripts/render_docs.py

## Python/JavaScript parity tests (needs node)
parity:
	$(PY) -m pytest -q tests/test_parity_*.py

test:
	$(PY) -m pytest -q tests

clean-results:
	rm -rf results/*.json results/figures results/models data/cache
