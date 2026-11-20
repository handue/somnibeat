PY ?= .venv/bin/python
CONFIG ?= config/default.toml
V2_CONFIG ?= config/v2.toml

.PHONY: reproduce reproduce-v1 setup download check features rpeaks evaluate web docs parity test clean-results
.PHONY: v2 v2-download v2-check v2-features v2-select v2-decide v2-evaluate

# Targets depend on each other through files in data/ and results/, not through make rules:
# never build them in parallel.
.NOTPARALLEL:

## One command: raw downloads -> results/*.json (v1) and results/v2/*.json (v2), web data, docs, tests.
## Three make runs, so that web, docs and the tests run again after v2 (make would otherwise skip
## targets that reproduce-v1 already made).
reproduce:
	$(MAKE) reproduce-v1
	$(MAKE) v2
	$(MAKE) web web-v2 docs parity test

## v1 only (the original pipeline, tag v1-results)
reproduce-v1: setup download check features rpeaks evaluate web docs parity test

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

## ---------------------------------------------------------------------------------------------- v2
## v2 (docs/v2_protocol.md): selection on training data only, frozen at tag v2-frozen, then one
## test-set evaluation. Needs the v1 caches and models (reproduce-v1). Durations measured on the
## development machine (results/v2/parts/run_*.log, data/cache/v2/build_*.log, results/v2/*.log).
v2: v2-download v2-check v2-features v2-select v2-decide v2-evaluate

## CAP (106 records of config/v2.toml) and HMC 1.1 (151 records, ~17 GB), SHA-256 verified
v2-download:
	$(PY) scripts/download.py --config $(V2_CONFIG) --datasets capslpdb hmc

## results/v2/data_check.json, results/v2/hmc_check.json (+ HMC RR cache), results/v2/slpdb_apnea_check.json
v2-check:
	$(PY) scripts/check_data.py --config $(V2_CONFIG) --out results/v2/data_check.json
	$(PY) scripts/check_hmc.py --config $(V2_CONFIG)
	$(PY) scripts/v2_select_apnea.py check

## data/cache/v2: CAP-106 and HMC v1-style tables (~5 + ~7 min), CAP-106 + slpdb training table,
## long-context extras (base, cap106 ~3.5 min, test ~4.5 min), slpdb apnea minute features
v2-features:
	$(PY) scripts/build_features.py --config $(V2_CONFIG) --datasets cap
	$(PY) scripts/build_features.py --config $(V2_CONFIG) --datasets hmc --prefix hmc_
	$(PY) scripts/v2_merge_training.py
	$(PY) scripts/v2_build_extras.py
	$(PY) scripts/v2_select_apnea.py features

## Selection on training data only (~4.4 h in total: sleep/wake 78 min, 3-stage 35 min,
## 4-stage 41 min, CAP-106 check 9.5 min, apnea 79 min)
v2-select:
	$(PY) scripts/v2_select.py --candidates default --tasks sleep_wake --out results/v2/parts/staging_sleep_wake_all.json
	$(PY) scripts/v2_select.py --candidates default --tasks wake_nrem_rem --out results/v2/parts/staging_wake_nrem_rem_all.json
	$(PY) scripts/v2_select.py --candidates default --tasks four_class --out results/v2/parts/staging_four_class_all.json
	$(PY) scripts/v2_select.py --candidates results/v2/parts/cap106_candidates.json --tasks sleep_wake --no-subject-cv \
		--features data/cache/v2/stage_features_cap106_slpdb.csv.gz \
		--extra data/cache/v2/stage_features_extra_cap106.csv.gz --out results/v2/selection_cap106.json
	$(PY) scripts/v2_select_apnea.py select

## results/v2/v2_selection.json, docs/v2_selection.md, results/v2/v2_final_spec.json, docs/v2_final_spec.md
v2-decide:
	$(PY) scripts/v2_finalize_selection.py
	$(PY) scripts/v2_decide_final.py

## One evaluation on the test sets (HMC, UCDDB) -> results/v2/test_results.json (~25 min).
## Refuses to run unless HEAD contains tag v2-frozen and the frozen files are unchanged.
v2-evaluate:
	$(PY) scripts/train_eval_v2.py --final

## ------------------------------------------------------------------------------------------- web
## Web demo data (web/data/*) from results/*.json and the trained models
web:
	$(PY) scripts/export_web.py

## v2 web models (web/data/models_v2.json) from results/v2/models/*.pkl (frozen v2)
.PHONY: web-v2
web-v2:
	$(PY) scripts/export_web_v2.py

## README.md and docs/WRITEUP.md, every number filled from results/*.json
docs:
	$(PY) scripts/render_docs.py

## Python/JavaScript parity tests (needs node)
parity:
	$(PY) -m pytest -q tests/test_parity_*.py

test:
	$(PY) -m pytest -q tests

clean-results:
	rm -rf results/*.json results/figures results/models results/v2 data/cache
