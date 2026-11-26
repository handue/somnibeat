# Web demo interface contract

The demo is a static site in `web/` (no build step, no server, no network
calls except loading its own files). Everything is computed in the browser.
Python is the reference implementation; JavaScript must match it (parity
tests in `tests/parity/`).

## Files

```
web/
  index.html            single page, hash routes: #/ (start), #/processing, #/report, #/method, #/problem
  package.json          {"type": "module"} so Node can import the same modules for parity tests
  css/app.css
  js/rr.js              CSV parsing + RR cleaning           (mirror of somnibeat/rr.py)
  js/features.js        epoch features                       (mirror of somnibeat/features.py)
  js/features_v2.js     v2 long-context features (600 s windows around each epoch; mirror of somnibeat/features_v2.py)
  js/model.js           HistGradientBoosting inference from JSON (v1)
  js/model_v2.js        v2 inference: staging trees + whole-night HMM Viterbi; apnea mean of trees and logistic regression
  js/smoothing.js       HMM smoothing                        (mirror of somnibeat/smoothing.py)
  js/report.js          v1 grid, predictions, night metrics  (mirror of somnibeat/report.py)
  js/report_v2.js       v2 report, same grid, metrics and shape (mirror of somnibeat/report_v2.py)
  js/worker.js          module Web Worker: runs report_v2.js (default) or report.js off the main thread
  js/app.js             UI, routing, rendering
  js/charts.js          SVG hypnogram, apnea strip, Bland-Altman, comparison bars
  data/models.json      v1 models, exported by scripts/export_web.py
  data/models_v2.json   frozen v2 models, exported by scripts/export_web_v2.py
  data/results.json     exported by scripts/export_web.py (subset of results/*.json + "v2" block)
  data/samples.json     sample metadata + attribution
  data/samples/*.csv    four sample RR files
  data/samples/*_reference.json  expert reference labels of the UCDDB samples (never a model output)
  assets/somnibeat-mark.svg, assets/somnibeat-lockup.svg, assets/favicon.svg (copies, unmodified)
```

## RR CSV format (upload and samples)

- Header row required. Columns `time_s` and `rr_ms` (case-insensitive,
  surrounding whitespace ignored). Extra columns are ignored.
- `time_s`: time of the beat that closes the interval, in seconds since the
  recording started. If the column is missing, times are the cumulative sum
  of `rr_ms` / 1000 and a warning is shown.
- `rr_ms`: interval in milliseconds.
- Rows with a non-numeric value are skipped and counted (warning).
- Errors (show the File-problem screen): no `rr_ms` column; fewer than 2 hours
  of beats (`MIN_HOURS = 2`); `time_s` not increasing in more than 1 % of rows;
  median `rr_ms` outside 300-2000 (wrong unit, e.g. seconds); `time_s` jumping
  back by more than 10 s even once (`TIME_RESET`, two nights in one file);
  median step of `time_s` (in ms) outside 0.5-2 times the median `rr_ms`
  (`BAD_TIME_UNIT`, e.g. `time_s` in milliseconds); the first interval starting
  more than 24 h after 0 (`BAD_TIME_START`, clock or Unix time). When no row is
  usable the message counts the non-numeric rows and points out decimal commas.
- Other accepted inputs, converted in the worker to the format above:
  an EDF recording (`.edf`, `.rec`) whose first channel labelled `ECG|EKG` is
  run through `js/ecg.js` (a port of `somnibeat/edf.py`, `somnibeat/rpeak.py`
  and the rounding of `scripts/ecg_to_rr.py`, parity-tested in
  `tests/test_parity_ecg.py`; every EDF recording: `results/ecg_parity.json`),
  and an RR-interval export with exactly one column named like `RR` and no
  `rr_ms` column (as `convert_rr_export`). EDF errors: `NOT_EDF`, `TRUNCATED`,
  `NO_ECG_CHANNEL` (with the channel labels), `LOW_FS` (below 100 Hz),
  `NO_BEATS`; RR exports: `NO_INTERVALS`; BDF files are refused.
- Warning (shown above the report): more than 10 % of intervals invalid after
  cleaning, reported as "[xx]% of beats were missing or unusable".

## Analysis grid

- `start_s` = lights-off offset if given, else 0 (recording start).
- `end_s` = lights-on offset if given, else the time of the last beat.
- Lights-off/on are entered as clock times together with the recording start
  clock time; offset = (clock - start) mod 24 h.
- Stage epochs: `start_s + 30 k` for k = 0 .. floor((end_s - start_s) / 30) - 1.
- Apnea minutes: `start_s + 60 m` for m = 0 .. floor((end_s - start_s) / 60) - 1.
- Features: `epoch_features(t, rr, valid, epoch_starts, 30, cfg.rr, cfg.features)`
  for staging and the same call with a 60 s grid for apnea, on the full RR
  series (beats outside the grid still feed the windows near its edges).

## Worker protocol (`js/worker.js`)

- In: `{type: "run", csvText | edf, fileName, lightsOffS, lightsOnS, model}` (`edf`: the File). `model` is
  `"v2"` (default when omitted) or `"v1"`; any other value is an `INTERNAL` error.
- Out: `{type: "progress", step, status}` for steps `read`, `ecg` (EDF only), `beats`, `features`,
  `report` (status `running` | `done`); `{type: "converted", csvText, info}` after an EDF or
  RR export was converted (the page keeps it for the model switch and the Save button);
  `{type: "result", report}`;
  `{type: "error", code, message, details}`.
- `report.model` is the model actually used (`"v2"` or `"v1"`). If
  `data/models_v2.json` cannot be loaded, a v2 request falls back to v1 and
  the reason is the first entry of `report.quality.warnings`. A dry-run v2
  export (`source.dry_run == true`) also adds a warning.
- The UI shows `Model: v2 (frozen <date>)` with the date from
  `results.json` `v2.frozen.date`, and a v1 / v2 switch that reruns the same
  file, clock times and sample with the other model. The `features` step
  reads "Computing 30-second and 10-minute features" for v2 and
  "Computing 30-second features" for v1.

## Predictions

- Task models: `sleep_wake` (labels `["sleep", "wake"]`, binary threshold on
  p(wake)), `wake_nrem_rem`, `four_class` (argmax), `apnea` (labels
  `["normal", "apnea"]`, threshold on p(apnea)).
- v1 primary model: gradient boosting (fixed in `config/default.toml`).
- v2 (default in the browser): the frozen spec of `results/v2/v2_final_spec.json`;
  staging adds the long-context features and smooths each task with an HMM
  (Viterbi) over the whole night; apnea averages gradient boosting and
  logistic regression. The format of `data/models_v2.json` is documented in
  `scripts/export_web_v2.py`.
- Low-quality epochs (`low_quality == 1`) are still predicted, exactly as in
  evaluation, and drawn hatched on the chart with the label
  "not enough usable data".

## Night metrics (identical to `somnibeat/evaluate.py`)

- Computed from the `sleep_wake` predictions over the grid.
- `tib_min` = epochs x 0.5; `tst_min` = sleep epochs x 0.5; `se_pct` = TST/TIB x 100.
- `awakenings` = runs of >= 2 wake epochs between the first and last sleep epoch.
- `apnea_min` = minutes predicted apnea.

## `data/models.json`

```
{
  "schema": 1,
  "config": {"rr": {...config [rr]}, "features": {...config [features]}, "epoch_s": 30, "apnea_epoch_s": 60},
  "tasks": {
    "<task>": {
      "labels": [...], "columns": [...feature names in model order...],
      "decision": {"type": "threshold", "positive": "wake"|"apnea", "threshold": 0.39} | {"type": "argmax"},
      "kind": "hgb",
      "baseline": [b_0, ..., b_{K-1}],          // K = 1 for binary, n_classes otherwise
      "trees": [[tree_k for k in 0..K-1] for each iteration],
      // tree = {"f": [...feature_idx], "t": [...num_threshold], "m": [...missing_go_to_left 0/1],
      //         "l": [...left], "r": [...right], "v": [...value], "leaf": [...is_leaf 0/1]}
    }
  }
}
```

Inference (matches `HistGradientBoostingClassifier.predict_proba` for
numeric features): for each row, raw_k = baseline_k + sum over iterations of
the leaf value of tree k. Traversal from node 0: if leaf -> value; if x[f] is
NaN -> left if m else right; else left if x[f] <= t else right.
Binary: p(positive) = 1 / (1 + exp(-raw_0)). Multiclass: softmax(raw).

## `data/results.json`

Exported subset of `results/*.json` and `results/v2/*.json` (no per-epoch
data). Every number shown on the page comes from this file, `samples.json`
or from the computation on the loaded RR file. Nothing is hard-coded in
HTML/JS. Write it with `python scripts/export_web.py --skip-models` to leave
`models.json` untouched. Keys (v1 keys unchanged; `datasets` gains `roles`
and an HMC row; `v2` is new):

```
{
  "generated_from": {"<task>.json": "<sha256>", ..., "v2/test_results.json": "<sha256>", "v2/v2_final_spec.json": ...,
                     "v2/v2_selection.json": ..., "v2/hmc_check.json": ..., "v2/selection_cap106.json": ...},
  "primary_model": "gradient_boosting",
  "tasks": {"<task>": {"models": {"<model>": {"external_ucddb": {"metrics": {...}, "bland_altman": {...}}, "internal_cv": {"metrics": {...}}, "threshold": ...}},
                       "baselines": {...metrics...}, "data": {...}}},                      // v1, UCDDB first look
  "per_record": {"sleep_wake": [{record, ref_tst_min, pred_tst_min, ref_se_pct, pred_se_pct, ...}], "apnea": [{record, ref_apnea_min, pred_apnea_min, ...}]},
  "rpeak_validation": {...pooled numbers...},
  "datasets": [{"id", "name", "short", "role" (v1 role), "roles": {"v1", "v2"}, "records", "subjects", "population",
                "license", "license_url", "citation", "doi", "url",
                // HMC row only: "records_primary", "male", "female", "scoring", "version", "license_short"
                // CAP row only: "records_v2"
               }],
  "limitations": [...v1 strings...],
  "v2": {
    "version": "v2",
    "frozen": {"tag": "v2-frozen", "date": "YYYY-MM-DD", "commit": "<sha>"},   // tag date in the tagger's time zone
    "note": "...", "primary_test_set": "hmc",
    "exclusion_rule": {"min_adequate_fraction": <from scripts/train_eval_v2.py>, "text": "..."},
    "spec": {"protocol", "staging", "apnea", "hyperparameters"},              // from v2_final_spec.json
    "selection": {"protocol", "training_datasets_only", "folds",
                  "staging": {"rule", "n_candidates", "chosen": {name, xds_sw_kappa, fold_a_sw_kappa, ...},
                              "v1_reference": {...same keys...}},
                  "apnea": {"rule", "n_candidates", "chosen": {name, xds_kappa, ...}, "v1_reference": {...}}},
    "test_sets": {"hmc" | "ucddb": {"records_all", "records_primary",
                  "excluded_low_coverage": [{"record", "adequate_fraction"}],
                  "tasks": {"sleep_wake" | "wake_nrem_rem" | "four_class": {"primary" | "all_records": {
                      "v2" | "v1" | "asleep_from_lights_off": {"n_records", "n_epochs", "metrics": {"<metric>": {point, ci_low, ci_high}},
                                                               "bland_altman" (sleep_wake), "labels", "confusion_matrix_rows_ref_cols_pred", "as_sleep_wake" (multiclass)}}}}}},
    "apnea_ucddb": {"threshold_p_apnea", "primary" | "all_records": {"v2" | "v1" | "no_apnea": {"n_records", "n_minutes", "metrics", "bland_altman"}}},
    "per_record": {"hmc_primary_sleep_wake": [{record, ref_tib_min, ref_tst_min, pred_tst_min, ref_se_pct, pred_se_pct, ref_awakenings, pred_awakenings, kappa}],
                   "ucddb_primary_sleep_wake": [...same...], "ucddb_primary_apnea": [{record, ref_apnea_min, pred_apnea_min, total_min, kappa}]},
    "baselines": {"stages": "asleep_from_lights_off", "apnea": "no_apnea"},
    "limitations": [...strings shown on the page...]
  }
}
```

Metrics are copied unchanged from `results/v2/test_results.json`; per-record
rows are v2 only and carry night totals, never epochs. `bland_altman` keeps
bias, MAE, SD and limits of agreement, without `per_record_error`. The frozen
date is read from the `v2-frozen` tag object in `.git` (no git command; only
the date and commit are kept), or from `--v2-frozen-date`; the export stops if
the tag does not point to the commit recorded in `test_results.json`. The
export also stops if a qualitative statement in `v2.limitations` (for example
"total sleep time error not smaller than v1's") no longer follows from the
numbers.

The page uses `v2` as follows: "How far can you trust this?" leads with the
report's model on HMC primary, a side-by-side row for v2, v1 and the trivial
rule on the same HMC nights, a plain list of what v2 improved, did not
improve and made worse, including the 3-stage REM-minute error (based on
whether the 95 % intervals overlap), one line for UCDDB (second
look) and one for apnea (UCDDB only), and the Bland-Altman plot of v2 HMC
primary per-record total sleep time. Method & evaluation shows the selection
story, v1-vs-v2 tables for HMC primary, HMC all records and UCDDB with
intervals, the apnea table, the excluded HMC records, and keeps the v1
tables labeled "v1, first look".

## `data/samples.json`

```
[{"id": "ordinary", "title": "An ordinary night", "summary": "...", "file": "samples/ucddb018.csv",
  "selection": "...rule...", "has_stage_reference": true, "reference": "samples/ucddb018_reference.json",
  "source": {"dataset": "...", "record": "...", "license": "ODC-By 1.0", "license_url": "...", "url": "...", "doi": "...",
             "note": "Part of the external test set..."}}]
```

Samples, in this order (the export stops if a rule picks another record):

| id | record | rule |
|---|---|---|
| `ordinary` | `ucddb018` | UCDDB record with the fewest reference apnea minutes (`results/apnea.json`) |
| `apnea` | `ucddb025` | UCDDB record with the most reference apnea minutes |
| `noisy` | Apnea-ECG `c04` | lowest fraction of valid RR intervals; no stage reference |
| `typical` | `ucddb015` | UCDDB primary record whose v2 absolute total sleep time error is the median (`results/v2/test_results.json`); with an even count, the lower record id of the two middle records |

Numbers in `summary` are formatted at export time from the JSON results.

## Parity

Split by layer; each Python test shells out to a Node CLI that imports the
same ES modules the browser uses.

| Test | Node CLI | Checks |
|---|---|---|
| `tests/test_parity_features.py` | `tests/parity/features_cli.mjs` | CSV parsing rules, RR cleaning, every feature on 30 s and 60 s grids, edge cases |
| `tests/test_parity_model.py` | `tests/parity/model_cli.mjs` | tree inference vs sklearn `predict_proba`, decisions, all four tasks |
| `tests/test_parity_report.py` | `tests/parity/report_cli.mjs` | grid, per-epoch labels, low-quality flags, apnea labels, night metrics, worker protocol, timing |
| `tests/test_parity_features_v2.py` | `tests/parity/features_v2_cli.mjs` | v2 long-context features |
| `tests/test_parity_model_v2.py` | `tests/parity/model_v2_cli.mjs` | v2 staging (trees + HMM) and apnea (trees + logistic) inference |
| `tests/test_parity_report_v2.py` | `tests/parity/report_v2_cli.mjs` | v2 report end to end, worker `model` field |

Export checks (no Node needed for most): `tests/test_reference_export.py`
(sample reference files) and `tests/test_web_results_v2.py` (results.json is a
fresh export, v2 numbers copied unchanged, no per-epoch arrays, HMC row, typical
sample rule).

Tolerance: features and probabilities within 1e-9 relative, except where
V8 and the platform libm differ in the last bit of sin/cos/log/atan2. Those
differences are bounded by `|a - b| <= 1e-9 * max(|a|, |b|, 1)`. A row whose
feature sits exactly on a tree split threshold (|x - t| <= 1e-9 max(|x|, 1))
may take the other branch; such rows must still produce the same label.
Predicted labels and night metrics must be identical.
