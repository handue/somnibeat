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
  js/model.js           HistGradientBoosting inference from JSON
  js/report.js          grid, predictions, night metrics     (mirror of somnibeat/report.py)
  js/worker.js          module Web Worker: runs report.js off the main thread, posts progress
  js/app.js             UI, routing, rendering
  js/charts.js          SVG hypnogram, apnea strip, Bland-Altman, comparison bars
  data/models.json      exported by scripts/export_web.py
  data/results.json     exported by scripts/export_web.py (subset of results/*.json)
  data/samples.json     sample metadata + attribution
  data/samples/*.csv    three sample RR files
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
  median `rr_ms` outside 300-2000 (wrong unit, e.g. seconds).
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

## Predictions

- Task models: `sleep_wake` (labels `["sleep", "wake"]`, binary threshold on
  p(wake)), `wake_nrem_rem`, `four_class` (argmax), `apnea` (labels
  `["normal", "apnea"]`, threshold on p(apnea)).
- Primary model: gradient boosting (fixed in `config/default.toml`).
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

Exported subset of `results/*.json` (no per-epoch data). Every number shown on
the page comes from this file or from the computation on the loaded RR file.
Nothing is hard-coded in HTML/JS. Keys:

```
{
  "generated_from": {"<task>.json": "<sha256>", ...},
  "primary_model": "gradient_boosting",
  "tasks": {"<task>": {"models": {"<model>": {"external_ucddb": {"metrics": {...}, "bland_altman": {...}}, "internal_cv": {"metrics": {...}}, "threshold": ...}},
                       "baselines": {...metrics...}, "data": {...}}},
  "per_record": {"sleep_wake": [{record, ref_tst_min, pred_tst_min, ref_se_pct, pred_se_pct}], "apnea": [{record, ref_apnea_min, pred_apnea_min}]},
  "rpeak_validation": {...pooled numbers...},
  "datasets": [{"name", "role", "records", "subjects", "population", "license", "citation", "url"}],
  "limitations": [...strings...]
}
```

## `data/samples.json`

```
[{"id": "ordinary", "title": "An ordinary night", "summary": "...", "file": "samples/ucddb018.csv",
  "source": {"dataset": "...", "record": "...", "license": "ODC-By 1.0", "url": "...", "note": "Part of the external test set"}}]
```

## Parity

Split by layer; each Python test shells out to a Node CLI that imports the
same ES modules the browser uses.

| Test | Node CLI | Checks |
|---|---|---|
| `tests/test_parity_features.py` | `tests/parity/features_cli.mjs` | CSV parsing rules, RR cleaning, every feature on 30 s and 60 s grids, edge cases |
| `tests/test_parity_model.py` | `tests/parity/model_cli.mjs` | tree inference vs sklearn `predict_proba`, decisions, all four tasks |
| `tests/test_parity_report.py` | `tests/parity/report_cli.mjs` | grid, per-epoch labels, low-quality flags, apnea labels, night metrics, worker protocol, timing |

Tolerance: features and probabilities within 1e-9 relative, except where
V8 and the platform libm differ in the last bit of sin/cos/log/atan2. Those
differences are bounded by `|a - b| <= 1e-9 * max(|a|, |b|, 1)`. A row whose
feature sits exactly on a tree split threshold (|x - t| <= 1e-9 max(|x|, 1))
may take the other branch; such rows must still produce the same label.
Predicted labels and night metrics must be identical.
