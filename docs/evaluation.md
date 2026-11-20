# Evaluation protocol

Fixed before any model tuning. Numbers in the README and demo are read from
`results/*.json`; nothing is typed by hand.

## Splits

| Task | Train (with subject-grouped CV) | External test |
|---|---|---|
| Sleep staging (2, 3, 4 classes) | CAP Sleep Database subset + MIT-BIH slpdb | UCDDB (25 records) |
| Apnea minutes | Apnea-ECG learning set (35 records) | UCDDB (25 records) |

- Splits are by record and by subject (slpdb slp01a/b and slp02a/b are the
  same people and always fall in the same fold).
- UCDDB is never used to fit, normalize across records, select features,
  select thresholds or pick models. Each model predicts it once.
- Decision thresholds are chosen on out-of-fold predictions from 5-fold
  subject-grouped CV on the training data (binary task: threshold maximizing
  epoch kappa). Multi-class tasks use argmax.

## Reference labels

- Stage mapping: Wake; REM; light = S1 + S2; deep = S3 + S4.
- Dropped epochs: UCDDB codes 6 (artifact), 7 (indeterminate) and 8
  (undocumented, 15 epochs in ucddb008/024); slpdb MT/M; CAP epochs missing
  from the hypnogram or marked unscored.
- UCDDB apnea minute: any APNEA-* or HYP-* event overlaps the minute;
  POSSIBLE events excluded. Apnea-ECG minutes include hypopneas by
  definition. The two databases may define hypopnea differently, which is
  a stated limitation.

## Night-level definitions

- Time in bed (TIB): the scored span (no lights-off markers in the data).
- Total sleep time (TST): sleep epochs x 30 s.
- Sleep efficiency: TST / TIB x 100.
- Awakenings: runs of >= 2 wake epochs after the first and before the last
  sleep epoch.
- Errors: prediction - reference.

## Reported statistics

- TST error (min), sleep-efficiency error (percentage points), awakenings
  error, apnea-minute error: bias and mean absolute error.
- Epoch-level Cohen's kappa (pooled, and mean per record), accuracy,
  per-class recall/precision.
- 95 % confidence intervals: 2000 record-level bootstrap resamples, percentile method.
- Bland-Altman: bias and 95 % limits of agreement (bias +- 1.96 SD, ddof = 1).
- Baselines on the same epochs: (a) asleep from lights-off (every epoch sleep),
  (b) majority class of the training labels.
- Internal CV numbers are reported alongside, labeled as internal.

## Reproducibility

`make reproduce` runs download -> data check -> features -> detector
validation -> training and evaluation -> tests for v1 (`make reproduce-v1`),
then the v2 pipeline (`make v2`, see the README). Seed and all parameters
live in `config/default.toml` (v1) and `config/v2.toml` (v2); each results
JSON stores the hashes of its inputs.

## v2 (second protocol)

v1 above is the pre-registered first result and is not changed. v2 follows a
separate protocol, [`docs/v2_protocol.md`](v2_protocol.md), written before
any v2 experiment. In short:

- **Test sets.** The HMC Sleep Staging Database (PhysioNet, CC BY 4.0) is the
  primary v2 test set and was never used before the single v2 test run.
  UCDDB, already used by v1, is reported only as a labeled "second look".
  Apnea minutes are tested on UCDDB only.
- **Selection on training data only.** Candidate changes (more CAP records,
  longer-context features, HMM smoothing, ensembles, class weighting,
  event-derived apnea labels from MIT-BIH) are compared by cross-dataset CV
  between the training databases (train CAP -> evaluate slpdb, and the
  reverse). Staging metric: mean cross-dataset sleep/wake kappa, with a
  total-sleep-time MAE guard (not worse than v1 by more than 10 %) and ties
  (< 0.01 kappa) going to the simpler candidate. Apnea: the same scheme with
  minute kappa and an apnea-minute MAE guard. The chosen staging
  specification is also used for 3 and 4 classes.
- **Freezing.** The choice is recorded in `results/v2/v2_selection.json` and
  `results/v2/v2_final_spec.json` (summaries: [`docs/v2_selection.md`](v2_selection.md),
  [`docs/v2_final_spec.md`](v2_final_spec.md)) and tagged `v2-frozen`
  before any test-set prediction.
- **Single test run.** `scripts/train_eval_v2.py --final` predicts HMC and
  UCDDB once and scores v1 on the same epochs at the same time
  (`results/v2/test_results.json`). Statistics are as for v1.
- **HMC data-quality rule (fixed before any HMC prediction).** The primary
  analysis excludes records in which fewer than 50 % of epochs have adequate
  RR coverage; the excluded IDs are listed in the results, and an
  all-records analysis is reported alongside. The rule uses only the RR
  series, never labels or predictions, and applies to UCDDB as well.
- **Reference labels.** AASM stages map as Wake; REM; light = N1 + N2;
  deep = N3.

Results and their discussion are in the README and in
[`docs/WRITEUP.md`](WRITEUP.md), rendered from the results JSON.
