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
validation -> training and evaluation -> tests. Seed and all parameters live
in `config/default.toml`; each results JSON stores the config SHA-256.
