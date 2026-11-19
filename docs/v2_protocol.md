# v2 protocol (written 2026-11-19, before any v2 experiment)

v1 is the pre-registered first result (`results/*.json` at git tag
`v1-results`). It stays in the repository and in the demo unchanged. v2 is an
attempt to improve it under rules fixed here, before any v2 number exists.

## Test sets

| Test set | Status for v2 | Use |
|---|---|---|
| HMC Sleep Staging Database (PhysioNet, CC BY 4.0, 148 recordings, AASM 2.4 scoring) | never used for anything | primary external test for staging v2 |
| UCDDB (25 recordings) | already used once by v1 | secondary, reported as "second look" |

HMC is checked with `scripts/check_data.py` before any use. No HMC data is
used for training, normalization across records, feature selection,
thresholds or model choice. AASM N1 + N2 = light, N3 = deep.

## Selection (training data only)

Candidates are compared only on the training databases:

- **Cross-dataset CV (primary):** two folds, (train CAP -> evaluate MIT-BIH
  slpdb) and (train slpdb -> evaluate CAP). The score is the mean of the two
  folds. This is a better proxy for transfer to an unseen hospital than
  random subject folds, which mixed databases in v1.
- **Subject-grouped 5-fold CV** within the training set (secondary, reported).

Selection metric for staging, fixed now: mean cross-dataset sleep/wake epoch
kappa. A candidate is only accepted if its mean cross-dataset total-sleep-time
MAE is not worse than v1's by more than 10 %. Ties (difference < 0.01 kappa)
go to the simpler candidate.

Apnea: candidates are compared by cross-dataset CV between Apnea-ECG and
event-derived minute labels from MIT-BIH slpdb (H, HA, OA, X, CA, CAA, ...
annotations), metric: minute kappa, with apnea-minute MAE as a guard (same
10 % rule).

## Candidate changes (planned; list may only shrink, not be tuned on test)

1. More CAP training records (the remaining non-NFLE and NFLE records).
2. Longer-context features (5-10 min windows, cyclic variation of heart rate).
3. Temporal smoothing of epoch probabilities (HMM / Viterbi over stages).
4. Ensembles of gradient boosting and logistic regression.
5. Class weighting and threshold rules chosen on out-of-fold training data.
6. Apnea: add MIT-BIH event-derived minute labels to training.

## Freezing and the single test run

1. Pick v2 by the selection rule above. Write the choice and its training
   scores to `results/v2_selection.json`.
2. Commit and tag `v2-frozen` before predicting any test set.
3. Predict HMC and UCDDB once. Report v1 and v2 side by side, with v2 on
   UCDDB labeled as a second look at that test set.
4. If v2 is worse on HMC than v1 would be, report that as well (v1 is also
   scored on HMC for the comparison, once, at the same time).

Deadline for freezing v2: 2026-11-27. Three- and four-stage work stops on
2026-11-29.
