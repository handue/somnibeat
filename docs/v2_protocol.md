# v2 protocol (written 2026-11-19, before any v2 experiment)

v1 is the pre-registered first result (`results/*.json` at git tag
`v1-results`). It stays in the repository and in the demo unchanged. v2 is an
attempt to improve it under rules fixed here, before any v2 number exists.

## Test sets

| Test set | Status for v2 | Use |
|---|---|---|
| HMC Sleep Staging Database (PhysioNet, CC BY 4.0, 151 recordings in release 1.1, AASM 2.4 scoring) | never used for anything | primary external test for staging v2 |
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

## Clarification for candidate 1 (added 2026-11-19, before the extra CAP data was downloaded or scored)

Adding CAP records changes only the training side of fold A (CAP -> slpdb);
fold B (slpdb -> CAP) would train on the same data and merely evaluate on a
larger CAP set, which is not comparable to v1. Candidate 1 is therefore judged
on fold A only, against the v1 reference's fold A score, with the same kappa
criterion and TST-MAE guard. Its subject-grouped CV is reported but not used
for the decision. If candidate 1 is accepted, it is combined with the best
candidate from the other changes and the combination is re-scored on fold A
the same way.

## HMC facts and a data-quality rule (added 2026-11-19, before any HMC prediction)

- The downloaded HMC 1.1 release contains 151 recordings (RECORDS, SHA256SUMS
  and the demographics file agree), not the 148 implied on the project page.
  All 151 pass SHA-256 and the stage/ECG alignment checks
  (`results/v2/hmc_check.json`).
- Two recordings have almost no usable ECG (SN098: essentially none; SN036:
  most 30 s blocks flat or clipped). The rule below is fixed now, uses only
  the RR series (never labels or predictions), and applies to every HMC and
  UCDDB record alike:
  - **Primary analysis:** records where fewer than 50 % of epochs have
    adequate RR coverage (`low_quality == 0`) are excluded. The number and IDs
    of excluded records are reported.
  - **Secondary analysis:** all records, including those, is reported too.
- No detector or RR-rule change is made because of HMC signal quality.

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
