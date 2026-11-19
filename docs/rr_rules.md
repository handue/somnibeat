# RR series: detection, artifact and missing-beat rules

Somnibeat's model only ever sees RR intervals. Raw ECG is used offline to
build training and validation data; the browser demo receives an RR CSV.

## 1. R-peak detection (offline only, `somnibeat/rpeak.py`)

Energy-envelope detector in the Pan-Tompkins family, run at each dataset's
native sampling rate (UCDDB 128 Hz, CAP 128-512 Hz, slpdb 250 Hz,
Apnea-ECG 100 Hz):

1. Zero-phase Butterworth band-pass, 5-25 Hz (upper edge capped at 0.45 fs).
2. Squared first difference, smoothed with a centered 120 ms moving average.
3. Adaptive threshold: 0.3 x running median (5 blocks) of the 98th percentile
   of the envelope in 2 s blocks, floored at 0.25 x the recording median of
   those values. Blocks below 0.1 x the recording median (lead-off, flat
   line) produce no beats, so those epochs get low coverage.
4. Envelope peaks with a 250 ms refractory period above the threshold.
5. Search-back in gaps longer than 1.66 x the running median RR, at half
   threshold, repeated (up to 5 passes) until no gap gains a beat.
6. Relocation to the extremum of the band-passed ECG within +-75 ms using the
   recording's dominant polarity, then parabolic sub-sample interpolation.

Validation numbers are in `results/rpeak_validation.json`
(`make rpeaks`). Cached beats store the SHA-256 of `rpeak.py` and are
recomputed whenever the detector changes.

## 2. RR construction (`somnibeat/rr.py`, mirrored in JavaScript)

- `rr_i = t_i - t_(i-1)`, stamped at the closing beat time `t_i`.
- Values are rounded to 0.0001 s (time) and 0.01 ms (RR) when written to CSV;
  features are computed from the rounded values so Python and the browser
  see the same numbers.

## 3. Artifact and missing-beat rules

Applied in order; parameters in `config/default.toml` `[rr]`.

| Rule | Condition | Action | Catches |
|---|---|---|---|
| R1 range | RR < 300 ms or RR > 2000 ms | invalid | doubled detections, long dropouts |
| R2 local median | \|RR - m\| / m > 0.30, m = median of R1-valid RR in a centered window of 11 intervals | invalid | one missed beat (~2x), one extra beat (~0.5x), ectopic beats |
| R3 no interpolation | - | invalid intervals are dropped, never filled | avoids inventing heart-rate variability |

- Successive-difference features (RMSSD, pNN50) only use pairs of adjacent
  intervals that are both valid, so a removed interval never creates a fake
  large difference.
- Coverage of a window = sum of valid RR / window length. Epochs with
  coverage below 0.70 are flagged `low_quality`: they are excluded from model
  fitting but are still predicted and still evaluated (the reported
  numbers include noisy epochs).
- Long gaps need no separate rule: an interval longer than 2000 ms is
  invalid (R1) and simply lowers the coverage of its windows.
- Spectral and DFA features need at least 32 valid intervals and 50 %
  coverage of their 300 s window, otherwise they are missing (NaN).
