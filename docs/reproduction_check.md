# Clean-clone reproduction check

## v1 (2026-11-19)

- Date: 2026-11-19
- Commit: `e0d2808` (tag `v1-results`), fresh `git clone` of https://github.com/handue/somnibeat into an empty directory
- Command: `make setup download` then `make reproduce` (raw PhysioNet downloads, SHA-256 verified)
- `make reproduce` (after downloads): 44:16.14 total wall time; tests: 41 passed in 178.56s (0:02:58)
- Working tree after the run: clean (every regenerated file is byte-identical to the committed one)

| File | SHA-256 (clone) | SHA-256 (committed) | Match |
|---|---|---|---|
| `results/sleep_wake.json` | `c91a99c9bd03f131` | `c91a99c9bd03f131` | yes |
| `results/wake_nrem_rem.json` | `bedafa6228c649b0` | `bedafa6228c649b0` | yes |
| `results/four_class.json` | `b7157b7b2d989866` | `b7157b7b2d989866` | yes |
| `results/apnea.json` | `987ebc8bdb086794` | `987ebc8bdb086794` | yes |
| `results/rpeak_validation.json` | `27cba216b957b49b` | `27cba216b957b49b` | yes |
| `results/data_check.json` | `136a180a8cc20797` | `136a180a8cc20797` | yes |
| `web/data/results.json` | `e83ee1ab1f2a03b9` | `e83ee1ab1f2a03b9` | yes |
| `web/data/models.json` | `9f320b6206dcf37d` | `9f320b6206dcf37d` | yes |
| `web/data/samples.json` | `a86af5a825bd146b` | `a86af5a825bd146b` | yes |
| `README.md` | `18f29c892f77bda6` | `18f29c892f77bda6` | yes |
| `docs/WRITEUP.md` | `678ca2cdefb75ebe` | `678ca2cdefb75ebe` | yes |

Tested on macOS (Apple silicon), Python 3.14.7, Node.js 24.20.0.

## v1 + v2 (2026-11-20)

- Fresh `git clone` of https://github.com/handue/somnibeat into an empty directory; final state at
  commit `75de7e6`. Command: `make reproduce` (v1, then `make v2`, then web, docs, parity and tests),
  with every raw file downloaded from PhysioNet and SHA-256 verified (~37 GB for v2).
- Result: every regenerated file below is byte-identical to the committed one, the working tree is
  clean after the run, and the tests pass (185 passed).
- Wall time on this machine: v1 about 45 min; v2 about 10 h (downloads and the model-selection grid).

What the runs found and what was changed (in order; no result number changed):

1. A v2 parity test needs the trained v2 models, which are git-ignored and are rebuilt by `make v2`,
   so the v1 stage stopped before v2 ran. The test now skips until `make v2` has built them; the
   final test step after v2 runs it (`b5237e8`).
2. PhysioNet answered one request with HTTP 502. The downloader now retries transient server errors
   with backoff (`10d6df1`).
3. The HMC data check stopped `make v2`: its beat-rate check failed on the two recordings already
   documented in `docs/v2_protocol.md` as having almost no usable ECG (SN036, SN098). The check now
   expects exactly those two and fails on any other. The same commit removed values that change on
   every run and so broke byte-identical output: a timestamp in `hmc_check.json`, wall-clock
   seconds in the selection files, and the checkout's HEAD commit in `test_results.json`
   (it now records the `v2-frozen` commit, which the freeze gate already requires the frozen files
   to match) (`d914bcd`).
4. The rebuilt `selection_cap106.json` contained subject-grouped CV numbers that the committed one
   did not: the committed file was made with `--no-subject-cv` (cross-dataset evaluation only),
   but the Makefile omitted the flag. The cross-dataset numbers, which drove the decision, were
   identical. The Makefile now runs the command as it was run. `models_v2.json` no longer stores
   SHA-256 hashes of the git-ignored pickles: pickle bytes include the thread count sklearn used
   when fitting (8 in the original run, 10 in the clone run), while every exported parameter was
   identical (`75de7e6`). After this commit only the cap106 selection step and the steps after it
   (decision, web, docs, parity, tests) were rerun in the clone; all earlier outputs come from the
   full run and were already identical.

| File | SHA-256 (clone) | SHA-256 (committed) | Match |
|---|---|---|---|
| `results/sleep_wake.json` | `c91a99c9bd03f131` | `c91a99c9bd03f131` | yes |
| `results/wake_nrem_rem.json` | `bedafa6228c649b0` | `bedafa6228c649b0` | yes |
| `results/four_class.json` | `b7157b7b2d989866` | `b7157b7b2d989866` | yes |
| `results/apnea.json` | `987ebc8bdb086794` | `987ebc8bdb086794` | yes |
| `results/rpeak_validation.json` | `27cba216b957b49b` | `27cba216b957b49b` | yes |
| `results/data_check.json` | `f3e25037f987d1ce` | `f3e25037f987d1ce` | yes |
| `results/v2/data_check.json` | `fa9d09ccbca596dc` | `fa9d09ccbca596dc` | yes |
| `results/v2/hmc_check.json` | `22721f273bdec93c` | `22721f273bdec93c` | yes |
| `results/v2/slpdb_apnea_check.json` | `455e21681f233522` | `455e21681f233522` | yes |
| `results/v2/selection_staging.json` | `5eee93794eaf1188` | `5eee93794eaf1188` | yes |
| `results/v2/selection_cap106.json` | `93072f5eca449fbf` | `93072f5eca449fbf` | yes |
| `results/v2/selection_apnea.json` | `ef63eaec908b866a` | `ef63eaec908b866a` | yes |
| `results/v2/v2_selection.json` | `6aecfdc3336addf1` | `6aecfdc3336addf1` | yes |
| `results/v2/v2_final_spec.json` | `7d3a990b49fe9798` | `7d3a990b49fe9798` | yes |
| `results/v2/test_results.json` | `3272f0640cdcf45d` | `3272f0640cdcf45d` | yes |
| `web/data/results.json` | `f89e53caabecae9a` | `f89e53caabecae9a` | yes |
| `web/data/models.json` | `9f320b6206dcf37d` | `9f320b6206dcf37d` | yes |
| `web/data/models_v2.json` | `7626e06995680eb0` | `7626e06995680eb0` | yes |
| `web/data/samples.json` | `2f3ea1cfb61ce4d6` | `2f3ea1cfb61ce4d6` | yes |
| `README.md` | `d752a9f56407c416` | `d752a9f56407c416` | yes |
| `docs/WRITEUP.md` | `a564edbef8f4e8b2` | `a564edbef8f4e8b2` | yes |
| `docs/v2_selection.md` | `a933eaa54b3b7dbf` | `a933eaa54b3b7dbf` | yes |
| `docs/v2_final_spec.md` | `1a05c17bdacb3cad` | `1a05c17bdacb3cad` | yes |

Tested on macOS (Apple silicon), Python 3.14.7, Node.js 24.20.0.
