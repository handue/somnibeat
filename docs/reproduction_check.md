# Clean-clone reproduction check

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
