# Sources and licenses

All datasets are public on PhysioNet and need no account. License text was
checked on each PhysioNet project page on 2026-11-18.

| Dataset | Version / DOI | License (as stated on PhysioNet) | Used for |
|---|---|---|---|
| St. Vincent's University Hospital / University College Dublin Sleep Apnea Database (UCDDB) | 1.0.0, doi:10.13026/C26C7D | Open Data Commons Attribution License v1.0 | External validation (staging, apnea) |
| CAP Sleep Database (capslpdb) | 1.0.0 | Open Data Commons Attribution License v1.0 | Staging training |
| MIT-BIH Polysomnographic Database (slpdb) | 1.0.0, doi:10.13026/C23K5S | Open Data Commons Attribution License v1.0 | Staging training |
| Apnea-ECG Database (apnea-ecg) | 1.0.0 | Open Data Commons Attribution License v1.0 | Apnea training |

License: https://physionet.org/content/<project>/1.0.0/ links to
https://opendatacommons.org/licenses/by/1-0/

## Citations

- PhysioNet platform: Pollard T, Moody BE, et al. "PhysioNet as a global platform for biomedical research." *Nature Health* (2026). https://doi.org/10.1038/s44360-026-00096-z
- MIT-BIH Polysomnographic Database: Ichimaru Y, Moody GB. "Development of the polysomnographic database on CD-ROM." *Psychiatry and Clinical Neurosciences* 53:175-177 (1999).
- Apnea-ECG Database: Penzel T, Moody GB, Mark RG, Goldberger AL, Peter JH. "The Apnea-ECG Database." *Computers in Cardiology* 27:255-258 (2000).
- CAP Sleep Database: Terzano MG, et al. "Atlas, rules, and recording techniques for the scoring of cyclic alternating pattern (CAP) in human sleep." *Sleep Medicine* 2:537-553 (2001).
- UCDDB: no original publication is listed on the PhysioNet page; cite the dataset DOI and the PhysioNet platform paper.

## Population notes (from the dataset pages)

- UCDDB: 25 adults (21 male, 4 female), suspected sleep-disordered breathing, age 50 ± 10 (28–68), AHI 1.7–90.9.
- slpdb: 18 records from 16 male subjects (slp01a/b and slp02a/b are the same two subjects), mean age 40 (32–56).
- CAP: 49 records used: 15 healthy controls (n1–n15; n16 has no ECG channel) and 34 patients (SDB, insomnia, PLM, narcolepsy, RBD, bruxism). NFLE records are not used.
- Apnea-ECG: 35 learning-set records (a01–a20, b01–b05, c01–c10).

## Code and models

- All Somnibeat code (Python package, scripts, web demo, tests) was written for
  this project and is released under the MIT License (`LICENSE`).
- No pretrained or third-party models are used. Every model in
  `results/models/` and `web/data/models.json` is trained from the datasets
  above by `scripts/train_eval.py`.
- The web demo has no third-party JavaScript, CSS or fonts and makes no
  network requests outside its own files.
- The R-peak detector follows the general energy-envelope approach of
  Pan J, Tompkins WJ. "A real-time QRS detection algorithm." *IEEE Trans Biomed
  Eng* 32(3):230-236 (1985); the implementation is our own.
- Lomb-Scargle periodogram: Lomb NR, *Astrophys Space Sci* 39:447-462 (1976);
  Scargle JD, *Astrophys J* 263:835-853 (1982). DFA: Peng C-K et al., *Chaos*
  5:82-87 (1995). Implementations are our own.

### Python libraries (pinned in `requirements.txt`)

License strings are read from each installed package's metadata.

| Package | Version | License | Home page |
|---|---|---|---|
| numpy | 2.5.5 | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org |
| scipy | 1.18.1 | BSD License | https://scipy.org |
| pandas | 3.1.1 | BSD-3-Clause | https://pandas.pydata.org |
| wfdb | 4.3.1 | MIT License | https://github.com/MIT-LCP/wfdb-python |
| pyEDFlib | 0.1.42 | BSD License | https://github.com/holgern/pyedflib |
| scikit-learn | 1.9.1 | BSD-3-Clause | https://scikit-learn.org |
| matplotlib | 3.11.2 | Matplotlib License (PSF-based) | https://matplotlib.org |
| pytest | 9.2.0 | MIT | https://docs.pytest.org |

### Build and hosting

- GitHub Actions (`actions/checkout`, `configure-pages`, `upload-pages-artifact`,
  `deploy-pages`, all MIT) deploy `web/` to GitHub Pages.
- Node.js runs the JavaScript side of the parity tests; no npm packages are used.

## Brand assets

`assets/somnibeat-mark.svg` and `assets/somnibeat-lockup.svg` were supplied by the project author.

## Demo samples

The web demo (`web/data/samples/`) ships three RR-interval files, copied
unmodified from `data/cache/rr/` by `scripts/export_web.py`. They are derived
data (RR intervals from R-peaks detected by this project's detector) and are
redistributed under the Open Data Commons Attribution License v1.0 of their
source databases. Attribution and the selection rules are also stored in
`web/data/samples.json`; the export stops if a rule no longer picks the
listed record.

| Sample id | File | Source record | Selection rule |
|---|---|---|---|
| `ordinary` | `ucddb018.csv` | UCDDB `ucddb018`, https://physionet.org/content/ucddb/1.0.0/ | UCDDB record with the fewest reference apnea minutes (`results/apnea.json`) |
| `apnea` | `ucddb025.csv` | UCDDB `ucddb025`, https://physionet.org/content/ucddb/1.0.0/ | UCDDB record with the most reference apnea minutes (`results/apnea.json`) |
| `noisy` | `apnea-ecg-c04.csv` | Apnea-ECG `c04`, https://physionet.org/content/apnea-ecg/1.0.0/ | Apnea-ECG record with the lowest fraction of valid RR intervals (`data/cache/record_summary.json`) |

- Both UCDDB samples are part of the external test set, so the demo shows
  nights the models never saw during training.
- `c04` is an Apnea-ECG training record and has no sleep-stage reference.
- License: Open Data Commons Attribution License v1.0,
  https://opendatacommons.org/licenses/by/1-0/
