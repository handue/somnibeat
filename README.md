<p align="center"><img src="assets/somnibeat-lockup.svg" alt="Somnibeat" width="480"></p>

# Somnibeat

**Research prototype · not a diagnostic tool.**

Somnibeat turns one night of heartbeat intervals (an RR-interval CSV) into an
overnight sleep report: total sleep time, sleep efficiency, wake bouts and
suspected sleep-apnea minutes, shown next to how far such estimates differ
from hospital sleep studies.

- Live demo: https://handue.github.io/somnibeat/ (everything runs in your browser; no file leaves the device)
- Evaluation results: [`results/`](results/) (all numbers on the demo are read from these files)
- Method: [`docs/evaluation.md`](docs/evaluation.md), [`docs/rr_rules.md`](docs/rr_rules.md)
- Data sources and licenses: [`SOURCES.md`](SOURCES.md)

Reproduce every number from the raw PhysioNet downloads:

```bash
make reproduce
```

A full README (setup, dependencies, configuration, evaluation) and the
written description are in progress.
