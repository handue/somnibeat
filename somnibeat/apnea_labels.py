"""Per-minute apnea labels for MIT-BIH slpdb, derived from the .st epoch annotations.

Source facts (data/raw/slpdb/README.st, checked by scripts/v2_select_apnea.py --check and written
to results/v2/slpdb_apnea_check.json):
- Every .st annotation marks one 30 s scoring epoch. Its aux string is the sleep stage token
  (W, 1, 2, 3, 4, R, MT, M) followed by zero or more event tokens:
  H hypopnea, HA hypopnea with arousal, OA obstructive apnea, X obstructive apnea with arousal,
  CA central apnea, CAA central apnea with arousal, L / LA leg movements (without / with arousal;
  the files also contain LL, LLL, LLLL), A unspecified arousal, MT movement time.
- Event tokens carry no onset or duration: they only say that the event occurs in that epoch.
- Annotation times sit on a 30 s grid from the start of the recording (some records start
  scoring later, e.g. slp16 at 780 s), so epoch g = round(time_s / 30).

Minute rule (v2, comparable to the UCDDB rule "any apnea or hypopnea in the minute"):
- Minute m covers [60 m, 60 m + 60) s, i.e. grid epochs 2m and 2m+1. It is the same 60 s grid
  from 0 that Apnea-ECG and the v1 feature code use.
- apnea = 1 if any of H, HA, OA, X, CA, CAA occurs in either epoch.
- apnea = 0 if both epochs are annotated and neither has one of those tokens. Leg movement,
  arousal-only and movement-time tokens do not count.
- apnea = -1 (unlabeled) otherwise: a minute with a missing epoch and no respiratory token
  (outside the scored span).
Unknown event tokens raise an error instead of being silently ignored.

How this differs from the other apnea references:
- Apnea-ECG: an expert labeled every minute (from respiration signals) as apnea/hypopnea or
  normal; the label is a per-minute judgement, not derived from events.
- UCDDB: events have an onset and a duration; a minute is apnea if any APNEA-* or HYP-* event
  overlaps it (POSSIBLE excluded).
- slpdb (this module): only per-epoch presence markers. An event that spans an epoch boundary
  may be marked in one epoch only, so a minute can be missed (labels lean towards fewer apnea
  minutes than an onset+duration rule), while a 10 s event anywhere in a 60 s minute makes the
  whole minute apnea, as in the other two rules. Hypopnea criteria of the 1990s scoring are
  not the same as AASM criteria used for UCDDB.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

EPOCH_S = 30.0
MINUTE_S = 60.0
STAGE_TOKENS = frozenset({"W", "1", "2", "3", "4", "R", "MT", "M"})
RESP_TOKENS = frozenset({"H", "HA", "OA", "X", "CA", "CAA"})
OTHER_TOKENS = frozenset({"L", "LA", "LL", "LLL", "LLLL", "A", "MT"})
KNOWN_EVENT_TOKENS = RESP_TOKENS | OTHER_TOKENS


def parse_aux(aux: str) -> tuple[str | None, list[str]]:
    """Split one .st aux string into (stage token or None, event tokens)."""
    tok = aux.replace("\x00", "").split()
    if not tok:
        return None, []
    if tok[0] in STAGE_TOKENS:
        return tok[0], tok[1:]
    return None, tok


def epoch_events(times_s, aux_notes) -> dict[int, list[str]]:
    """{grid epoch index: event tokens} for every annotated epoch (empty list = no events)."""
    out: dict[int, list[str]] = {}
    for t, aux in zip(times_s, aux_notes):
        g = int(round(float(t) / EPOCH_S))
        if abs(float(t) - g * EPOCH_S) > 1.0:
            raise ValueError(f"annotation at {t} s is not on the 30 s grid")
        _, ev = parse_aux(aux)
        unknown = [e for e in ev if e not in KNOWN_EVENT_TOKENS]
        if unknown:
            raise ValueError(f"unknown event tokens {unknown} at {t} s")
        if g in out:
            raise ValueError(f"two annotations in grid epoch {g}")
        out[g] = ev
    return out


def minute_labels(events: dict[int, list[str]], n_minutes: int) -> np.ndarray:
    """Apply the minute rule (module docstring) to minutes 0..n_minutes-1."""
    lab = np.full(n_minutes, -1, dtype=int)
    for m in range(n_minutes):
        e0, e1 = events.get(2 * m), events.get(2 * m + 1)
        resp = any(tok in RESP_TOKENS for e in (e0, e1) if e is not None for tok in e)
        if resp:
            lab[m] = 1
        elif e0 is not None and e1 is not None:
            lab[m] = 0
    return lab


def read_slpdb_st(root: Path, rid: str) -> tuple[np.ndarray, list[str], float, int]:
    """(annotation times in s, aux strings, fs, signal length in samples) of one slpdb record."""
    import wfdb

    ann = wfdb.rdann(str(Path(root) / rid), "st")
    hdr = wfdb.rdheader(str(Path(root) / rid))
    fs = float(hdr.fs)
    return np.asarray(ann.sample, dtype=float) / fs, list(ann.aux_note), fs, int(hdr.sig_len)


def slpdb_minute_labels(root: Path, rid: str) -> np.ndarray:
    """Minute labels on the 60 s grid from 0 with floor(duration / 60) minutes (as Apnea-ECG)."""
    t, aux, fs, n = read_slpdb_st(root, rid)
    n_min = int(n / fs // MINUTE_S)
    return minute_labels(epoch_events(t, aux), n_min)


def record_facts(root: Path, rid: str) -> dict:
    """Plain facts about one record's annotations (no model, no features)."""
    t, aux, fs, n = read_slpdb_st(root, rid)
    ev = epoch_events(t, aux)
    tok_count = Counter(tok for e in ev.values() for tok in e)
    epochs_with = Counter(tok for e in ev.values() for tok in set(e))
    resp_epochs = sorted(g for g, e in ev.items() if any(x in RESP_TOKENS for x in e))
    runs, run = [], 0
    for i, g in enumerate(resp_epochs):
        run = run + 1 if i and g == resp_epochs[i - 1] + 1 else 1
        if i == len(resp_epochs) - 1 or resp_epochs[i + 1] != g + 1:
            runs.append(run)
    stages = Counter(parse_aux(a)[0] or "" for a in aux)
    lab = slpdb_minute_labels(root, rid)
    return {
        "fs": fs,
        "duration_s": n / fs,
        "n_annotations": len(aux),
        "first_annotation_s": float(t[0]) if len(t) else None,
        "last_annotation_s": float(t[-1]) if len(t) else None,
        "max_offset_from_30s_grid_s": float(np.max(np.abs(t - np.round(t / EPOCH_S) * EPOCH_S))) if len(t) else None,
        "stage_token_counts": dict(sorted(stages.items())),
        "event_token_counts": dict(sorted(tok_count.items())),
        "epochs_with_token": dict(sorted(epochs_with.items())),
        "max_repeats_of_a_token_in_one_epoch": max((max(Counter(e).values()) for e in ev.values() if e), default=0),
        "epochs_with_respiratory_token": len(resp_epochs),
        "respiratory_epoch_run_lengths": dict(sorted(Counter(runs).items())),
        "minutes_total": int(len(lab)),
        "minutes_apnea": int((lab == 1).sum()),
        "minutes_normal": int((lab == 0).sum()),
        "minutes_unlabeled": int((lab == -1).sum()),
    }
