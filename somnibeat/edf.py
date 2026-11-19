"""Minimal EDF reader for a single channel.

Strict readers reject some CAP Sleep Database files (out-of-spec digital
min/max fields), so the fixed-width EDF header is parsed directly and only
the requested channel is decoded.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EdfHeader:
    start: dt.datetime
    labels: list[str]
    units: list[str]
    phys_min: np.ndarray
    phys_max: np.ndarray
    dig_min: np.ndarray
    dig_max: np.ndarray
    samples_per_record: np.ndarray
    n_records: int
    record_duration_s: float
    header_bytes: int

    @property
    def fs(self) -> np.ndarray:
        return self.samples_per_record / self.record_duration_s

    @property
    def duration_s(self) -> float:
        return self.n_records * self.record_duration_s


def read_header(path: Path) -> EdfHeader:
    path = Path(path)
    with open(path, "rb") as fh:
        h = fh.read(256).decode("ascii", "replace")
        ns = int(h[252:256])
        sh = fh.read(256 * ns).decode("ascii", "replace")
    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    fields, pos = [], 0
    for w in widths:
        fields.append([sh[pos + i * w: pos + (i + 1) * w].strip() for i in range(ns)])
        pos += w * ns
    labels, _, units, pmin, pmax, dmin, dmax, _, spr, _ = fields
    d, mo, y = (int(x) for x in re.split(r"[.:]", h[168:176]))
    hh, mi, ss = (int(x) for x in re.split(r"[.:]", h[176:184]))
    start = dt.datetime(2000 + y if y < 85 else 1900 + y, mo, d, hh, mi, ss)
    rec_dur = float(h[244:252])
    spr_a = np.array([int(x) for x in spr])
    header_bytes = 256 * (ns + 1)
    n_rec = int(h[236:244])
    from_size = (path.stat().st_size - header_bytes) // (2 * int(spr_a.sum()))
    if n_rec <= 0 or n_rec > from_size:  # header count missing or larger than the file holds
        n_rec = int(from_size)
    return EdfHeader(start, labels, units,
                     np.array([float(x) for x in pmin]), np.array([float(x) for x in pmax]),
                     np.array([float(x) for x in dmin]), np.array([float(x) for x in dmax]),
                     spr_a, n_rec, rec_dur, header_bytes)


def read_channel(path: Path, label_pattern: str) -> tuple[np.ndarray, float, EdfHeader, str]:
    """Return (signal in physical units, fs, header, matched label) for the first matching channel."""
    hdr = read_header(path)
    idx = [i for i, lab in enumerate(hdr.labels) if re.search(label_pattern, lab, re.I)]
    if not idx:
        raise KeyError(f"{path}: no channel matching {label_pattern!r} in {hdr.labels}")
    i = idx[0]
    total = int(hdr.samples_per_record.sum())
    raw = np.memmap(path, dtype="<i2", mode="r", offset=hdr.header_bytes,
                    shape=(hdr.n_records, total))
    start = int(hdr.samples_per_record[:i].sum())
    dig = np.asarray(raw[:, start:start + hdr.samples_per_record[i]], dtype=np.float64).ravel()
    span_d = hdr.dig_max[i] - hdr.dig_min[i]
    gain = (hdr.phys_max[i] - hdr.phys_min[i]) / span_d if span_d else 1.0
    x = (dig - hdr.dig_min[i]) * gain + hdr.phys_min[i]
    return x, float(hdr.fs[i]), hdr, hdr.labels[i]
