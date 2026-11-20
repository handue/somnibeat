"""Keep result files reproducible byte for byte across runs."""

from __future__ import annotations

# Wall-clock durations differ on every run; they are printed to the console but not written to results.
TIMING_KEYS = frozenset({"seconds", "fit_seconds", "total_seconds"})


def drop_timing(obj):
    """Return a copy of a JSON-like object without wall-clock timing keys (at any depth)."""
    if isinstance(obj, dict):
        return {k: drop_timing(v) for k, v in obj.items() if k not in TIMING_KEYS}
    if isinstance(obj, list):
        return [drop_timing(v) for v in obj]
    return obj
