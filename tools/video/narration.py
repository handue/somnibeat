"""Build the demo-video narration from results JSON (no hand-typed result numbers).

Writes tools/video/out/narration.json: a list of beats {id, scene, text}.
Numbers spoken in the narration are read from web/data/results.json and the
sample reference files, then spelled out for the text-to-speech voice.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/video/out"

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
        "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + words(rest))
    raise ValueError(n)


def leading_wake_hours(reference: dict) -> float:
    """Hours of expert-scored wake before the first scored sleep epoch (sleep_wake mode)."""
    stages = reference["stages"]["sleep_wake"] if isinstance(reference["stages"], dict) else reference["stages"]
    n = 0
    for s in stages:
        if s == "wake":
            n += 1
        elif s is None:
            continue
        else:
            break
    return n * 30 / 3600


def main() -> None:
    res = json.loads((ROOT / "web/data/results.json").read_text())
    hmc = res["v2"]["test_sets"]["hmc"]
    n_primary = int(hmc["records_primary"])
    ref = json.loads((ROOT / "web/data/samples/ucddb018_reference.json").read_text())
    lead_h = leading_wake_hours(ref)
    lead_txt = "almost two hours" if 1.5 <= lead_h < 2 else f"about {words(round(lead_h))} hours"

    sens = float(res["v2"]["apnea_ucddb"]["primary"]["v2"]["metrics"]["sensitivity"]["point"])
    sens_txt = "fewer than half" if sens < 0.5 else "only some"

    beats = [
        ("N0", "intro", "Somnibeat. An overnight sleep report, from heartbeat intervals alone."),
        ("N1", "start", "A night in a sleep lab is the gold standard, but it is costly, and hard to get. "
                        "Many people already have a much simpler signal: the time between heartbeats."),
        ("N2", "report", "Load one night, and Somnibeat estimates total sleep time, sleep efficiency, wake bouts, "
                         "and minutes with suspected sleep apnea. Everything is computed in your browser. "
                         "The file never leaves your device."),
        ("N3", "compare", "For public test nights, the estimate sits right above what sleep experts scored "
                          "from the full polysomnography, so you can see where it agrees, and where it does not."),
        ("N4", "failure", f"It is not always right. On this night, the person lay awake for {lead_txt} "
                          "before falling asleep, and the model missed it. We show that night too."),
        ("N5", "apnea", "On a night with many breathing pauses, suspected apnea minutes appear as red marks "
                        f"under the chart. It is a screening hint, not a diagnosis: on the test nights, "
                        f"the model found {sens_txt} of the apnea minutes the experts scored."),
        ("N6", "noisy", "Real recordings are messy. When many beats are missing, the report warns you, "
                        "and hatches the stretches where the data are too thin to trust."),
        ("N7", "trust", f"Every number comes with its measured error, on {words(n_primary)} nights from a sleep clinic "
                        "that the model never saw. Version two agrees better with the experts, thirty seconds at a time. "
                        "But its total sleep time error did not improve, and the page says so."),
        ("N8", "method", "We fixed the evaluation before training, chose models on training data only, "
                         "froze them, and tested once. One command rebuilds every number, byte for byte."),
        ("N9", "harm", "Who could be harmed? Someone falsely reassured, someone needlessly worried, "
                       "and the people our data barely cover. Most recordings come from sleep clinics, "
                       "and home wearables have not been tested."),
        ("N10", "outro", "Somnibeat is a research prototype, not a diagnostic tool."),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"source": {"results_json": "web/data/results.json", "hmc_primary_records": n_primary,
                       "ucddb018_leading_wake_hours": round(lead_h, 3),
                       "ucddb_v2_apnea_sensitivity": round(sens, 3)},
            "beats": [{"id": i, "scene": s, "text": t} for i, s, t in beats]}
    (OUT / "narration.json").write_text(json.dumps(data, indent=2))
    for b in data["beats"]:
        print(b["id"], len(b["text"].split()), "words:", b["text"])


if __name__ == "__main__":
    main()
