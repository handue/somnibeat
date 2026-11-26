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

    sel = json.loads((ROOT / "results/v2/selection_cap106.json").read_text())
    n_train = sum(len(v) for v in sel["records"].values())  # recordings the frozen v2 staging model was trained on
    n_beats = len((ROOT / "web/data/samples/ucddb015.csv").read_text().split()) - 1
    beats_txt = f"about {words(round(n_beats / 1000))} thousand heartbeats"
    facts = json.loads((ROOT / "tools/video/facts.json").read_text())  # external facts with sources
    lab = facts["sleep_lab_night_usd_average"]
    lab_txt = f"{words(lab // 1000)} thousand dollars" if lab % 1000 == 0 else f"{lab} dollars"
    holter_txt = f"{words(facts['holter_cash_price_usd'][0])} dollars"
    strap_txt = words(facts["ecg_chest_strap_usd_about"])

    parity = json.loads((ROOT / "results/ecg_parity.json").read_text())["summary"]
    n_rec_parity = parity["recordings"]
    n_diff = parity["intervals"] - parity["identical_intervals"]
    millions = parity["intervals"] / 1e6
    millions_txt = f"more than {words(int(millions))} million"

    beats = [
        ("N0", "intro", "Somnibeat. An overnight sleep report, from your heart's ECG."),
        ("N1", "start", "A night in a sleep lab costs about "
                        f"{lab_txt} in the United States, and the wait can take months. "
                        "Somnibeat is for anyone with an overnight ECG, from a Holter monitor or a chest strap, "
                        "and for researchers with ECG data."),
        ("N2", "report", "Here is a real overnight ECG from an open research dataset, the kind of file a sleep lab "
                         "or Holter recorder gives you. We drop it onto the page as it is. "
                         f"Right in the browser, Somnibeat finds {beats_txt}, and in seconds the report appears: "
                         "total sleep time, sleep efficiency, wake bouts, and minutes with suspected sleep apnea. "
                         "Nothing is sent to a server."),
        ("N3", "compare", "The AI behind it is a set of machine-learning models that read heart-rate variability, "
                          f"trained on {words(n_train)} overnight recordings scored by sleep experts. "
                          "They label every thirty seconds as wake, light, deep, or REM sleep. "
                          "For public test nights, the experts' own scoring sits right below."),
        ("NE", "ecg", "An ECG records the heart's electrical signal: every beat is a sharp spike, "
                      "and the time between spikes is the heartbeat interval. "
                      f"At home, a Holter monitor costs from about {holter_txt}, "
                      f"and an ECG chest strap about {strap_txt}. "
                      "With Somnibeat, a night recorded this way becomes a sleep report."),
        ("NH", "how", "To use it, record your heart overnight, then drop the file on the page: "
                      "the EDF file from the recorder, or the RR export from a chest strap app. "
                      "Somnibeat finds every heartbeat, and its AI reads the night thirty seconds at a time."),
        ("N4", "apnea", "On a night with many breathing pauses, suspected apnea minutes appear as red marks "
                        "under the chart, right next to the experts' own scoring."),
        ("N5", "noisy", "Somnibeat checks every single beat. When a recording is noisy, the report tells you, "
                        "and clearly marks the stretches it could not read."),
        ("NF", "fail", "And when a file can't be used, it says why. Here the times are in milliseconds instead of seconds, "
                       "so Somnibeat stops and explains, instead of making a wrong report."),
        ("N6", "trust", f"Every number comes with its measured accuracy, from {words(n_primary)} nights "
                        "in a sleep clinic that the model had never seen. Version two agrees more closely "
                        "with the sleep experts than version one, and the page shows exactly how much."),
        ("NL", "limits", "Somnibeat is a research prototype, not a diagnosis. Its accuracy was measured on clinical ECG "
                         "recordings, not yet on chest straps, and the three- and four-stage views are experimental."),
        ("N8", "outro", "Somnibeat is open source. Try it with your own file, or a sample night, at the address on screen."),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"source": {"results_json": "web/data/results.json", "hmc_primary_records": n_primary,
                       "ucddb018_leading_wake_hours": round(lead_h, 3),
                       "v2_training_recordings": n_train, "ucddb015_beats": n_beats,
                       "ecg_parity": "results/ecg_parity.json"},
            "beats": [{"id": i, "scene": s, "text": t} for i, s, t in beats]}
    (OUT / "narration.json").write_text(json.dumps(data, indent=2))
    for b in data["beats"]:
        print(b["id"], len(b["text"].split()), "words:", b["text"])


if __name__ == "__main__":
    main()
