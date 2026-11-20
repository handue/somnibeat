"""Narration with Kokoro-82M (hexgrad, Apache-2.0), offline from the local Hugging Face cache.

Reads tools/video/out/narration.json, writes one WAV per beat (24 kHz mono) to
tools/video/out/narr/<id>.wav and per-sentence timings to out/narr/timings.json
(used for subtitles).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from kokoro import KPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/video/out"
VOICE = "af_heart"
SPEED = 0.92
SR = 24000
GAP_S = 0.35  # pause between sentences
# Pronunciation fixes, applied only to the speech input (captions keep the plain text).
PRONOUNCE = {"Somnibeat": "[Somnibeat](/sˈɑmnibˌit/)"}


def speech_text(s: str) -> str:
    for word, markup in PRONOUNCE.items():
        s = s.replace(word, markup)
    return s


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def main() -> None:
    beats = json.loads((OUT / "narration.json").read_text())["beats"]
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    (OUT / "narr").mkdir(parents=True, exist_ok=True)
    timings = {}
    for b in beats:
        parts, cues, t = [], [], 0.0
        for s in sentences(b["text"]):
            audio = np.concatenate([np.asarray(a, dtype=np.float32) for _, _, a in pipe(speech_text(s), voice=VOICE, speed=SPEED)])
            dur = len(audio) / SR
            cues.append({"text": s, "start": round(t, 3), "end": round(t + dur, 3)})
            parts += [audio, np.zeros(int(GAP_S * SR), dtype=np.float32)]
            t += dur + GAP_S
        wav = np.concatenate(parts[:-1])
        sf.write(OUT / "narr" / f"{b['id']}.wav", wav, SR)
        timings[b["id"]] = {"duration": round(len(wav) / SR, 3), "cues": cues}
        print(b["id"], f"{len(wav) / SR:.1f}s")
    (OUT / "narr" / "timings.json").write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
