"""Assemble the Somnibeat demo video: title cards, recorded scenes, narration, captions.

Inputs (tools/video/out): narration.json, narr/*.wav + narr/timings.json (tts.py),
scenes/<name>/frames.json + JPEGs (record.py).
Outputs: out/somnibeat_demo.mp4 (1920x1080, 30 fps, H.264 + AAC) and
out/somnibeat_demo.srt (the same captions as a separate file).

Logos: assets/somnibeat-lockup.svg and assets/somnibeat-mark.svg are drawn
unmodified by Chrome onto the title cards.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/video/out"
TMP = OUT / "build"
W, H, FPS = 1920, 1080, 30
LEAD_IN, TAIL = 0.6, 1.0
ORDER = [("intro", "N0"), ("start", "N1"), ("ecg", "NE"), ("how", "NH"), ("report", "N2"), ("compare", "N3"), ("apnea", "N4"),
         ("noisy", "N5"), ("trust", "N6"), ("method", "N7"), ("outro", "N8")]
CARD_LEN = {"intro": 7.0, "outro": 8.0}
SCENE_FADE = 0.45  # each recorded scene fades in from and out to black (cards fade too)
FAST_TARGET_S = 2.5  # the in-browser computation is shown in this many seconds
FONT = "/System/Library/Fonts/Helvetica.ttc"
SITE_URL = "handue.github.io/somnibeat"
REPO_URL = "github.com/handue/somnibeat"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


# ----------------------------------------------------------------------------- cards (rendered by Chrome)

CARD_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:#000;width:1920px;height:1080px;overflow:hidden;
font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif;color:#F2F4F7}}
.wrap{{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:28px}}
.sub{{font-size:40px;color:#C9CED6;font-weight:500}}
.small{{font-size:28px;color:#969FAC}}
.pill{{border:1px solid #1A1F27;border-radius:999px;padding:12px 26px;font-size:26px;color:#969FAC}}
.url{{font-size:44px;color:#4FD1C5;font-weight:600;font-variant-numeric:tabular-nums}}
.credits{{position:absolute;bottom:250px;width:100%;text-align:center;font-size:22px;color:#969FAC;line-height:1.6}}
</style></head><body><div class="wrap">{body}</div>{credits}</body></html>"""


def render_cards() -> dict[str, Path]:
    lockup = (ROOT / "assets/somnibeat-lockup.svg").read_text()
    intro = CARD_HTML.format(body=f'<div style="width:1120px">{lockup.replace("width=\"640\" height=\"160\"", "width=\"1120\" height=\"280\"")}</div>'
                                  '<div class="sub">An overnight sleep report from heartbeat intervals</div>'
                                  '<div class="pill">Research prototype · not a diagnostic tool</div>', credits="")
    credits = ('<div class="credits">Data: UCDDB, CAP, MIT-BIH Polysomnographic, Apnea-ECG (PhysioNet, ODC-By 1.0) · '
               'HMC Sleep Staging (PhysioNet, CC BY 4.0)<br>Narration: Kokoro-82M text to speech · '
               'The file uploaded in this video is a public UCDDB research recording</div>')
    outro = CARD_HTML.format(body=f'<div style="width:880px">{lockup.replace("width=\"640\" height=\"160\"", "width=\"880\" height=\"220\"")}</div>'
                                  f'<div class="url">{SITE_URL}</div><div class="small">{REPO_URL}</div>'
                                  '<div class="pill">Research prototype · not a diagnostic tool</div>', credits=credits)
    paths = {}
    hows = [how_card(k) for k in range(4)]  # 0: steps dimmed, k: steps 1..k lit
    ecgs = [ecg_card(k) for k in range(3)]  # 0: trace, 1: + prices, 2: + Somnibeat line
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        for name, html in (("intro", intro), ("outro", outro), *((f"how{k}", h) for k, h in enumerate(hows)),
                           *((f"ecg{k}", h) for k, h in enumerate(ecgs))):
            pg.set_content(html)
            pg.wait_for_timeout(200)
            paths[name] = TMP / f"card_{name}.png"
            pg.screenshot(path=str(paths[name]))
        b.close()
    return paths


HOW_CSS = """<style>
.title{font-size:30px;color:#969FAC;letter-spacing:.08em;text-transform:uppercase}
.steps{display:flex;gap:28px;align-items:stretch}
.step{width:500px;background:#0A0C10;border:1px solid #1A1F27;border-radius:18px;padding:34px 34px 30px;box-sizing:border-box}
.step.dim{opacity:.22}
.n{font-size:22px;color:#4FD1C5;font-weight:600}
.h{font-size:38px;font-weight:600;margin:10px 0 18px}
.t{font-size:24px;line-height:1.5;color:#C9CED6}
pre{margin:18px 0 0;padding:16px 18px;background:#000;border:1px solid #1A1F27;border-radius:10px;
font:22px/1.55 ui-monospace,Menlo,monospace;color:#C9CED6}
pre b{color:#4FD1C5}
.chips{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0 18px}
.chip{padding:6px 14px;border-radius:999px;font-size:21px;color:#000;font-weight:600}
ul{margin:0;padding-left:26px;font-size:24px;line-height:1.6;color:#C9CED6}
.arrow{align-self:center;font-size:40px;color:#4FD1C5}
</style>"""


def how_card(lit: int) -> str:
    rows = (ROOT / "web/data/samples/ucddb015.csv").read_text().split()[:4]
    csv = "<b>" + rows[0] + "</b>\n" + "\n".join(rows[1:]) + "\n…"
    ecg = ('<svg width="300" height="70" viewBox="0 0 300 70"><polyline fill="none" stroke="#4FD1C5" stroke-width="3" '
           'points="0,40 40,40 50,34 58,40 70,40 78,8 86,62 94,40 130,40 140,34 148,40 160,40 168,8 176,62 184,40 '
           '220,40 230,34 238,40 250,40 258,8 266,62 274,40 300,40"/></svg>')
    steps = [
        ("1", "Record overnight", f'{ecg}<div class="t">An ECG through the night: a Holter monitor '
                                  'or an ECG chest strap.</div>'),
        ("2", "Convert to one CSV", '<div class="t"><code>ecg_to_rr.py</code> turns an ECG file (EDF) or an RR export '
                                    'into two columns:</div>'
                                    f'<pre>{csv}</pre>'),
        ("3", "AI report", '<div class="t">Every 30 seconds is labelled:</div><div class="chips">'
                           '<span class="chip" style="background:#F5A524">Wake</span>'
                           '<span class="chip" style="background:#A78BFA">REM</span>'
                           '<span class="chip" style="background:#5B7FA6">Light</span>'
                           '<span class="chip" style="background:#3B5BDB;color:#F2F4F7">Deep</span></div>'
                           '<ul><li>Total sleep time</li><li>Sleep efficiency</li><li>Wake bouts</li>'
                           '<li>Suspected sleep-apnea minutes</li></ul>'),
    ]
    cards = []
    for i, (n, h, body) in enumerate(steps):
        if i:
            cards.append('<div class="arrow">&rarr;</div>')
        cards.append(f'<div class="step{"" if i < lit else " dim"}"><div class="n">STEP {n}</div><div class="h">{h}</div>{body}</div>')
    html = CARD_HTML.format(body='<div class="title">How it works</div><div class="steps">' + "".join(cards) + "</div>"
                            '<div class="pill">The file is read in your browser and never sent to a server</div>',
                            credits="")
    return html.replace("</head>", HOW_CSS + "</head>")


ECG_CSS = """<style>
.grid{display:flex;gap:40px;align-items:stretch}
.panel{background:#0A0C10;border:1px solid #1A1F27;border-radius:18px;padding:34px 38px;box-sizing:border-box}
.k{font-size:22px;color:#4FD1C5;font-weight:600;letter-spacing:.06em}
.h{font-size:36px;font-weight:600;margin:8px 0 14px}
.t{font-size:23px;line-height:1.5;color:#C9CED6}
.row{display:grid;grid-template-columns:300px 260px 1fr;align-items:center;gap:18px;margin:16px 0}
.row .name{font-size:23px;color:#F2F4F7;line-height:1.3}.row .name span{display:block;font-size:18px;color:#969FAC}
.bar{height:22px;border-radius:6px;background:#5B7FA6}
.price{font-size:26px;font-weight:600;text-align:right;font-variant-numeric:tabular-nums}
.hide{visibility:hidden}
.note{font-size:24px;color:#4FD1C5;font-weight:600;margin-top:22px}
.foot{font-size:17px;color:#969FAC;margin-top:18px}
</style>"""


def ecg_card(stage: int) -> str:
    facts = json.loads((ROOT / "tools/video/facts.json").read_text())
    lab, (h_lo, h_hi), strap = (facts["sleep_lab_night_usd_average"], facts["holter_cash_price_usd"],
                                facts["ecg_chest_strap_usd_about"])
    # ECG trace with R peaks and one marked interval
    beats = [70, 270, 465, 665]
    pts = []
    for b in beats:
        pts += [(b - 70, 120), (b - 40, 112), (b - 30, 120), (b - 10, 120), (b, 30), (b + 9, 160), (b + 18, 120),
                (b + 55, 120), (b + 75, 106), (b + 95, 120)]
    poly = " ".join(f"{x},{y}" for x, y in pts) + " 760,120"
    svg = (f'<svg width="760" height="230" viewBox="0 0 760 230"><polyline fill="none" stroke="#4FD1C5" stroke-width="3.5" '
           f'stroke-linejoin="round" points="{poly}"/>'
           + "".join(f'<circle cx="{b}" cy="30" r="7" fill="#F2F4F7"/>' for b in beats)
           + '<line x1="270" y1="196" x2="465" y2="196" stroke="#F2F4F7" stroke-width="2.5"/>'
           '<line x1="270" y1="184" x2="270" y2="208" stroke="#F2F4F7" stroke-width="2.5"/>'
           '<line x1="465" y1="184" x2="465" y2="208" stroke="#F2F4F7" stroke-width="2.5"/>'
           '<text x="367" y="226" fill="#F2F4F7" font-size="21" text-anchor="middle" '
           'font-family="-apple-system,Helvetica">heartbeat interval (RR), e.g. 850 ms</text></svg>')
    left = (f'<div class="panel" style="width:820px"><div class="k">WHAT IS AN ECG?</div><div class="h">The heart\'s electrical signal</div>'
            f'<div class="t">Pads on the skin record each beat as a sharp spike (white dots).</div>{svg}</div>')
    full = 260

    def row(name, sub, usd, label, color="#5B7FA6", hidden=False):
        w = max(4, round(full * usd / lab)) if usd else 0
        bar = f'<div class="bar" style="width:{w}px;background:{color}"></div>' if w else '<div></div>'
        return (f'<div class="row{" hide" if hidden else ""}"><div class="name">{name}<span>{sub}</span></div>{bar}'
                f'<div class="price" style="color:{color}">{label}</div></div>')
    rows = (row("Sleep lab night", "US average; waits can take months", lab, f"${lab:,}", "#969FAC", stage < 1)
            + row("Holter monitor", "portable ECG worn at home, doctor's order", h_hi, f"${h_lo}–{h_hi}", "#5B7FA6", stage < 1)
            + row("ECG chest strap", "sold for sport", strap, f"about ${strap}", "#5B7FA6", stage < 1)
            + f'<div class="note{" hide" if stage < 2 else ""}">With Somnibeat, a night recorded this way '
              'becomes a sleep report.</div>')
    right = (f'<div class="panel" style="width:860px"><div class="k">WHAT IT COSTS TO RECORD A NIGHT</div>{rows}'
             '<div class="foot">US prices, 2026; they vary by insurance, state and seller. Sources: Sleep Foundation, MDsave, '
             'Polar (see SOURCES.md). Somnibeat\'s accuracy was measured on clinical ECG recordings.</div></div>')
    html = CARD_HTML.format(body=f'<div class="grid">{left}{right}</div>', credits="")
    return html.replace("</head>", ECG_CSS + "</head>")


def steps_clip(name: str, pngs: list[Path], switches: list[float], length: float) -> Path:
    """A card whose image changes at the given times (seconds from the clip start)."""
    lst = TMP / f"{name}.txt"
    times = [0.0, *switches, length]
    lines = []
    for png, a, b in zip(pngs, times[:-1], times[1:]):
        lines += [f"file '{png}'", f"duration {b - a:.4f}"]
    lines.append(f"file '{pngs[-1]}'")
    lst.write_text("\n".join(lines) + "\n")
    clip = TMP / f"{name}.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-t", f"{length:.3f}",
         "-vf", f"fps={FPS},format=yuv420p,fade=t=in:st=0:d=0.5,fade=t=out:st={length - 0.5:.2f}:d=0.5",
         "-c:v", "libx264", "-preset", "slow", "-crf", "16", str(clip)])
    return clip


# ----------------------------------------------------------------------------- scenes

def scene_clip(name: str, min_len: float) -> tuple[Path, float, tuple[float, float] | None]:
    """CFR clip from captured frames. Returns (path, length, fast-forward window in output time)."""
    meta = json.loads((OUT / "scenes" / name / "frames.json").read_text())
    frames = [(f["t"], OUT / "scenes" / name / f["file"]) for f in meta["frames"]]
    length = meta["length"]
    fast = None
    remap = lambda t: t  # noqa: E731
    marks = meta.get("marks", {})
    if "processing" in marks and "report" in marks:
        a, b = marks["processing"] + 0.6, marks["report"] - 0.2
        if b - a > FAST_TARGET_S:
            k = (b - a) / FAST_TARGET_S

            def remap(t, a=a, b=b, k=k):
                return t if t <= a else (a + (t - a) / k if t <= b else a + (b - a) / k + (t - b))
            fast = (a, a + FAST_TARGET_S)
            length = remap(length)
    out_len = max(length, min_len)
    lst = TMP / f"{name}.txt"
    lines = []
    times = [remap(t) for t, _ in frames] + [out_len]
    for i, (_, f) in enumerate(frames):
        d = max(times[i + 1] - times[i], 0.0)
        if i == 0:
            d += times[0]  # first frame also covers the time before it
        lines += [f"file '{f}'", f"duration {d:.4f}"]
    lines.append(f"file '{frames[-1][1]}'")
    lst.write_text("\n".join(lines) + "\n")
    clip = TMP / f"{name}.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-vf", f"{crop_filter(meta, frames[0][1])}scale={W}:{H}:flags=lanczos,fps={FPS},format=yuv420p,"
                f"fade=t=in:st=0:d={SCENE_FADE},fade=t=out:st={out_len - SCENE_FADE:.3f}:d={SCENE_FADE}",
         "-t", f"{out_len:.3f}",
         "-c:v", "libx264", "-preset", "slow", "-crf", "16", str(clip)])
    return clip, out_len, fast


def crop_filter(meta: dict, first_frame: Path) -> str:
    """Optional crop recorded by record.py (CSS px of the 1440 x 810 viewport) -> frame pixels."""
    if not meta.get("crop"):
        return ""
    fw, fh = Image.open(first_frame).size
    k = fw / 1440.0
    x, y, w, h = (int(round(v * k)) for v in meta["crop"])
    w, h = min(w, fw - x), min(h, fh - y)
    return f"crop={w}:{h}:{x}:{y},"


def card_clip(name: str, png: Path, length: float) -> Path:
    clip = TMP / f"{name}.mp4"
    fade = f"fade=t=in:st=0:d=0.8,fade=t=out:st={length - 1.0:.2f}:d=1.0" if name == "outro" else \
           f"fade=t=in:st=0:d=0.8,fade=t=out:st={length - 0.5:.2f}:d=0.5"
    run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(png), "-t", f"{length:.3f}",
         "-vf", f"fps={FPS},format=yuv420p,{fade}", "-c:v", "libx264", "-preset", "slow", "-crf", "16", str(clip)])
    return clip


# ----------------------------------------------------------------------------- captions

def wrap(draw, text, fnt, width):
    lines, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def caption_png(text: str, path: Path, size: int = 40) -> None:
    fnt = ImageFont.truetype(FONT, size)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lines = wrap(d, text, fnt, 1500)
    lh = int(size * 1.35)
    box_h = lh * len(lines) + 36
    box_w = int(max(d.textlength(l, font=fnt) for l in lines)) + 64
    x0, y0 = (W - box_w) // 2, H - 70 - box_h
    d.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=16, fill=(0, 0, 0, 200))
    for i, l in enumerate(lines):
        tw = d.textlength(l, font=fnt)
        d.text(((W - tw) / 2, y0 + 18 + i * lh), l, font=fnt, fill=(242, 244, 247, 255))
    img.save(path)


def label_png(text: str, path: Path) -> None:
    fnt = ImageFont.truetype(FONT, 30)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    tw = d.textlength(text, font=fnt)
    x0, y0 = W - int(tw) - 110, 120
    d.rounded_rectangle([x0, y0, x0 + tw + 48, y0 + 58], radius=29, fill=(10, 12, 16, 230), outline=(79, 209, 197), width=2)
    d.text((x0 + 24, y0 + 12), text, font=fnt, fill=(79, 209, 197))
    img.save(path)


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


# ----------------------------------------------------------------------------- main

def main() -> None:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    timings = json.loads((OUT / "narr/timings.json").read_text())
    cards = render_cards()

    clips, starts, cues, fast_windows = [], {}, [], []
    t = 0.0
    for name, beat in ORDER:
        need = LEAD_IN + timings[beat]["duration"] + TAIL
        if name == "ecg":
            length = need
            sw = [LEAD_IN + c["start"] for c in timings[beat]["cues"]]
            # prices with the third sentence, the Somnibeat line with the fourth
            clip = steps_clip(name, [cards[f"ecg{k}"] for k in range(3)], [sw[2], sw[3]], length)
        elif name == "how":
            length = need
            sw = [LEAD_IN + c["start"] for c in timings[beat]["cues"]]
            # step 1 lights as the voice starts, steps 2 and 3 with their sentences
            clip = steps_clip(name, [cards[f"how{k}"] for k in range(4)], [LEAD_IN * 0.5, *sw[1:3]], length)
        elif name in cards:
            length = max(CARD_LEN[name], need)
            clip = card_clip(name, cards[name], length)
        else:
            clip, length, fast = scene_clip(name, need)
            if fast:
                fast_windows.append((t + fast[0], t + fast[1]))
        clips.append(clip)
        starts[beat] = t + LEAD_IN
        for c in timings[beat]["cues"]:
            cues.append((starts[beat] + c["start"], starts[beat] + c["end"] + 0.15, c["text"]))
        t += length
    total = t

    # picture
    concat = TMP / "picture.txt"
    concat.write_text("".join(f"file '{c}'\n" for c in clips))
    picture = TMP / "picture.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(picture)])

    # captions + fast-forward label as overlays
    inputs = ["-i", str(picture)]
    filt, last = [], "[0:v]"
    # One transparent overlay track holds every caption and the fast-forward label:
    # a timeline of full-frame RGBA PNGs (blank between cues), encoded once with alpha.
    blank = TMP / "blank.png"
    Image.new("RGBA", (W, H), (0, 0, 0, 0)).save(blank)
    events = [(a, b, "cap", text) for a, b, text in cues]
    if fast_windows:
        events.append((fast_windows[0][0], fast_windows[0][1], "label", "Computing in the browser · shown faster"))
    cuts = sorted({0.0, total, *[e[0] for e in events], *[e[1] for e in events]})
    seg_lines, n = [], 0
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 1e-3:
            continue
        mid = (a + b) / 2
        active = [e for e in events if e[0] <= mid < e[1]]
        if active:
            img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            for _, _, kind, text in active:
                part = TMP / f"part_{n:03d}_{kind}.png"
                (caption_png if kind == "cap" else label_png)(text, part)
                img.alpha_composite(Image.open(part))
            frame = TMP / f"ov_{n:03d}.png"
            img.save(frame)
            n += 1
        else:
            frame = blank
        seg_lines += [f"file '{frame}'", f"duration {b - a:.4f}"]
    seg_lines.append(f"file '{blank}'")
    (TMP / "overlay.txt").write_text("\n".join(seg_lines) + "\n")
    overlay = TMP / "overlay.mov"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(TMP / "overlay.txt"),
         "-vf", f"fps={FPS},format=argb", "-t", f"{total:.3f}", "-c:v", "qtrle", str(overlay)])
    inputs += ["-i", str(overlay)]
    filt.append(f"{last}[1:v]overlay=0:0:format=auto[vcap]")
    last = "[vcap]"

    # audio: narration placed at its start times (no music)
    n0 = inputs.count("-i")
    beats = [b for _, b in ORDER]
    for b in beats:
        inputs += ["-i", str(OUT / "narr" / f"{b}.wav")]
    for j, b in enumerate(beats):
        filt.append(f"[{n0 + j}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={int(starts[b] * 1000)}:all=1[n{j}]")
    filt.append("".join(f"[n{j}]" for j in range(len(beats))) +
                f"amix=inputs={len(beats)}:normalize=0,apad=whole_dur={total:.3f},"
                "loudnorm=I=-16:TP=-1.5:LRA=11[aout]")

    # Render picture and sound separately, then mux. Encoding both from one filter graph
    # left gaps in the audio timestamps (narration drifted by ~8 s); a WAV render is exact.
    vfilt = [x for x in filt if x.endswith("[vcap]")]
    afilt = [x for x in filt if not x.endswith("[vcap]")]
    video_only = TMP / "video_only.mp4"
    run(["ffmpeg", "-v", "error", "-y", *inputs[:n0 * 2], "-filter_complex", ";".join(vfilt),
         "-map", last, "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
         "-t", f"{total:.3f}", str(video_only)])
    audio_inputs = inputs[n0 * 2:]
    afilt_rebased = []
    for x in afilt:
        for j in range(len(beats)):
            x = x.replace(f"[{n0 + j}:a]", f"[A{j}:a]")
        for j in range(len(beats)):
            x = x.replace(f"[A{j}:a]", f"[{j}:a]")
        afilt_rebased.append(x)
    audio_wav = TMP / "audio.wav"
    run(["ffmpeg", "-v", "error", "-y", *audio_inputs, "-filter_complex", ";".join(afilt_rebased),
         "-map", "[aout]", "-ar", "48000", "-t", f"{total:.3f}", str(audio_wav)])
    out = OUT / "somnibeat_demo.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-i", str(video_only), "-i", str(audio_wav), "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{total:.3f}",
         "-movflags", "+faststart", str(out)])
    (TMP / "timeline.json").write_text(json.dumps({"total": total, "narration_starts": starts}, indent=2))
    srt = "".join(f"{i + 1}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n" for i, (a, b, text) in enumerate(cues))
    (OUT / "somnibeat_demo.srt").write_text(srt)
    print(f"wrote {out} ({total:.1f}s), {len(cues)} captions")


if __name__ == "__main__":
    main()
