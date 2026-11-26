"""Record demo scenes from the live site with Chrome's screencast (CDP), frame by frame.

Each scene is a scripted interaction on https://handue.github.io/somnibeat/.
Frames are saved as JPEG with their capture timestamps so the editor can build
a constant-frame-rate clip. Scene minimum lengths come from the narration
timings (out/narr/timings.json) so the picture always covers the voice.

Usage: tools/video/.venv/bin/python tools/video/record.py [scene ...]
"""

from __future__ import annotations

import base64
import json
import shutil
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/video/out"
SITE = "https://handue.github.io/somnibeat/"
VIEW = {"width": 1440, "height": 810}
LEAD_IN, TAIL = 0.6, 1.0  # seconds of picture before and after the narration of a scene

SMOOTH = """
async ([y, ms]) => {
  const y0 = window.scrollY, dy = y - y0, t0 = performance.now();
  await new Promise(res => {
    const step = now => {
      const k = Math.min(1, (now - t0) / ms);
      const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
      window.scrollTo(0, y0 + dy * e);
      k < 1 ? requestAnimationFrame(step) : res();
    };
    requestAnimationFrame(step);
  });
}
"""
TOP_OF = """([sel, text, offset]) => {
  const els = [...document.querySelectorAll(sel)].filter(e => !text || e.textContent.includes(text));
  const vis = els.filter(x => x.offsetParent !== null);
  const pool = vis.length ? vis : els;
  const e = pool.sort((a, b) => a.textContent.length - b.textContent.length)[0];  // innermost match
  return e ? Math.max(0, e.getBoundingClientRect().top + window.scrollY - offset) : null;
}"""


class Recorder:
    def __init__(self, page, name: str):
        self.page, self.name = page, name
        self.dir = OUT / "scenes" / name
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        self.frames: list[tuple[float, str]] = []
        self.marks: dict[str, float] = {}
        self.crop: list[float] | None = None  # [x, y, w, h] in CSS px of the viewport
        self.cdp = page.context.new_cdp_session(page)
        self.cdp.on("Page.screencastFrame", self._frame)

    def _frame(self, ev):
        i = len(self.frames)
        fn = f"{i:05d}.jpg"
        (self.dir / fn).write_bytes(base64.b64decode(ev["data"]))
        self.frames.append((time.monotonic(), fn))
        try:
            self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})
        except Exception:
            pass

    def start(self):
        self.t0 = time.monotonic()
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": 2880,
                                               "maxHeight": 1620, "everyNthFrame": 1})

    def mark(self, name: str):
        self.marks[name] = time.monotonic() - self.t0

    def hold(self, seconds: float):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.page.wait_for_timeout(50)

    def stop(self, min_len: float):
        elapsed = time.monotonic() - self.t0
        if elapsed < min_len:
            self.hold(min_len - elapsed)
        t_end = time.monotonic()
        self.cdp.send("Page.stopScreencast")
        meta = {"t0": self.t0, "t_end": t_end, "length": t_end - self.t0, "marks": self.marks, "crop": self.crop, "scale": None,
                "frames": [{"t": t - self.t0, "file": f} for t, f in self.frames]}
        (self.dir / "frames.json").write_text(json.dumps(meta))
        print(self.name, f"{meta['length']:.1f}s", len(self.frames), "frames")


def scroll_to(page, sel, text=None, offset=90, ms=1400):
    y = page.evaluate(TOP_OF, [sel, text, offset])
    if y is not None:
        page.evaluate(SMOOTH, [y, ms])


def click_sample(page, title):
    page.get_by_role("button", name=f"Open this sample: {title}").click()


def wait_report(page, timeout=120000):
    page.wait_for_function("location.hash.startsWith('#/report')", timeout=timeout)
    page.wait_for_timeout(400)


def scene_start(page, rec, need):
    page.goto(SITE + "#/", wait_until="networkidle")
    page.wait_for_timeout(800)
    rec.start()
    rec.hold(2.5)
    scroll_to(page, "h2", "Sample nights", offset=60, ms=2200)
    rec.hold(2.5)
    page.evaluate(SMOOTH, [0, 1800])
    rec.stop(need)


SAMPLE_FILE = ROOT / "web/data/samples/ucddb015.csv"
EDF_FILE = ROOT / "data/raw/ucddb/ucddb015.rec"  # the public UCDDB recording behind that sample
DEMO_URL = "https://handue.github.io/somnibeat/_demo/"  # served by page.route, never fetched from the site


def ecg_excerpt_svg(start_s: float = 7200.0, seconds: float = 7.0, width: int = 700, height: int = 190) -> str:
    """The real ECG of EDF_FILE for a few seconds, with the beats the detector found (white dots)."""
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from somnibeat.edf import read_channel
    x, fs, _, _ = read_channel(EDF_FILE, "ECG|EKG")
    a, b = int(start_s * fs), int((start_s + seconds) * fs)
    seg = x[a:b] - np.median(x[a:b])
    lo, hi = np.percentile(seg, 0.5), np.percentile(seg, 99.8)
    def y(v):
        return height - 16 - (v - lo) / (hi - lo) * (height - 40)
    pts = " ".join(f"{i / (b - a) * width:.1f},{y(v):.1f}" for i, v in enumerate(seg))
    beats = np.load(ROOT / "data/cache/beats/ucddb_ucddb015.npy")
    beats = beats[(beats >= start_s) & (beats < start_s + seconds)]
    dots = "".join(f'<circle cx="{(t - start_s) / seconds * width:.1f}" cy="{y(seg[min(len(seg) - 1, int(round((t - start_s) * fs)))]):.1f}" '
                   f'r="6" fill="#F2F4F7"/>' for t in beats)
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<polyline fill="none" stroke="#4FD1C5" stroke-width="2.2" stroke-linejoin="round" points="{pts}"/>{dots}</svg>')


EDF_VIEW = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:#000;height:100%;font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif;color:#F2F4F7}}
.wrap{{height:100%;display:flex;align-items:flex-start;justify-content:center;padding-top:70px;box-sizing:border-box}}
.win{{width:760px;background:#0A0C10;border:1px solid #1A1F27;border-radius:14px;overflow:hidden}}
.bar{{display:flex;justify-content:space-between;padding:16px 22px;border-bottom:1px solid #1A1F27;font-size:17px}}
.bar span{{color:#969FAC}}
.body{{padding:18px 30px 6px}}
.kv{{display:grid;grid-template-columns:190px 1fr;gap:6px 12px;font-size:17px;color:#C9CED6;margin-bottom:12px}}
.kv b{{color:#969FAC;font-weight:500}}
.note{{padding:4px 30px 18px;font-size:15px;color:#969FAC}}
</style></head><body><div class="wrap"><div class="win">
<div class="bar"><strong>{name}</strong><span>EDF recording · {mb} MB</span></div>
<div class="body"><div class="kv"><b>Signal</b><span>ECG (channel "{label}"), {fs:g} Hz</span>
<b>Length</b><span>{hours:.1f} hours, started {start}</span></div>{svg}</div>
<div class="note">A few seconds of the night's ECG · white dots: heartbeats<br>
Recording ucddb015 from the open UCDDB research dataset (PhysioNet)</div>
</div></div></body></html>"""

# A plain view of the file being uploaded (its real first rows), in the site's colours.
FILE_VIEW = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:#000;height:100%;font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif;color:#F2F4F7}}
.wrap{{height:100%;display:flex;align-items:flex-start;justify-content:center;padding-top:70px;box-sizing:border-box}}
.win{{width:760px;background:#0A0C10;border:1px solid #1A1F27;border-radius:14px;overflow:hidden}}
.bar{{display:flex;justify-content:space-between;padding:16px 22px;border-bottom:1px solid #1A1F27;font-size:17px}}
.bar span{{color:#969FAC}}
pre{{margin:0;padding:20px 22px;font:17px/1.65 ui-monospace,Menlo,monospace;color:#C9CED6}}
pre b{{color:#4FD1C5;font-weight:600}}
.note{{padding:0 22px 18px;font-size:15px;color:#969FAC}}
</style></head><body><div class="wrap"><div class="win">
<div class="bar"><strong>{name}</strong><span>{rows:,} rows · {kb} KB</span></div>
<pre>{lines}</pre>
<div class="note">time_s: time of each beat in seconds · rr_ms: time since the previous beat in milliseconds<br>
Recording from the open UCDDB research dataset (PhysioNet)</div>
</div></div></body></html>"""



# Drag-and-drop as a person does it: a file window, the cursor picks up the file and drops it on the
# page's drop zone. The window, icon and cursor are drawn for the video; the drop itself is a real
# DragEvent carrying the real file, handled by the site's own drop handler (web/js/app.js).
DRAG_DROP = """async ([name, text, url, kind]) => {
  const css = document.createElement('style');
  css.textContent = `
  #demo-win{position:fixed;left:70px;top:470px;width:420px;background:#14171C;border:1px solid #2A303A;border-radius:12px;
    box-shadow:0 18px 50px rgba(0,0,0,.6);z-index:99990;font:14px -apple-system,Helvetica,sans-serif;color:#C9CED6;
    opacity:0;transform:translateY(16px);transition:opacity .45s,transform .45s}
  #demo-win.on{opacity:1;transform:none}
  #demo-win .bar{display:flex;align-items:center;gap:8px;padding:11px 14px;border-bottom:1px solid #2A303A}
  #demo-win .dot{width:12px;height:12px;border-radius:50%}
  #demo-win .ttl{margin-left:10px;color:#F2F4F7;font-weight:600}
  #demo-win .body{padding:22px 26px;display:flex;gap:30px}
  .demo-file{width:96px;text-align:center;font-size:13px;color:#F2F4F7}
  .demo-file .ic{width:54px;height:66px;margin:0 auto 8px;background:#F2F4F7;border-radius:4px;position:relative;
    clip-path:polygon(0 0,72% 0,100% 22%,100% 100%,0 100%)}
  .demo-file .ic b{position:absolute;bottom:9px;left:0;right:0;font:700 11px Helvetica;color:#0A7A6F}
  .demo-file.sel .nm{background:#2B5FD9;border-radius:4px;padding:1px 4px}
  .demo-file.dim{opacity:.35}
  #demo-ghost{position:fixed;left:0;top:0;z-index:99998;pointer-events:none;opacity:.85}
  #demo-cursor{position:fixed;left:0;top:0;width:22px;height:22px;border-radius:50%;z-index:99999;pointer-events:none;
    background:rgba(242,244,247,.92);box-shadow:0 0 0 3px rgba(0,0,0,.6)}`;
  document.head.appendChild(css);
  const win = document.createElement('div');
  win.id = 'demo-win';
  win.innerHTML = `<div class="bar"><span class="dot" style="background:#FF5F57"></span><span class="dot" style="background:#FEBC2E"></span>
    <span class="dot" style="background:#28C840"></span><span class="ttl">Downloads</span></div>
    <div class="body"><div class="demo-file" id="demo-src"><div class="ic"><b>${kind || 'CSV'}</b></div><div class="nm">${name}</div></div>
    <div class="demo-file dim"><div class="ic"><b>PDF</b></div><div class="nm">sleep_notes.pdf</div></div></div>`;
  document.body.appendChild(win);
  const cur = document.createElement('div'); cur.id = 'demo-cursor'; document.body.appendChild(cur);
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  let cx = 900, cy = 760;
  const place = (el, x, y) => { el.style.transform = `translate(${x}px, ${y}px)`; };
  place(cur, cx, cy);
  const move = async (x, y, ms, ghost) => {
    const x0 = cx, y0 = cy, t0 = performance.now();
    await new Promise(res => { const step = now => {
      const k = Math.min(1, (now - t0) / ms), e = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
      cx = x0 + (x - x0) * e; cy = y0 + (y - y0) * e; place(cur, cx - 11, cy - 11);
      if (ghost) place(ghost, cx - 40, cy - 30);
      k < 1 ? requestAnimationFrame(step) : res(); }; requestAnimationFrame(step); });
  };
  await sleep(100); win.classList.add('on'); await sleep(700);
  const src = document.getElementById('demo-src').getBoundingClientRect();
  await move(src.left + src.width / 2, src.top + 34, 900);
  document.getElementById('demo-src').classList.add('sel');
  await sleep(350);
  const ghost = document.getElementById('demo-src').cloneNode(true); ghost.id = 'demo-ghost'; ghost.classList.remove('sel');
  document.body.appendChild(ghost); place(ghost, cx - 40, cy - 30);
  const dt = new DataTransfer();
  const content = url ? await (await fetch(url)).arrayBuffer() : text;
  dt.items.add(new File([content], name, {type: url ? 'application/octet-stream' : 'text/csv'}));
  const zone = document.getElementById('drop-zone');
  const z = zone.getBoundingClientRect();
  const tx = z.left + z.width / 2, ty = z.top + z.height / 2;
  await move(src.left + 160, src.top - 60, 350, ghost);
  zone.dispatchEvent(new DragEvent('dragenter', {dataTransfer: dt, bubbles: true, cancelable: true}));
  await move(tx, ty, 1100, ghost);
  zone.dispatchEvent(new DragEvent('dragover', {dataTransfer: dt, bubbles: true, cancelable: true}));
  await sleep(450);
  for (const el of [ghost, win, cur]) el.remove();
  zone.dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true, cancelable: true}));
}"""


def serve_edf(page):
    data = EDF_FILE.read_bytes()
    page.route(DEMO_URL + "**", lambda route: route.fulfill(status=200, body=data,
                                                             headers={"Content-Type": "application/octet-stream"}))


def scene_report(page, rec, need):
    """Show the ECG file, then drag it from a file window onto the start page and wait for the report."""
    sys.path.insert(0, str(ROOT))
    from somnibeat.edf import read_header
    hdr = read_header(EDF_FILE)
    i = [k for k, lab in enumerate(hdr.labels) if "ECG" in lab.upper()][0]
    page.goto("about:blank")  # set_content keeps the URL; leave the site so the later goto reloads it
    page.set_content(EDF_VIEW.format(name=EDF_FILE.name, mb=round(EDF_FILE.stat().st_size / 1e6), label=hdr.labels[i],
                                     fs=float(hdr.fs[i]), hours=hdr.duration_s / 3600,
                                     start=hdr.start.strftime("%H:%M"), svg=ecg_excerpt_svg()))
    page.wait_for_timeout(300)
    serve_edf(page)
    rec.start()
    rec.hold(5.0)
    page.goto(SITE + "#/", wait_until="networkidle")
    page.wait_for_timeout(600)
    rec.hold(1.0)
    page.evaluate(DRAG_DROP, [EDF_FILE.name, "", DEMO_URL + EDF_FILE.name, "EDF"])
    rec.mark("processing")
    page.wait_for_function("document.querySelector('#steps li[data-step=ecg]')?.dataset.status === 'done'", timeout=120000)
    rec.mark("ecg_done")
    wait_report(page)
    rec.mark("report")
    rec.hold(2.5)
    scroll_to(page, ".metric, [class*=tile]", None, offset=140, ms=900)
    rec.hold(3.0)
    scroll_to(page, "#conversion-line", None, offset=420, ms=1400)
    rec.hold(3.0)
    rec.stop(need)


def scene_fail(page, rec, need):
    """Drop a copy of the sample file whose times are in milliseconds: the page refuses it and says why."""
    rows = SAMPLE_FILE.read_text().split()
    body = [f"{float(t) * 1000:.1f},{r}" for t, r in (row.split(",") for row in rows[1:])]
    text = rows[0] + "\n" + "\n".join(body) + "\n"
    page.goto(SITE + "#/", wait_until="networkidle")
    page.wait_for_timeout(600)
    rec.start()
    rec.hold(0.8)
    page.evaluate(DRAG_DROP, ["night_times_in_ms.csv", text, None, "CSV"])
    page.wait_for_function("location.hash.startsWith('#/problem')", timeout=60000)
    rec.hold(1.0)
    rec.stop(need)


def scene_limits(page, rec, need):
    page.goto(SITE + "#/method", wait_until="networkidle")
    page.wait_for_timeout(600)
    y = page.evaluate(TOP_OF, ["h2", "Who benefits, and who could be harmed", 30])
    page.evaluate("y => window.scrollTo(0, y)", y or 0)
    page.wait_for_timeout(400)
    rec.start()
    rec.hold(1.5)
    scroll_to(page, "h2", "Limitations", offset=60, ms=2200)
    rec.stop(need)


def scene_compare(page, rec, need):
    open_report_quietly(page, "A typical night")
    y = page.evaluate(TOP_OF, ["h2", "Night report", 30])
    page.evaluate("y => window.scrollTo(0, y)", y or 0)
    page.wait_for_timeout(300)
    rec.start()
    rec.hold(1.0)
    scroll_to(page, "h2", "Sleep stages across the night", offset=40, ms=1600)
    rec.hold(1.5)
    page.get_by_role("button", name="4 stages (experimental)").click()
    rec.hold(3.0)
    scroll_to(page, "h3, h2, p, div", "Sleep lab (polysomnography", offset=420, ms=1600)
    rec.stop(need)


def scene_failure(page, rec, need):
    page.goto(SITE + "#/", wait_until="networkidle")
    page.wait_for_timeout(500)
    click_sample(page, "An ordinary night")
    wait_report(page)
    page.evaluate(TOP_OF, ["h2", "Sleep stages across the night", 40])
    y = page.evaluate(TOP_OF, ["h2", "Sleep stages across the night", 40])
    page.evaluate("y => window.scrollTo(0, y)", y)
    page.wait_for_timeout(1500)
    rec.start()
    rec.hold(2.0)
    scroll_to(page, "h3, h2, p, div", "Sleep lab (polysomnography", offset=330, ms=1800)
    rec.hold(2.5)
    scroll_to(page, "h3, h2", "Compare with the sleep lab", offset=380, ms=1400)
    rec.stop(need)


def scene_trust(page, rec, need):
    # continues on the same report
    rec.start()
    scroll_to(page, "h2", "How far can you trust this?", offset=24, ms=1600)
    # Frame only the trust card (the editor crops to this 16:9 box and enlarges it).
    box = page.evaluate("""() => { const r = document.querySelector('.trust-card').getBoundingClientRect();
                                    return [r.left, r.width]; }""")
    x, w = box[0] - 16, box[1] + 32
    rec.crop = [x, 0, w, w * 9 / 16]
    rec.hold(4.5)
    # then the side-by-side table (v2, v1, trivial rule)
    # bring the agreement (kappa) rows of the side-by-side table above the caption area
    scroll_to(page, "#trust-compare td, #trust-compare th, #trust-compare div", "Epoch agreement", offset=195, ms=2400)
    rec.hold(4.0)
    rec.stop(need)


def scene_method(page, rec, need):
    page.goto(SITE + "#/method", wait_until="networkidle")
    page.wait_for_timeout(800)
    rec.start()
    rec.hold(1.5)
    scroll_to(page, "h2", "v2: chosen on training data", offset=30, ms=2200)
    rec.hold(4.0)
    scroll_to(page, "h2", "Reproduce", offset=200, ms=2400)
    rec.stop(need)


def open_report_quietly(page, title):
    """Open a sample and wait for its report before recording starts (no processing screen)."""
    page.goto(SITE + "#/", wait_until="networkidle")
    page.wait_for_timeout(500)
    click_sample(page, title)
    wait_report(page)
    page.wait_for_timeout(1500)


def scene_apnea(page, rec, need):
    open_report_quietly(page, "Many suspected apnea minutes")
    y = page.evaluate(TOP_OF, ["h2", "Night report", 30])
    page.evaluate("y => window.scrollTo(0, y)", y or 0)
    page.wait_for_timeout(300)
    rec.start()
    rec.hold(2.5)
    scroll_to(page, "h2", "Sleep stages across the night", offset=40, ms=1800)
    rec.hold(4.0)
    scroll_to(page, "h3, h2, p, div", "Sleep lab (polysomnography", offset=360, ms=1600)
    rec.stop(need)


def scene_noisy(page, rec, need):
    open_report_quietly(page, "A noisy night")
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(300)
    rec.start()
    rec.hold(3.0)
    scroll_to(page, "h2", "Sleep stages across the night", offset=40, ms=1800)
    rec.stop(need)


def scene_trust_on(page, rec, need):
    open_report_quietly(page, "A typical night")
    y = page.evaluate(TOP_OF, ["h2", "Sleep stages across the night", 40])
    page.evaluate("y => window.scrollTo(0, y)", y or 0)
    page.wait_for_timeout(500)
    scene_trust(page, rec, need)


def scene_harm(page, rec, need):
    page.goto(SITE + "#/method", wait_until="networkidle")
    page.wait_for_timeout(600)
    y = page.evaluate(TOP_OF, ["h2", "Who benefits, and who could be harmed", 30])
    page.evaluate("y => window.scrollTo(0, y)", y or 0)
    page.wait_for_timeout(500)
    rec.start()
    rec.hold(5.0)
    scroll_to(page, "h2", "Limitations", offset=200, ms=2200)
    rec.stop(need)


SCENES = {"start": ("N1", scene_start), "report": ("N2", scene_report), "compare": ("N3", scene_compare),
          "apnea": ("N4", scene_apnea), "noisy": ("N5", scene_noisy), "fail": ("NF", scene_fail),
          "trust": ("N6", scene_trust_on), "limits": ("NL", scene_limits)}
ORDER = ["start", "report", "compare", "apnea", "noisy", "fail", "trust", "limits"]


def main(names: list[str]) -> None:
    timings = json.loads((OUT / "narr/timings.json").read_text())
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport=VIEW, device_scale_factor=2, color_scheme="dark")
        page = ctx.new_page()
        for name in ORDER:
            beat, fn = SCENES[name]
            need = LEAD_IN + timings[beat]["duration"] + TAIL
            if names and name not in names:
                continue
            rec = Recorder(page, name)
            fn(page, rec, need)
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1:])
