"""File checks of the web page that the audit added (web/js/rr.js, web/js/ecg.js).

Each case is a file that used to pass and give a huge or wrong report, or an app failure.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "web/data/samples/ucddb018.csv"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

SCRIPT = """
import { readFileSync } from "node:fs";
import { parseRRCsv } from "./web/js/rr.js";
import { readEdfHeader } from "./web/js/ecg.js";
const [kind, path] = process.argv.slice(1);
try {
  if (kind === "csv") { const p = parseRRCsv(readFileSync(path, "utf8")); console.log(JSON.stringify({ ok: p.rr_ms.length })); }
  else { await readEdfHeader(new Uint8Array(readFileSync(path))); console.log(JSON.stringify({ ok: true })); }
} catch (e) { console.log(JSON.stringify({ code: e.code || "INTERNAL", message: e.message })); }
"""


def check(kind: str, path: Path) -> dict:
    r = subprocess.run(["node", "--input-type=module", "-e", SCRIPT, kind, str(path)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120, check=True)
    return json.loads(r.stdout)


def rows():
    lines = SAMPLE.read_text().strip().split("\n")
    return lines[0], [tuple(map(float, l.split(","))) for l in lines[1:]]


def write(tmp: Path, text: str) -> Path:
    p = tmp / "f.csv"
    p.write_text(text)
    return p


def test_sample_still_passes(tmp_path):
    assert "ok" in check("csv", SAMPLE)


@pytest.mark.parametrize("case, code, words", [
    ("ms", "BAD_TIME_UNIT", "milliseconds"),
    ("unix", "BAD_TIME_START", "since the recording started"),
    ("reset", "TIME_RESET", "jumps back"),
    ("comma", "TOO_SHORT", "Decimal commas"),
])
def test_bad_time_columns_are_file_errors(tmp_path, case, code, words):
    head, data = rows()
    if case == "ms":
        body = [f"{t * 1000!r},{r!r}" for t, r in data]
    elif case == "unix":
        body = [f"{t + 1.7e9!r},{r!r}" for t, r in data]
    elif case == "reset":
        body = [f"{(t - 5000 if i > 10000 else t)!r},{r!r}" for i, (t, r) in enumerate(data)]
    else:
        head = head.replace(",", ";")
        body = [f"{t!r};{r!r}".replace(".", ",") for t, r in data]
    res = check("csv", write(tmp_path, head + "\n" + "\n".join(body) + "\n"))
    assert res.get("code") == code and words in res["message"], res


def test_cut_off_edf_header_is_a_file_error(tmp_path):
    b = bytearray(b" " * 300)
    b[0:1] = b"0"
    b[252:256] = b"2   "
    p = tmp_path / "cut.edf"
    p.write_bytes(bytes(b))
    assert check("edf", p)["code"] == "TRUNCATED"
