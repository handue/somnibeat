"""Pipeline wiring: Makefile target order, the v2 freeze gate, the HMC download plan and the v2
cache-rebuild scripts. Nothing here trains, evaluates or downloads; tracked files are never edited
(changes are simulated with monkeypatch and temporary files)."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = ".venv/bin/python"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------- Makefile

def make_n(target: str) -> list[str]:
    if shutil.which("make") is None:
        pytest.skip("make not installed")
    out = subprocess.run(["make", "-n", "--no-print-directory", target], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    # Join backslash-continued recipe lines, keep the commands that run a script or pytest.
    joined = out.replace("\\\n", " ").splitlines()
    return [" ".join(l.split()) for l in joined if l.startswith(PY)]


V1_STEPS = [
    "-m pip install -q -r requirements.txt",
    "scripts/download.py --config config/default.toml",
    "scripts/check_data.py",
    "scripts/build_features.py --config config/default.toml",
    "scripts/validate_rpeaks.py",
    "scripts/train_eval.py --config config/default.toml --tasks sleep_wake wake_nrem_rem four_class apnea",
    "scripts/export_web.py",
    "scripts/render_docs.py",
    "-m pytest -q tests/test_parity_*.py",
    "-m pytest -q tests",
]
V2_STEPS = [
    "scripts/download.py --config config/v2.toml --datasets capslpdb hmc",
    "scripts/check_data.py --config config/v2.toml --out results/v2/data_check.json",
    "scripts/check_hmc.py --config config/v2.toml",
    "scripts/v2_select_apnea.py check",
    "scripts/build_features.py --config config/v2.toml --datasets cap",
    "scripts/build_features.py --config config/v2.toml --datasets hmc --prefix hmc_",
    "scripts/v2_merge_training.py",
    "scripts/v2_build_extras.py",
    "scripts/v2_select_apnea.py features",
    "scripts/v2_select.py --candidates default --tasks sleep_wake --out results/v2/parts/staging_sleep_wake_all.json",
    "scripts/v2_select.py --candidates default --tasks wake_nrem_rem --out results/v2/parts/staging_wake_nrem_rem_all.json",
    "scripts/v2_select.py --candidates default --tasks four_class --out results/v2/parts/staging_four_class_all.json",
    "scripts/v2_select.py --candidates results/v2/parts/cap106_candidates.json --tasks sleep_wake --no-subject-cv "
    "--features data/cache/v2/stage_features_cap106_slpdb.csv.gz "
    "--extra data/cache/v2/stage_features_extra_cap106.csv.gz --out results/v2/selection_cap106.json",
    "scripts/v2_select_apnea.py select",
    "scripts/v2_finalize_selection.py",
    "scripts/v2_decide_final.py",
    "scripts/train_eval_v2.py --final",
]
AFTER_V2 = ["scripts/export_web.py", "scripts/export_web_v2.py", "scripts/render_docs.py",
            "-m pytest -q tests/test_parity_*.py", "-m pytest -q tests"]


def test_make_reproduce_v1_unchanged():
    assert make_n("reproduce-v1") == [f"{PY} {s}" for s in V1_STEPS]


def test_make_v2_order():
    assert make_n("v2") == [f"{PY} {s}" for s in V2_STEPS]


def test_make_reproduce_full_sequence():
    # v1, then v2, then web/docs/tests again so they reflect v2 (docs after the v2 evaluation).
    assert make_n("reproduce") == [f"{PY} {s}" for s in V1_STEPS + V2_STEPS + AFTER_V2]


def test_v2_select_commands_match_recorded_runs():
    import json
    sel = json.loads((ROOT / "results/v2/selection_cap106.json").read_text())
    cmd = next(c for c in make_n("v2-select") if "cap106_candidates" in c)
    assert f"--features {sel['features_path']}" in cmd and f"--extra {sel['extra_features_path']}" in cmd
    parts = json.loads((ROOT / "results/v2/selection_staging.json").read_text())["merged_from"]
    outs = [c.split("--out ")[1] for c in make_n("v2-select") if "--candidates default" in c]
    assert sorted(outs) == sorted(parts)


# ---------------------------------------------------------------------------------- freeze gate

def git_ok() -> bool:
    if shutil.which("git") is None:
        return False
    r = subprocess.run(["git", "rev-parse", "-q", "--verify", "refs/tags/v2-frozen"], cwd=ROOT,
                       capture_output=True)
    return r.returncode == 0


needs_git = pytest.mark.skipif(not git_ok(), reason="needs git and the tag v2-frozen")


@pytest.fixture
def tev2():
    return load("train_eval_v2")


@needs_git
def test_check_frozen_accepts_current_head(tev2):
    r = tev2.check_frozen()
    tag = subprocess.run(["git", "rev-parse", "v2-frozen^{commit}"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    assert r == tag  # HEAD is not recorded: it differs between checkouts


def fake_git(monkeypatch, tev2, handler):
    """Route git calls made by check_frozen through handler(args, real_result) -> result."""
    real = subprocess.run

    def run(args, *a, **kw):
        res = real(args, *a, **kw)
        if isinstance(args, list) and args[:1] == ["git"]:
            return handler(args[1:], res)
        return res

    monkeypatch.setattr(tev2.subprocess, "run", run)


def replace_stdout(res, data: bytes, code: int = 0):
    return subprocess.CompletedProcess(res.args, code, data, b"")


@needs_git
@pytest.mark.parametrize("path", ["somnibeat/rr.py", "config/v2.toml", "results/v2/v2_final_spec.json"])
def test_check_frozen_rejects_changed_file_at_head(monkeypatch, tev2, path):
    def handler(args, res):
        if args == ["show", f"HEAD:{path}"]:
            return replace_stdout(res, res.stdout + b"\n# simulated change\n")
        return res

    fake_git(monkeypatch, tev2, handler)
    with pytest.raises(SystemExit, match=path.replace(".", r"\.")):
        tev2.check_frozen()


@needs_git
def test_check_frozen_rejects_changed_working_tree(monkeypatch, tev2):
    target = (ROOT / "somnibeat/evaluate.py").resolve()
    real = Path.read_bytes

    def read_bytes(self):
        data = real(self)
        return data + b"\n# simulated edit\n" if self.resolve() == target else data

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(SystemExit, match="somnibeat/evaluate.py.*working tree"):
        tev2.check_frozen()


@needs_git
def test_check_frozen_ignores_only_its_own_function(monkeypatch, tev2):
    script = "scripts/train_eval_v2.py"

    def edit(old: bytes, new: bytes):
        def handler(args, res):
            if args == ["show", f"HEAD:{script}"]:
                assert old in res.stdout
                return replace_stdout(res, res.stdout.replace(old, new, 1))
            return res
        return handler

    # A change inside check_frozen (whatever version HEAD holds) is allowed ...
    sig = b"def check_frozen()"
    fake_git(monkeypatch, tev2, edit(sig, b"def check_frozen(*, reworded=True)"))
    tag = subprocess.run(["git", "rev-parse", "v2-frozen^{commit}"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    assert tev2.check_frozen() == tag
    monkeypatch.undo()
    # ... a change anywhere else in the script is not.
    fake_git(monkeypatch, tev2, edit(b"MIN_GOOD_FRACTION = 0.5", b"MIN_GOOD_FRACTION = 0.4"))
    with pytest.raises(SystemExit, match="scripts/train_eval_v2.py"):
        tev2.check_frozen()


@needs_git
def test_check_frozen_rejects_missing_tag_or_unrelated_head(monkeypatch, tev2):
    fake_git(monkeypatch, tev2, lambda args, res: replace_stdout(res, b"", 1) if args[:1] == ["rev-parse"]
             and "refs/tags/v2-frozen^{commit}" in args else res)
    with pytest.raises(SystemExit, match="does not exist"):
        tev2.check_frozen()
    monkeypatch.undo()
    fake_git(monkeypatch, tev2, lambda args, res: replace_stdout(res, b"", 1) if args[:1] == ["merge-base"] else res)
    with pytest.raises(SystemExit, match="contain the tag"):
        tev2.check_frozen()


# -------------------------------------------------------------------------------- HMC download plan

def fake_hmc(tmp_path: Path) -> tuple[Path, dict]:
    raw = tmp_path / "raw"
    root = raw / "hmc"
    (root / "recordings").mkdir(parents=True)
    files = {"RECORDS": b"recordings/SN001.edf\nrecordings/SN002.edf\n", "LICENSE.txt": b"license",
             "Subjects_info_aggregated.txt": b"subjects"}
    for r in ("SN001", "SN002"):
        files[f"recordings/{r}.edf"] = f"edf {r}".encode()
        files[f"recordings/{r}_sleepscoring.txt"] = f"scoring {r}".encode()
    for name, data in files.items():
        (root / name).write_bytes(data)
    sums = "".join(f"{hashlib.sha256(d).hexdigest()} {n}\n" for n, d in files.items())
    (root / "SHA256SUMS.txt").write_text(sums)
    return raw, files


def test_hmc_plan_lists_every_record(tmp_path):
    dl = load("download")
    raw, files = fake_hmc(tmp_path)
    entries = dl.plan("hmc", {}, raw)
    names = [e["name"] for e in entries]
    assert names == ["RECORDS", "SHA256SUMS.txt", "LICENSE.txt", "Subjects_info_aggregated.txt",
                     "recordings/SN001.edf", "recordings/SN001_sleepscoring.txt",
                     "recordings/SN002.edf", "recordings/SN002_sleepscoring.txt"]
    assert all(e["url"].startswith("https://physionet.org/files/hmc-sleep-staging/1.1/") for e in entries)
    assert all(e["sha256"] for e in entries if e["name"] != "SHA256SUMS.txt")
    assert all(e["present"] for e in entries)


def test_hmc_verify_detects_mismatch_and_missing(tmp_path):
    dl = load("download")
    raw, files = fake_hmc(tmp_path)
    r = dl.verify("hmc", {}, raw)
    assert len(r["ok"]) == len(files) and r["no_checksum"] == ["SHA256SUMS.txt"]
    (raw / "hmc/recordings/SN002.edf").write_bytes(b"corrupted")
    (raw / "hmc/recordings/SN001_sleepscoring.txt").unlink()
    r = dl.verify("hmc", {}, raw)
    assert r["mismatch"] == ["recordings/SN002.edf"] and r["missing"] == ["recordings/SN001_sleepscoring.txt"]


def test_hmc_plan_matches_raw_files_on_disk():
    root = ROOT / "data/raw/hmc"
    if not (root / "RECORDS").exists():
        pytest.skip("HMC raw data not downloaded")
    dl = load("download")
    entries = dl.plan("hmc", {}, ROOT / "data/raw")
    n = len((root / "RECORDS").read_text().split())
    assert len(entries) == 4 + 2 * n
    assert sum(e["sha256"] is not None for e in entries) == 3 + 2 * n


# ------------------------------------------------------------------------------- v2 cache rebuilds

def write_gz(path: Path, text: str) -> None:
    with gzip.open(path, "wt") as f:
        f.write(text)


def test_merge_training_text_order(tmp_path):
    mt = load("v2_merge_training")
    hdr = "dataset,record,epoch,x\n"
    write_gz(tmp_path / "cap.csv.gz", hdr + "cap,n1,0,1.5\ncap,n1,1,0.30000000000000004\n")
    write_gz(tmp_path / "v1.csv.gz", hdr + "ucddb,u1,0,9\ncap,n1,0,7\nslpdb,s1,0,2\nslpdb,s1,1,\n")
    out = tmp_path / "out.csv.gz"
    r = mt.merge(tmp_path / "cap.csv.gz", tmp_path / "v1.csv.gz", out)
    assert (r["cap_rows"], r["slpdb_rows"]) == (2, 2)
    assert gzip.open(out, "rt").read() == hdr + "cap,n1,0,1.5\ncap,n1,1,0.30000000000000004\nslpdb,s1,0,2\nslpdb,s1,1,\n"
    assert out.read_bytes()[4:8] == b"\0\0\0\0"  # gzip mtime 0
    write_gz(tmp_path / "bad.csv.gz", "dataset,record,epoch,y\n")
    with pytest.raises(SystemExit):
        mt.merge(tmp_path / "cap.csv.gz", tmp_path / "bad.csv.gz", out)


def test_merge_training_reproduces_frozen_cache(tmp_path):
    """Byte identity with the cache used by selection and the frozen evaluation (~15 s)."""
    import json
    v2 = ROOT / "data/cache/v2"
    ref = v2 / "stage_features_cap106_slpdb.csv.gz"
    if not (ref.exists() and (v2 / "stage_features.csv.gz").exists() and (ROOT / "data/cache/stage_features.csv.gz").exists()):
        pytest.skip("caches not built")
    mt = load("v2_merge_training")
    out = tmp_path / ref.name  # the gzip header stores the file name
    r = mt.merge(v2 / "stage_features.csv.gz", ROOT / "data/cache/stage_features.csv.gz", out)
    recorded = json.loads((ROOT / "results/v2/selection_cap106.json").read_text())["features_sha256"]
    assert r["sha256"] == recorded
    assert out.read_bytes() == ref.read_bytes()


def test_build_extras_commands(tmp_path):
    be = load("v2_build_extras")
    cmds = [c[2:] for c in be.commands(tmp_path)]
    assert len(cmds) == 3
    base, cap106, test = cmds
    assert base[:6] == ["--datasets", "cap", "slpdb", "apnea-ecg", "ucddb", "--v1-grids-only"]
    assert base[base.index("--stage-out") + 1] == str(tmp_path / "stage_features_extra.csv.gz")
    assert base[base.index("--apnea-out") + 1] == str(tmp_path / "apnea_features_extra.csv.gz")
    for cmd, ds, out in ((cap106, "cap", "stage_features_extra_cap106.csv.gz"),
                         (test, "hmc", "stage_features_extra_test.csv.gz")):
        assert cmd[:2] == ["--datasets", ds]
        assert cmd[cmd.index("--stage-base") + 1] == str(tmp_path / "stage_features_extra.csv.gz")
        assert cmd[cmd.index("--stage-out") + 1] == str(tmp_path / out)
        assert cmd[cmd.index("--stage-gzip-name") + 1] == "stage_features_extra.csv.gz"
        assert cmd[cmd.index("--apnea-out") + 1] == "none"


def test_build_features_v2_gzip_name(tmp_path):
    """--stage-gzip-name only changes the name stored in the gzip header, not the table."""
    import pandas as pd
    bf = load("build_features_v2")
    df = pd.DataFrame({"dataset": ["b", "a"], "record": ["r", "r"], "epoch": [0, 0], "x": [0.1, 2.0]})
    p1, p2 = tmp_path / "one.csv.gz", tmp_path / "two.csv.gz"
    bf.write(p1, [df], {"a", "b"}, "epoch", base="none")
    bf.write(p2, [df], {"a", "b"}, "epoch", base="none", gzip_name="stage_features_extra.csv.gz")
    assert gzip.open(p1).read() == gzip.open(p2).read()
    assert b"one.csv\0" in p1.read_bytes()[:40] and b"stage_features_extra.csv\0" in p2.read_bytes()[:40]
    assert sorted(x.name for x in tmp_path.iterdir()) == ["one.csv.gz", "two.csv.gz"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
