"""Merge the v2 selection runs, apply the fixed selection rule and write the selection record.

Inputs (training data only; written by scripts/v2_select.py and scripts/v2_select_apnea.py):
  results/v2/parts/staging_<task>_<group>.json   (staging grid, split into parallel runs)
  results/v2/selection_apnea.json
Outputs:
  results/v2/selection_staging.json   merged staging grid (every candidate, every task)
  results/v2/v2_selection.json        chosen staging and apnea candidates, scores, ranking, skipped
  docs/v2_selection.md                summary; every number is read from v2_selection.json

Rule (docs/v2_protocol.md): maximize mean cross-dataset sleep/wake kappa (pooled); reject a
candidate whose mean cross-dataset TST MAE is > 10 % worse than the v1 reference; among the
accepted candidates within 0.01 kappa of the best, take the simplest (fewest of: ensemble,
smoothing, extra features, class weighting), then the higher kappa. The same model spec is used
for the 3- and 4-class tasks; the multi-class maximizer is reported for information only.

Usage: python scripts/v2_finalize_selection.py [--docs-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from somnibeat import v2_models as vm  # noqa: E402

RES = ROOT / "results" / "v2"
PARTS = RES / "parts"
STAGING_OUT = RES / "selection_staging.json"
APNEA_IN = RES / "selection_apnea.json"
SELECTION_OUT = RES / "v2_selection.json"
DOC_OUT = ROOT / "docs" / "v2_selection.md"
TASKS = ("sleep_wake", "wake_nrem_rem", "four_class")
TIE = 0.01
GUARD = 1.10


def staging_complexity(spec: dict) -> int:
    return int(spec["model"] == "ensemble_mean") + int(spec["smoothing"] != "none") + \
        int(spec["feature_set"] != "v1") + int(spec["class_weight"] is not None)


def apnea_complexity(spec: dict) -> int:
    return int(spec["model"] == "ensemble_mean") + int(spec["class_weight"] is not None) + \
        int(spec["feature_set"] != "v1") + int(spec["train_on"] != "apnea-ecg")


def merge_staging() -> dict:
    parts = sorted(PARTS.glob("staging_*.json"))
    assert parts, "no staging runs found"
    by_name: dict[str, dict] = {}
    meta = None
    shas = set()
    for p in parts:
        d = json.loads(p.read_text())
        assert set(d["datasets_loaded"]) == {"cap", "slpdb"}, d["datasets_loaded"]
        shas.add((d["features_sha256"], d["extra_features_sha256"], d["v1_config_sha256"]))
        meta = meta or {k: v for k, v in d.items() if k not in ("candidates", "n_model_fits", "fit_seconds", "total_seconds")}
        for c in d["candidates"]:
            assert "skipped" not in c, c["spec"]
            e = by_name.setdefault(c["spec"]["name"], {"spec": c["spec"], "tasks": {}})
            assert e["spec"] == c["spec"]
            for t, r in c["tasks"].items():
                assert t not in e["tasks"], (c["spec"]["name"], t)
                e["tasks"][t] = r
    assert len(shas) == 1, "runs used different inputs"
    order = [vm.validate_spec(s, True)["name"] for s in vm.default_grid(True)]
    assert sorted(order) == sorted(by_name), "grid incomplete or extra candidates"
    for n in order:
        assert set(by_name[n]["tasks"]) == set(TASKS), (n, sorted(by_name[n]["tasks"]))
    out = dict(meta)
    out["merged_from"] = [str(p.relative_to(ROOT)) for p in parts]
    out["candidates"] = [by_name[n] for n in order]
    return out


def xds(c: dict, task: str, block: str, key: str) -> float:
    return c["tasks"][task]["cross_dataset"]["mean"][block][key]


def select_staging(st: dict) -> dict:
    cands = st["candidates"]
    ref = next(c for c in cands if c["spec"]["name"] == "v1_reference")
    ref_mae = xds(ref, "sleep_wake", "sleep_wake", "tst_mae_min")
    rows = []
    for c in cands:
        s = c["spec"]
        sw = c["tasks"]["sleep_wake"]
        a = sw["cross_dataset"]["train_cap_eval_slpdb"]["sleep_wake"]
        b = sw["cross_dataset"]["train_slpdb_eval_cap"]["sleep_wake"]
        rows.append({
            "name": s["name"], "model": s["model"], "class_weight": s["class_weight"],
            "smoothing": s["smoothing"], "feature_set": s["feature_set"],
            "complexity": staging_complexity(s),
            "xds_sw_kappa": xds(c, "sleep_wake", "sleep_wake", "kappa_pooled"),
            "fold_a_sw_kappa": a["kappa_pooled"]["point"], "fold_b_sw_kappa": b["kappa_pooled"]["point"],
            "xds_tst_mae_min": xds(c, "sleep_wake", "sleep_wake", "tst_mae_min"),
            "xds_awakenings_mae": xds(c, "sleep_wake", "sleep_wake", "awakenings_mae"),
            "xds_se_mae_pctpt": xds(c, "sleep_wake", "sleep_wake", "se_mae_pctpt"),
            "subject_cv_sw_kappa": sw["subject_cv"]["sleep_wake"]["kappa_pooled"]["point"],
            "subject_cv_tst_mae_min": sw["subject_cv"]["sleep_wake"]["tst_mae_min"]["point"],
            "xds_3class_kappa": xds(c, "wake_nrem_rem", "multiclass", "kappa_pooled"),
            "xds_4class_kappa": xds(c, "four_class", "multiclass", "kappa_pooled"),
            "subject_cv_3class_kappa": c["tasks"]["wake_nrem_rem"]["subject_cv"]["multiclass"]["kappa_pooled"]["point"],
            "subject_cv_4class_kappa": c["tasks"]["four_class"]["subject_cv"]["multiclass"]["kappa_pooled"]["point"],
            "guard_ok": xds(c, "sleep_wake", "sleep_wake", "tst_mae_min") <= GUARD * ref_mae,
        })
    ok = [r for r in rows if r["guard_ok"]]
    best = max(r["xds_sw_kappa"] for r in ok)
    tied = [r for r in ok if best - r["xds_sw_kappa"] < TIE]
    pick = min(tied, key=lambda r: (r["complexity"], -r["xds_sw_kappa"]))
    ranking = sorted(rows, key=lambda r: -r["xds_sw_kappa"])
    for i, r in enumerate(ranking, 1):
        r["rank"] = i
        r["within_tie_of_best"] = r["guard_ok"] and best - r["xds_sw_kappa"] < TIE
    refrow = next(r for r in rows if r["name"] == "v1_reference")
    mc_best = {t: max(rows, key=lambda r: r[k])
               for t, k in (("wake_nrem_rem", "xds_3class_kappa"), ("four_class", "xds_4class_kappa"))}
    v1chk = ref["tasks"]["sleep_wake"]["v1_protocol_check"]
    v1json = json.loads((ROOT / "results" / "sleep_wake.json").read_text())["models"]["gradient_boosting"]
    v1_internal = {"threshold_p_wake": v1json["threshold_p_wake"],
                   "kappa_pooled": v1json["internal_cv"]["metrics"]["kappa_pooled"]["point"],
                   "tst_mae_min": v1json["internal_cv"]["metrics"]["tst_mae_min"]["point"]}
    return {
        "rule": {"metric": "cross_dataset.mean.sleep_wake.kappa_pooled (task sleep_wake)",
                 "guard": f"cross_dataset.mean.sleep_wake.tst_mae_min <= {GUARD} x v1_reference",
                 "tie": f"accepted candidates within {TIE} kappa of the best -> fewest of "
                        "{ensemble, smoothing, extra features, class weighting}; then higher kappa",
                 "multiclass": "the chosen spec is also the 3- and 4-class model (no separate choice)"},
        "v1_reference_xds_tst_mae_min": ref_mae, "guard_tst_mae_limit_min": GUARD * ref_mae,
        "best_accepted_xds_sw_kappa": best,
        "tied_candidates": [r["name"] for r in sorted(tied, key=lambda r: -r["xds_sw_kappa"])],
        "chosen": pick, "v1_reference": refrow,
        "improvement_over_v1": {
            "xds_sw_kappa_abs": pick["xds_sw_kappa"] - refrow["xds_sw_kappa"],
            "xds_sw_kappa_rel_pct": 100 * (pick["xds_sw_kappa"] / refrow["xds_sw_kappa"] - 1),
            "xds_tst_mae_min_change": pick["xds_tst_mae_min"] - refrow["xds_tst_mae_min"],
            "xds_3class_kappa_abs": pick["xds_3class_kappa"] - refrow["xds_3class_kappa"],
            "xds_4class_kappa_abs": pick["xds_4class_kappa"] - refrow["xds_4class_kappa"],
            "subject_cv_sw_kappa_abs": pick["subject_cv_sw_kappa"] - refrow["subject_cv_sw_kappa"],
        },
        "multiclass_maximizers_info_only": {
            t: {"name": r["name"], "xds_kappa": r["xds_3class_kappa" if t == "wake_nrem_rem" else "xds_4class_kappa"]}
            for t, r in mc_best.items()},
        "v1_reproduction": {"v1_protocol_check": v1chk, "results_sleep_wake_json_internal_cv": v1_internal,
                            "identical": all(abs(v1chk[k] - v1_internal[k]) < 1e-12 for k in v1_internal)},
        "ranking": ranking,
    }


def select_apnea(ap: dict) -> dict:
    assert set(ap["datasets_loaded"]) == {"apnea-ecg", "slpdb"}, ap["datasets_loaded"]
    cands = ap["candidates"]
    ref = next(c for c in cands if c["spec"]["name"] == "v1_reference")
    ref_mae = ref["result"]["cross_dataset"]["mean"]["apnea_min_mae"]
    rows = []
    for c in cands:
        s, r = c["spec"], c["result"]
        m = r["cross_dataset"]["mean"]
        u = r["subject_cv_union"]["all"]["metrics"]
        rows.append({"name": s["name"], "model": s["model"], "class_weight": s["class_weight"],
                     "feature_set": s["feature_set"], "train_on": s["train_on"],
                     "complexity": apnea_complexity(s),
                     "xds_kappa": m["kappa_pooled"], "xds_apnea_min_mae": m["apnea_min_mae"],
                     "fold_a_kappa": r["cross_dataset"]["train_apnea-ecg_eval_slpdb"]["metrics"]["kappa_pooled"]["point"],
                     "fold_b_kappa": r["cross_dataset"]["train_slpdb_eval_apnea-ecg"]["metrics"]["kappa_pooled"]["point"],
                     "union_cv_kappa": u["kappa_pooled"]["point"], "union_cv_apnea_min_mae": u["apnea_min_mae"]["point"],
                     "guard_ok": m["apnea_min_mae"] <= GUARD * ref_mae})
    ok = [r for r in rows if r["guard_ok"]]
    best = max(r["xds_kappa"] for r in ok)
    tied = [r for r in ok if best - r["xds_kappa"] < TIE]
    pick = min(tied, key=lambda r: (r["complexity"], -r["xds_kappa"]))
    assert pick["name"] == ap["ranking"]["pick_by_cross_dataset_rule"], "rule mismatch with v2_select_apnea.py"
    ranking = sorted(rows, key=lambda r: (-r["xds_kappa"], r["complexity"]))
    for i, r in enumerate(ranking, 1):
        r["rank"] = i
    refrow = next(r for r in rows if r["name"] == "v1_reference")
    return {
        "rule": {"metric": "cross_dataset.mean.kappa_pooled (minute kappa; fold A apnea-ecg -> slpdb, fold B slpdb -> apnea-ecg)",
                 "guard": f"cross_dataset.mean.apnea_min_mae <= {GUARD} x v1_reference",
                 "tie": f"within {TIE} kappa -> fewest of {{ensemble, class weighting, extra features, slpdb in training}}",
                 "note": ap["ranking"]["note"]},
        "v1_reference_xds_apnea_min_mae": ref_mae, "guard_apnea_min_mae_limit": GUARD * ref_mae,
        "best_accepted_xds_kappa": best,
        "tied_candidates": [r["name"] for r in sorted(tied, key=lambda r: -r["xds_kappa"])],
        "chosen": pick, "v1_reference": refrow,
        "improvement_over_v1": {"xds_kappa_abs": pick["xds_kappa"] - refrow["xds_kappa"],
                                "xds_kappa_rel_pct": 100 * (pick["xds_kappa"] / refrow["xds_kappa"] - 1),
                                "xds_apnea_min_mae_change": pick["xds_apnea_min_mae"] - refrow["xds_apnea_min_mae"]},
        "v1_reproduction": ap["v1_protocol_check"],
        "ranking": ranking,
    }


def skipped_list(st: dict) -> list[dict]:
    return [
        {"candidate": "1. More CAP training records (remaining non-NFLE and NFLE records)",
         "reason": "Extra CAP records were not available (no features or RR cache for them) when the grid "
                   "was run; per the task it is skipped, not scored. If it becomes available it is judged on "
                   "fold A only (protocol clarification) and combined with the chosen candidate."},
        {"candidate": "3. Temporal smoothing for apnea minutes",
         "reason": "The protocol's smoothing candidate is an HMM over sleep stages; the apnea grid in "
                   "scripts/v2_select_apnea.py has no smoothing option, so none was added."},
        {"candidate": "5. Threshold rules other than the v1 rule",
         "reason": "Only class weighting varies; every binary threshold uses v1's rule (max kappa on a 0.05-0.95 "
                   "grid of out-of-fold training predictions, nested inside each training side). No alternative "
                   "threshold rules are in the protocol list."},
    ]


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def write_doc(sel: dict) -> None:
    s, a = sel["staging"], sel["apnea"]
    c, r, imp = s["chosen"], s["v1_reference"], s["improvement_over_v1"]
    ac, ar, aimp = a["chosen"], a["v1_reference"], a["improvement_over_v1"]
    L = []
    L += ["# v2 selection (training data only)", "",
          "Generated by `scripts/v2_finalize_selection.py` from `results/v2/v2_selection.json`. Every number "
          "below is read from that file. No UCDDB or HMC data was loaded, predicted or scored. Rule and folds: "
          "`docs/v2_protocol.md`.", ""]
    L += ["## Staging choice", "",
          f"- Chosen: **`{c['name']}`** (model `{c['model']}`, class weight `{c['class_weight']}`, "
          f"smoothing `{c['smoothing']}`, features `{c['feature_set']}`).",
          f"- Mean cross-dataset sleep/wake kappa: {fmt(c['xds_sw_kappa'])} vs v1 reference {fmt(r['xds_sw_kappa'])} "
          f"({imp['xds_sw_kappa_abs']:+.3f}, {imp['xds_sw_kappa_rel_pct']:+.1f} %).",
          f"- Mean cross-dataset TST MAE: {fmt(c['xds_tst_mae_min'], 1)} min vs v1 {fmt(r['xds_tst_mae_min'], 1)} min "
          f"(guard limit {fmt(s['guard_tst_mae_limit_min'], 1)} min).",
          f"- Best accepted kappa {fmt(s['best_accepted_xds_sw_kappa'])}; candidates within 0.01 of it: "
          f"{len(s['tied_candidates'])} ({', '.join('`' + n + '`' for n in s['tied_candidates'])}). "
          "The simplest of them was taken.",
          f"- Same spec for 3 and 4 classes: cross-dataset kappa {fmt(c['xds_3class_kappa'])} (v1 {fmt(r['xds_3class_kappa'])}) "
          f"and {fmt(c['xds_4class_kappa'])} (v1 {fmt(r['xds_4class_kappa'])}).",
          ]
    for t, lab in (("wake_nrem_rem", "3-class"), ("four_class", "4-class")):
        m = s["multiclass_maximizers_info_only"][t]
        L.append(f"- For information only, the {lab} cross-dataset kappa would be maximized by `{m['name']}` "
                 f"({fmt(m['xds_kappa'])}); the protocol does not allow a separate choice.")
    v = s["v1_reproduction"]
    L += [f"- v1 reproduction check (v1's own pooled-OOF threshold): threshold {v['v1_protocol_check']['threshold_p_wake']}, "
          f"kappa {fmt(v['v1_protocol_check']['kappa_pooled'], 4)}, TST MAE {fmt(v['v1_protocol_check']['tst_mae_min'], 2)} min; "
          f"identical to results/sleep_wake.json: {v['identical']}.", ""]
    L += ["### Staging ranking (all 36 candidates)", "",
          "xds = mean of fold A (train CAP, evaluate slpdb) and fold B (train slpdb, evaluate CAP). "
          "Subject CV = subject-grouped 5-fold CV on CAP + slpdb (secondary). Complexity counts ensemble, "
          "smoothing, extra features and class weighting.", "",
          "| # | Candidate | Cx | xds SW kappa | Fold A | Fold B | xds TST MAE (min) | Guard | xds awak. MAE | "
          "Subject-CV SW kappa | xds 3-cl kappa | xds 4-cl kappa |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in s["ranking"]:
        name = f"**{row['name']}**" if row["name"] == c["name"] else row["name"]
        L.append(f"| {row['rank']} | {name} | {row['complexity']} | {fmt(row['xds_sw_kappa'])} | {fmt(row['fold_a_sw_kappa'])} | "
                 f"{fmt(row['fold_b_sw_kappa'])} | {fmt(row['xds_tst_mae_min'], 1)} | {'ok' if row['guard_ok'] else 'rejected'} | "
                 f"{fmt(row['xds_awakenings_mae'], 1)} | {fmt(row['subject_cv_sw_kappa'])} | {fmt(row['xds_3class_kappa'])} | "
                 f"{fmt(row['xds_4class_kappa'])} |")
    L += ["", "## Apnea choice", "",
          f"- Chosen: **`{ac['name']}`** (model `{ac['model']}`, class weight `{ac['class_weight']}`, "
          f"features `{ac['feature_set']}`, trained on `{ac['train_on']}`).",
          f"- Mean cross-dataset minute kappa: {fmt(ac['xds_kappa'])} vs v1 reference {fmt(ar['xds_kappa'])} "
          f"({aimp['xds_kappa_abs']:+.3f}, {aimp['xds_kappa_rel_pct']:+.1f} %).",
          f"- Mean cross-dataset apnea-minute MAE: {fmt(ac['xds_apnea_min_mae'], 1)} vs v1 {fmt(ar['xds_apnea_min_mae'], 1)} "
          f"(guard limit {fmt(a['guard_apnea_min_mae_limit'], 1)}).",
          f"- Candidates within 0.01 of the best accepted kappa ({fmt(a['best_accepted_xds_kappa'])}): "
          f"{', '.join('`' + n + '`' for n in a['tied_candidates'])}.",
          f"- Note: {a['rule']['note']}",
          f"- v1 reproduction (apnea internal CV): identical to results/apnea.json: {a['v1_reproduction']['identical']}.", "",
          "### Apnea ranking", "",
          "| # | Candidate | Cx | xds kappa | Fold A | Fold B | xds apnea-min MAE | Guard | Union-CV kappa | Union-CV MAE |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for row in a["ranking"]:
        name = f"**{row['name']}**" if row["name"] == ac["name"] else row["name"]
        L.append(f"| {row['rank']} | {name} | {row['complexity']} | {fmt(row['xds_kappa'])} | {fmt(row['fold_a_kappa'])} | "
                 f"{fmt(row['fold_b_kappa'])} | {fmt(row['xds_apnea_min_mae'], 1)} | {'ok' if row['guard_ok'] else 'rejected'} | "
                 f"{fmt(row['union_cv_kappa'])} | {fmt(row['union_cv_apnea_min_mae'], 1)} |")
    L += ["", "## Skipped", ""]
    for k in sel["skipped"]:
        L.append(f"- {k['candidate']}: {k['reason']}")
    L += ["", "## Next steps (protocol)", "",
          "Freeze this choice (tag `v2-frozen`) before any test-set prediction, then predict HMC and UCDDB "
          "once, side by side with v1.", ""]
    DOC_OUT.write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-only", action="store_true")
    a = ap.parse_args()
    if a.docs_only:
        write_doc(json.loads(SELECTION_OUT.read_text()))
        return 0
    st = merge_staging()
    STAGING_OUT.write_text(json.dumps(st, indent=2))
    apn = json.loads(APNEA_IN.read_text())
    sel = {
        "protocol": "docs/v2_protocol.md",
        "note_path": "The protocol names results/v2_selection.json; v2 outputs live under results/v2/, so the "
                     "selection record is results/v2/v2_selection.json.",
        "training_datasets_only": {"staging": st["datasets_loaded"], "apnea": apn["datasets_loaded"]},
        "inputs": {"staging": str(STAGING_OUT.relative_to(ROOT)), "apnea": str(APNEA_IN.relative_to(ROOT)),
                   "stage_features_sha256": st["features_sha256"],
                   "stage_extra_features_sha256": st["extra_features_sha256"],
                   "apnea_inputs_sha256": apn["inputs_sha256"], "v1_config_sha256": st["v1_config_sha256"]},
        "hyperparameters": st["hyperparameters"],
        "staging": select_staging(st),
        "apnea": select_apnea(apn),
        "skipped": skipped_list(st),
    }
    SELECTION_OUT.write_text(json.dumps(sel, indent=2))
    write_doc(sel)
    s, p = sel["staging"], sel["apnea"]
    print(f"staging pick {s['chosen']['name']}: xds kappa {s['chosen']['xds_sw_kappa']:.3f} "
          f"(v1 {s['v1_reference']['xds_sw_kappa']:.3f}); apnea pick {p['chosen']['name']}: "
          f"xds kappa {p['chosen']['xds_kappa']:.3f} (v1 {p['v1_reference']['xds_kappa']:.3f})")
    print(f"wrote {STAGING_OUT}, {SELECTION_OUT}, {DOC_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
