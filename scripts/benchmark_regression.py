"""Regression guard for the 300-question acceptance benchmark: compare a new run's verdicts with the locked baseline.

The baseline (artifacts/benchmark_baseline/<run>/baseline.json) holds every question's verdict as scored. Rule:
  * a baseline PASS must stay PASS;
  * a baseline PARTIALLY PASS must not fall to PARTIALLY FAIL or FAIL (it may rise to PASS).
Any other row may move either way; a fall is reported, but only a protected row's fall is a regression.
A question missing from the new run is reported as not run, never as kept.

    python scripts/benchmark_regression.py --new <verdicts.json | results.xlsx> [--baseline <baseline.json>]
                                           [--approved <file>] [--subset <subset.json>] [--json <out.json>]

--baseline: a locked baseline.json (default: full300e), or a verdict file of another run (Test A, for a later arm).
--subset: compare only the subset's questions (artifacts/benchmark_subsets/evidence_complete_159.json, ...).

--new: grade.py's verdicts.json ({"S-001": {"verdict": "PASS", ...}}), a plain {"S-001": "PASS"} map, or a results
workbook with a "Results" sheet holding "ID" and "Verdict" columns (read with the standard library).
--approved: a JSON list of question IDs whose regression the user has explicitly approved; each is still listed.
Exit status: 0 when no protected row regressed (outside --approved) and every baseline question was run; 1 otherwise.
The verdicts, the key and the scoring rules are read, never changed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "artifacts" / "benchmark_baseline" / "full300e_20261102" / "baseline.json"
RANK = {"FAIL": 0, "PARTIALLY FAIL": 1, "PARTIALLY PASS": 2, "PASS": 3}
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
       "rel": "http://schemas.openxmlformats.org/package/2006/relationships"}


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_results_workbook(path: Path, sheet: str = "Results") -> dict[str, str]:
    """{ID: Verdict} from a results workbook's sheet, with the standard library only."""
    rows = read_sheet(path, sheet)
    if rows and ("ID" not in rows[0] or "Verdict" not in rows[0]):
        raise SystemExit(f"{path.name}: sheet {sheet!r} has no ID and Verdict columns")
    return {r["ID"]: r.get("Verdict") for r in rows if r.get("ID")}


def read_sheet(path: Path, sheet: str = "Results") -> list[dict[str, str]]:
    """A workbook sheet as one {header: cell text} dict per row below the header, with the standard library only."""
    with zipfile.ZipFile(path) as z:
        sst = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
                sst.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")).findall("rel:Relationship", _NS)}
        target = None
        for s in wb.find("m:sheets", _NS).findall("m:sheet", _NS):
            if s.get("name") == sheet:
                target = rels[s.get(f"{{{_NS['r']}}}id")].lstrip("/")
                target = target if target.startswith("xl/") else "xl/" + target
        if target is None:
            raise SystemExit(f"{path.name}: no sheet named {sheet!r}")
        rows = []
        for row in ET.fromstring(z.read(target)).iter(f"{{{_NS['m']}}}row"):
            vals = {}
            for c in row.findall("m:c", _NS):
                v = c.find("m:v", _NS)
                if c.get("t") == "s" and v is not None:
                    vals[_col(c.get("r"))] = sst[int(v.text)]
                elif c.get("t") == "inlineStr":
                    vals[_col(c.get("r"))] = "".join(t.text or "" for t in c.iter(f"{{{_NS['m']}}}t"))
                elif v is not None:
                    vals[_col(c.get("r"))] = v.text
            rows.append(vals)
    if not rows:
        return []
    names = rows[0]
    return [{names[i]: v for i, v in r.items() if i in names} for r in rows[1:]]


def read_verdicts(path: Path) -> dict[str, str | None]:
    if path.suffix.lower() == ".xlsx":
        return read_results_workbook(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {qid: (v.get("verdict") if isinstance(v, dict) else v) for qid, v in data.items()}


def read_baseline(path: Path) -> tuple[str, dict[str, str]]:
    """(run name, {ID: verdict}) from a locked baseline.json, or from any verdict file read_verdicts accepts - so a
    later arm can be checked against Test A as well as against the official baseline."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "questions" in data:
            return data.get("run", path.stem), {q["id"]: q["verdict"] for q in data["questions"]}
    return path.stem, {q: v for q, v in read_verdicts(path).items() if v is not None}


def read_subset(path: Path) -> frozenset[str]:
    """The question IDs of a subset file (artifacts/benchmark_subsets/*.json: {"ids": [...]}) or a plain JSON list."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return frozenset(data["ids"] if isinstance(data, dict) else data)


def compare(baseline: dict[str, str], new: dict[str, str | None], approved: frozenset[str] = frozenset()) -> dict:
    out = {"kept_pass": [], "kept_partially_pass": [], "newly_pass_or_pp": [], "improved": [], "regressed": [],
           "approved_regressions": [], "worsened_unprotected": [], "unchanged_other": [], "not_run": [], "unknown_ids": [],
           "invalid_verdicts": []}
    for qid, old in baseline.items():
        v = new.get(qid)
        if v is None:
            out["not_run"].append(qid)
            continue
        if v not in RANK:
            out["invalid_verdicts"].append(f"{qid}: {v!r}")
            continue
        protected = old in ("PASS", "PARTIALLY PASS")
        fell = (old == "PASS" and v != "PASS") or (old == "PARTIALLY PASS" and RANK[v] < RANK["PARTIALLY PASS"])
        item = {"id": qid, "baseline": old, "new": v}
        if protected and fell:
            out["approved_regressions" if qid in approved else "regressed"].append(item)
        elif old == "PASS":
            out["kept_pass"].append(qid)
        elif old == "PARTIALLY PASS" and v == "PARTIALLY PASS":
            out["kept_partially_pass"].append(qid)
        elif RANK[v] > RANK[old]:
            out["improved"].append(item)
            if not protected and RANK[v] >= RANK["PARTIALLY PASS"]:
                out["newly_pass_or_pp"].append(qid)
        elif RANK[v] < RANK[old]:
            out["worsened_unprotected"].append(item)
        else:
            out["unchanged_other"].append(qid)
    out["unknown_ids"] = sorted(set(new) - set(baseline))
    n = len(baseline)

    def share(d):
        return sum(1 for q in d if d[q] in ("PASS", "PARTIALLY PASS"))

    out["summary"] = {
        "baseline_pass_pp": share(baseline), "baseline_total": n,
        "new_pass_pp": share({q: new[q] for q in baseline if new.get(q) in RANK}),
        "new_counts": {k: sum(1 for q in baseline if new.get(q) == k) for k in RANK},
        "ok": not out["regressed"] and not out["not_run"] and not out["invalid_verdicts"],
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--approved", type=Path)
    ap.add_argument("--subset", type=Path, help="compare only these questions (artifacts/benchmark_subsets/*.json)")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    run, baseline = read_baseline(a.baseline)
    new = read_verdicts(a.new)
    if a.subset:
        ids = read_subset(a.subset)
        missing = sorted(ids - set(baseline))
        if missing:
            raise SystemExit(f"subset {a.subset.name}: {len(missing)} IDs not in the baseline ({', '.join(missing[:5])})")
        baseline = {q: v for q, v in baseline.items() if q in ids}
        new = {q: v for q, v in new.items() if q in ids}
        run = f"{run} [{a.subset.stem}]"
    approved = frozenset(json.loads(a.approved.read_text(encoding="utf-8"))) if a.approved else frozenset()
    res = compare(baseline, new, approved)
    s = res["summary"]
    print(f"baseline {run}: PASS+PP {s['baseline_pass_pp']}/{s['baseline_total']}; new: PASS+PP {s['new_pass_pp']} "
          f"{s['new_counts']}")
    print(f"kept PASS {len(res['kept_pass'])}, kept PP {len(res['kept_partially_pass'])}, improved {len(res['improved'])} "
          f"(newly PASS/PP {len(res['newly_pass_or_pp'])}), worsened unprotected {len(res['worsened_unprotected'])}")
    for key in ("regressed", "approved_regressions"):
        for it in res[key]:
            print(f"  {key.upper()}: {it['id']} {it['baseline']} -> {it['new']}")
    for key in ("not_run", "invalid_verdicts", "unknown_ids"):
        if res[key]:
            print(f"  {key}: {res[key]}")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("RESULT:", "no protected regression" if s["ok"] else "REGRESSION OR INCOMPLETE RUN")
    return 0 if s["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
