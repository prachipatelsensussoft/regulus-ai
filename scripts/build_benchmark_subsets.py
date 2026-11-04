"""The frozen 300-question benchmark subsets, recomputed from the locked full300e results workbook (no model, no DB).

    python scripts/build_benchmark_subsets.py            write artifacts/benchmark_subsets/*.json
    python scripts/build_benchmark_subsets.py --check    recompute and compare with the committed files (exit 1 if
                                                         they differ)

Rule (the 3 Nov root-cause review's `all_expected_supplied`): the CFR section numbers in a row's "Expected citation
(verified key)" and in its "Sections supplied" are read with one pattern (NNN.NNN, optional letter). A row is
EVIDENCE-COMPLETE when it names at least one section and every one of them was supplied to the model in full300e.
  evidence_complete_159   every key section supplied (protects evidence-complete answers; cannot show retrieval gains)
  complement_141          the rest: key_section_missing (some key section not supplied) and no_section_key (the key
                          names a part, an Act section or nothing) - where retrieval gains have to be measured
The workbook is read, never changed; the subsets are fixed for the whole programme of work, whatever later runs supply.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "artifacts" / "benchmark_baseline" / "full300e_20261102"
WORKBOOK = BASELINE_DIR / "Regulus_AI_Acceptance_Benchmark_REV02_300Q_RESULTS_2026-11-02.xlsx"
OUT = ROOT / "artifacts" / "benchmark_subsets"
SEC = re.compile(r"(?<![\d.])(\d{1,4})\.(\d{1,4}[a-z]?)(?![\d])")


def _guard():
    spec = importlib.util.spec_from_file_location("benchmark_regression", ROOT / "scripts" / "benchmark_regression.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sections(text: str | None) -> list[str]:
    out: list[str] = []
    for m in SEC.finditer(str(text or "")):
        s = f"{m.group(1)}.{m.group(2)}"
        if s not in out:
            out.append(s)
    return out


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def build(workbook: Path = WORKBOOK) -> dict[str, dict]:
    rows = _guard().read_sheet(workbook)
    source = {"workbook": workbook.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
              "sheet": "Results", "run": "full300e"}
    complete, missing, nokey = [], [], []
    detail = {}
    for r in rows:
        exp, sup = sections(r.get("Expected citation (verified key)")), sections(r.get("Sections supplied"))
        qid = r["ID"]
        detail[qid] = {"split": r.get("Split"), "verdict": r.get("Verdict"), "expected_sections": exp,
                       "supplied_sections": sup, "missing_sections": [s for s in exp if s not in sup]}
        (nokey if not exp else complete if all(s in sup for s in exp) else missing).append(qid)

    def summary(ids):
        return {"questions": len(ids), "verdicts": dict(Counter(detail[q]["verdict"] for q in ids)),
                "pass_pp": sum(detail[q]["verdict"] in ("PASS", "PARTIALLY PASS") for q in ids),
                "split": dict(Counter(detail[q]["split"] for q in ids))}

    rule = ("every CFR section named in 'Expected citation (verified key)' appears in 'Sections supplied' (full300e); "
            "rows whose key names no section are excluded")
    ec = sorted(complete)
    comp = sorted(missing + nokey)
    return {
        "evidence_complete_159": {"name": "evidence_complete_159", "rule": rule, "source": source, "ids": ec,
                                  "ids_sha256": ids_sha256(ec), "summary": summary(ec),
                                  "questions": {q: detail[q] for q in ec}},
        "complement_141": {"name": "complement_141", "rule": "the 300 benchmark questions not in evidence_complete_159",
                           "source": source, "ids": comp, "ids_sha256": ids_sha256(comp), "summary": summary(comp),
                           "categories": {"key_section_missing": sorted(missing), "no_section_key": sorted(nokey)},
                           "questions": {q: detail[q] for q in comp}},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    built = build()
    if len(built["evidence_complete_159"]["ids"]) != 159 or len(built["complement_141"]["ids"]) != 141:
        print("unexpected subset sizes:", {k: len(v["ids"]) for k, v in built.items()})
        return 1
    bad = 0
    for name, data in built.items():
        path = OUT / f"{name}.json"
        text = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
        if a.check:
            same = path.is_file() and path.read_text(encoding="utf-8") == text
            bad += not same
            print(f"{name}: {'matches' if same else 'DIFFERS FROM'} {path.relative_to(ROOT).as_posix()}")
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            s = data["summary"]
            print(f"{name}: {s['questions']} questions, PASS+PP {s['pass_pp']}, {s['verdicts']}, ids sha256 {data['ids_sha256'][:16]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
