"""scripts/build_benchmark_subsets.py: the frozen 159 / 141 subsets are what the locked full300e workbook gives."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_benchmark_subsets", ROOT / "scripts" / "build_benchmark_subsets.py")
BS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BS)


def _load(name):
    return json.loads((BS.OUT / f"{name}.json").read_text(encoding="utf-8"))


def test_the_committed_subsets_equal_a_fresh_recomputation():
    assert BS.main(["--check"]) == 0


def test_the_two_subsets_partition_the_locked_baseline():
    base = json.loads((BS.BASELINE_DIR / "baseline.json").read_text(encoding="utf-8"))
    every = {q["id"] for q in base["questions"]}
    ec, comp = set(_load("evidence_complete_159")["ids"]), set(_load("complement_141")["ids"])
    assert len(ec) == 159 and len(comp) == 141 and not ec & comp and ec | comp == every


def test_the_subset_baseline_figures_are_the_documented_ones():
    ec = _load("evidence_complete_159")
    assert ec["ids_sha256"].startswith("c2056be34239142f")   # the 3 Nov root-cause review's 159 rows
    assert ec["summary"]["pass_pp"] == 114
    assert ec["summary"]["verdicts"] == {"PASS": 110, "PARTIALLY PASS": 4, "PARTIALLY FAIL": 37, "FAIL": 8}
    cats = _load("complement_141")["categories"]
    assert len(cats["key_section_missing"]) == 94 and len(cats["no_section_key"]) == 47


def test_section_reading_follows_the_review():
    assert BS.sections("21 CFR 111.14; cross-ref 21 CFR 111.605 and 111.610") == ["111.14", "111.605", "111.610"]
    assert BS.sections("21 CFR 820.30(a)(1), 820.30") == ["820.30"]
    assert BS.sections("FD&C Act 201(h); Part 820") == []
    assert BS.sections("21 CFR 1271.3(d)") == ["1271.3"]
