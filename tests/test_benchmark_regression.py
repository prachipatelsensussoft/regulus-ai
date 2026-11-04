"""scripts/benchmark_regression.py: the protected-row rule and the locked 2 Nov baseline."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark_regression", ROOT / "scripts" / "benchmark_regression.py")
BR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BR)

BASE = {"A": "PASS", "B": "PARTIALLY PASS", "C": "PARTIALLY FAIL", "D": "FAIL", "E": "PASS"}


def test_a_pass_that_drops_is_a_regression():
    res = BR.compare(BASE, {**BASE, "A": "PARTIALLY PASS"})
    assert [r["id"] for r in res["regressed"]] == ["A"] and not res["summary"]["ok"]


def test_a_partially_pass_may_rise_but_not_fall_below_it():
    assert BR.compare(BASE, {**BASE, "B": "PASS"})["summary"]["ok"]
    res = BR.compare(BASE, {**BASE, "B": "PARTIALLY FAIL"})
    assert [r["id"] for r in res["regressed"]] == ["B"]


def test_unprotected_rows_move_freely_and_gains_are_counted():
    res = BR.compare(BASE, {**BASE, "C": "PASS", "D": "PARTIALLY FAIL"})
    assert res["summary"]["ok"]
    assert res["newly_pass_or_pp"] == ["C"] and {r["id"] for r in res["improved"]} == {"C", "D"}
    res = BR.compare(BASE, {**BASE, "C": "FAIL"})
    assert res["summary"]["ok"] and [r["id"] for r in res["worsened_unprotected"]] == ["C"]


def test_a_question_not_run_fails_the_check_and_is_never_counted_as_kept():
    new = dict(BASE)
    del new["E"]
    res = BR.compare(BASE, new)
    assert res["not_run"] == ["E"] and "E" not in res["kept_pass"] and not res["summary"]["ok"]


def test_approved_regressions_are_listed_apart_and_do_not_fail():
    res = BR.compare(BASE, {**BASE, "A": "FAIL"}, approved=frozenset({"A"}))
    assert res["summary"]["ok"] and [r["id"] for r in res["approved_regressions"]] == ["A"]


def test_an_unknown_verdict_label_is_invalid():
    res = BR.compare(BASE, {**BASE, "A": "OK"})
    assert res["invalid_verdicts"] and not res["summary"]["ok"]


def test_grade_py_verdicts_json_is_read(tmp_path):
    p = tmp_path / "verdicts.json"
    p.write_text(json.dumps({"A": {"verdict": "PASS", "sub": 2}, "B": {"verdict": "FAIL"}}), encoding="utf-8")
    assert BR.read_verdicts(p) == {"A": "PASS", "B": "FAIL"}


def test_another_runs_verdicts_can_be_the_baseline(tmp_path):
    p = tmp_path / "testA_verdicts.json"
    p.write_text(json.dumps({"A": {"verdict": "PASS"}, "B": {"verdict": "FAIL"}, "C": {"verdict": None}}), encoding="utf-8")
    assert BR.read_baseline(p) == ("testA_verdicts", {"A": "PASS", "B": "FAIL"})
    assert BR.read_baseline(BR.DEFAULT_BASELINE)[0] == "full300e"


def test_a_subset_limits_the_comparison_to_its_questions(tmp_path, capsys):
    base = tmp_path / "base.json"
    base.write_text(json.dumps(BASE), encoding="utf-8")
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"A": "PASS", "B": "PARTIALLY PASS", "C": "FAIL"}), encoding="utf-8")   # D, E not run
    subset = tmp_path / "abc.json"
    subset.write_text(json.dumps({"ids": ["A", "B", "C"]}), encoding="utf-8")
    assert BR.main(["--new", str(new), "--baseline", str(base), "--subset", str(subset)]) == 0
    assert "PASS+PP 2/3" in capsys.readouterr().out
    assert BR.main(["--new", str(new), "--baseline", str(base)]) == 1   # without the subset D and E are not run
    subset.write_text(json.dumps(["A", "Z"]), encoding="utf-8")
    with pytest.raises(SystemExit):
        BR.main(["--new", str(new), "--baseline", str(base), "--subset", str(subset)])


def test_the_locked_baseline_reproduces_the_2_nov_result():
    base = json.loads(BR.DEFAULT_BASELINE.read_text(encoding="utf-8"))
    verdicts = {q["id"]: q["verdict"] for q in base["questions"]}
    assert len(verdicts) == 300
    counts = {k: sum(1 for v in verdicts.values() if v == k) for k in BR.RANK}
    assert counts == {"PASS": 161, "PARTIALLY PASS": 23, "PARTIALLY FAIL": 89, "FAIL": 27}
    # the baseline compared with itself: nothing lost, nothing gained
    res = BR.compare(verdicts, verdicts)
    assert res["summary"]["ok"] and res["summary"]["new_pass_pp"] == 184
    assert base["protected"] == sorted(q for q, v in verdicts.items() if v in ("PASS", "PARTIALLY PASS"))


@pytest.mark.skipif(not (BR.DEFAULT_BASELINE.parent / "Regulus_AI_Acceptance_Benchmark_REV02_300Q_RESULTS_2026-11-02.xlsx").is_file(),
                    reason="baseline workbook copy not present")
def test_the_baseline_matches_its_source_workbook():
    wb = BR.DEFAULT_BASELINE.parent / "Regulus_AI_Acceptance_Benchmark_REV02_300Q_RESULTS_2026-11-02.xlsx"
    base = json.loads(BR.DEFAULT_BASELINE.read_text(encoding="utf-8"))
    assert BR.read_results_workbook(wb) == {q["id"]: q["verdict"] for q in base["questions"]}
