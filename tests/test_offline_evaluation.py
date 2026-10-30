"""Offline evaluation (regulus.evaluation.offline_evaluation) assesses the SAVED answer against the SAVED evidence.

Pinned: the default run never evaluates; the offline pass never answers or generates; the evaluator
receives the record's own answer and evidence, in prompt order, plus a ValidationReport and AnswerPlan
rebuilt losslessly; records are matched by question ID and anything missing, corrupt or inconsistent is
NOT EVALUABLE with a reason, never reconstructed; results go to a new file and the run's files stay byte
for byte; one row's failure stays in that row; and on every deterministic pipeline path the offline
assessment equals the live one given the same (committed) evaluator interface.

Answers are SCRIPTED in place of the model over the in-memory table: no model, no database, no GPU.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import regulus.accuracy.evaluator as EV
from regulus.accuracy import evaluate_safely
from regulus.accuracy.alignment import LexicalAligner
from regulus.accuracy.review import FAIL, NOT_EVALUATED, PASS
from regulus.evaluation import offline_evaluation as O
from regulus.rag import RagPipeline
from regulus.traceability import measurement as M
from tests.test_accuracy_observer import CASES, Scripted, answer_one
from tests.test_measurement_record import A, B, C, INVENTED, RUN, build, focused, runner

# scripted model drafts: the model-drafted path (tests/conftest.py)
pytestmark = pytest.mark.usefixtures("model_drafted_path")

ROOT = Path(__file__).resolve().parents[1]
CLEAN, ONLY_C = f"{A} [1] {B} [2] {C} [3]", f"{C} [3]"

# (what the pipeline is asked, how the scripted answer is produced)
SCRIPT = [
    (CASES[0][0], lambda: answer_one(*CASES[0])),        # clean, verbatim
    (focused(CLEAN).question, lambda: focused(CLEAN)),    # focused, every requirement
    (focused(INVENTED).question, lambda: focused(INVENTED)),  # invented condition removed by repair
    (CASES[3][0], lambda: answer_one(*CASES[3])),        # no evidence: the model is not called
    (focused(ONLY_C).question, lambda: focused(ONLY_C)),  # focused, two requirements omitted
]


class Queue:
    """A pipeline stand-in: hands out the scripted answers in order; None raises, as a database error would."""

    def __init__(self, makers):
        self.makers = list(makers)

    def answer(self, question):
        make = self.makers.pop(0)
        if make is None:
            raise RuntimeError("database cancelled the query")
        out = make()
        assert out.question == question
        return out


def make_run(tmp_path, script=SCRIPT, fail=()):
    run_dir = tmp_path / RUN["run_id"]
    run_dir.mkdir()
    qs = [SimpleNamespace(question_id=f"Q-{i:03}", question=q) for i, (q, _) in enumerate(script, start=1)]
    makers = [None if q.question_id in fail else make for q, (_, make) in zip(qs, script)]
    runner.run_questions(Queue(makers), qs, {q.question_id: i for i, q in enumerate(qs, start=1)}, RUN, run_dir)
    return run_dir


def evaluate(run_dir, out, **kw):
    return O.evaluate_run(run_dir, out, root=ROOT, **kw)


def read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def results(path):
    return [line["result"] for line in read(path)]


def live(out):
    """The live assessment, as serve_ui makes it (repair record included; no identifier index)."""
    d = evaluate_safely(out.question, out.answer, package=out.package, evidence=out.evidence_snapshot,
                        citations=out.citations, validation=out.validation, coverage=out.coverage,
                        model_called=out.model_called, fallback_type=out.metadata.get("fallback_type"),
                        scope_status=(out.metadata.get("question_analysis") or {}).get("scope_status"),
                        answerability=out.metadata.get("answerability"), repair=out.metadata.get("validation"),
                        aligner=LexicalAligner()).to_dict()
    d.pop("elapsed_ms")
    return json.loads(json.dumps(d, default=str))


def reseal(path, qid, edit):
    """Rewrite one record in a COPY of a run, re-sealing its hash, as an editor who knows the scheme would."""
    lines = read(path)
    for rec in lines:
        if rec["question_id"] == qid:
            edit(rec)
            rec["record_sha256"] = M.record_hash(rec)
    Path(path).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in lines), encoding="utf-8")


# ------------------------------------------------------------------ 1-2. separation from answering


def test_the_default_run_never_evaluates(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("the answering run called the evaluator")

    monkeypatch.setattr(EV, "evaluate", forbidden)
    monkeypatch.setattr(O, "evaluate_safely", forbidden)
    run_dir = make_run(tmp_path)
    assert {m["evaluation"]["state"] for m in read(run_dir / M.MEASUREMENT_FILE)} == {"not_run"}


def test_importing_the_runner_loads_neither_the_offline_evaluator_nor_runs_anything():
    code = ("import sys; sys.path.insert(0, 'scripts'); import run_acceptance_test;"
            "print('regulus.evaluation.offline_evaluation' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_offline_evaluation_never_answers_or_generates(tmp_path, monkeypatch):
    run_dir = make_run(tmp_path)

    def forbidden(*a, **k):
        raise AssertionError("offline evaluation reached the answering pipeline")

    monkeypatch.setattr(RagPipeline, "answer", forbidden)
    monkeypatch.setattr(Scripted, "generate", forbidden)
    summary = evaluate(run_dir, tmp_path / "eval.jsonl")
    assert summary["counts"].get(O.EVALUATION_ERROR, 0) == 0
    assert all(r["outcome"] == O.EVALUATED for r in results(tmp_path / "eval.jsonl"))


# ------------------------------------------------------------------ 3-4. exactly the saved data


def test_the_saved_answer_and_evidence_reach_the_evaluator_in_prompt_order(tmp_path, monkeypatch):
    run_dir = make_run(tmp_path)
    seen = []
    real = O.evaluate_safely

    def spy(question, answer, **kw):
        seen.append((question, answer, kw))
        return real(question, answer, **kw)

    monkeypatch.setattr(O, "evaluate_safely", spy)
    evaluate(run_dir, tmp_path / "eval.jsonl")
    records = read(run_dir / M.MEASUREMENT_FILE)
    probes = [s for s in seen if "identifier_index" in s[2]]       # the discarded parity probe, one per record
    seen = [s for s in seen if "identifier_index" not in s[2]]     # the evaluation that is recorded
    assert len(seen) == len(probes) == len(records)
    assert all(isinstance(kw["identifier_index"], O._IndexProbe) for _, _, kw in probes)
    for (question, answer, kw), rec in zip(seen, records):
        assert question == rec["question"]["text"]
        assert answer.encode() == rec["text"]["delivered_answer"]["text"].encode()
        saved = [i["text"] for i in rec["evidence"]["snapshot"]["items"]]
        assert [i["text"] for i in kw["evidence"]["items"]] == saved
        assert [i["order"] for i in kw["evidence"]["items"]] == list(range(len(saved)))
        if kw["package"] is not None:
            assert [s.text for s in kw["package"].sources] == saved
        assert "identifier_index" not in kw
        v = rec["validation"]                                       # the live UI's metadata["validation"] shape
        assert kw["repair"] == ({"initial": v["initial"], "final": v["final"], "repair": v["repair"]}
                                if v["state"] == M.CAPTURED else None)


@pytest.mark.parametrize("make", [
    *[(lambda q=q, r=r: answer_one(q, r)) for q, r in CASES],
    lambda: focused(CLEAN), lambda: focused(INVENTED), lambda: focused(ONLY_C),
])
def test_the_offline_assessment_equals_the_live_one(make):
    out = make()
    rec = json.loads(json.dumps(build(out)))
    row = {"question_id": "Q-1", "question": out.question, "answer": out.answer}
    result, _ = O.evaluate_record(rec, row, run_id=RUN["run_id"], ordinal=1, evaluator={})
    expected = live(out)
    assert result["outcome"] == O.EVALUATED
    assert result["assessment"] == expected                 # every dimension, claim verdict, completeness decision
    assert result["score"] == expected["accuracy"] and result["review"]["status"] == result["review_status"]


def test_the_answer_plan_is_what_decides_a_focused_omission():
    """Why the record captures it: without the plan, the focus-gap check cannot run and the answer passes."""
    out = focused(ONLY_C)
    rec = json.loads(json.dumps(build(out)))
    row = {"question_id": "Q-1", "question": out.question, "answer": out.answer}
    kwargs = O._inputs(rec, row, RUN["run_id"])
    with_plan = evaluate_safely(kwargs.pop("question"), kwargs.pop("answer"), aligner=LexicalAligner(), **kwargs)
    kwargs["package"] = O.SavedPackage(O.load_evidence(rec["evidence"]["snapshot"]), None, [])
    without = evaluate_safely(out.question, out.answer, aligner=LexicalAligner(), **kwargs)
    assert with_plan.to_dict()["completeness"] is False and without.to_dict()["completeness"] is not False


# ------------------------------------------------------------------ lossless rebuilding


@pytest.mark.parametrize("make", [*[(lambda q=q, r=r: answer_one(q, r)) for q, r in CASES], lambda: focused(INVENTED)])
def test_validation_report_and_answer_plan_rebuild_exactly(make):
    out = make()
    if out.validation is not None:
        saved = json.loads(json.dumps(out.validation.to_dict()))
        rebuilt = O.validation_report_from_dict(saved)
        assert rebuilt.to_dict() == saved
        assert (rebuilt.status, rebuilt.problems, rebuilt.unaddressed) == \
               (out.validation.status, out.validation.problems, out.validation.unaddressed)
    if out.package is not None and out.package.answer_plan is not None:
        assert O.answer_plan_from_dict(json.loads(json.dumps(out.package.answer_plan.to_dict()))) == out.package.answer_plan


def test_an_unknown_or_inconsistent_field_is_refused_not_dropped():
    out = focused(INVENTED)
    report = json.loads(json.dumps(out.validation.to_dict()))
    with pytest.raises(O.UnsupportedRecord):
        O.validation_report_from_dict({**report, "ok": not report["ok"]})
    with pytest.raises(O.UnsupportedRecord):
        O.validation_report_from_dict({**report, "problems": [{**report["problems"][0], "new_field": 1}]}
                                      if report["problems"] else {k: v for k, v in report.items() if k != "status"})
    plan = json.loads(json.dumps(out.package.answer_plan.to_dict()))
    with pytest.raises(O.UnsupportedRecord):
        O.answer_plan_from_dict({**plan, "new_field": 1})
    with pytest.raises(O.UnsupportedRecord):
        O.answer_plan_from_dict({**plan, "groups": [{**plan["groups"][0], "items": [999]}]})


def test_the_adapter_provides_every_package_attribute_the_evaluator_reads():
    read_attrs = set()
    for path in (ROOT / "src/regulus/accuracy").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        read_attrs |= set(re.findall(r"\bpackage\.([a-z_]+)", text))
        read_attrs |= set(re.findall(r"getattr\(package, \"([a-z_]+)\"", text))
    assert read_attrs <= set(O.SavedPackage.__slots__), read_attrs - set(O.SavedPackage.__slots__)


# ------------------------------------------------------------------ 5-8. matching and refusal


def test_records_are_matched_by_question_id_not_position(tmp_path):
    run_dir = make_run(tmp_path)
    measured = (run_dir / M.MEASUREMENT_FILE).read_text(encoding="utf-8").splitlines()
    (run_dir / M.MEASUREMENT_FILE).write_text("\n".join(reversed(measured)) + "\n", encoding="utf-8")
    evaluate(run_dir, tmp_path / "eval.jsonl")
    by_id = {r["question_id"]: r["record_sha256"] for r in read(run_dir / M.MEASUREMENT_FILE)}
    got = results(tmp_path / "eval.jsonl")
    assert [r["question_id"] for r in got] == [f"Q-{i:03}" for i in range(1, len(SCRIPT) + 1)]   # answers order
    assert all(r["source"]["measurement_record_sha256"] == by_id[r["question_id"]] for r in got)
    assert all(r["outcome"] == O.EVALUATED for r in got)


def test_an_answers_row_that_disagrees_with_its_record_is_not_evaluable(tmp_path):
    run_dir = make_run(tmp_path)
    rows = read(run_dir / "answers.jsonl")
    rows[1]["answer"] = rows[1]["answer"] + " (edited)"
    rows[2]["question_id"] = rows[0]["question_id"]                    # a duplicated ID
    (run_dir / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    got = results(evaluate(run_dir, tmp_path / "eval.jsonl")["output"])
    assert [r["outcome"] for r in got[:3]] == [O.NOT_EVALUABLE] * 3
    assert "answer differs" in got[1]["reason"]
    assert "more than once" in got[0]["reason"] and "more than once" in got[2]["reason"]
    assert [r["outcome"] for r in got[3:]] == [O.EVALUATED] * (len(SCRIPT) - 3)


@pytest.mark.parametrize("edit,why", [
    (lambda r: r["evidence"].update(snapshot=None, provenance=M.MISSING, reason="not captured"), "no exact evidence"),
    (lambda r: r["evidence"]["snapshot"]["items"][0].update(text="tampered"), "integrity"),
    (lambda r: r.pop("evaluator_inputs"), "predates evaluator-input capture"),
    (lambda r: r["validation"]["final"].update(ok=not r["validation"]["final"]["ok"]), "round-trip"),
    (lambda r: r["question"].update(matches_pipeline_question=False), "different question"),
    (lambda r: r["validation"].update(initial=None), "no initial report"),
    (lambda r: r["validation"].update(initial={"status": "valid"}), "initial validation report is malformed"),
    (lambda r: r["validation"]["initial"]["problems"].append({"code": "x", "severity": "error", "detail": "",
                                                             "sentence": None, "new_field": 1}), "malformed"),
    (lambda r: r["validation"].update(repair=["invented_conditions_removed"]), "repair kind is malformed"),
])
def test_missing_corrupt_or_unsupported_data_is_not_evaluable(tmp_path, edit, why):
    run_dir = make_run(tmp_path)
    reseal(run_dir / M.MEASUREMENT_FILE, "Q-002", edit)
    got = results(evaluate(run_dir, tmp_path / "eval.jsonl")["output"])
    assert got[1]["outcome"] == O.NOT_EVALUABLE and why in got[1]["reason"] and got[1]["assessment"] is None
    assert all(r["outcome"] == O.EVALUATED for i, r in enumerate(got) if i != 1)


def test_an_invalid_record_hash_is_rejected(tmp_path):
    run_dir = make_run(tmp_path)
    lines = read(run_dir / M.MEASUREMENT_FILE)
    lines[0]["text"]["delivered_answer"]["text"] = "edited"           # not re-sealed
    (run_dir / M.MEASUREMENT_FILE).write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    got = results(evaluate(run_dir, tmp_path / "eval.jsonl")["output"])
    assert got[0]["outcome"] == O.NOT_EVALUABLE and "record_sha256 does not match" in got[0]["reason"]


ARCHIVED = sorted(p for p in (ROOT / "artifacts/acceptance_test").glob("acceptance_*") if (p / "answers.jsonl").is_file())


@pytest.mark.parametrize("run_dir", ARCHIVED, ids=[p.name for p in ARCHIVED])
def test_archived_runs_are_not_evaluable_and_untouched(tmp_path, run_dir):
    listing = sorted(p.name for p in run_dir.iterdir())
    summary = evaluate(run_dir, tmp_path / "eval.jsonl")
    got = results(tmp_path / "eval.jsonl")
    assert summary["inputs_unchanged"] and summary["rows"] == len(got) == 300
    assert {r["outcome"] for r in got} == {O.NOT_EVALUABLE}
    assert all("no measurement record" in r["reason"] and "nothing is reconstructed" in r["reason"] for r in got)
    assert sorted(p.name for p in run_dir.iterdir()) == listing


# ------------------------------------------------------------------ 9-11. files and failure isolation


def test_the_run_files_stay_byte_identical_and_an_earlier_result_is_never_overwritten(tmp_path):
    run_dir = make_run(tmp_path)
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    first = evaluate(run_dir, tmp_path / "eval.jsonl")
    kept = (tmp_path / "eval.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        evaluate(run_dir, tmp_path / "eval.jsonl")
    assert (tmp_path / "eval.jsonl").read_bytes() == kept and first["inputs_unchanged"]
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before


def test_the_cli_writes_a_new_file_and_leaves_the_run_alone(tmp_path):
    run_dir = make_run(tmp_path)
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    for name in ("a.jsonl", "b.jsonl"):
        assert runner.evaluate_offline(run_dir, str(tmp_path / name)) == 0
    assert results(tmp_path / "a.jsonl") == results(tmp_path / "b.jsonl")
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before


def test_one_failing_row_stays_in_that_row(tmp_path, monkeypatch):
    run_dir = make_run(tmp_path, fail={"Q-002"})                       # the pipeline itself failed on Q-002
    lines = (run_dir / M.MEASUREMENT_FILE).read_text(encoding="utf-8").splitlines()
    lines.insert(3, "{not json")                                       # and a corrupt line mid-file
    (run_dir / M.MEASUREMENT_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    real, target = EV.evaluate, read(run_dir / "answers.jsonl")[4]["answer"]

    def boom(question, answer, **kw):                                  # the evaluator fails on Q-005's answer
        if answer == target:
            raise RuntimeError("evaluator defect")
        return real(question, answer, **kw)

    monkeypatch.setattr(EV, "evaluate", boom)
    summary = evaluate(run_dir, tmp_path / "eval.jsonl")
    got = results(tmp_path / "eval.jsonl")
    assert [r["question_id"] for r in got] == [f"Q-{i:03}" for i in range(1, 6)]
    assert got[1]["outcome"] == O.NOT_EVALUABLE and "pipeline_error" in got[1]["reason"]
    assert got[4]["outcome"] == O.EVALUATION_ERROR and got[4]["review_status"] == NOT_EVALUATED
    assert "evaluator defect" in got[4]["reason"]
    assert [r["outcome"] for r in (got[0], got[2], got[3])] == [O.EVALUATED] * 3
    assert any("unreadable" in p for p in summary["problems"])


def test_a_failure_is_distinguishable_from_a_legitimate_not_evaluated():
    out = answer_one(*CASES[1])                                        # the evaluator abstains on this one
    rec = json.loads(json.dumps(build(out)))
    result, _ = O.evaluate_record(rec, {"question_id": "Q-1", "question": out.question, "answer": out.answer},
                                  run_id=RUN["run_id"], ordinal=1, evaluator={})
    assert (result["outcome"], result["review_status"]) == (O.EVALUATED, NOT_EVALUATED)


# ------------------------------------------------------------------ 12. hashes and reproducibility


def test_results_are_reproducible_and_their_hash_verifies(tmp_path):
    run_dir = make_run(tmp_path)
    evaluate(run_dir, tmp_path / "a.jsonl")
    evaluate(run_dir, tmp_path / "b.jsonl")
    a, b = read(tmp_path / "a.jsonl"), read(tmp_path / "b.jsonl")
    assert [l["result_sha256"] for l in a] == [l["result_sha256"] for l in b]
    assert a[0]["execution"]["execution_id"] != b[0]["execution"]["execution_id"]   # outside the hash
    assert all(O.verify_result(l) == [] for l in a)
    a[0]["result"]["review_status"] = PASS if a[0]["result"]["review_status"] != PASS else FAIL
    assert O.verify_result(a[0]) == ["result_sha256 does not match"]
    statuses = [r["review_status"] for r in results(tmp_path / "a.jsonl")]
    # Q-003's repair removed a statement carrying 211.25(a) text: completeness is then not established
    assert statuses == [PASS, PASS, NOT_EVALUATED, PASS, FAIL]         # the last omits two requirements
    identity = a[1]["result"]["evaluator"]
    assert (identity["aligner"], identity["identifier_index"], identity["repair_argument"]) == \
           ("lexical", "not_used", "passed")
