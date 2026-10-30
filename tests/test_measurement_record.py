"""The acceptance runner's measurement record (regulus.traceability.measurement) is exact, verifiable and inert.

Exact: the raw output, the draft repair received, the delivered answer and the evidence texts read back
byte for byte, in prompt order, with both validation reports and the repair kept apart. Verifiable: every
hash is recomputable from the record alone, here with hashlib and json directly rather than the module's own
helpers. Inert: answering with capture on delivers the same answer, row and assessment as answering without
it, and a capture failure is a visible record for that one question. Historical rows read back as what they
hold - never upgraded to a measurement record.

Answers are SCRIPTED in place of the model over the in-memory table: no model, no database, no GPU.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from regulus.accuracy import evaluate_safely
from regulus.accuracy.alignment import LexicalAligner
from regulus.accuracy.review import question_review
from regulus.config import load_settings
from regulus.rag import RagPipeline
from regulus.retrieval import LexicalRetriever
from regulus.system_identity import configured_rag_config
from regulus.traceability import EXACT, MISSING
from regulus.traceability import measurement as M
from tests.test_accuracy_observer import CASES, Scripted, answer_one
from tests.test_accuracy_pass_integrity import A, B, C, FOCUSED, S211
from tests.test_retrieval_plan import TableDatabase

# scripted model drafts: the model-drafted path (tests/conftest.py)
pytestmark = pytest.mark.usefixtures("model_drafted_path")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
runner = importlib.import_module("run_acceptance_test")

RUN = {"run_id": "acceptance_test_20260101T000000Z", "run_started_utc": "2026-01-01T00:00:00+00:00",
       "model": {"model_id": "scripted"}, "code_revision": {"git_commit": None, "git_dirty": None}}
AT = "2026-01-01T00:00:01+00:00"
INVENTED = f"{A[:-1]}, unless the batch was destroyed. [1] {B} [2] {C} [3]"

# The answers.jsonl row keys every archived run has (acceptance_c123/_sft2_v1): consumers depend on them.
ROW_KEYS = {"question_id", "question", "execution_status", "seconds", "answer", "raw_model_output", "model_called",
            "no_context", "citations", "evidence", "evidence_snapshot", "answerability", "scope", "fallback_type",
            "validation_initial", "validation_initial_codes", "validation_final", "repair", "stop_reason",
            "generated_tokens", "error"}


def focused(response):
    return RagPipeline(LexicalRetriever(TableDatabase(S211)), Scripted(response),
                       configured_rag_config(load_settings())).answer(FOCUSED)


def build(out, **kw):
    return M.build_record(question_id=kw.pop("qid", "Q-1"), ordinal=kw.pop("ordinal", 1), question=out.question,
                          out=out, run=RUN, captured_utc=AT, **kw)


def assess(out):
    """The evaluator call serve_ui makes, without the uncommitted `repair` argument."""
    d = evaluate_safely(out.question, out.answer, package=out.package, evidence=out.evidence_snapshot,
                        citations=out.citations, validation=out.validation, coverage=out.coverage,
                        model_called=out.model_called, fallback_type=out.metadata.get("fallback_type"),
                        scope_status=(out.metadata.get("question_analysis") or {}).get("scope_status"),
                        answerability=out.metadata.get("answerability"), aligner=LexicalAligner()).to_dict()
    d.pop("elapsed_ms")
    return d


def independent_hash(record):
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def roundtrip(record):
    return json.loads(json.dumps(record, ensure_ascii=False, sort_keys=True))


# ------------------------------------------------------------------ 1. exact content


def test_a_record_reads_back_with_the_exact_texts_and_evidence():
    out = focused(INVENTED)
    rec = roundtrip(build(out))
    assert M.verify_record(rec) == [] and rec["complete"] and rec["status"] == M.STATUS_CAPTURED
    t = rec["text"]
    assert t["raw_model_output"]["text"] == out.metadata["raw_model_output"] == INVENTED
    assert t["validation_input"]["text"] == out.metadata["repair_input"]
    assert t["repair_output"]["text"] == out.metadata["repair_output"]
    assert t["delivered_answer"]["text"].encode() == out.answer.encode()
    items = rec["evidence"]["snapshot"]["items"]
    assert [i["text"] for i in items] == [s.text for s in out.evidence_snapshot.items]
    assert rec["evidence"]["provenance"] == EXACT and rec["evidence"]["fingerprint"] == out.evidence_snapshot.fingerprint


# ------------------------------------------------------------------ 2-3. hashes


def test_hashes_are_deterministic_and_independently_verifiable():
    a, b = build(focused(INVENTED)), build(focused(INVENTED))
    assert a["record_sha256"] == b["record_sha256"] == independent_hash(a)
    for block in a["text"].values():
        if block["text"] is not None:
            assert block["sha256"] == hashlib.sha256(block["text"].encode("utf-8")).hexdigest()
    items = a["evidence"]["snapshot"]["items"]
    assert a["evidence"]["item_text_sha256"] == [hashlib.sha256(i["text"].encode("utf-8")).hexdigest() for i in items]


@pytest.mark.parametrize("path,value", [
    (("text", "delivered_answer", "text"), "A different answer."),
    (("text", "raw_model_output", "text"), "A different draft."),
    (("validation", "repair"), None),
    (("run", "model", "model_id"), "another-model"),
    (("evaluation", "state"), "captured"),
    (("question_id",), "Q-2"),
])
def test_changing_a_covered_field_changes_the_record_hash(path, value):
    rec = copy.deepcopy(build(focused(INVENTED)))
    target = rec
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert M.record_hash(rec) != rec["record_sha256"]
    assert "record_sha256 does not match the record's content" in M.verify_record(rec)


def test_an_edited_evidence_text_fails_verification_even_with_a_rehashed_record():
    rec = copy.deepcopy(build(focused(INVENTED)))
    rec["evidence"]["snapshot"]["items"][0]["text"] += " (edited)"
    rec["record_sha256"] = M.record_hash(rec)                 # an editor who re-seals the record
    problems = M.verify_record(rec)
    assert any(p.startswith("evidence.snapshot") for p in problems)


# ------------------------------------------------------------------ 4. ordering


def test_evidence_order_is_the_prompt_order_and_reordering_is_detected():
    out = focused(f"{A} [1] {B} [2] {C} [3]")
    rec = build(out)
    items = rec["evidence"]["snapshot"]["items"]
    assert [i["order"] for i in items] == list(range(len(items))) == [0, 1, 2]
    assert [i["index"] for i in items] == [s.index for s in out.package.sources]
    assert [i["text"] for i in items] == [s.text for s in out.package.sources]
    swapped = copy.deepcopy(rec)
    snap = swapped["evidence"]["snapshot"]["items"]
    snap[0]["text"], snap[1]["text"] = snap[1]["text"], snap[0]["text"]
    assert M.verify_record(swapped)


# ------------------------------------------------------------------ 5-6. validation and repair


def test_initial_and_final_validation_are_kept_apart():
    out = focused(INVENTED)
    rec = build(out)
    v = rec["validation"]
    assert v["state"] == M.CAPTURED
    assert v["initial"] == out.metadata["validation"]["initial"] and v["final"] == out.metadata["validation"]["final"]
    assert v["initial"]["problems"] and v["initial"] != v["final"]
    assert v["initial"]["status"] != v["final"]["status"] or v["initial"]["problems"] != v["final"]["problems"]


def test_a_repair_is_recorded_from_the_pipeline_not_inferred():
    out = focused(INVENTED)
    rec = build(out)
    assert rec["validation"]["repair"] == out.metadata["validation"]["repair"] == "invented_conditions_removed"
    assert rec["text"]["validation_input"]["state"] == rec["text"]["repair_output"]["state"] == M.CAPTURED
    assert "unless the batch was destroyed" in rec["text"]["validation_input"]["text"]
    assert "unless the batch was destroyed" not in rec["text"]["delivered_answer"]["text"]
    assert rec["inferred"]["label"].startswith("inferred")


def test_without_a_repair_the_validated_draft_is_marked_not_exposed_not_reconstructed():
    rec = build(focused(f"{A} [1] {B} [2] {C} [3]"))
    assert rec["validation"]["repair"] is None
    assert rec["text"]["validation_input"] == {"state": M.NOT_EXPOSED, "text": None, "sha256": None, "chars": None,
                                               "reason": rec["text"]["validation_input"]["reason"]}
    assert rec["text"]["repair_output"]["state"] == M.NOT_APPLICABLE
    assert [(g["field"], g["state"]) for g in rec["gaps"]] == [("text.validation_input", M.NOT_EXPOSED)]
    assert rec["complete"]                                   # not a required rescoring input


@pytest.mark.parametrize("case", range(len(CASES)))
def test_every_pipeline_path_yields_a_verified_record(case):
    out = answer_one(*CASES[case])
    rec = roundtrip(build(out))
    assert M.verify_record(rec) == [] and rec["complete"]
    raw = rec["text"]["raw_model_output"]["state"]
    assert raw == (M.CAPTURED if out.model_called else M.NOT_APPLICABLE)


def test_a_missing_evidence_snapshot_makes_the_record_incomplete():
    out = focused(f"{A} [1] {B} [2] {C} [3]")
    out.evidence_snapshot = None
    rec = build(out)
    assert rec["evidence"]["provenance"] == MISSING and rec["evidence"]["snapshot"] is None
    assert not rec["complete"] and any(g["field"] == "evidence" for g in rec["gaps"])
    assert M.verify_record(rec) == []


# ------------------------------------------------------------------ 7-8. historical rows


def legacy_row(qid, **over):
    row = {"question_id": qid, "question": "What does 21 CFR 211.25 require?", "execution_status": "OK", "seconds": 1.0,
           "answer": "Personnel shall have training. [1]", "raw_model_output": "Personnel shall have training. [1]",
           "model_called": True, "no_context": False, "citations": [{"index": 1, "label": "21 CFR 211.25(a)"}],
           "evidence": ["21 CFR 211.25(a)"], "answerability": "ANSWERABLE", "scope": None, "fallback_type": None,
           "validation_initial": "valid", "validation_initial_codes": [], "validation_final": "valid", "repair": None,
           "stop_reason": "eos", "generated_tokens": 12, "error": None}
    row.update(over)
    return row


def write_answers(run_dir, rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_historical_rows_load_without_fabricated_fields(tmp_path):
    error_row = {"question_id": "Q-2", "question": "q", "execution_status": "ERROR", "seconds": 0.1, "answer": None,
                 "error": "QueryCanceled: x", "traceback": "..."}
    write_answers(tmp_path, [legacy_row("Q-1"), error_row])
    before = (tmp_path / "answers.jsonl").read_bytes()
    loaded = M.load_run(tmp_path)
    assert loaded["problems"] == [] and [r["question_id"] for r in loaded["rows"]] == ["Q-1", "Q-2"]
    first, second = loaded["rows"]
    assert first["source"] == "legacy_answers_row" and first["complete"] is False and first["schema"] is None
    assert first["evidence"] == {"provenance": MISSING, "reason": "the row holds no evidence snapshot",
                                 "labels": ["21 CFR 211.25(a)"]}            # labels only, never text
    assert first["validation"]["state"] == "status_only" and first["validation"]["problems"] == M.NOT_RECORDED
    assert first["text"]["validation_input"]["state"] == M.NOT_RECORDED
    assert first["evaluation"]["state"] == M.NOT_RECORDED
    assert second["text"]["raw_model_output"]["state"] == M.NOT_RECORDED and second["validation"]["state"] == M.NOT_RECORDED
    assert (tmp_path / "answers.jsonl").read_bytes() == before and not (tmp_path / M.MEASUREMENT_FILE).exists()


def test_a_row_with_an_exact_snapshot_but_no_record_is_still_not_complete(tmp_path):
    out = focused(f"{A} [1] {B} [2] {C} [3]")
    write_answers(tmp_path, [legacy_row("Q-1", evidence_snapshot=out.evidence_snapshot.to_dict())])
    view = M.load_run(tmp_path)["rows"][0]
    assert view["evidence"]["provenance"] == EXACT and view["complete"] is False
    assert view["validation"]["problems"] == M.NOT_RECORDED


@pytest.mark.parametrize("run", sorted(p.name for p in (Path(__file__).resolve().parents[1] / "artifacts/acceptance_test")
                                       .glob("acceptance_*") if (p / "answers.jsonl").is_file()))
def test_the_archived_runs_read_back_as_legacy_and_are_left_untouched(run):
    run_dir = Path(__file__).resolve().parents[1] / "artifacts/acceptance_test" / run
    before = (run_dir / "answers.jsonl").read_bytes()
    loaded = M.load_run(run_dir)
    assert len(loaded["rows"]) == 300 and all(r["source"] == "legacy_answers_row" and not r["complete"]
                                              for r in loaded["rows"])
    assert (run_dir / "answers.jsonl").read_bytes() == before and not (run_dir / M.MEASUREMENT_FILE).exists()


# ------------------------------------------------------------------ 9-11. the runner loop: failure isolation, ordering


class Pipe:
    """Answers each question with the focused pipeline; raises for the ones listed."""

    def __init__(self, fail=()):
        self.fail = set(fail)

    def answer(self, question):
        if question in self.fail:
            raise RuntimeError("database cancelled the query")
        return focused({"q-clean": f"{A} [1] {B} [2] {C} [3]", "q-invented": INVENTED}.get(question, f"{A} [1]"))


def questions(*texts):
    return [SimpleNamespace(question_id=f"Q-{i:03}", question=t) for i, t in enumerate(texts, start=1)]


def read(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_a_capture_failure_is_visible_and_changes_neither_the_answer_nor_the_assessment(tmp_path, monkeypatch):
    qs = questions("q-clean", "q-invented")
    ordinals = {q.question_id: i for i, q in enumerate(qs, start=1)}
    (tmp_path / "ok").mkdir()
    (tmp_path / "broken").mkdir()
    runner.run_questions(Pipe(), qs, ordinals, RUN, tmp_path / "ok")
    real = M.build_record

    def broken(**kw):
        if kw["question_id"] == "Q-001":
            raise ValueError("serialisation defect")
        return real(**kw)

    monkeypatch.setattr(M, "build_record", broken)
    runner.run_questions(Pipe(), qs, ordinals, RUN, tmp_path / "broken")
    ok, bad = read(tmp_path / "ok/answers.jsonl"), read(tmp_path / "broken/answers.jsonl")
    strip = lambda rows: [{k: v for k, v in r.items() if k != "seconds"} for r in rows]   # noqa: E731
    assert strip(ok) == strip(bad)                            # the answers rows are untouched by the failure
    measured = read(tmp_path / "broken" / M.MEASUREMENT_FILE)
    assert [m["question_id"] for m in measured] == ["Q-001", "Q-002"]
    assert measured[0]["status"] == M.STATUS_CAPTURE_ERROR and measured[0]["complete"] is False
    assert "ValueError: serialisation defect" in measured[0]["error"] and M.verify_record(measured[0]) == []
    assert measured[1]["status"] == M.STATUS_CAPTURED and measured[1]["complete"] and M.verify_record(measured[1]) == []
    # and the verdict of the same answer is what it is with capture working
    out = focused(f"{A} [1] {B} [2] {C} [3]")
    first = assess(out)
    M.capture_safely(question_id="Q-001", ordinal=1, question=out.question, out=out, run=RUN, captured_utc=AT)
    assert assess(out) == first


def test_one_failing_row_does_not_disturb_the_rows_after_it_and_order_is_stable(tmp_path):
    qs = questions("q-clean", "q-boom", "q-invented", "q-other")
    ordinals = {q.question_id: i for i, q in enumerate(qs, start=1)}
    runner.run_questions(Pipe(fail={"q-boom"}), qs, ordinals, RUN, tmp_path)
    rows, measured = read(tmp_path / "answers.jsonl"), read(tmp_path / M.MEASUREMENT_FILE)
    assert [r["question_id"] for r in rows] == [m["question_id"] for m in measured] == ["Q-001", "Q-002", "Q-003", "Q-004"]
    assert [m["ordinal"] for m in measured] == [1, 2, 3, 4]
    assert rows[1]["execution_status"] == "ERROR" and measured[1]["status"] == M.STATUS_PIPELINE_ERROR
    assert [m["status"] for m in measured[2:]] == [M.STATUS_CAPTURED] * 2
    assert measured[2]["validation"]["repair"] == "invented_conditions_removed"
    loaded = M.load_run(tmp_path)
    assert loaded["problems"] == [] and all(r["verified"] for r in loaded["rows"])
    assert [r["complete"] for r in loaded["rows"]] == [True, False, True, True]


def test_resuming_appends_and_never_rewrites_completed_rows(tmp_path):
    qs = questions("q-clean", "q-invented", "q-other")
    ordinals = {q.question_id: i for i, q in enumerate(qs, start=1)}
    runner.run_questions(Pipe(), qs[:2], ordinals, RUN, tmp_path)
    answers_before = (tmp_path / "answers.jsonl").read_bytes()
    measured_before = (tmp_path / M.MEASUREMENT_FILE).read_bytes()
    runner.run_questions(Pipe(), qs[2:], ordinals, RUN, tmp_path)
    assert (tmp_path / "answers.jsonl").read_bytes().startswith(answers_before)
    assert (tmp_path / M.MEASUREMENT_FILE).read_bytes().startswith(measured_before)
    assert [m["ordinal"] for m in read(tmp_path / M.MEASUREMENT_FILE)] == [1, 2, 3]


def test_a_measurement_without_an_answer_row_and_a_duplicate_are_reported(tmp_path):
    runner.run_questions(Pipe(), questions("q-clean"), {"Q-001": 1}, RUN, tmp_path)
    line = (tmp_path / M.MEASUREMENT_FILE).read_text(encoding="utf-8")
    orphan = json.loads(line)
    orphan["question_id"] = "Q-999"
    (tmp_path / M.MEASUREMENT_FILE).write_text(line + line + json.dumps(orphan) + "\n{not json\n", encoding="utf-8")
    problems = M.load_run(tmp_path)["problems"]
    assert any("duplicate" in p for p in problems) and any("Q-999" in p for p in problems)
    assert any("not JSON" in p for p in problems)


# ------------------------------------------------------------------ 12. backward compatibility


def test_the_answers_row_keeps_its_shape():
    out = focused(INVENTED)
    q = SimpleNamespace(question_id="Q-1", question=FOCUSED)
    assert set(runner.record(q, out, 1.0)) == ROW_KEYS


# ------------------------------------------------------------------ capture is observational


def snapshot_of(out):
    return json.loads(json.dumps({**out.to_dict(), "snapshot": out.evidence_snapshot.to_dict()
                                  if out.evidence_snapshot else None}, default=str))


@pytest.mark.parametrize("make", [
    *[(lambda q=q, r=r: answer_one(q, r)) for q, r in CASES],
    lambda: focused(f"{A} [1] {B} [2] {C} [3]"),
    lambda: focused(INVENTED),
])
def test_capture_changes_no_answer_row_evidence_validation_or_assessment(make):
    plain, captured = make(), make()
    before = snapshot_of(captured)
    rec = build(captured)
    assert snapshot_of(captured) == before                    # the RagAnswer was not touched
    q = SimpleNamespace(question_id="Q-1", question=plain.question)
    row_plain, row_captured = runner.record(q, plain, 1.0), runner.record(q, captured, 1.0)
    assert row_plain == row_captured                          # draft, answer, citations, evidence, validation
    a, b = assess(plain), assess(captured)
    assert a == b                                             # every dimension, claim verdict and the evidence used
    assert question_review(a, question=plain.question) == question_review(b, question=captured.question)
    # and the record holds what the pipeline did, not something adjacent to it
    assert rec["text"]["delivered_answer"]["text"] == row_captured["answer"]
    assert rec["text"]["raw_model_output"]["text"] == row_captured["raw_model_output"]
    assert rec["validation"]["repair"] == row_captured["repair"]
    assert [c["index"] for c in rec["citations"]] == [c["index"] for c in row_captured["citations"]]


def test_an_attached_assessment_is_stored_as_given_and_covered_by_the_hash():
    out = focused(f"{A} [1] {B} [2] {C} [3]")
    d = assess(out)
    review = question_review(d, question=out.question)
    rec = roundtrip(build(out, evaluation=d, review=review))
    assert rec["evaluation"] == {"state": M.CAPTURED, "assessment": json.loads(json.dumps(d)), "review": review}
    assert M.verify_record(rec) == []
    assert build(out)["evaluation"]["state"] == "not_run"     # the runner's default: no evaluator call
