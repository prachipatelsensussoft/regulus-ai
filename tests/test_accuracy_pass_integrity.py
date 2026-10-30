"""PASS means every applicable check passed, not that some part of the answer is right.

Probed 2026-10-09 (artifacts/accuracy/pass_integrity_*): every mixed, fabricated,
mis-cited or cross-subsection answer below already failed to PASS, and is pinned
here. One confirmed false-PASS path was fixed: a FOCUSED question ("What are the
personnel qualification requirements in 21 CFR 211.25?") answered with one of the
section's on-topic requirements was PASS 100, because the pipeline checks group
coverage only for a synthesis and the evaluator's completeness took any supported
answering statement as covering the question. Rubric O=0: "a material governing
requirement is omitted such that a user acting on the answer would be out of
compliance".

Evidence: VERBATIM snapshot paragraphs (tests/fixtures/accuracy_definition_citation_rows.json),
served through the real plan, retrieval, package, validator and - for the focused
cases - the PRODUCTION configuration (configured_rag_config, as serve_ui). Answers:
CONSTRUCTED, scripted in place of the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from regulus.accuracy import evaluate, evaluate_safely
from regulus.accuracy.alignment import LexicalAligner
from regulus.accuracy.review import FAIL, PASS, question_review
from regulus.config import load_settings
from regulus.inference.generate import GenerationResult
from regulus.planning import plan_query
from regulus.rag import RagPipeline
from regulus.rag.coverage import check_coverage
from regulus.rag.validation import validate_answer
from regulus.retrieval import LexicalRetriever
from regulus.system_identity import configured_rag_config
from tests.test_accuracy_observer import Scripted
from tests.test_generation_contract import package_for
from tests.test_retrieval_plan import TableDatabase, row

# scripted model drafts: the model-drafted path (tests/conftest.py)
pytestmark = pytest.mark.usefixtures("model_drafted_path")

TEXT = {p["citation"]: p["text"] for p in json.loads(
    (Path(__file__).parent / "fixtures" / "accuracy_definition_citation_rows.json").read_text(encoding="utf-8"))["paragraphs"]}


def _row(citation, ordinal, heading):
    section = citation.replace("21 CFR ", "").split("(", 1)[0]
    para = "(" + citation.split("(", 1)[1] if "(" in citation else ""
    return row(section, para, ordinal, text=TEXT[citation], section_heading=heading)


def body(citation):
    t = TEXT[citation]
    return t.split(") ", 1)[1] if t.startswith("(") else t


S211 = [_row(c, i, "Personnel qualifications.") for i, c in enumerate(("21 CFR 211.25(a)", "21 CFR 211.25(b)", "21 CFR 211.25(c)"), 1)]
S820 = [_row(c, i, "Control of records.") for i, c in enumerate((
    "21 CFR 820.35(a)", "21 CFR 820.35(a)(1)", "21 CFR 820.35(a)(2)", "21 CFR 820.35(a)(3)", "21 CFR 820.35(a)(4)",
    "21 CFR 820.35(a)(5)", "21 CFR 820.35(a)(6)", "21 CFR 820.35(a)(7)", "21 CFR 820.35(b)", "21 CFR 820.35(b)(1)",
    "21 CFR 820.35(b)(3)"), 1)]
A, B, C = body("21 CFR 211.25(a)"), body("21 CFR 211.25(b)"), body("21 CFR 211.25(c)")
FOCUSED = "What are the personnel qualification requirements in 21 CFR 211.25?"


def production(question, response, table):
    """As serve_ui: the configured pipeline (validation, repair) delivers; the evaluator assesses what it delivered."""
    out = RagPipeline(LexicalRetriever(TableDatabase(table)), Scripted(response), configured_rag_config(load_settings())).answer(question)
    d = evaluate_safely(out.question, out.answer, package=out.package, evidence=out.evidence_snapshot, citations=out.citations,
                        validation=out.validation, coverage=out.coverage, model_called=out.model_called,
                        fallback_type=out.metadata.get("fallback_type"),
                        scope_status=(out.metadata.get("question_analysis") or {}).get("scope_status"),
                        answerability=out.metadata.get("answerability"), repair=out.metadata.get("validation"),
                        aligner=LexicalAligner()).to_dict()
    return question_review(d, question=out.question), d, out


def completeness(d):
    return next(x for x in d["dimensions"] if x["dimension"] == "completeness")


# ---------------------------------------------------- the fixed path: a focused answer that omits on-topic requirements


def test_a_focused_answer_covering_every_on_topic_requirement_passes():
    v, d, out = production(FOCUSED, f"{A} [1] {B} [2] {C} [3]", S211)
    assert (out.metadata.get("answer_shape") or {}).get("mode") == "focused"
    assert (v["status"], v["live_check_score"]) == (PASS, 100)


@pytest.mark.parametrize("response,omitted", [
    (f"{C} [3]", ("21 CFR 211.25(a)", "21 CFR 211.25(b)")),
    (f"{A} [1]", ("21 CFR 211.25(b)", "21 CFR 211.25(c)")),
])
def test_a_focused_answer_omitting_an_on_topic_requirement_fails_completeness(response, omitted):
    v, d, out = production(FOCUSED, response, S211)
    assert out.answer.startswith(response.split(" [")[0][:40])     # delivered as generated: nothing repaired it
    assert v["status"] == FAIL and completeness(d)["state"] == "fail"
    for label in omitted:
        assert label in completeness(d)["detail"]
    assert v["substantive_correctness"]["state"] == "pass"         # what it says is right; it is still not a PASS


def test_an_unverified_paraphrase_of_a_requirement_is_not_called_an_omission():
    v, d, _ = production(FOCUSED, f"Each person making drugs needs suitable schooling and experience for the job [1]. {C} [3]", S211)
    assert v["status"] != PASS
    # (a) is addressed - by a paraphrase the evaluator cannot verify - so it is not called omitted;
    # (b), which nothing addresses, is
    assert completeness(d)["state"] == "fail"
    assert "211.25(b)" in completeness(d)["detail"] and "211.25(a)" not in completeness(d)["detail"]


def test_a_group_addressed_only_by_an_unverifiable_paraphrase_is_not_established():
    v, d, _ = production(FOCUSED, f"Each person making drugs needs suitable schooling and experience for the job [1]. {B} [2] {C} [3]", S211)
    assert v["status"] != PASS
    assert completeness(d)["state"] == "not_established" and "211.25(a)" in completeness(d)["detail"]


def test_a_requirement_off_the_questions_topic_is_not_demanded(monkeypatch):
    # complaint records: 820.35(b) servicing records are in the same section but not what was asked. The model's answer
    # is what is assessed here: LS1 would answer this list question verbatim (completeness "not_applicable")
    from regulus.rag import validator_switches as VS
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"LS1"})
    items = " ".join(f"{body(c)} [{i}]" for i, c in enumerate(
        ("21 CFR 820.35(a)", "21 CFR 820.35(a)(1)", "21 CFR 820.35(a)(2)", "21 CFR 820.35(a)(3)", "21 CFR 820.35(a)(4)",
         "21 CFR 820.35(a)(5)", "21 CFR 820.35(a)(6)", "21 CFR 820.35(a)(7)"), 1))
    v, d, _ = production("What must complaint records include under 21 CFR 820.35?", items, S820)
    assert completeness(d)["state"] == "pass"


def test_an_unrestricted_section_question_is_repaired_by_the_pipeline_before_it_is_assessed():
    v, d, out = production("What does 21 CFR 211.25 require?", f"{C} [3]", S211)
    assert (out.metadata.get("answer_shape") or {}).get("mode") == "synthesis"
    assert body("21 CFR 211.25(a)")[:40] in out.answer                # the missing groups were appended verbatim
    assert v["status"] == PASS


# ---------------------------------------------------- mixtures that must never PASS (pinned)


def validated(question, answer, table):
    pkg = package_for(question, db=TableDatabase(table))
    report = validate_answer(answer, pkg, GenerationResult("p", answer, 500, 60, 1.0, finished=True))
    cov = check_coverage(LexicalRetriever(TableDatabase(table)).run_plan(plan_query(question), top_k=8).registry)
    d = evaluate(question, answer, package=pkg, validation=report, coverage=cov).to_dict()
    return question_review(d, question=question), d


Q211 = "What do 21 CFR 211.25(a), (b), and (c) require?"


@pytest.mark.parametrize("answer", [
    f"{A} [1] {B} [2] {C} [3] Each person shall also complete an annual CGMP proficiency test [1].",   # fabricated requirement
    f"Each person shall pass an annual written CGMP examination.\n\n{A} [1] {B} [2] {C} [3]",         # uncited fabrication
    f"{A} [1] {B} [2] {C} Personnel records shall be retained for ten years [3].",                     # marker supports neighbour only
    f"{A} [1] {B} [2] There shall be an adequate number of qualified personnel to supervise the holding of each drug product, "
    f"and at least three supervisors per shift [3].",                                                   # a number with no source
    f"{C} [3]",                                                                                         # one of three requested
])
def test_an_answer_with_a_material_defect_beside_correct_statements_never_passes(answer):
    v, _ = validated(Q211, answer, S211)
    assert v["status"] != PASS


def test_servicing_items_stated_as_complaint_record_contents_never_pass():
    v, _ = validated("What must complaint records include under 21 CFR 820.35?",
                     "Complaint records must include the date of service and the service performed [10].", S820)
    assert v["status"] != PASS
