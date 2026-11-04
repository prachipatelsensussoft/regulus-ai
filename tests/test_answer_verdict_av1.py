"""AV1 (report-only): the delivered answer's verdict - complete, incomplete, unsupported, incorrect, insufficient."""
from __future__ import annotations

from regulus.rag import RagPipeline
from regulus.rag import answer_verdict as AV
from regulus.rag import fallback as FB
from regulus.rag import validator_switches as VS
from regulus.retrieval import LexicalRetriever
from tests.test_answer_validation import ScriptedGenerator
from tests.test_citation_verdicts_cc1 import QUESTION, TABLE, _package
from tests.test_retrieval_plan import TableDatabase

BOTH = ("21 CFR 820.10(b)(2) requires records of traceability for each device lot, including the distribution records "
        "[1]. 21 CFR 820.10(b)(4) requires advisory notices issued to customers about device corrections and removals [2].")


def _verdict(answer, **kw):
    return AV.answer_verdict(answer, _package(), question=QUESTION, **kw)


def test_av1_is_off_by_default():
    assert "AV1" not in VS.DEFAULT.split(",")


def test_an_answer_that_covers_both_requested_paragraphs_from_their_text_is_complete():
    res = _verdict(BOTH)
    assert res["verdict"] == AV.COMPLETE, res
    assert res["citations"]["valid"] == 2


def test_a_requested_paragraph_left_out_is_correct_but_incomplete():
    res = _verdict(BOTH.split(" 21 CFR 820.10(b)(4)")[0])
    assert res["verdict"] == AV.INCOMPLETE and "requirement_not_addressed" in res["reasons"]


def test_a_citation_of_the_wrong_paragraph_is_unsupported():
    res = _verdict("21 CFR 820.10(b)(2) requires records of traceability for each device lot [2]. " + BOTH.split("[1]. ")[1])
    assert res["verdict"] == AV.UNSUPPORTED
    assert "provision_citation_mismatch" in res["reasons"]


def test_a_refusal_or_a_withheld_answer_is_insufficient_evidence():
    for fb in (FB.FB_PROVISION_NOT_FOUND, FB.FB_UNVALIDATED, FB.FB_OUT_OF_SCOPE, FB.FB_PRODUCT_CLARIFICATION):
        assert _verdict("anything", fallback_type=fb)["verdict"] == AV.INSUFFICIENT
    assert AV.answer_verdict("", _package(), question=QUESTION)["verdict"] == AV.INSUFFICIENT


def test_a_verbatim_fallback_and_an_evidence_limit_are_reasons_for_incomplete():
    res = _verdict(BOTH, fallback_type=FB.FB_FOCUSED_QUOTE)
    assert res["verdict"] == AV.INCOMPLETE and any("verbatim fallback" in r for r in res["reasons"])
    # the pipeline passes its closing notes apart from the body: they set reasons, and are never validated as claims
    notice = ("\nEvidence limit: not all of the retrieved text of 21 CFR 820.10 fit in the evidence supplied for this "
              "answer (2 paragraphs left out). The answer above does not cover that text; consult the full provisions.")
    res = _verdict(BOTH, notes=notice)
    assert res["verdict"] == AV.INCOMPLETE and any("evidence limit" in r for r in res["reasons"])
    assert "unsupported_claim" not in res["codes"]
    res = _verdict(BOTH, notes='\nNot answered: "the import fees". The available regulatory sources do not cover this.')
    assert res["verdict"] == AV.INCOMPLETE and any("outside the available sources" in r for r in res["reasons"])


def test_a_contradiction_outranks_every_other_finding(monkeypatch):
    from regulus.rag import validation as V
    real = V.validate_answer

    def with_false_denial(text, package, generation=None):
        rep = real(text, package, generation)
        extra = V.ValidationProblem("false_denial", "error", "the cited paragraph states it")
        return V.ValidationReport(**{**rep.__dict__, "problems": rep.problems + (extra,)})

    monkeypatch.setattr(V, "validate_answer", with_false_denial)
    res = _verdict(BOTH.split(" 21 CFR 820.10(b)(4)")[0])   # also incomplete; incorrect wins
    assert res["verdict"] == AV.INCORRECT and res["reasons"] == ["false_denial"]


def test_a_failure_is_reported_not_raised():
    assert "error" in AV.safe_answer_verdict("x [1].", object(), question=QUESTION)


def test_the_pipeline_answer_is_unchanged_and_only_av1_adds_the_verdict(monkeypatch):
    draft = "21 CFR 820.10(b)(2) requires traceability [1]. 21 CFR 820.10(b)(4) requires advisory notices [2]."

    def answer():
        return RagPipeline(LexicalRetriever(TableDatabase()), ScriptedGenerator(draft)).answer(QUESTION)

    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"AV1", "CC1"})
    off = answer()
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"AV1"})
    on = answer()
    assert on.answer == off.answer and "answer_verdict" not in off.metadata
    assert on.metadata["answer_verdict"]["verdict"] in AV.VERDICTS and "answer_verdict" in on.diagnostics()
    assert "citation_verdicts" not in on.metadata   # AV1 computes its own citation verdicts when CC1 is off
