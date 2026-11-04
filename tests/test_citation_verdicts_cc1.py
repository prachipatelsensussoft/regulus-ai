"""CC1 (report-only): every citation of the delivered answer gets one of four verdicts; no answer changes."""
from __future__ import annotations

import pytest

from regulus.rag import RagPipeline
from regulus.rag import citation_verdicts as CCV
from regulus.rag import validator_switches as VS
from regulus.retrieval import LexicalRetriever
from regulus.retrieval.programmes import Programme
from tests.test_answer_validation import ScriptedGenerator
from tests.test_generation_contract import package_for
from tests.test_retrieval_plan import TableDatabase, row

QUESTION = "What do 820.10(b)(2) and 820.10(b)(4) require?"
TABLE = [
    row("820.10", "(b)", 0, text="(b) The manufacturer shall document the following records:"),
    row("820.10", "(b)(2)", 1, text="(2) Records of traceability for each device lot, including the distribution records."),
    row("820.10", "(b)(4)", 2, text="(4) Advisory notices issued to customers about device corrections and removals."),
]
SECTION_TABLE = [row("820.30", "", 0, text="Each manufacturer shall establish and maintain procedures to control the "
                                           "design of the device in order to ensure that design requirements are met.")]


def _package(question=QUESTION, table=TABLE):
    return package_for(question, db=TableDatabase(table))


def _verdicts(answer, question=QUESTION, table=TABLE):
    return CCV.citation_verdicts(answer, _package(question, table), question)


def _by_marker(res):
    return {i["marker"]: i for i in res["items"] if i["marker"] is not None}


def test_cc1_is_off_by_default():
    assert "CC1" not in VS.DEFAULT.split(",")


def test_the_package_cites_the_two_requested_paragraphs():
    labels = [s.citation for s in _package().sources]
    assert labels[:2] == ["21 CFR 820.10(b)(2)", "21 CFR 820.10(b)(4)"]


def test_a_citation_of_the_named_paragraph_that_states_it_is_valid():
    res = _verdicts("21 CFR 820.10(b)(2) requires records of traceability for each device lot [1].")
    item = _by_marker(res)[1]
    assert item["verdict"] == CCV.VALID and item["source"] == "21 CFR 820.10(b)(2)" and item["overlap"] >= 0.5
    assert res["summary"]["citations"] == 1 and res["summary"]["valid_share"] == 1.0


def test_naming_one_paragraph_and_citing_another_is_the_wrong_section():
    res = _verdicts("21 CFR 820.10(b)(2) requires records of traceability for each device lot [2].")
    item = _by_marker(res)[2]
    assert item["verdict"] == CCV.WRONG and any("provision_citation_mismatch" in r for r in item["reasons"])


def test_a_marker_with_no_supplied_source_cannot_be_verified():
    res = _verdicts("21 CFR 820.10(b)(2) requires records of traceability for each device lot [7].")
    assert _by_marker(res)[7]["verdict"] == CCV.UNVERIFIABLE


def test_a_sentence_the_source_does_not_state_is_not_supported_by_it():
    res = _verdicts("21 CFR 820.10(b)(4) requires advisory notices, and manufacturers must also pay annual registration "
                    "fees and keep invoices for ten years [2].")
    item = _by_marker(res)[2]
    assert item["verdict"] == CCV.NOT_SUPPORTING and item["overlap"] < CCV.WEAK_OVERLAP


def test_a_fabricated_paragraph_of_a_section_level_source_cannot_be_verified():
    # the validator accepts it (a section-level source is "related" to any path of its section); CC1 reports it
    q = "What does 21 CFR 820.30 require?"
    res = _verdicts("21 CFR 820.30(z)(9) requires each manufacturer to establish procedures to control the design of "
                    "the device [1].", question=q, table=SECTION_TABLE)
    item = _by_marker(res)[1]
    assert item["verdict"] == CCV.UNVERIFIABLE and "(z)(9)" in item["reasons"][0]


def test_provisions_named_without_a_marker_are_checked_against_the_supplied_sources():
    res = _verdicts("21 CFR 820.10(b)(2): Records of traceability for each device lot, including the distribution "
                    "records.\n21 CFR 820.30(a) requires design controls.")
    named = {i["named"]: i["verdict"] for i in res["items"] if i["marker"] is None}
    assert named == {"21 CFR 820.10(b)(2)": CCV.VALID, "21 CFR 820.30(a)": CCV.UNVERIFIABLE}


def test_an_appended_notice_is_not_a_claim():
    res = _verdicts("21 CFR 820.10(b)(2) requires records of traceability for each device lot [1].\n"
                    "Evidence limit: not all of the retrieved text of 21 CFR 820.30 fit in the evidence supplied for this "
                    "answer (3 paragraphs left out).")
    assert [i.get("named") for i in res["items"]] == [None] and res["summary"]["citations"] == 1


def test_a_part_outside_the_programme_the_question_names_is_the_wrong_programme(monkeypatch):
    import re

    import regulus.retrieval.programmes as programmes
    supplements = Programme(name="Dietary supplements", parts=frozenset({"111"}), act=(), words=re.compile("x"), acronyms=None)
    monkeypatch.setattr(programmes, "programme_for", lambda q, extended=False: supplements)
    res = _verdicts("21 CFR 820.10(b)(2) requires records of traceability for each device lot [1].")
    item = _by_marker(res)[1]
    assert item["verdict"] == CCV.WRONG and "Dietary supplements" in item["reasons"][0]
    assert res["summary"]["programme"] == "Dietary supplements"


def test_an_empty_answer_and_a_failure_are_reported_not_raised():
    assert CCV.citation_verdicts("", _package(), QUESTION)["summary"]["citations"] == 0
    assert "error" in CCV.safe_citation_verdicts("x [1].", object(), QUESTION)


# ------------------------------------------------------------------------------- the pipeline: same answer

DRAFT = "21 CFR 820.10(b)(2) requires traceability [1]. 21 CFR 820.10(b)(4) requires advisory notices [2]."


def _answer():
    from tests.test_trace_tr1 import MULTI
    return RagPipeline(LexicalRetriever(TableDatabase()), ScriptedGenerator(DRAFT)).answer(MULTI)


def test_the_pipeline_answer_is_unchanged_and_only_cc1_adds_the_verdicts(monkeypatch):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"CC1"})
    off = _answer()
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"CC1"})
    on = _answer()
    assert on.answer == off.answer and "citation_verdicts" not in off.metadata
    cv = on.metadata["citation_verdicts"]
    assert cv["version"] == CCV.VERSION and cv["summary"]["citations"] >= 1 and "citation_verdicts" in on.diagnostics()
