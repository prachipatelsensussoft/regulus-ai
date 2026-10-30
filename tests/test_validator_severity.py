"""Validator severity fix (16 Oct 2026): unsupported_claim, condition_not_in_cited_source and
applicability_dropped are errors, mapped to the `unsupported` status, so an answer carrying one is not
delivered.

Scoped exactly as the paired offline severity experiment tested it
(~/regulus_eval_runs/severity_experiment_20261016T115314Z): two Stage 7 SEVERITY entries, the Stage 12
applicability_dropped severity at its source, and two _CODE_STATUS entries. Detection is unchanged. As
warnings, the three fired only on delivered answers the rubric scored FAIL; on the recorded 300-question run
the change withholds 31 answers, all FAIL, and no PASS. It is a safety change: it raises no answer to correct.

The recorded benchmark rows are pinned in tests/integration/test_validator_severity_recorded.py (database).
Here: in-memory tables and scripted answers, no model, no database.
"""

from __future__ import annotations

import pytest

from regulus.config import load_settings
from regulus.rag import RagConfig, RagPipeline
from regulus.rag import claim_validation as CV
from regulus.rag import validation as V
from regulus.rag.shape_validation import APPLICABILITY_DROPPED, QUALIFIER_DROPPED
from regulus.rag.validation import STATUS_UNSUPPORTED, STATUS_VALID, validate_answer
from regulus.retrieval import LexicalRetriever
from regulus.system_identity import configured_rag_config
from tests.test_answer_shape import CONDITIONAL, TABLE, gen, idx, package
from tests.test_answer_validation import ScriptedGenerator
from tests.test_retrieval_plan import TableDatabase
from tests.test_stage7 import pipeline, validate

TREATED = (CV.UNSUPPORTED_CLAIM, CV.CONDITION_NOT_IN_CITED, APPLICABILITY_DROPPED)
WITHHELD = "could not be validated"
ONE = "What does 21 CFR 820.10(b)(2) require?"
TWO = "What do 21 CFR 820.10(b)(2) and 21 CFR 820.10(b)(4) require?"
SOUND = "21 CFR 820.10(b)(2) requires traceability procedures under part 821, if applicable [1]."
UNSUPPORTED = " Quarterly biometric turbine calibration logbooks stay mandatory [1]."
# (b)(4)'s real condition-free text, cited for a condition only (b)(2) carries
MISPLACED = ("21 CFR 820.10(b)(4) requires advisory notices to be handled under part 806, if applicable [2]. "
             "21 CFR 820.10(b)(2) requires traceability procedures under part 821, if applicable [1].")
NO_LIFE_LIMIT = "21 CFR 820.10(d) requires manufacturers to comply with Clause 7.5.9.2 in ISO 13485, as appropriate [{i}]."


def errors(report):
    return {p.code for p in report.problems if p.severity == "error"}


def treated(report):
    return {p.code for p in report.problems if p.code in TREATED}


def withheld(out, draft):
    """Not delivered: the draft's own words are gone and the pipeline's withheld notice stands instead."""
    return WITHHELD in out.answer and draft.split(" [")[0] not in out.answer


# ------------------------------------------------------------ severity and status propagation


def test_the_three_findings_are_errors_mapped_to_unsupported():
    assert CV.SEVERITY[CV.UNSUPPORTED_CLAIM] == CV.SEVERITY[CV.CONDITION_NOT_IN_CITED] == "error"
    assert all(V._CODE_STATUS[code] == STATUS_UNSUPPORTED for code in TREATED)


def test_the_fix_changes_no_other_severity_or_status():
    # OBLIGATION_STRENGTHENED is raised only with fix E1 switched on (regulus.rag.validator_switches, 23 Oct)
    # FIGURE_MISATTRIBUTED (FC2) and FALSE_DENIAL (FD1), 29 Oct: raised only with their switches on
    assert CV.SEVERITY == {CV.PARAGRAPH_MISATTRIBUTION: "error", CV.INVENTED_CONDITION: "error", CV.CONDITION_NOT_IN_CITED: "error",
                           CV.UNSUPPORTED_PREMISE: "error", CV.UNSUPPORTED_CLAIM: "error", CV.OBLIGATION_STRENGTHENED: "error",
                           CV.FIGURE_MISATTRIBUTED: "error", CV.FALSE_DENIAL: "error"}
    assert V._CODE_STATUS[CV.INVENTED_CONDITION] == V.STATUS_EVIDENCE_MISMATCHED   # still repaired in place
    assert V.LOW_OVERLAP not in V._CODE_STATUS and QUALIFIER_DROPPED not in V._CODE_STATUS


def test_unsupported_claim_alone_makes_the_report_unsupported():
    rep, _ = validate(SOUND + UNSUPPORTED, ONE)
    assert errors(rep) == {CV.UNSUPPORTED_CLAIM} and treated(rep) == {CV.UNSUPPORTED_CLAIM}
    assert rep.status == STATUS_UNSUPPORTED and not rep.ok


def test_condition_not_in_cited_source_alone_makes_the_report_unsupported():
    rep, _ = validate(MISPLACED, TWO)
    assert errors(rep) == {CV.CONDITION_NOT_IN_CITED}
    assert rep.status == STATUS_UNSUPPORTED and not rep.ok


def test_applicability_dropped_alone_makes_the_report_unsupported():
    pkg = package(CONDITIONAL)
    rep = validate_answer(NO_LIFE_LIMIT.format(i=idx(pkg, "21 CFR 820.10(d)")), pkg, gen())
    assert [(p.code, p.severity) for p in rep.problems] == [(APPLICABILITY_DROPPED, "error")]
    assert rep.status == STATUS_UNSUPPORTED and not rep.ok


# ------------------------------------------------------------ what must not trigger


@pytest.mark.usefixtures("delivered_as_drafted")   # delivered exactly as validated / repaired
@pytest.mark.parametrize("question, answer", [
    (ONE, SOUND),                                                                               # verbatim substance
    (ONE, "21 CFR 820.10(b)(2) requires the manufacturer to document traceability procedures meeting part 821, if applicable [1]."),  # paraphrase
    ("Does 21 CFR 820.10(b)(2) require annual traceability audits?",                            # a correct "no"
     "No. 21 CFR 820.10(b)(2) does not require annual audits; it requires traceability procedures under part 821, if applicable [1]."),
])
def test_correct_answers_and_a_close_paraphrase_are_valid_and_delivered(question, answer):
    p, _ = pipeline(None, response=answer)
    out = p.answer(question)
    assert out.metadata["validation"]["initial"]["status"] == STATUS_VALID and not set(out.metadata["validation"]["initial"]["codes"]) & set(TREATED)
    assert out.answer == answer


def test_keeping_the_applicability_limit_is_valid():
    pkg = package(CONDITIONAL)
    i = idx(pkg, "21 CFR 820.10(d)")
    rep = validate_answer(f"21 CFR 820.10(d) requires manufacturers of devices that support or sustain life to comply with "
                          f"Clause 7.5.9.2 in ISO 13485, as appropriate [{i}].", pkg, gen())
    assert rep.ok and not treated(rep)


@pytest.mark.parametrize("answer", [
    "The available regulatory sources do not provide enough information to answer this.",
    "The retrieved sources do not establish whether this applies.",
])
def test_refusals_and_abstentions_are_not_claims(answer):
    rep, _ = validate(answer, ONE)
    assert not treated(rep)


# ------------------------------------------------------------ delivery


@pytest.mark.parametrize("question, draft, finding", [
    (ONE, SOUND + UNSUPPORTED, CV.UNSUPPORTED_CLAIM),
    (TWO, MISPLACED, CV.CONDITION_NOT_IN_CITED),
])
def test_each_claim_finding_withholds_the_answer(question, draft, finding):
    p, _ = pipeline(None, response=draft)
    out = p.answer(question)
    assert finding in out.metadata["validation"]["initial"]["codes"]
    assert out.metadata["validation"]["initial"]["status"] == STATUS_UNSUPPORTED
    assert withheld(out, draft) and "Quarterly biometric" not in out.answer


def test_applicability_dropped_withholds_the_answer():
    draft = NO_LIFE_LIMIT.format(i=1)
    out = RagPipeline(LexicalRetriever(TableDatabase(TABLE)), ScriptedGenerator(draft), RagConfig(answer_shape="adaptive")).answer(CONDITIONAL)
    assert out.metadata["validation"]["initial"]["codes"] == [APPLICABILITY_DROPPED]
    assert withheld(out, draft)


def test_production_configuration_withholds_through_an_existing_whole_answer_path():
    """As serve_ui (configured_rag_config). Over this table the existing whole-answer replacement runs; over the
    recorded database rows it is the unvalidated-answer gate (pinned in the integration test)."""
    draft = SOUND + UNSUPPORTED
    out = RagPipeline(LexicalRetriever(TableDatabase(TABLE)), ScriptedGenerator(draft), configured_rag_config(load_settings())).answer(ONE)
    assert out.metadata["validation"]["repair"] in ("extractive_replacement", "unvalidated_answer_withheld")
    assert withheld(out, draft) and "Quarterly biometric" not in out.answer


def test_a_blocking_finding_that_survives_repair_still_withholds():
    # the invalid citation is repairable; the unsupported claim it carries is not, so nothing is delivered
    draft = SOUND + " Quarterly biometric turbine calibration logbooks stay mandatory [9]."
    p, _ = pipeline(None, response=draft)
    out = p.answer(ONE)
    assert {"invalid_citation", CV.UNSUPPORTED_CLAIM} <= set(out.metadata["validation"]["initial"]["codes"])
    assert withheld(out, draft) and "[9]" not in out.answer


# ------------------------------------------------------------ existing repair behaviour preserved


@pytest.mark.usefixtures("delivered_as_drafted")   # delivered exactly as validated / repaired
def test_an_invented_condition_alone_is_still_repaired_in_place():
    """D-013 passed only after this repair: an invented condition with nothing else wrong is removed in place."""
    p, _ = pipeline(None, response=SOUND + " If appropriate, the rule also covers imports [1].")
    out = p.answer(ONE)
    assert out.metadata["validation"]["repair"] == "invented_conditions_removed"
    assert out.answer == SOUND


def test_an_invented_condition_beside_a_treated_finding_is_no_longer_repaired_in_place():
    """D-019, H-002, M-017 and M-022 in the experiment: the treated error sends the whole answer down the
    withholding path instead of the in-place repair (all four were scored FAIL)."""
    draft = MISPLACED + " If appropriate, the rule also covers imports [1]."
    p, _ = pipeline(None, response=draft)
    out = p.answer(TWO)
    initial = out.metadata["validation"]["initial"]
    assert {CV.INVENTED_CONDITION, CV.CONDITION_NOT_IN_CITED} <= set(initial["codes"]) and initial["status"] == STATUS_UNSUPPORTED
    assert "invented_conditions_removed" not in (out.metadata["validation"]["repair"] or "")
    assert withheld(out, draft)


# ------------------------------------------------------------ known costs, measured and accepted for decision


@pytest.mark.usefixtures("model_drafted_path")   # the cost is on the model-drafted path
def test_known_cost_a_faithful_paraphrase_with_little_source_wording_is_withheld():
    """Lexical support cannot judge paraphrase (Stage 7's reason for keeping this a warning). This faithful
    paraphrase of the verbatim 21 CFR 211.25(a) text shares under 30% of its content words with the sources,
    so the answer is withheld. The recorded run had no such PASS answer (0 of 7 lost); this pins that the
    risk is real. A closer paraphrase (above) is delivered."""
    from tests.test_accuracy_pass_integrity import C, FOCUSED, S211

    paraphrase = "Each person making drugs needs suitable schooling and experience for the job [1]."
    draft = f"{paraphrase} {C} [3]"
    out = RagPipeline(LexicalRetriever(TableDatabase(S211)), ScriptedGenerator(draft), configured_rag_config(load_settings())).answer(FOCUSED)
    assert CV.UNSUPPORTED_CLAIM in out.metadata["validation"]["initial"]["codes"]
    # withheld; with Q1 (on by default since 27 Oct) the provision is quoted instead - the paraphrase is never shown
    assert out.metadata["fallback_type"] in ("unvalidated_answer_withheld", "focused_provision_quote")
    assert "schooling" not in out.answer


def test_known_interaction_the_truncation_note_is_scored_as_a_claim():
    """Open for decision. Re-validating the repaired answer reads the pipeline's own truncation note as part
    of the cut sentence. After a one-word fragment it has no source words, so unsupported_claim (now an
    error) fails the repair and the answer is withheld, where it used to be delivered with its note
    (tests/test_answer_length_policy.py test_e_*, tests/test_answer_validation.py, which now fail). The
    three recorded truncated answers are unaffected (their fragments carry source wording), as is the
    realistic case below."""
    short = ScriptedGenerator("21 CFR 820.10(b)(2): traceability [1]. 21 CFR 820.10(b)(4): advis", finished=False)
    out = RagPipeline(LexicalRetriever(TableDatabase()), short).answer(TWO)
    assert out.metadata["validation"]["repair"] == "truncation_noted+failed_repair_fallback" and WITHHELD in out.answer
    candidate = validate_answer(out.metadata["repair_output"], out.package)
    assert [p.sentence for p in candidate.errors if p.code == CV.UNSUPPORTED_CLAIM][0].endswith(
        "(Incomplete: generation stopped at the token limit before this answer finished, so wording after this point is missing.)")


def test_a_realistically_truncated_answer_is_still_delivered_with_its_note():
    draft = (SOUND + " 21 CFR 820.10(b)(4): the manufacturer must document procedures for advisory notices to be handled "
             "under part 806 of this chap")
    out = RagPipeline(LexicalRetriever(TableDatabase(TABLE)), ScriptedGenerator(draft, finished=False)).answer(TWO)
    assert out.metadata["validation"]["repair"] == "truncation_noted"
    assert out.answer.startswith(SOUND) and "Incomplete: generation stopped at the token limit" in out.answer
    assert out.validation.truncated and WITHHELD not in out.answer
