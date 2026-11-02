"""The per-question evaluation status shown with a live answer.

A reading of `AccuracyAssessment.to_dict()`, never a second judgement: every state here is one the evaluator already
decided. What this module adds is the one-word status a reader sees first.

Calibrated rule (default since 23 Oct 2026; REGULUS_REVIEW_RULE=strict restores the earlier one):

    FAIL            a check found a DEFECT: the answer verdict is incorrect, off-topic or a wrong refusal; a statement
                    is contradicted by the governing text or the question went unanswered (substantive correctness
                    failed); a fabricated identifier or figure; a citation that points to a source that does not
                    support it; scope; a requested provision or a named element of the question left out.
    PASS            no defect, every citation checked points to a supporting source, no figure, identifier or scope
                    left undecided, and the answer was verified against the retrieved regulatory text: at least two
                    statements that answer the question were matched to evidence and at most a third of its material
                    statements could not be checked (or every one was verified). Never with a statement no evidence
                    corresponds to, an unverified statement carrying a quantity, or completeness in doubt (a possible
                    omission, or the question's subject not addressed - the mark of an answer built on the wrong section).
    NOT EVALUATED   everything else, shown as "NOT FULLY EVALUATED" when no defect was found but too little could be
                    checked; never a pass. Also: no assessment, the evaluator abstained or failed, no requirement stated,
                    incomplete metadata, another question's result, a decline (one the evidence bears out may still be
                    wrong: the evidence can lack the provision that answers it), and an answer replaced by verbatim
                    provision text because the generated answer could not be validated (it does not apply them).

Why: on the 22 Oct holdout (97 answers with blind two-lens verdicts) the strict rule - PASS only when every dimension
passed - called 0 of 37 blind-PASS answers PASS and 35 of them FAIL, because statements the aligner cannot verify (a
correct paraphrase) made dimensions "partial" and "partial" counted as failure; "partial" was in fact more common on
blind-PASS answers (31/37) than on the rest (14/60). Uncertainty is now reported as uncertainty, and FAIL is kept for
defects. Measured on 182 blind-scored answers (holdout 22 Oct; G5 arms, incl. 39 answers to dev questions the rule was
not tuned on): blind-PASS answers shown FAIL 44 -> 8; PASS shown 13 times, 11 of them blind PASS and none misleading
(the strict rule showed PASS twice, wrongly). Most correct answers are NOT FULLY EVALUATED: without a reference answer the
evaluator can establish that an answer restates the retrieved text correctly, not that the text is the right section.
See regulus_eval_runs/evaluator_calibration_20261023 for the measurement.

The assessment is the live evidence assessment of this one answer (`regulus_live_evidence_assessment_v2`): it
compares the answer with the regulatory evidence retrieved for it. No reference answer is involved, and it is not the
300-question acceptance-test result.
"""

from __future__ import annotations

import os
import re
from typing import Any

from regulus.accuracy import rubric as R

PASS = "pass"
FAIL = "fail"
NOT_EVALUATED = "not_evaluated"

LABEL = {PASS: "PASS", FAIL: "FAIL", NOT_EVALUATED: "NOT EVALUATED"}
NOT_FULLY_EVALUATED = "NOT FULLY EVALUATED"
# fallback types whose answer is the provision's own text, quoted and cited (regulus.rag.fallback)
QUOTED_FALLBACKS = frozenset({"focused_provision_quote", "composed_records_silence", "composed_qmsr",
                              "composed_premise_figure", "composed_definitions"})

STATE_LABEL = {R.PASS: "Pass", R.FAIL: "Fail", R.PARTIAL: "Partial", R.NOT_ESTABLISHED: "Not established",
               R.NOT_APPLICABLE: "Not applicable"}

_PASSING_VERDICTS = frozenset({"correct", "abstention_correct"})
_FAILING_VERDICTS = frozenset({"incorrect", "partial", "off_topic", "abstention_incorrect"})

BASIS = ("Live evidence assessment of this one answer: it is compared with the regulatory evidence retrieved for "
         "it. No reference answer is used, and this is not the 300-question acceptance-test result.")
_MAX_EXPLANATION = 400
RULE = os.environ.get("REGULUS_REVIEW_RULE", "calibrated").strip().lower()
# dimensions whose FAIL is a defect in the answer itself (substantive correctness and completeness are read separately)
_DEFECT_DIMENSIONS = (R.FABRICATION_FREE, R.CITATION, R.SCOPE, R.QUESTION_RELEVANCE)
_SUBJECT_ELEMENT = "the question's subject"
_ANSWERING = frozenset({"relevant", "partially_relevant"})
_UNVERIFIED = "unverifiable"
# an unverified statement is tolerated only as a paraphrase: never when it carries a quantity nothing confirmed
# ("at least three supervisors per shift", "retained for ten years", "an annual examination")
# (a bare number word is not one: "21 CFR 82.1328 contains one requirement" restates the section; the invented cases carry
# a digit, a unit of time, "per" or a frequency word - "annual", "ten years", "three supervisors per shift")
_QUANTITY = re.compile(r"\d|\b(?:twice|annual(?:ly)?|daily|weekly|monthly|quarterly|yearly|hourly|per|"
                       r"each\s+(?:day|week|month|year|shift)|days?|weeks?|months?|years?|hours?|percent)\b", re.I)
# citation labels and markers are not quantities ("21 CFR 820.10(b)(2)", "§ 605(e)", "section 721", "[3]")
_CITE = re.compile(r"(?:\b21\s*CFR\s*|§+\s*|\bsection\s+|\bpart\s+|\bFD&C Act\s*§?\s*)\d[\w.]*(?:\s*\([a-z0-9]{1,5}\))*|"
                   r"\b\d{1,4}\.\d+[a-z]?(?:\s*\([a-z0-9]{1,5}\))*|\[\d{1,2}\]|\b\d+\s*U\.S\.C\.\s*\d+\w*", re.I)
# completeness left undecided because content may be missing (repair removed it, or required paragraphs were not in
# the evidence, or only unverifiable statements address a requirement): never a pass
_POSSIBLE_OMISSION = re.compile(r"removed before delivery|were not in the evidence|states a requirement", re.I)


def _dimension(dims: dict[str, dict], name: str) -> dict[str, Any] | None:
    d = dims.get(name)
    if d is None:
        return None
    return {"state": d["state"], "label": STATE_LABEL[d["state"]], "detail": str(d.get("detail") or "")}


def _score(value: Any) -> int | None:
    """The evaluator's score as displayed: `rubric.score()` is max(0, 100 - deductions),
    an integer from 0 to 100. Anything else is not that score and is not shown."""
    return value if type(value) is int and 0 <= value <= 100 else None


def _short(text: Any) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= _MAX_EXPLANATION else text[: _MAX_EXPLANATION - 1].rstrip() + "…"


def question_review(assessment: Any, *, question: str) -> dict[str, Any]:
    """The status to display for `question`, from its own assessment dict."""

    def out(status: str, explanation: str, *, label: str | None = None, dims: dict[str, dict] | None = None,
            score: int | None = None) -> dict[str, Any]:
        dims = dims or {}
        return {
            "question": question,
            "status": status,
            "label": label or LABEL[status],
            "substantive_correctness": _dimension(dims, R.SUBSTANTIVE),
            "citation_correctness": _dimension(dims, R.CITATION),
            "explanation": _short(explanation),
            # the evaluator's own score (rubric.score: 100 minus deductions, 0-100). An evaluation
            # score, never a percentage of factual accuracy; None when there is no valid one.
            "live_check_score": _score(score),
            "rubric_id": R.RUBRIC_ID,
            "basis": BASIS,
        }

    if not isinstance(assessment, dict):
        return out(NOT_EVALUATED, "No evaluation result was returned for this answer.")
    checked = assessment.get("checked") if isinstance(assessment.get("checked"), dict) else {}
    if checked.get("question") is not None and checked.get("question") != question:
        return out(NOT_EVALUATED, "The evaluation result does not belong to this question, so it is not shown.")
    status = assessment.get("status")
    reason = assessment.get("reason") or ""
    if status == R.ERROR:
        return out(NOT_EVALUATED, f"The evaluator failed, so this answer was not evaluated. {reason}")
    if status == R.NOT_SCORED:
        return out(NOT_EVALUATED, reason or "The answer states no regulatory requirement, so there is nothing to evaluate.")
    if status == R.UNABLE:
        return out(NOT_EVALUATED, reason or "Not enough could be established from the retrieved evidence.")
    if status != R.EVALUATED:
        return out(NOT_EVALUATED, "The evaluation result is incomplete, so it is not shown.")

    dims: dict[str, dict] = {}
    for d in assessment.get("dimensions") or ():
        if isinstance(d, dict) and d.get("dimension") in R.DIMENSIONS and d.get("state") in R.STATES:
            dims[d["dimension"]] = d
    verdict = (assessment.get("answer_verdict") or {}).get("verdict") if isinstance(assessment.get("answer_verdict"), dict) else None
    score = assessment.get("accuracy")
    if set(dims) != set(R.DIMENSIONS) or verdict is None or not isinstance(score, int):
        return out(NOT_EVALUATED, "The evaluation result is incomplete, so it is not shown.")

    if RULE != "strict":
        return _calibrated(assessment, dims, verdict, score, reason, out)
    failed = [d for d in R.DIMENSIONS if dims[d]["state"] in (R.FAIL, R.PARTIAL)]
    if failed or verdict in _FAILING_VERDICTS:
        # the evaluator's reason leads with the answer verdict, which can be
        # "Correct" while a check still failed: name the failed checks instead
        why = reason if verdict in _FAILING_VERDICTS else "; ".join(
            f"{R.LABELS[d]} {STATE_LABEL[dims[d]['state']].lower()}: {dims[d].get('detail') or ''}".rstrip(": ")
            for d in failed)
        return out(FAIL, why, dims=dims, score=score)
    undecided = [R.LABELS[d] for d in R.DIMENSIONS if dims[d]["state"] == R.NOT_ESTABLISHED]
    if undecided or verdict not in _PASSING_VERDICTS:
        why = (f"No check failed, but these could not be decided from the retrieved evidence: {', '.join(undecided)}."
               if undecided else f"No check failed, but the answer verdict is '{verdict}'.")
        return out(NOT_EVALUATED, why, label=NOT_FULLY_EVALUATED, dims=dims, score=score)
    return out(PASS, reason, dims=dims, score=score)


def _calibrated(assessment: dict, dims: dict[str, dict], verdict: str, score: int, reason: str, out) -> dict[str, Any]:
    """The calibrated rule (module docstring): FAIL for a defect, PASS for a verified answer, otherwise not fully
    evaluated."""
    state = lambda d: dims[d]["state"]
    detail = lambda d: str(dims[d].get("detail") or "")
    checked = assessment.get("checked") if isinstance(assessment.get("checked"), dict) else {}
    # the provision text shown verbatim because the generated answer could not be validated: the regulation's own words,
    # not applied to the question - neither a pass nor, unless a check below finds a real defect in it, a failure (the
    # evaluator reads it as a decline; blind scorers passed 82.1336's and failed 700.3-Q2's, 22 Oct holdout)
    verbatim = any(str(x.get("origin")) == "pipeline_sentence" and "could not be validated" in str(x.get("text") or "").lower()
                   for x in checked.get("excluded_sentences") or () if isinstance(x, dict))
    # the answers composed from the provision's own quoted text (Q1, FP1, RS1, QM1; 26 Oct): judged by word relevance
    # they read as "answers a different question" - 5 of 9 good follow-up answers shown FAIL were these
    quoted = checked.get("fallback_type") in QUOTED_FALLBACKS
    if quoted:
        # the provision's text under its own citation: the citation and relevance checks misread a quotation (710.4's
        # own words name "§ 700.3(j)"), and correctness is the quoted text's - it is never a pass and never a failure
        return out(NOT_EVALUATED, "The answer quotes the provision's own text, verbatim and cited; whether it is the "
                                  "right provision for the question is not something this check can establish.",
                   label=NOT_FULLY_EVALUATED, dims=dims, score=score)
    defects = []
    # a statement the operators found to drop a condition its evidence carries - the event a time limit runs from
    # ("7 calendar days" without "after the sponsor's initial receipt", 312.32(c)(2)), a scope extension ("whether or not
    # ...", 211.192) - is a defect, not paraphrase uncertainty: "partial" for that reason was counted as verified, so
    # such an answer passed (2 Nov 2026; the four tests of those fixes had failed since the calibration of 23 Oct)
    dropped = [c for c in checked.get("claims") or () if isinstance(c, dict)
               and c.get("support") == "partially_supported" and c.get("support_basis") == "drops_a_condition"]
    if not verbatim and (verdict in ("incorrect", "off_topic", "abstention_incorrect") or state(R.SUBSTANTIVE) == R.FAIL
                         or dropped):
        defects.append(R.SUBSTANTIVE)
    defects += [d for d in _DEFECT_DIMENSIONS if state(d) == R.FAIL]
    missing = [str(x) for x in (dims[R.COMPLETENESS].get("evidence") or ())]
    if verbatim:
        pass                                    # completeness of quoted text is not the answer's own omission
    elif state(R.COMPLETENESS) == R.FAIL and any(x != _SUBJECT_ELEMENT for x in missing or ("?",)):
        defects.append(R.COMPLETENESS)          # a named part of the question or a requested provision left out
    elif state(R.COMPLETENESS) == R.PARTIAL and "requested provision" in detail(R.COMPLETENESS):
        defects.append(R.COMPLETENESS)          # a requested provision had no text, said so: still not complete
    if defects:
        why = reason if R.SUBSTANTIVE in defects and verdict in _FAILING_VERDICTS else "; ".join(
            f"{R.LABELS[d]} {STATE_LABEL[state(d)].lower()}: {detail(d)}".rstrip(": ") for d in defects)
        return out(FAIL, why, dims=dims, score=score)

    if verdict == "abstention_correct":
        # a decline the retrieved evidence bears out is not shown as a pass: the evidence may lack the provision that
        # answers it (FDCA612-Q2: 612(a) states the figure, and it was not retrieved; blind scoring fails every decline)
        return out(NOT_EVALUATED, "The answer declined. Declining is consistent with the retrieved evidence, which does "
                                  "not state what was asked; that does not establish that the regulation is silent.",
                   label=NOT_FULLY_EVALUATED, dims=dims, score=score)
    if verbatim:
        return out(NOT_EVALUATED, "The generated answer could not be validated, so the provision text was shown "
                                  "verbatim instead; it does not apply the provisions to the question.",
                   label=NOT_FULLY_EVALUATED, dims=dims, score=score)
    claims = [c for c in checked.get("claims") or () if isinstance(c, dict)]
    answering = [c for c in claims if (c.get("relevance") or {}).get("verdict") in _ANSWERING]
    verified = sum(1 for c in answering if c.get("final") != _UNVERIFIED)
    # what could not be checked counts every material statement not judged off the question: an unverifiable
    # statement the relevance check left undetermined ("It regulates all food sold worldwide") is still a claim
    considered = [c for c in claims if (c.get("relevance") or {}).get("verdict") != "irrelevant"]
    unverified = sum(1 for c in considered if c.get("final") == _UNVERIFIED)
    # an identifier or figure nothing accounts for (fabrication not established), or scope not established, is never
    # passed: "This duty is restated in 21 CFR 820.777" names a section the evidence cannot vouch for
    vouched = all(state(d) == R.PASS for d in (R.CITATION, R.FABRICATION_FREE, R.SCOPE))
    # a statement with no corresponding evidence at all (the invented "annual CGMP proficiency test" beside verbatim
    # 211.25 text is "unverifiable / no_evidence", and not counted as answering), an unverified statement with a
    # quantity, or completeness undecided for a possible omission: never a pass (tests/test_accuracy_pass_integrity)
    invented = [c for c in claims if c.get("final") == _UNVERIFIED and c.get("support_basis") == "no_evidence"]
    unconfirmed_quantity = [c for c in answering if c.get("final") == _UNVERIFIED and _QUANTITY.search(_CITE.sub(" ", str(c.get("claim") or "")))]
    omission = state(R.COMPLETENESS) == R.NOT_ESTABLISHED and _POSSIBLE_OMISSION.search(detail(R.COMPLETENESS))
    # the question's subject not addressed: noise on right-section answers (14 blind-PASS holdout answers), but the mark
    # of an answer built on the wrong section (G5 held-out: 70.3-Q1, 71.25-Q1, FDCA606-Q2, FDCA602-Q1 restate the
    # evidence faithfully and answer another question) - never a failure on its own, never a pass
    omission = omission or (state(R.COMPLETENESS) in (R.FAIL, R.PARTIAL) and not defects)
    vouched = vouched and not invented and not unconfirmed_quantity and not omission
    enough = ((verified >= 2 and 3 * unverified <= len(considered))
              or (verified >= 1 and unverified == 0 and state(R.EVIDENCE_SUPPORT) == R.PASS))
    if vouched and enough:
        why = (f"No defect found: {verified} of {len(considered)} statement(s) were checked "
               f"against the retrieved regulatory text and their citations point to supporting sources"
               + (f"; {unverified} could not be checked" if unverified else "") + ".")
        return out(PASS, why, dims=dims, score=score)
    undecided = [f"{R.LABELS[d]} ({detail(d)})" for d in R.DIMENSIONS if state(d) == R.NOT_ESTABLISHED]
    undecided += [f"a statement no retrieved evidence corresponds to (\"{str(c.get('claim'))[:120]}\")" for c in invented[:2]]
    undecided += [f"an unconfirmed quantity (\"{str(c.get('claim'))[:120]}\")" for c in unconfirmed_quantity[:2]]
    why = ("No defect was found, but " + (f"only {verified} of {len(considered)} statement(s) could be checked against "
           f"the retrieved regulatory text" if vouched else "not everything could be checked")
           + (f"; not decided: {'; '.join(undecided)}" if undecided else "") + ".")
    return out(NOT_EVALUATED, why, label=NOT_FULLY_EVALUATED, dims=dims, score=score)
