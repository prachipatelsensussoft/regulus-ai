"""AV1 (4 Nov, report-only): one verdict for the delivered answer as a whole.

    complete                 no check finds an omission, an unsupported statement or a contradiction
    correct_but_incomplete   what it states is supported, but a requested provision, list item or part of the evidence
                             is not covered, or the answer is a verbatim quote standing in for an answer
    unsupported              a statement, figure, condition or citation is not what the supplied sources state
    incorrect                a statement contradicts its source: a false denial, a duty made stronger than the text, an
                             applicability dropped, a yes/no premise the provision does not state, a missing provision
                             described as if it existed
    insufficient_evidence    a safe refusal or a decline: out of scope, provision not found, value not established,
                             answer withheld, a clarifying question

The checks are deterministic and are the validator's own, re-run on the delivered text, plus CC1's citation verdicts.
AV1 is a proxy and not the grade: its agreement with the blind scorer's verdicts is reported per arm (it is calibrated,
never trusted on its own). Nothing in the answer changes; QP1 is the switch that acts on question-part coverage.
"""
from __future__ import annotations

import re
from typing import Any

from regulus.rag import citation_verdicts as CCV
from regulus.rag import claim_validation as CV
from regulus.rag import fallback as FB
from regulus.rag import shape_validation as SV
from regulus.rag import validation as V

VERSION = 1
COMPLETE = "complete"
INCOMPLETE = "correct_but_incomplete"
UNSUPPORTED = "unsupported"
INCORRECT = "incorrect"
INSUFFICIENT = "insufficient_evidence"
VERDICTS = (COMPLETE, INCOMPLETE, UNSUPPORTED, INCORRECT, INSUFFICIENT)

REFUSALS = frozenset({FB.FB_OUT_OF_SCOPE, FB.FB_UNSUPPORTED_REQUEST, FB.FB_AMBIGUOUS, FB.FB_WRONG_PROVISION,
                      FB.FB_PROVISION_NOT_FOUND, FB.FB_NO_EVIDENCE, FB.FB_VALUE_NOT_ESTABLISHED, FB.FB_UNVALIDATED,
                      FB.FB_PREMISE_NOT_ESTABLISHED, FB.FB_PRODUCT_CLARIFICATION})
VERBATIM_FALLBACKS = frozenset({FB.FB_FOCUSED_QUOTE, FB.FB_VERBATIM_PROVISION, FB.FB_DEFINITIONS_QUOTED})
INCORRECT_CODES = frozenset({CV.FALSE_DENIAL, CV.OBLIGATION_STRENGTHENED, SV.APPLICABILITY_DROPPED, CV.UNSUPPORTED_PREMISE,
                             V.ASSERTED_MISSING_PROVISION})
UNSUPPORTED_CODES = frozenset({CV.UNSUPPORTED_CLAIM, CV.INVENTED_CONDITION, CV.CONDITION_NOT_IN_CITED, V.UNSUPPORTED_FIGURE,
                               CV.PARAGRAPH_MISATTRIBUTION, CV.FIGURE_MISATTRIBUTED, V.PROVISION_CITATION_MISMATCH,
                               V.CITATION_MISMATCH, V.DISCOVERY_CITED_FOR_REQUIREMENT, V.INVALID_CITATION})
INCOMPLETE_CODES = frozenset({V.REQUIREMENT_NOT_ADDRESSED, SV.ENUMERATION_INCOMPLETE, SV.MATERIAL_GROUP_UNCOVERED,
                              V.TRUNCATED})
_EVIDENCE_LIMIT = re.compile(r"^\s*Evidence limit:", re.M)
_NOT_FOUND = re.compile(r"^\s*Not found in the available eCFR data:", re.M)
_NOT_ANSWERED = re.compile(r"^\s*Not answered: ", re.M)   # answer_quality.mixed_scope_note


def answer_verdict(answer: str, package, *, question: str | None = None, fallback_type: str | None = None,
                   gate_state: str | None = None, evidence_gap: Any = None,
                   citation_verdicts: dict[str, Any] | None = None, notes: str = "") -> dict[str, Any]:
    """`answer`: the delivered answer body; `notes`: the closing notes the pipeline appended after it (the limitation,
    not-found, evidence-limit and status notes) - read for the incomplete reasons, never validated as claims."""
    text = (answer or "").strip()
    reasons: list[str] = []
    out: dict[str, Any] = {"version": VERSION, "verdict": None, "reasons": reasons}
    if fallback_type in REFUSALS:
        reasons.append(f"fallback: {fallback_type}")
        out["verdict"] = INSUFFICIENT
        return out
    if not text or package is None:
        reasons.append("no answer text" if not text else "no evidence package")
        out["verdict"] = INSUFFICIENT
        return out
    report = V.validate_answer(text, package)
    codes = {p.code: p.severity for p in report.problems}
    out["codes"] = dict(sorted(codes.items()))
    cv = citation_verdicts if citation_verdicts and "summary" in citation_verdicts else \
        CCV.citation_verdicts(text, package, question)
    out["citations"] = cv.get("summary")

    incorrect = sorted(c for c, sev in codes.items() if c in INCORRECT_CODES and sev == "error")
    unsupported = sorted(c for c, sev in codes.items() if c in UNSUPPORTED_CODES and sev == "error")
    bad_citations = {v: (cv.get("summary") or {}).get(v, 0) for v in (CCV.NOT_SUPPORTING, CCV.WRONG, CCV.UNVERIFIABLE)}
    incomplete = sorted(c for c in codes if c in INCOMPLETE_CODES)   # an optional provision's omission counts here too
    if incorrect:
        reasons += incorrect
        out["verdict"] = INCORRECT
    elif unsupported or any(bad_citations.values()):
        reasons += unsupported + [f"citations {v}: {n}" for v, n in bad_citations.items() if n]
        out["verdict"] = UNSUPPORTED
    else:
        reasons += incomplete
        if fallback_type in VERBATIM_FALLBACKS:
            reasons.append(f"verbatim fallback: {fallback_type}")
        closing = f"{text}\n{notes or ''}"
        if _EVIDENCE_LIMIT.search(closing) or evidence_gap:
            reasons.append("evidence limit: part of a requested provision was left out of the evidence")
        if _NOT_FOUND.search(closing):
            reasons.append("a requested provision was not found in the corpus")
        if _NOT_ANSWERED.search(closing):
            reasons.append("a part of the question is outside the available sources")
        if gate_state == "PARTIALLY_ANSWERABLE":
            reasons.append("gate: partially answerable")
        out["verdict"] = INCOMPLETE if reasons else COMPLETE
    return out


def safe_answer_verdict(answer: str, package, **kw) -> dict[str, Any]:
    """answer_verdict(), with any failure recorded instead of raised: AV1 is report-only and never breaks an answer."""
    try:
        return answer_verdict(answer, package, **kw)
    except Exception as exc:  # noqa: BLE001 - report-only
        return {"version": VERSION, "error": f"{type(exc).__name__}: {exc}"[:300]}
