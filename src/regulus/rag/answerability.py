"""Answerability gate (Stage 5): may the model be asked to answer, given this evidence?

Retrieval success is not answerability. The decision is taken after retrieval,
coverage and selection and before generation, from signals the pipeline
already computes (plan, question analysis, coverage, the selected evidence).
It is deterministic, and every decision carries its reason, the signals that
produced it, a confidence and the evidence it looked at.

    ANSWERABLE             every requested provision has exact evidence; nothing checkable is missing
    PARTIALLY_ANSWERABLE   some of the request is supported: provisions found and not found, a requested
                           value missing from one part of a multi-part question, or an unrelated clause
    INSUFFICIENT_EVIDENCE  the evidence cannot establish the request: no requested provision was found,
                           nothing was retrieved, or the question asks for a kind of value no source contains
    OUT_OF_SCOPE           a named non-U.S. framework with no U.S. reference (regulus.planning.intent)
    AMBIGUOUS              the question points at a provision without saying which one ("paragraph (b)")
    WRONG_PROVISION        a citation lookup whose only named provision is the one being REFERRED TO
                           ("meeting part 803 criteria"), so the evidence is the reference, not the answer
    UNSUPPORTED_REQUEST    a request for a determination the sources cannot make (is my product compliant)
    UNKNOWN                a general question: whether the sources answer it is not decidable by rule

Blocked (the model is not called): INSUFFICIENT_EVIDENCE, OUT_OF_SCOPE, AMBIGUOUS,
WRONG_PROVISION, UNSUPPORTED_REQUEST. Everything else is generated as before;
PARTIALLY_ANSWERABLE carries a limitation note into the prompt and the answer.
There are no topic, number, question or provision rules here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from regulus.planning.intent import SCOPE_MIXED, SCOPE_OUT, TYPE_CITATION
from regulus.rag.evidence_values import (
    evidence_has_value,
    requested_value_kind,
    value_kind_label,
    value_request_is_whole_question,
    with_article,
)

ANSWERABLE = "ANSWERABLE"
PARTIALLY_ANSWERABLE = "PARTIALLY_ANSWERABLE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
OUT_OF_SCOPE = "OUT_OF_SCOPE"
AMBIGUOUS = "AMBIGUOUS"
WRONG_PROVISION = "WRONG_PROVISION"
UNSUPPORTED_REQUEST = "UNSUPPORTED_REQUEST"
UNKNOWN = "UNKNOWN"
STATES = (ANSWERABLE, PARTIALLY_ANSWERABLE, INSUFFICIENT_EVIDENCE, OUT_OF_SCOPE, AMBIGUOUS, WRONG_PROVISION,
          UNSUPPORTED_REQUEST, UNKNOWN)
BLOCKING = (INSUFFICIENT_EVIDENCE, OUT_OF_SCOPE, AMBIGUOUS, WRONG_PROVISION, UNSUPPORTED_REQUEST)

PROVEN = "PROVEN"        # decided by deterministic facts (scope rule, coverage, value absent from every source)
PROBABLE = "PROBABLE"    # decided by a wording pattern that can misfire
UNDETERMINED = "UNDETERMINED"

# a mention that is the object of a reference, not the thing asked for
_REFERENCE_CUE = re.compile(r"\b(?:refers?\s+to|referenc\w*|cross-referenc\w*|meeting|meets|criteria\s+(?:of|in)|in\s+accordance\s+with"
                            r"|pursuant\s+to|as\s+defined\s+in|described\s+in|listed\s+in)\s+(?:the\s+)?(?:\S+\s+){0,2}$", re.I)
# a provision pointed at without saying which one
_AMBIGUOUS_RE = re.compile(r"\b(?:paragraph|subsection|clause)\s*\((?:[a-z0-9]{1,4}|[ivxl]{5,8})\)(?:\((?:[a-z0-9]{1,4}|[ivxl]{5,8})\))*|\b(?:this|that|the\s+same)\s+"
                           r"(?:section|paragraph|provision|regulation|rule)\b", re.I)
# a determination about a particular product, company or case, or a prediction
_DETERMINATION_RE = re.compile(r"\b(?:is|are|am|was|would|will)\s+(?:my|our|this|their)\s+(?:device|product|company|firm|facility|"
                               r"process|label|submission)\s+(?:be\s+)?(?:compliant|in\s+compliance|approved|cleared|legal|acceptable)\b"
                               r"|\bwill\s+(?:the\s+)?FDA\s+(?:approve|clear|accept|reject)\b|\b(?:am|are)\s+(?:i|we)\s+compliant\b", re.I)


@dataclass(frozen=True)
class AnswerabilityDecision:
    state: str
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)
    confidence: str = UNDETERMINED
    evidence_ids: tuple[str, ...] = ()
    blocking_conditions: tuple[str, ...] = ()
    limitation: str | None = None      # PARTIALLY_ANSWERABLE: what cannot be answered, said to the model and the reader

    @property
    def blocked(self) -> bool:
        return self.state in BLOCKING

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "reason": self.reason, "signals": self.signals, "confidence": self.confidence,
                "evidence_ids": list(self.evidence_ids), "blocking_conditions": list(self.blocking_conditions),
                "blocked": self.blocked, "limitation": self.limitation}


def reference_targets(plan) -> list[str]:
    """Canonical labels of provisions that the question mentions as the object of a reference."""
    out = []
    for req in plan.requirements:
        m = plan.scan.mentions[req.mention] if req.mention < len(plan.scan.mentions) else None
        if m is not None and _REFERENCE_CUE.search(plan.scan.text[max(0, m.start - 40):m.start]):
            out.append(req.canonical)
    return out


# FJ1: how a named framework is referred to in the note
_FOREIGN_LABEL = {"health canada": "Health Canada", "canadian": "Canadian", "eu": "EU", "e.u.": "EU", "european union": "EU",
                  "european": "European", "mhra": "UK MHRA", "uk mdr": "UK MDR", "pmda": "Japanese PMDA", "japanese": "Japanese",
                  "tga": "Australian TGA", "australian": "Australian", "nmpa": "Chinese NMPA", "chinese": "Chinese",
                  "cdsco": "Indian CDSCO", "anvisa": "Brazilian ANVISA", "swissmedic": "Swissmedic", "ivdr": "EU IVDR"}


def decide(question: str, plan, analysis, *, coverage, sources, no_context: bool, general_knowledge: str | None = None,
           resolved_statute: tuple[str, ...] = (), unknown_guidance: dict | None = None,
           premise_fact: str | None = None, product_choices: dict | None = None) -> AnswerabilityDecision:
    """The answerability of one question. `sources` are the context sources that would be placed
    in the prompt (anything with .text, and .result.key or .evidence_id to identify it)."""
    texts = [s.text for s in sources]
    ids = tuple(str(getattr(getattr(s, "result", None), "key", None) or getattr(s, "evidence_id", "")) for s in sources)
    signals: dict[str, Any] = {"question_type": analysis.question_type, "scope": analysis.scope_status,
                               "requested_provisions": [r.canonical for r in plan.requirements], "sources": len(texts)}

    def d(state, reason, confidence, blocking=(), limitation=None, **extra):
        signals.update(extra)
        return AnswerabilityDecision(state, reason, dict(signals), confidence, ids, tuple(blocking), limitation)

    # 1. scope: decided from the question (a named non-U.S. framework, no U.S. reference, no CFR provision)
    if analysis.scope_status == SCOPE_OUT:
        return d(OUT_OF_SCOPE, f"the question {analysis.scope_reason}", PROVEN, ("scope",))
    if general_knowledge:
        # Stage 7 scope policy (regulus.retrieval.domain): three independent signals agree
        return d(OUT_OF_SCOPE, f"the question {general_knowledge}", PROBABLE, ("general_knowledge",))
    # 2. requests the sources cannot answer by their nature
    from regulus.rag.validator_switches import enabled as precision_fix

    boundary = None
    if precision_fix("BR1"):
        from regulus.rag.boundary import classify

        boundary = classify(question)
    if unknown_guidance:
        # GT1: a guidance document asked about by title that FDA's listing, as held, does not contain
        detail = f"{unknown_guidance['title']}|{unknown_guidance['documents']}|{unknown_guidance['retrieved']}"
        return d(INSUFFICIENT_EVIDENCE, f"no FDA guidance titled \"{unknown_guidance['title']}\" is in the guidance listing",
                 PROVEN, ("boundary_unknown_guidance",), boundary={"kind": "unknown_guidance", "detail": detail,
                                                                   "question": question})
    if premise_fact:
        # NX2: what the database says about a subpart the question names - absent or reserved - stated as the answer
        return d(INSUFFICIENT_EVIDENCE, "a subpart the question names is absent or reserved in its part", PROVEN,
                 ("boundary_premise_fact",), boundary={"kind": "premise_fact", "detail": premise_fact, "question": question})
    if boundary is not None:
        # BR1: a request a research tool must not carry out as asked (regulus.rag.boundary), answered by its kind
        kind, detail = boundary
        state = OUT_OF_SCOPE if kind == "other_agency" else UNSUPPORTED_REQUEST
        return d(state, f"the question asks for something outside what the sources can do ({kind})", PROBABLE,
                 (f"boundary_{kind}",), boundary={"kind": kind, "detail": detail, "question": question})
    if _DETERMINATION_RE.search(question):
        return d(UNSUPPORTED_REQUEST, "the question asks for a determination about a particular case or a prediction, "
                 "which regulatory text cannot establish", PROBABLE, ("determination_request",))
    if product_choices:
        # AMB1: the same provision written for several product types, and the question names none (product_choices:
        # {family: [[section, heading, subpart], ...]}, regulus.rag.product_clarification)
        return d(AMBIGUOUS, "the requirement is set separately for each product type and the question does not say which",
                 PROBABLE, ("product_unspecified",), product_choices=product_choices)
    # `resolved_statute`: FD&C Act section(s) the question itself names ("subsection (e) of section 605"): identified
    if not plan.requirements and not plan.docket and not resolved_statute and _AMBIGUOUS_RE.search(question):
        return d(AMBIGUOUS, "the question points at a provision without identifying it (no CFR part or section)",
                 PROBABLE, ("unidentified_provision",))
    # 3. evidence that exists at all
    explicit = coverage is not None and coverage.explicit
    if explicit and not coverage.satisfied_ids:
        return d(INSUFFICIENT_EVIDENCE, "no requested provision was found in the available eCFR data: "
                 + ", ".join(coverage.missing_labels), PROVEN, ("requested_provision_not_found",), missing=list(coverage.missing_labels))
    if no_context or not texts:
        return d(INSUFFICIENT_EVIDENCE, "no source was retrieved for the question", PROVEN, ("no_evidence",))
    # 4. provision identity: a citation lookup whose only named provision is the one being referred to
    targets = reference_targets(plan)
    if targets:
        signals["reference_targets"] = targets
        if analysis.question_type == TYPE_CITATION and len(targets) == len(plan.requirements):
            return d(WRONG_PROVISION, f"{', '.join(targets)} is named as the provision being referred to, not as the one asked "
                     "for; the retrieved evidence is that provision itself, so it cannot show which provision answers",
                     PROBABLE, ("reference_target_only",))
    # 5. a requested kind of value that no supplied source contains
    kind = requested_value_kind(question)
    if kind:
        present = evidence_has_value(kind, texts)
        signals.update(requested_value_kind=kind, value_kind_in_evidence=present)
        if not present:
            label = value_kind_label(kind)
            if value_request_is_whole_question(question, kind):
                return d(INSUFFICIENT_EVIDENCE, f"the question asks for {with_article(label)}, and none of the {len(texts)} retrieved "
                         f"sources contains one", PROVEN, ("requested_value_absent",))
            return d(PARTIALLY_ANSWERABLE, f"one part of the question asks for {with_article(label)} that no retrieved source contains",
                     PROVEN, limitation=f"The retrieved evidence does not establish the {label} asked for.")
    # 6. partial coverage and mixed scope: answer what is supported, say what is not
    if explicit and coverage.missing_ids:
        return d(PARTIALLY_ANSWERABLE, "some requested provisions were not found: " + ", ".join(coverage.missing_labels), PROVEN,
                 missing=list(coverage.missing_labels))
    if analysis.scope_status == SCOPE_MIXED and analysis.out_of_scope_part:
        return d(PARTIALLY_ANSWERABLE, f"part of the question is not covered by the sources: \"{analysis.out_of_scope_part}\"",
                 PROBABLE, limitation=f'The available regulatory sources do not cover "{analysis.out_of_scope_part}".')
    # 7. what remains - FJ1 (29 Oct): a question that names a non-U.S. framework beside FDA ("Health Canada's Medical Devices
    # Regulations require an MDSAP certificate. Does FDA require the same thing?", SD-013) is answered for FDA only, and
    # says so: the other jurisdiction's requirement is neither confirmed nor denied
    foreign = None
    if precision_fix("FJ1") and analysis.scope_status != SCOPE_OUT:
        from regulus.planning.intent import _FOREIGN_RE

        # a jurisdiction or regulator, not an actor or mark (SD-011, PASS: "a certificate issued by a notified body" asks
        # about FDA's QMSR only)
        foreign = next((m.group(0) for m in _FOREIGN_RE.finditer(question or "") if m.group(0).lower() in _FOREIGN_LABEL), None)
    note = (f"The {_FOREIGN_LABEL.get(foreign.lower(), foreign)} requirement is outside Regulus AI's sources (U.S. FDA "
            "regulations and FDA guidance only), so it is neither confirmed nor described here; only the U.S. FDA side is "
            "answered." if foreign else None)
    if not explicit:
        weak = all(getattr(s, "tier", None) == "discovery" for s in sources)
        return d(UNKNOWN, "a general question: whether the retrieved sources answer it cannot be decided by rule"
                 + ("; every source came from a broad search" if weak else ""), UNDETERMINED, limitation=note,
                 weak_evidence=weak, **({"foreign_framework": foreign} if foreign else {}))
    return d(ANSWERABLE, "every requested provision has exact evidence and nothing checkable is missing "
             "(claim-level support is checked after generation by validation)", PROVEN, limitation=note,
             **({"foreign_framework": foreign} if foreign else {}))
