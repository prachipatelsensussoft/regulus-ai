"""Safe fallback answers (Stage 5): what the reader gets when no generated answer can be shown.

One place builds every non-generated answer that the answerability gate or a
failed repair produces, so the rules hold everywhere:

  * never present retrieved text as the answer unless it is the text of the
    provision the question asked about, at paragraph or section level, for a
    question that asks what that provision says (`quoting_allowed`); a part-level
    or topical request, a citation lookup, or a request for a specific value
    gets no quotation, because the first matching passage is not the answer
  * say what the evidence does not establish ("the retrieved evidence does not
    establish ..."), never that the regulation does not contain something:
    retrieval can miss text that exists
  * name the sources that were consulted, as sources, not as an answer

Texts are deterministic and contain no model output.
"""

from __future__ import annotations

import re

from regulus.planning.intent import TYPE_DEFINITION, TYPE_EXPLANATION, TYPE_REQUIREMENT
from regulus.rag import answerability as A
from regulus.rag.answer_quality import out_of_scope_answer
from regulus.rag.evidence_values import requested_value_kind, value_kind_label
from regulus.rag.generation import UNVALIDATED_PREFIX, extractive_answer, verbatim_requested
from regulus.rag.prompt import NO_CONTEXT_ANSWER, incomplete_coverage_answer

# fallback types (recorded in metadata and traces)
FB_OUT_OF_SCOPE = "out_of_scope"
FB_UNSUPPORTED_REQUEST = "unsupported_request"
FB_AMBIGUOUS = "ambiguous"
FB_WRONG_PROVISION = "wrong_provision"
FB_PROVISION_NOT_FOUND = "provision_not_found"
FB_NO_EVIDENCE = "no_evidence"
FB_VALUE_NOT_ESTABLISHED = "value_not_established"
FB_VERBATIM_PROVISION = "verbatim_requested_provision"   # the repair quoted the requested provision (allowed)
FB_UNVALIDATED = "unvalidated_answer_withheld"           # generated answer withheld, nothing quoted
FB_COMPOSED_LISTING = "composed_from_listing"   # a color-additive listing question answered from the listing text (rag.color_listing)
FB_COMPOSED_SILENCE = "composed_practice_silence"   # a practice no cosmetic provision mentions, answered from the Act (rag.practice_silence)
FB_DEFINITIONS_QUOTED = "definitions_quoted"   # a general question the model declined: its terms' CFR definitions, verbatim
FB_PREMISE_NOT_ESTABLISHED = "premise_not_established"  # Stage 7: a yes/no premise the identified provision does not state
FB_COMPOSED_PREMISE = "composed_premise_figure"   # FP1: a confirmation question asserting a figure the provision does not state
FB_COMPOSED_RECORDS = "composed_records_silence"   # RS1: a records question about a section that requires none
FB_COMPOSED_QMSR = "composed_qmsr"   # QM1: a former part 820 section or a non-existent ISO 13485 clause (regulus.rag.qmsr)
FB_COMPOSED_LIST = "composed_list"   # LS1: a "what must it contain" question answered with the regulation's list (rag.list_answers)
FB_COMPOSED_WARNING = "composed_warning_statement"   # WC1: a Part 740 label statement quoted verbatim (rag.warning_statements)
FB_COMPOSED_DEFINITIONS = "composed_definitions"   # D1: the supplied paragraphs defining the terms asked about
FB_FOCUSED_QUOTE = "focused_provision_quote"   # Q1: no verified answer; the governing provision's relevant paragraphs, verbatim
FB_PRODUCT_CLARIFICATION = "product_clarification"   # AMB1: which product type? the parallel provisions, cited (rag.product_clarification)
FALLBACK_TYPES = (FB_OUT_OF_SCOPE, FB_UNSUPPORTED_REQUEST, FB_AMBIGUOUS, FB_WRONG_PROVISION, FB_PROVISION_NOT_FOUND,
                  FB_NO_EVIDENCE, FB_VALUE_NOT_ESTABLISHED, FB_VERBATIM_PROVISION, FB_UNVALIDATED, FB_PREMISE_NOT_ESTABLISHED,
                  FB_PRODUCT_CLARIFICATION)

QUOTABLE_QUESTION_TYPES = (TYPE_REQUIREMENT, TYPE_EXPLANATION, TYPE_DEFINITION)


def _join(labels) -> str:
    labels = list(dict.fromkeys(labels))
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def consulted(sources, limit: int = 5) -> str:
    """'Sources consulted: A, B.' - labels only, never their text, and no [n] markers
    (a fallback answer has no citations for them to point at)."""
    seen = list(dict.fromkeys(s.result.citation.label for s in sources))
    if not seen:
        return ""
    items = seen[:limit]
    more = len(seen) - len(items)
    return "Sources consulted: " + ", ".join(items) + (f" and {more} more" if more > 0 else "") + "."


def quoting_allowed(question: str, plan, shape, analysis, package) -> bool:
    """May the verbatim text of the requested provisions stand in for the answer?

    Only when that text IS what was asked for: every requested provision is a
    section or paragraph (a part is too broad; its first matching passage is
    arbitrary), the question asks what the provision says (not a citation
    lookup, applicability or procedure question), it is not restricted to a
    topic inside the provision, and it does not ask for a specific value."""
    if package is None or not package.satisfied_ids or not plan.requirements:
        return False
    if any(r.reference.section is None for r in plan.requirements):
        return False
    if shape is not None and shape.focus_terms:
        return False
    if analysis is None or analysis.question_type not in QUOTABLE_QUESTION_TYPES:
        return False
    return requested_value_kind(question) is None


def provision_is_identified(plan, package) -> bool:
    """Every requested provision is a section or paragraph and has exact evidence: its text is
    relevant by identity (it is the provision asked about), unlike a part's first matching passage.
    Sections OC1 moved to context are no longer requested (30 Oct: full300d, 10 answers that had quoted the provision
    were withheld because satisfied_ids lacked the moved companions' ids)."""
    if package is None:
        return False
    context = set(getattr(package, "context_ids", ()))
    requested = [r for r in plan.requirements if r.id not in context]
    return (bool(requested) and all(r.reference.section is not None for r in requested)
            and set(package.satisfied_ids) == {r.id for r in requested})


def blocked_answer(decision: A.AnswerabilityDecision, sources, notes=(), missing_labels=(), *, plan=None,
                   package=None) -> tuple[str, str]:
    """(answer text, fallback type) for a decision the gate blocked."""
    state, conditions = decision.state, decision.blocking_conditions
    listed = consulted(sources)
    boundary = decision.signals.get("boundary")
    if boundary:
        # BR1 (regulus.rag.boundary): what the tool cannot do, why, and what it can do instead
        from regulus.rag.boundary import answer as boundary_answer

        return (boundary_answer(boundary["kind"], boundary.get("detail", ""), boundary.get("question", "")),
                FB_OUT_OF_SCOPE if state == A.OUT_OF_SCOPE else FB_NO_EVIDENCE if state == A.INSUFFICIENT_EVIDENCE
                else FB_UNSUPPORTED_REQUEST)
    if "product_unspecified" in conditions:
        # AMB1: which product type the question is about (regulus.rag.product_clarification)
        from regulus.rag.product_clarification import Choice, answer as clarification

        found = {k: tuple(Choice(*c) for c in v) for k, v in (decision.signals.get("product_choices") or {}).items()}
        return clarification(found), FB_PRODUCT_CLARIFICATION
    if state == A.OUT_OF_SCOPE and "general_knowledge" in conditions:
        return ("This does not appear to be a regulatory question. Regulus AI answers questions about U.S. FDA regulations "
                "(21 CFR) and FDA guidance from verified sources, and none of them addresses this question, so no answer is "
                "given."), FB_OUT_OF_SCOPE
    if state == A.OUT_OF_SCOPE:
        return out_of_scope_answer(decision.reason.removeprefix("the question ")), FB_OUT_OF_SCOPE
    if state == A.UNSUPPORTED_REQUEST:
        return ("Regulus AI cannot make this determination. It answers what U.S. FDA regulations (21 CFR) and FDA guidance "
                "say; whether a particular product, firm or submission complies, or what FDA will decide, depends on facts "
                "the regulatory text does not contain. Ask what the requirements are, and apply them with qualified "
                "regulatory advice."), FB_UNSUPPORTED_REQUEST
    if state == A.AMBIGUOUS:
        return ("The question refers to a provision without identifying it (for example a paragraph with no CFR part or "
                "section), so the evidence cannot be matched to it. Please name the provision, e.g. \"21 CFR 820.10(b)\"."), FB_AMBIGUOUS
    if state == A.WRONG_PROVISION:
        targets = _join(decision.signals.get("reference_targets") or ())
        return (f"The question names {targets} as the provision being referred to, not the one it asks for. The retrieved "
                f"evidence is the text of {targets} itself, which does not establish which provision refers to it, so no "
                f"provision is cited. {listed}").strip(), FB_WRONG_PROVISION
    if "requested_provision_not_found" in conditions:
        return incomplete_coverage_answer(missing_labels), FB_PROVISION_NOT_FOUND
    if "no_evidence" in conditions:
        return (NO_CONTEXT_ANSWER + (" " + " ".join(notes) if notes else "")), FB_NO_EVIDENCE
    if "requested_value_absent" in conditions:
        label = value_kind_label(decision.signals.get("requested_value_kind"))
        head = (f"The retrieved evidence does not establish the {label} asked for: none of the retrieved sources states "
                f"one, so no {label} is given.")
        if plan is not None and provision_is_identified(plan, package):
            # the question named the provision: what it does say is relevant, and is quoted as such
            return (f"{head} What the requested provision says, verbatim:\n{extractive_answer(package)}"), FB_VALUE_NOT_ESTABLISHED
        return (f"{head} The regulation may address the topic in general terms without stating one, or the "
                f"relevant text may not have been retrieved. {listed}").strip(), FB_VALUE_NOT_ESTABLISHED
    return f"The retrieved evidence does not establish an answer to this question. {listed}".strip(), FB_NO_EVIDENCE


def unvalidated_answer(package, sources, *, quote: bool, requirement_ids=None) -> tuple[str, str]:
    """(answer text, fallback type) when a generated answer failed validation and cannot be shown.
    With `quote` the verbatim text of the requested provisions is given, which is grounded by construction."""
    if quote:
        if requirement_ids is None:
            return f"{UNVALIDATED_PREFIX}\n{verbatim_requested(package)}", FB_VERBATIM_PROVISION
        return f"{UNVALIDATED_PREFIX}\n{extractive_answer(package, requirement_ids)}", FB_VERBATIM_PROVISION
    return ("The generated answer could not be validated against the retrieved regulatory evidence, so it is not shown. "
            "The retrieved sources were not verified to answer the question as asked, so their text is not quoted as an "
            f"answer. {consulted(sources)}").strip(), FB_UNVALIDATED


def premise_not_established(package, terms) -> tuple[str, str]:
    """Stage 7: a yes/no answer affirmed terms that the identified provision's text does not contain. The
    grounded statement is that the retrieved text does not state them; the provision is quoted verbatim.
    Only for identified provisions (the caller checks `provision_is_identified`)."""
    listed = ", ".join(f'"{t}"' for t in terms)
    return (f"The retrieved text of the requested provision does not state {listed}. What it says, verbatim:\n"
            f"{extractive_answer(package)}"), FB_PREMISE_NOT_ESTABLISHED


# Q1 (26 Oct 2026, regulus.rag.validator_switches): when no generated answer can be shown - withheld, or the model
# declined although the provision the question is about was supplied - the reader gets that provision's paragraphs that
# share most subject words with the question, verbatim and cited, instead of a bare "not shown". Only for a provision
# identified by the question, by name or by section resolution (the caller decides); never an application of it.
FOCUSED_QUOTE_LEAD = "What the provision this question concerns says on this point, verbatim:"
# Q2: the provision was chosen by the picker among candidates, which can be wrong: the lead does not claim it is the
# provision the question concerns
FOCUSED_QUOTE_LEAD_PICKED = "The provision in this system that most closely matches the question says, verbatim:"
FOCUSED_QUOTE_TAIL = ("This is the provision's own text, not an application of it to your facts: a fuller answer could not "
                      "be verified sentence by sentence against the regulation text.")
FOCUSED_QUOTE_PARAGRAPHS = 3
FOCUSED_QUOTE_CHARS = 1600
# FQ1 (28 Oct, off until measured): the quote takes more of the requested provision. 300Q 27 Oct: 27 focused quotes,
# 0 PASS or PARTIALLY PASS - 22 scored "correct verbatim but omits (b)-(g)" (G-001: 201.1(a), (h)(1), (j) of 201.1)
FQ1_PARAGRAPHS = 40   # the character budget binds, not the count: 111.15 is 39 short paragraphs (S-006 got 8 = 1,928 chars)
FQ1_CHARS = 6500


# LQ1 (2 Nov 2026, off until measured): the quote kept a list's lead-in and dropped its items - 4.4(a)(1) "...; or"
# without (a)(2) (X-002), 4.102(c)(1) "you must submit:" without the five-day and malfunction reports (X-004); 14 of the
# 18 focused quotes of 29 Oct's 300Q end a quoted paragraph on ":" or "; or"/"; and" without its list
_LIST_LEAD_IN = re.compile(r":\s*$")
_LIST_ITEM_JOINED = re.compile(r"[;,]\s*(?:or|and)\s*$")
_PARAGRAPH = re.compile(r"21 CFR (\d+\.\d+[a-z]?)((?:\([^)\s]+\))*)$")


def list_members(bodies) -> list[set[int]]:
    """For each (source, text) of `bodies`, the indices it must be quoted with: a paragraph ending with ":" brings its
    supplied children, one ending with "; or" / "; and" the rest of its list (its parent's lead-in and children)."""
    keys = []
    for s, _ in bodies:
        m = _PARAGRAPH.match(s.citation or "") if s.corpus != "usc" else None
        keys.append((m.group(1), tuple(re.findall(r"\(([^)]+)\)", m.group(2)))) if m else None)
    at = {k: i for i, k in enumerate(keys) if k}

    def children(section, path):
        return [i for i, k in enumerate(keys) if k and k[0] == section and len(k[1]) == len(path) + 1
                and k[1][:len(path)] == path]

    def close(i, seen):
        if i in seen:
            return
        seen.add(i)
        k, text = keys[i], bodies[i][1]
        if k is None or not k[1]:
            return
        if _LIST_LEAD_IN.search(text):
            for c in children(*k):
                close(c, seen)
        elif _LIST_ITEM_JOINED.search(text):
            parent = at.get((k[0], k[1][:-1]))
            if parent is not None and _LIST_LEAD_IN.search(bodies[parent][1]):
                close(parent, seen)
            for c in children(k[0], k[1][:-1]):
                close(c, seen)

    out = []
    for i in range(len(bodies)):
        seen: set[int] = set()
        close(i, seen)
        out.append(seen)
    return out


def focused_quote(package, question: str, *, limit: int = FOCUSED_QUOTE_PARAGRAPHS,
                  max_chars: int = FOCUSED_QUOTE_CHARS, lead: str = FOCUSED_QUOTE_LEAD) -> str | None:
    """The requested provisions' EXACT paragraphs (and promoted FD&C Act text) that share most subject words with the
    question, at most `limit`, in document order, each cited and quoted whole; None when no paragraph shares one."""
    from regulus.rag.conversation import subject_words
    from regulus.rag.generation import _verbatim, whole_sentences
    from regulus.retrieval.evidence import TIER_EXACT, TIER_SUPPORTING

    if package is None:
        return None
    from regulus.rag.quote_selection import rank
    from regulus.rag.validator_switches import enabled

    if enabled("FQ1"):
        limit, max_chars = max(limit, FQ1_PARAGRAPHS), max(max_chars, FQ1_CHARS)

    asked = subject_words(question)
    pool = [s for s in package.sources if not s.is_docket and (
        (s.tier == TIER_EXACT and s.requirement_ids) or (s.tier == TIER_SUPPORTING and s.corpus == "usc"))]
    bodies = [(s, whole_sentences(s.text or "")) for s in pool]
    bodies = [(s, b) for s, b in bodies if b]
    if not bodies or not any(asked & subject_words(b) for _, b in bodies):
        return None   # nothing supplied speaks to the question at all
    lists = list_members(bodies) if enabled("LQ1") else None
    kept, used, taken = [], 0, set()
    for i in rank(question, [(s.citation, b) for s, b in bodies]):
        if i in taken:
            continue
        unit = sorted(lists[i] - taken) if lists else [i]
        cost = sum(len(bodies[j][1]) for j in unit)
        if len(unit) > 1 and (len(kept) + len(unit) > limit or used + cost > max_chars):
            if kept:
                continue      # LQ1: a list is quoted whole or not at all, never its lead-in alone
            unit, cost = [i], len(bodies[i][1])
        if len(kept) >= limit or (kept and used + cost > max_chars):
            break
        kept += [bodies[j] for j in unit]
        taken.update(unit)
        used += cost
    lines = [lead]
    for s, body in sorted(kept, key=lambda x: x[0].index):
        kind = "FD&C Act, verbatim" if s.corpus == "usc" else "eCFR, binding, verbatim"
        lines.append(f"{s.citation} ({kind}): {_verbatim(body, s.index)}")
    lines.append(FOCUSED_QUOTE_TAIL)
    return "\n".join(lines)
