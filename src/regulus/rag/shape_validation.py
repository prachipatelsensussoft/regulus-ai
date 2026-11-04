"""Answer-shape checks (Stage 12), run by validate_answer() when the package
carries an AnswerPlan.

These check whether the answer has the shape the question asked for and
whether synthesis lost anything on the way: a qualifier, an applicability
limit, a requirement group, a citation's support. They are deterministic and
read only the answer, the package and its plan.

Nothing here measures or penalises length. A long answer to an exhaustive
question is correct; a short one that drops an item is not. Every finding is
a warning except `enumeration_incomplete`, which is raised only when the user
explicitly asked for every item and one is missing, and
`material_group_uncovered`, raised when a synthesis leaves out a whole
requirement group. Both map to the existing `incomplete` status and repair
appends the missing text verbatim. `prompt_echo` (the answer copies the
prompt's instructions) maps to `degenerate`. `applicability_dropped` is an
error mapped to `unsupported` (16 Oct 2026 severity experiment: as a warning
it fired only on delivered answers scored FAIL).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from regulus.evaluation.grounding_eval import split_sentences
from regulus.planning.shape import MODE_COMPARISON, MODE_DIRECT, MODE_ENUMERATION, MODE_FOCUSED, MODE_SYNTHESIS
from regulus.rag.answer_plan import (
    CONDITIONAL_QUALIFIERS,
    GROUP_DOCUMENT,
    ROLE_LEAD_IN,
    ROLE_REQUIREMENT,
    ROLE_SUBORDINATE,
    AnswerPlan,
    extract_cross_references,
)
from regulus.retrieval.references import parse_cfr_references

EVIDENCE_BY_EVIDENCE = "evidence_by_evidence"
SCOPE_EXPANSION = "scope_expansion"
ENUMERATION_INCOMPLETE = "enumeration_incomplete"
MATERIAL_GROUP_UNCOVERED = "material_group_uncovered"
QUALIFIER_DROPPED = "qualifier_dropped"
APPLICABILITY_DROPPED = "applicability_dropped"
SYNTHESIS_CITATION_GAP = "synthesis_citation_gap"
REPEATED_STATEMENT = "repeated_statement"
PROMPT_ECHO = "prompt_echo"
QUALIFIER_MISAPPLIED = "qualifier_misapplied"
SHAPE_CODES = (PROMPT_ECHO, QUALIFIER_MISAPPLIED, EVIDENCE_BY_EVIDENCE, SCOPE_EXPANSION, ENUMERATION_INCOMPLETE, MATERIAL_GROUP_UNCOVERED,
               QUALIFIER_DROPPED, APPLICABILITY_DROPPED, SYNTHESIS_CITATION_GAP, REPEATED_STATEMENT)

# wording in an answer that keeps a source's qualifier (phrases only: a bare
# "applicable" or "appropriate" also occurs in unconditional sentences)
KEEPS_QUALIFIER: dict[str, tuple[str, ...]] = {
    "if applicable": ("if applicable", "where applicable", "when applicable", "as applicable", "to the extent applicable",
                      "if it applies", "where it applies", "if they apply", "where they apply", "if relevant", "where relevant"),
    "as appropriate": ("as appropriate", "where appropriate", "when appropriate", "if appropriate", "to the extent appropriate"),
    "to the extent": ("to the extent",),
    "unless": ("unless", "except"),
    "except": ("except", "unless", "exempt", "other than", "excluding"),
}
KEEPS_QUALIFIER["where applicable"] = KEEPS_QUALIFIER["as applicable"] = KEEPS_QUALIFIER["if applicable"]
KEEPS_QUALIFIER["where appropriate"] = KEEPS_QUALIFIER["as appropriate"]

MARK_RE = re.compile(r"(?<![A-Za-z0-9])\[(\d{1,2})\]")
_OBLIGATION_RE = re.compile(r"\b(must|shall|required?|requires|needs? to|has to|have to|obligat\w*|mandat\w*)\b", re.I)
_NEGATION_RE = re.compile(r"\b(not found|does not|do not|cannot|no exact|not among|not available|not provided)\b", re.I)
_LEAD_LABEL = re.compile(
    r"^\s*(?:[-*•]\s*|\d+[.)]\s+)?(?:\*\*)?\s*(?:21\s*C\.?F\.?R\.?\s*)?(?:§+\s*)?(?:\d{3,4}\.\d+)?\s*((?:\((?:[a-z0-9]{1,4}|[ivxl]{5,8})\))+)", re.I)
_SHORT_REF_RE = re.compile(r"(?<![\w)\]])((?:\([a-z]\))(?:\(\d{1,2}\))*)(?!\s*\()")
_WORD_RE = re.compile(r"[a-z][a-z'-]{2,}")
_STOP = {
    "the", "and", "for", "with", "that", "this", "must", "shall", "are", "was", "were", "have", "has", "not", "its", "any",
    "all", "such", "from", "into", "under", "per", "requires", "required", "require", "requirement", "requirements",
    "cfr", "section", "paragraph", "provision", "chapter", "part", "accordance", "applicable", "manufacturer",
    "manufacturers", "device", "devices", "which", "when", "where", "then", "than", "also", "ecfr", "binding",
    "regulation", "other", "including", "limited", "following", "listed", "those", "these", "their", "comply", "complies",
    "compliance", "iso", "clause", "clauses", "subclauses", "document", "documents", "system", "systems",
}
# prompt scaffolding that must never appear in an answer (the model copying its instructions)
_ECHO_RE = re.compile(
    r"\bSOURCES? \d{1,2}\b|\bGROUP [A-Z]\b|\bSIDE \d\b|\bTier: (?:EXACT|STRUCTURAL|SUPPORTING|DISCOVERY)\b"
    r"|\bANSWER (?:FORMAT|PLAN|SHAPE)\b|^\s*[-*]?\s*(?:Lead-in|Context only|Related material)\s*:|\bREQUESTED PROVISIONS\b"
    r"|do not repeat these instructions"
    # the REQUESTED PROVISIONS label of a general question, copied as the "answer" ("None (general question) [1]")
    r"|(?i:\bnone \(general question\))",
    re.MULTILINE)
NEAR_DUPLICATE = 0.8
# repetition compares ordinary wording, so only function words are ignored
_FUNCTION = {"the", "and", "for", "with", "that", "this", "are", "was", "were", "has", "have", "its", "any", "from",
             "into", "under", "per", "which", "when", "where", "then", "than", "also", "such", "their", "these", "those"}
ITEM_WORD_COVERAGE = 0.5


@dataclass(frozen=True)
class ShapeFinding:
    code: str
    severity: str
    detail: str
    citation: int | None = None
    sentence: str | None = None


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(MARK_RE.sub("", (text or "").lower()))) - _STOP


def _keeps(text: str, qualifier: str) -> bool:
    low = text.lower()
    return any(p in low for p in KEEPS_QUALIFIER.get(qualifier, (qualifier,)))


def _anchor_patterns(anchor: str) -> list[re.Pattern[str]]:
    """Identity patterns for a cross-reference: the number(s) it names, in their own form."""
    a = anchor.lower()
    if a.startswith("part"):
        return [re.compile(rf"\bparts?\s+{re.escape(n)}\b", re.I) for n in re.findall(r"\d{1,4}", a)]
    if a.startswith("clause"):
        return [re.compile(rf"(?<![\d.]){re.escape(n)}(?![\d]|\.\d)") for n in re.findall(r"\d+(?:\.\d+)+", a)]
    if a.startswith("section"):
        return [re.compile(re.escape(a.split(None, 1)[1]), re.I)]
    n = re.search(r"\d{1,4}\.\d{1,4}[a-z]?", a)
    return [re.compile(rf"(?<![\d.]){re.escape(n.group(0))}(?![\d])")] if n else []


_LABEL_RE = re.compile(r"(?:\d{3,4}\.\d+)?(?:\((?:[a-z0-9]{1,4}|[ivxl]{5,8})\))+|\d{3,4}\.\d+", re.I)


def _is_label_line(line: str) -> bool:
    """A line that holds only a provision label and its marker: "21 CFR 820.10(b)(3) [6]"."""
    bare = MARK_RE.sub("", line).strip(" -*\u2022:")
    return bool(MARK_RE.search(line)) and bool(_LABEL_RE.search(bare)) and len(bare.split()) <= 6 and not bare.endswith(".")


def _blocks(answer: str) -> list[tuple[str, list[str]]]:
    """(block, its sentences) for every non-empty line: a sentence's paragraph is its line.
    A label-only line heads the line after it, and its marker is moved to the end
    of that text, so the marker is checked against the text it introduces."""
    merged: list[str] = []
    label = ""
    for line in (ln.strip() for ln in re.split(r"\n+", answer) if ln.strip()):
        if _is_label_line(line):
            if label:
                merged.append(label)
            label = line
            continue
        if label:
            marks = "".join(f"[{m}]" for m in MARK_RE.findall(label))
            head = MARK_RE.sub("", label).strip()
            line = f"{head} {line.rstrip(' .;:')} {marks}." if not MARK_RE.search(line) else f"{head} {line}"
            label = ""
        merged.append(line)
    if label:
        merged.append(label)
    return [(b, split_sentences(b)) for b in merged]


def _related(a: tuple[str | None, str], b: tuple[str | None, str]) -> bool:
    if a[0] != b[0]:
        return False
    x, y = a[1], b[1]
    return x == y or not x or not y or x.startswith(y + "(") or y.startswith(x + "(")


def check_shape(answer: str, package, plan: AnswerPlan) -> tuple[list[ShapeFinding], dict[str, Any]]:
    """Shape findings for an answer (markers already attached to their sentences)."""
    text = answer or ""
    low = text.lower()
    findings: list[ShapeFinding] = []
    by_index = {s.index: s for s in package.sources}
    cited = {int(m) for m in MARK_RE.findall(text) if int(m) in by_index}
    blocks = _blocks(text)
    sentences = [s for _, ss in blocks for s in ss]
    mode = plan.mode
    sections = {s.section for s in plan.sources if s.section}
    single_section = len(sections) == 1
    answer_words = _words(text)

    # -- which evidence items does the answer represent?
    def label_named(i: int) -> bool:
        s = plan.source(i)
        if s is None or not s.section or not s.paragraph:
            return False
        tokens = re.findall(r"\(([a-z0-9]+)\)", s.paragraph, re.I)
        body = "".join(rf"\s*\(\s*{re.escape(t)}\s*\)" for t in tokens)
        if re.search(rf"(?<![\d.]){re.escape(s.section)}{body}(?!\s*\()", text, re.I):
            return True
        return single_section and bool(re.search(rf"(?<![\w)]){body.lstrip()}(?!\s*\()", text, re.I))

    def covered(i: int) -> bool:
        if i in cited or label_named(i):
            return True
        s, src = plan.source(i), by_index.get(i)
        if s is None or src is None:
            return False
        for ref in s.cross_references:
            pats = _anchor_patterns(ref)
            if pats and all(p.search(text) for p in pats):
                return True
        words = _words(src.text)
        return bool(words) and len(words & answer_words) / len(words) >= ITEM_WORD_COVERAGE

    requested_heads = {g.head for g in plan.groups if g.head is not None and g.key in dict(package.requirements).values()}
    group_status = []
    for g in plan.groups:
        members_covered = [i for i in g.members if covered(i)]
        group_status.append({"key": g.key, "label": g.label, "members": list(g.members), "covered": members_covered})

    # -- 0. the answer reproduces prompt scaffolding instead of answering
    echo = _ECHO_RE.findall(text)
    if echo:
        findings.append(ShapeFinding(PROMPT_ECHO, "error",
                                     f"the answer reproduces prompt instructions ({', '.join(dict.fromkeys(e.strip() for e in echo if e.strip()))}) "
                                     "instead of answering"))

    # -- 1. synthesis failure: one block per evidence row
    led: list[str] = []
    for block, ss in blocks:
        for piece in [block] + ss:
            m = _LEAD_LABEL.match(piece)
            if m and m.group(1).lower() not in led:
                led.append(m.group(1).lower())
    material_rows = sum(1 for s in plan.sources if s.role != ROLE_LEAD_IN and s.section)
    if mode in (MODE_SYNTHESIS, MODE_DIRECT) and len(led) >= 4 and len(led) > len(plan.groups):
        findings.append(ShapeFinding(EVIDENCE_BY_EVIDENCE, "warning",
                                     f"{len(led)} blocks each open with their own provision label for {len(plan.groups)} requirement "
                                     f"groups ({material_rows} evidence rows): the evidence is reproduced row by row, not synthesized"))

    # -- 2. scope: a same-section provision that was not requested (nor a parent/child of one) is described
    scope_refs: list[str] = []
    if mode in (MODE_FOCUSED, MODE_COMPARISON) and not plan.shape.focus_terms:
        requested = [(r_sec, r_para) for r_sec, r_para in (
            (s.section, s.paragraph) for s in plan.sources if s.index in requested_heads)]
        requested += [(ref.section, ref.paragraph_path) for _, label in package.requirements for ref in parse_cfr_references(label)]
        req_sections = {r[0] for r in requested if r[0]}
        for sentence in sentences:
            if _NEGATION_RE.search(sentence):
                continue
            named = [(ref.section, ref.paragraph_path, ref.canonical) for ref in parse_cfr_references(sentence) if ref.section in req_sections]
            if len(req_sections) == 1:
                sec = next(iter(req_sections))
                named += [(sec, m.group(1).lower(), f"21 CFR {sec}{m.group(1).lower()}") for m in _SHORT_REF_RE.finditer(sentence)]
            for sec, para, canon in named:
                if para and not any(_related((sec, para), r) for r in requested) and canon not in scope_refs:
                    scope_refs.append(canon)
                    findings.append(ShapeFinding(SCOPE_EXPANSION, "warning",
                                                 f"the answer describes {canon}, which the question did not ask about", sentence=sentence))

    # -- 3. enumeration: every material item of an explicitly exhaustive request
    uncovered_items: list[int] = []
    if mode == MODE_ENUMERATION and plan.shape.exhaustive:
        for i in plan.material_items():
            if i in requested_heads:
                continue  # a requested provision: the requirement check already covers it
            if not covered(i):
                uncovered_items.append(i)
                findings.append(ShapeFinding(ENUMERATION_INCOMPLETE, "error",
                                             f"{by_index[i].citation} is part of what was explicitly requested but the answer does not represent it",
                                             citation=i))

    # -- 4. synthesis: a whole requirement group is missing - a completeness error
    #       repaired by appending the group verbatim (not for a direct
    #       answer, which says what something is rather than covering all of it)
    uncovered_groups: list[str] = []
    if mode == MODE_SYNTHESIS:
        for g, st in zip(plan.groups, group_status):
            if g.kind == GROUP_DOCUMENT or not g.requirement_ids:
                continue
            if not st["covered"]:
                uncovered_groups.append(g.label)
                findings.append(ShapeFinding(MATERIAL_GROUP_UNCOVERED, "error",
                                             f"{g.label}{' (' + g.heading + ')' if g.heading else ''} is not represented in the answer",
                                             citation=g.head))

    # -- 5/6. qualifiers and applicability kept with the statements that rely on them
    flagged: set[tuple[int, str]] = set()
    for block, ss in blocks:
        for sentence in ss:
            marks = [int(m) for m in MARK_RE.findall(sentence) if int(m) in by_index]
            if not marks or _NEGATION_RE.search(sentence):
                continue
            for i in marks:
                s = plan.source(i)
                if s is None:
                    continue
                for q in s.qualifiers:
                    if q in CONDITIONAL_QUALIFIERS and (i, q) not in flagged and _OBLIGATION_RE.search(sentence) \
                            and not _keeps(sentence, q) and not _keeps(block, q):
                        flagged.add((i, q))
                        findings.append(ShapeFinding(QUALIFIER_DROPPED, "warning",
                                                     f'SOURCE {i} says "{q}" but the statement citing it does not keep that condition',
                                                     citation=i, sentence=sentence))
                if s.applicability and s.role in (ROLE_REQUIREMENT, ROLE_SUBORDINATE) and (i, "applicability") not in flagged:
                    limit = _words(s.applicability) - {"class", "those", "listed", "table", "paragraph"}
                    if limit and not (limit & (_words(sentence) | _words(block))):
                        flagged.add((i, "applicability"))
                        findings.append(ShapeFinding(APPLICABILITY_DROPPED, "error",
                                                     f"SOURCE {i} applies only to {s.applicability[:120]}, which the statement citing it does not say",
                                                     citation=i, sentence=sentence))
    # a qualifier-bearing item the answer represents without citing it, whose qualifier appears nowhere
    for i in sorted(by_index):
        s = plan.source(i)
        if s is None or i in cited or not covered(i):
            continue
        for q in s.qualifiers:
            if q in CONDITIONAL_QUALIFIERS and (i, q) not in flagged and not _keeps(text, q):
                flagged.add((i, q))
                findings.append(ShapeFinding(QUALIFIER_DROPPED, "warning",
                                             f'the answer states the content of SOURCE {i} without its condition "{q}"', citation=i))

    # -- 6b. a condition attached to a requirement that does not carry it ("if applicable"
    #        on the part 830 obligation when only the part 821 one says it)
    for clause in re.split(r"(?<=[.;])\s+|,\s+and\s+|;\s*", text):
        low_clause = clause.lower()
        for q in ("if applicable", "as appropriate"):
            if not _keeps(clause, q):
                continue
            for anchor in extract_cross_references(MARK_RE.sub("", clause)):
                pats = _anchor_patterns(anchor)
                holders = [s for s in package.sources if pats and all(p.search(s.text or "") for p in pats)]
                if holders and not any(any(k in (h.text or "").lower() for k in KEEPS_QUALIFIER[q]) for h in holders):
                    findings.append(ShapeFinding(QUALIFIER_MISAPPLIED, "warning",
                                                 f'"{q}" is attached to {anchor}, but the source for {anchor} '
                                                 f"(SOURCE {holders[0].index}) carries no such condition", citation=holders[0].index,
                                                 sentence=clause.strip()))
                    break

    # -- 7. synthesis citation gap: a named cross-reference the cited sources do not contain
    gaps = 0
    for sentence in sentences:
        marks = [int(m) for m in MARK_RE.findall(sentence) if int(m) in by_index]
        if not marks:
            continue
        for anchor in extract_cross_references(MARK_RE.sub("", sentence)):
            pats = _anchor_patterns(anchor)
            if not pats:
                continue
            if any(all(p.search(by_index[i].text or "") for p in pats) for i in marks):
                continue
            holders = [s.index for s in package.sources if all(p.search(s.text or "") for p in pats)]
            if holders:
                gaps += 1
                findings.append(ShapeFinding(SYNTHESIS_CITATION_GAP, "warning",
                                             f"{anchor} comes from SOURCE {', '.join(map(str, holders))}, which the statement does not cite "
                                             f"(it cites {', '.join(map(str, marks))})", citation=holders[0], sentence=sentence))

    # -- 8. repetition: materially identical statements
    pairs = []
    def plain(s: str) -> set[str]:
        words = set(_WORD_RE.findall(MARK_RE.sub("", s.lower()))) - _FUNCTION
        return {w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w for w in words}

    long_sentences = [s for s in sentences if len(plain(s)) >= 6]
    for a in range(len(long_sentences)):
        for b in range(a + 1, len(long_sentences)):
            x, y = plain(long_sentences[a]), plain(long_sentences[b])
            if len(x & y) / len(x | y) >= NEAR_DUPLICATE:
                pairs.append((long_sentences[a], long_sentences[b]))
    for x, y in pairs:
        findings.append(ShapeFinding(REPEATED_STATEMENT, "warning", f"near-duplicate of an earlier statement: {x[:80]!r}", sentence=y))

    diagnostics = {
        "mode": mode,
        "shape": plan.shape.to_dict(),
        "groups": len(plan.groups),
        "groups_covered": sum(1 for st in group_status if st["covered"]),
        "uncovered_groups": uncovered_groups,
        "material_items": len(plan.material_items()),
        "material_items_covered": sum(1 for i in plan.material_items() if covered(i)),
        "uncovered_items": uncovered_items,
        "provision_led_blocks": led,
        "out_of_scope_named": scope_refs,
        "citation_gaps": gaps,
        "repeated_pairs": len(pairs),
        "prompt_echo": bool(echo),
        "codes": list(dict.fromkeys(f.code for f in findings)),
    }
    return findings, diagnostics
