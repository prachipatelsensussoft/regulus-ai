"""EQ1 (5 Nov 2026): a question asking which exemptions a regulation has - or who and what it covers and exempts - is
answered by quoting the sections that set them out, verbatim.

The 4B model paraphrases exemptions badly: "exempt from all of Part 117" for "subparts C and G ... do not apply",
conditions added, scope widened. The validator's exception check (condition_not_in_cited_source) stops those drafts -
two blind labellers found 21 of its "exempt" alarms genuine against 9 false (5 Nov calibration, 92 sentences) - and
every attempt to relax it let wrong answers through (EX2 on 3 Nov and again on 5 Nov: R-004, G-008). So the answer is
not the model's: the frame sections (Applicability, Exemptions, "Which farms are subject / eligible", "What produce is
not covered" - SectionPromoter.frame_companions brings them, as AE1) are quoted paragraph by paragraph, each cited,
with leads that make no claim of their own. 300Q: F-001 ("Who is subject to Part 117 and what are the principal
exemptions?", FAIL, draft withheld), Y-001 ("Which farms and produce are covered ... and what exemptions apply?",
FAIL, withheld), R-006.
"""
from __future__ import annotations

import re

COMPOSED_TYPE = "composed_exemptions"
NOTE = "exemption question"
LEAD = "The provisions that set out what these requirements cover and the exemptions from them, each verbatim:"

# a question asking for the exemptions as such (not how to obtain, request or lose one, and not a name that contains
# the word: "Exemption from SE request", "Investigational Device Exemption")
_ASKS = re.compile(
    r"\b(?:what|which)\s+(?:are\s+the\s+)?(?:principal\s+|main\s+|key\s+|available\s+)?exemptions\b"
    r"|\bwhat\s+exemptions?\s+(?:apply|are\s+(?:available|provided))\b"
    r"|\b(?:who|what|which)\b[^?]{0,60}?\b(?:is|are)\s+(?:exempt(?:ed)?|excluded|not\s+covered)\b"
    r"|\b(?:who|what|which)\b[^?]{0,60}?\b(?:is|are)\s+covered\b[^?]*\bexempt", re.I)
_NOT = re.compile(r"\b(?:withdraw\w*|obtain\w*|request\w*|petition\w*|apply\s+for|qualif\w+\s+for)\b[^?]{0,40}\bexempt"
                  r"|\bexemption\s+from\s+(?:SE|substantial)\b|\binvestigational\s+device\s+exemption\b"
                  # one named provision or a defined term: the answer is in that text, not in the regulation's frame
                  # ("Which records are excluded from 610(a)?", "What products are excluded from 'suntanning preparations'?")
                  r"|(?<![\d.])\d{1,4}\.\d{1,4}(?![\d])|\b\d{1,4}\([a-z0-9]{1,4}\)|§\s*\d|\bsection\s+\d"
                  r"|\bdefinition\b|\bexclu\w*\s+from\s+(?:the\s+)?(?:term\s+)?['\"\u2018\u201c]", re.I)
# a frame section that states an exemption or what is not covered (at least one must be supplied)
_EXEMPTING = re.compile(r"\bexempt|\bexclu|\bnot\s+covered\b|\beligible\b", re.I)
_SECTION = re.compile(r"21 CFR (\d+\.\d+[a-z]?)")


def asks_exemptions(question: str) -> bool:
    q = question or ""
    return bool(_ASKS.search(q)) and not _NOT.search(q)


def compose(package, question: str) -> str | None:
    """The EQ1 answer: every supplied frame section, all its supplied passages in document order, verbatim and cited;
    None unless the question asks for exemptions and a section stating one is supplied. No cap of its own: the evidence
    budget bounds the text, and a cap cut F-001's 117.5 at (g)(1), before the on-farm exemptions the question asks for
    (blind, 5 Nov). A passage that only introduces a list ("... the following:") whose items were not supplied is left
    out - it says nothing on its own."""
    from regulus.rag.generation import whole_sentences
    from regulus.rag.section_promotion import _FRAME_SECTION
    from regulus.rag.validation import _related_identity
    from regulus.retrieval.references import parse_cfr_references

    if package is None or not asks_exemptions(question):
        return None
    passages: dict[str, list] = {}
    for s in sorted((s for s in package.sources if not s.is_docket), key=lambda s: s.index):
        m = _SECTION.match((s.citation or "").strip())
        if m and _FRAME_SECTION.match((getattr(s, "heading", None) or "")) and (s.text or "").strip():
            passages.setdefault(m.group(1), []).append(s)
    if not any(_EXEMPTING.search(ps[0].heading or "") for ps in passages.values()):
        return None
    ident = {}
    for s in package.sources:
        refs = [] if s.is_docket else [r for r in parse_cfr_references(s.citation or "") if r.section]
        if refs:
            ident[s.index] = (refs[0].title, refs[0].section, refs[0].paragraph_path)

    def cross_marks(body: str, own: int) -> list[int]:
        """A provision the quoted text itself names ("under § 112.2(b)(3)(i)", "in § 117.201") is cited to its own
        supplied text as well - the validator's provision_citation_mismatch asks a named provision's source (or its
        parent/child) among the sentence's markers (Y-001, 5 Nov: 112.2(b)(4) names 112.2(b)(3)(i))."""
        out: list[int] = []
        for ref in parse_cfr_references(body):
            if not ref.section:
                continue
            named = (ref.title, ref.section, ref.paragraph_path)
            if any(i in ident and _related_identity(named, ident[i]) for i in [own] + out):
                continue
            related = [i for i in sorted(ident) if _related_identity(named, ident[i])]
            exact = [i for i in related if ident[i] == named]
            if exact or related:
                out.append((exact or related)[0])
        return out
    lines = [LEAD]
    for sec in sorted(passages, key=lambda x: [int(p) if p.isdigit() else 0 for p in re.split(r"[.]", x)]):
        kept = list(passages[sec])
        while kept and (kept[-1].text or "").rstrip().endswith(":"):
            kept.pop()        # a list lead whose items did not arrive (Y-001: 112.2(b)(3)(ii)(B) "... it will either:")
        for k, s in enumerate(kept):
            body = whole_sentences(s.text or "") or (s.text or "").strip()
            heading = (s.heading or "").strip().rstrip(".") if k == 0 else ""
            label = f"{s.citation}" + (f" ({heading})" if heading else "")
            marks = "".join(f"[{i}]" for i in [s.index] + cross_marks(body, s.index))
            quoted = f"{body[:-1]} {marks}{body[-1]}" if body.endswith((".", "!", "?")) else f"{body} {marks}."
            lines.append(f"{label} (eCFR, binding, verbatim): {quoted}")
    return "\n".join(lines) if len(lines) > 1 else None
