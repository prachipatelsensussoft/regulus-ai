"""SP4O (5 Nov 2026): a pure overview question about one large subpart is answered with the subpart's outline.

"What current good manufacturing practice requirements apply to ... animal food?" asks for 21 CFR 507 Subpart B as a
whole (key: 507.14 through 507.28 - personnel, plant and grounds, sanitation, water supply, equipment, processes,
warehousing, by-products). The subpart is 17,626 characters, more than the prompt holds, and the model answered from
the three sections it was given (PARTIALLY PASS, "personnel, water supply, warehousing omitted"). Each section's heading
and opening provision, verbatim and cited, is the subpart's outline - what an expert gives for an overview question,
and every governing section is surfaced. The promoter (SectionPromoter.outlined_subpart) decides the question asks for
the subpart and nothing narrower and supplies the sections; this module only writes the answer from what was supplied.
"""
from __future__ import annotations

import re

COMPOSED_TYPE = "composed_outline"
SECTION_CHARS = 700   # each section's quoted passages, after the first (always quoted)
MIN_SHARE = 0.75   # sections of the subpart whose opening provision must be supplied for an outline to be given
# any section number in the quoted text ("§§ 112.43(c)(4)(ii) and 112.44"); only numbers of the subpart's own sections
# are used, so a figure like "0.5" never becomes a citation
_SECTION_REF = re.compile(r"(?<![\d.])(\d{1,4}\.\d{1,4}[a-z]?)(?![\d])")
LEAD = "The {title} subpart of 21 CFR Part {part} sets out these requirements section by section; each section's opening provision, verbatim:"


def outline_request(promotion) -> tuple[str, list[str]] | None:
    """(subpart title, its sections) from the promotion's SP4O note, or None."""
    from regulus.rag.section_promotion import SUBPART_OUTLINE_NOTE

    for note in getattr(promotion, "notes", ()) or ():
        if note.startswith(SUBPART_OUTLINE_NOTE + ": "):
            title, _, secs = note[len(SUBPART_OUTLINE_NOTE) + 2:].rpartition(": ")
            members = [s.strip() for s in secs.split(",") if s.strip()]
            if title and members:
                return title, members
    return None


def compose(package, question: str, title: str, members: list[str]) -> str | None:
    """The outline from the supplied evidence: for each section, its supplied passages in document order (by source
    number) up to SECTION_CHARS - always the first, so a lead-in ("The following apply to the water supply:") comes with
    its items (5 Nov blind scoring: "openers only, some contentless") - the first carrying the section heading; None when
    fewer than MIN_SHARE of the sections are supplied."""
    from regulus.rag.generation import whole_sentences

    if package is None or not members:
        return None
    passages: dict[str, list] = {}
    for s in sorted((s for s in package.sources if not s.is_docket), key=lambda s: s.index):
        m = re.match(r"21 CFR (\d+\.\d+[a-z]?)", (s.citation or "").strip())
        if m and m.group(1) in members and (s.text or "").strip():
            passages.setdefault(m.group(1), []).append(s)
    if len(passages) < MIN_SHARE * len(members):
        return None
    first = {sec: ps[0] for sec, ps in passages.items()}
    part = members[0].split(".")[0]
    lines = [LEAD.format(part=part, title=title)]
    for sec in members:
        used = 0
        for k, s in enumerate(passages.get(sec, [])):
            body = whole_sentences(s.text or "") or (s.text or "").strip()
            if k and used + len(body) > SECTION_CHARS:
                break
            used += len(body)
            heading = (getattr(s, "heading", None) or "").strip().rstrip(".") if k == 0 else ""
            label = f"{s.citation}" + (f" ({heading})" if heading else "")
            # a section of the subpart the quoted provision itself refers to ("tests required under § 112.44") is cited to
            # its own supplied text as well (the validator's citation_mismatch, 5 Nov: 112.47(a), 4.104(b))
            refs = [first[m].index for m in dict.fromkeys(_SECTION_REF.findall(body))
                    if m != sec and m in first and first[m].index != s.index]
            marks = "".join(f"[{i}]" for i in [s.index] + refs)
            quoted = f"{body[:-1]} {marks}{body[-1]}" if body.endswith((".", "!", "?")) else f"{body} {marks}."
            lines.append(f"{label} (eCFR, binding, verbatim): {quoted}")
    return "\n".join(lines)
