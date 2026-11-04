"""Two presentation fixes to a drafted answer, applied before it is validated (4 Nov 2026, both off until measured).

DI1 - a denial opener that the answer itself contradicts. Asked "How does Part 820 extend the standard's traceability
requirements for implantable devices ...?" the model opened "Part 820 does not extend traceability requirements ...
beyond the scope of the standard's own provisions. Instead, it specifies that manufacturers of devices that support or
sustain life ... must comply with Clause 7.5.9.2" - the second sentence IS the extension (21 CFR 820.10(d)), so the first
is false (M-018, scored "opens by wrongly denying that Part 820 extends the requirement"; M-006 the same shape with
"provide"). The opener is dropped only when all of these hold: it is the answer's first sentence, it denies the very verb
the question asks with ("How does X extend ...?" -> "X does not extend ..."), the question asks How/What does X ..., and
the next sentence begins "Instead," or "Rather,". A question asking WHICH provision requires something (a negative
control, where a denial is the answer) never qualifies. 300Q (combo4, 3 Nov): fires on M-006 and M-018 only.

MP1 - a paragraph letter written as a citation marker. The model cited 21 CFR 820.45(b) as "[b]" (M-026: "... the release
of labeling for use must be documented in accordance with Clause 4.2.5 of ISO 13485 [b]."), which no check reads, so the
statement looked unsupported. "[x]" becomes the [n] marker of the one supplied top-level paragraph (x) - of a section the
sentence names, or of the only supplied section that has a paragraph (x). Ambiguous or unsupplied: left as written.
300Q (combo4): fires on M-026 only.
"""
from __future__ import annotations

import re

_ASKS_HOW_WHAT = re.compile(r"^\s*(?:how|what)\s+(?:does|do)\b", re.I)
_DENIAL_VERBS = "extend|provide|require|specify|address|establish|set|impose|apply|contain|include|permit"
_DENIAL_OPENER = re.compile(
    r"\A\s*(?P<subject>[^.\n]{2,80}?)\s+(?:does|do)\s+not\s+(?P<verb>" + _DENIAL_VERBS + r")\b[^.\n]*\.\s+"
    r"(?:Instead|Rather),\s+(?P<rest>\S)", re.I)
_PRONOUN = re.compile(r"(?:it|they)\b", re.I)


def drop_denial_opener(text: str, question: str | None) -> tuple[str, str | None]:
    """(text, the dropped opener or None) - DI1, see the module docstring."""
    if not text or not question or not _ASKS_HOW_WHAT.search(question):
        return text, None
    m = _DENIAL_OPENER.match(text)
    if m is None:
        return text, None
    verb = m.group("verb").lower()
    if not re.search(rf"\b{verb}(?:s|ed|ing)?\b", question, re.I):
        return text, None   # the opener denies something the question did not ask about
    subject = m.group("subject").strip()
    rest = text[m.start("rest"):]
    pronoun = _PRONOUN.match(rest)
    if pronoun:   # "Instead, it specifies ..." -> "Part 820 specifies ..."
        rest = subject + rest[pronoun.end():]
    else:
        rest = rest[0].upper() + rest[1:]
    return rest, text[:m.start("rest")].strip()


_LETTER_MARK = re.compile(r"(?<![A-Za-z0-9\]])\[([a-z])\]")
_TOP_PARAGRAPH = re.compile(r"^21 CFR (\d{1,4}\.\d{1,5}[a-z]?)\(([a-z])\)$")
_SECTION = re.compile(r"(?<![\d.])(\d{1,4}\.\d{1,5}[a-z]?)(?![\d])")


def resolve_letter_markers(text: str, sources) -> tuple[str, int]:
    """(text, markers resolved) - MP1, see the module docstring. `sources` are the package's EvidenceSources."""
    if not text or not _LETTER_MARK.search(text):
        return text, 0
    top: dict[tuple[str, str], int] = {}
    for s in sources:
        m = _TOP_PARAGRAPH.match((getattr(s, "citation", "") or "").strip())
        if m:
            top.setdefault((m.group(1), m.group(2)), s.index)
    n = 0

    def repl(match: re.Match) -> str:
        nonlocal n
        letter = match.group(1)
        having = {sec: i for (sec, x), i in top.items() if x == letter}
        if not having:
            return match.group(0)
        before = match.string[:match.start()]
        start = max(before.rfind(". "), before.rfind("\n")) + 1
        named = [sec for sec in _SECTION.findall(before[start:]) if sec in having]
        choice = set(named) if named else set(having)
        if len(choice) != 1:
            return match.group(0)
        n += 1
        return f"[{having[choice.pop()]}]"

    return _LETTER_MARK.sub(repl, text), n
