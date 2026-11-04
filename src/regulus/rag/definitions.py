"""A question asking what terms mean, answered from the paragraphs that define them (fix D1, 26 Oct 2026).

"Our lake supplier mentions a 'gloss white' substratum and says our foot cream is 'externally applied.' What do these
terms mean under Part 82?" - the definitions were in the evidence (21 CFR 82.3(i), (n)) and the answer was withheld
(COS-82.3-Q1, 22 Oct holdout). A definition answers such a question by itself: the paragraph that reads "The term gloss
white means ..." is the answer, verbatim.

The terms asked about are the quoted ones in a question that asks for a meaning ("mean", "define", "definition",
"term"), or X in "what does X mean" / "the definition of X". Every one must have a supplied paragraph that defines it -
"The term(s) ... X ... mean(s)", "X means", "X is defined as" - or nothing is composed (the model answers). The match is
the paragraph naming the asked term as the thing it defines, so the answer cannot quote the wrong definition; it needs
no section resolution and works for every definitions section in Title 21. Switched by "D1"
(regulus.rag.validator_switches).
"""
from __future__ import annotations

import re

_QUOTED = re.compile(r"(?<![A-Za-z])['\"‘“]([^'\"’”]{2,60}?)[.,]?['\"’”](?![A-Za-z])")
# an explicit request for a meaning: "what do these terms mean", "what does X mean", "the meaning of", "define X". The
# words "definition" or "term" alone are not one: "Under the definitions, is our premix an 'ingredient'?" asks to apply
# them (COS-700.3-Q2: the model's answer was right; quoting the definitions instead would not apply them)
_ASKS_MEANING = re.compile(r"\bwhat\s+(?:do|does|did)\b[^?]*\bmean\b|\bmeaning\s+of\b|\bwhat\s+is\s+meant\b|\bdefine\b|"
                           r"\bdefinitions?\s+of\b|\bhow\s+(?:does|do)\b[^?]*\bdefine\b|\bwhat\s+meaning\b", re.I)
_WHAT_MEANS = re.compile(r"\bwhat\s+(?:does|do|is meant by)\s+(?:the\s+term\s+)?(?P<t>[A-Za-z][\w\s&'-]{1,60}?)\s+(?:mean|refer to)\b"
                         r"|\b(?:definition|meaning)\s+of\s+(?:the\s+term\s+|a\s+|an\s+|the\s+)?(?P<u>[A-Za-z][\w\s&'-]{1,60}?)(?=[?.,;]|\s+(?:under|in|for)\b|$)",
                         re.I)
LEAD_ONE = "What the regulation defines this term as, verbatim:"
LEAD_MANY = "What the regulation defines these terms as, verbatim:"


def asked_terms(question: str) -> list[str]:
    q = question or ""
    if not _ASKS_MEANING.search(q):
        return []
    terms = [t.strip() for t in _QUOTED.findall(q) if len(t.strip()) >= 2]
    if not terms:
        terms = [(m.group("t") or m.group("u") or "").strip() for m in _WHAT_MEANS.finditer(q)]
    return list(dict.fromkeys(t for t in terms if t and len(t.split()) <= 8))


def _defines(text: str, term: str) -> bool:
    t = r"\s+".join(re.escape(w) for w in term.lower().split())
    body = (text or "").lower()
    return bool(re.search(rf"\bthe\s+terms?\s+(?:[\w,'-]+\s+){{0,6}}?\W?{t}\W?(?:\s+[\w,'-]+){{0,8}}?\s+(?:means?|refers? to|is defined)\b", body)
                or re.search(rf"^\W*(?:\((?:[a-z0-9]{{1,4}}|[ivxl]{{5,8}})\)\s*)*\W?{t}\W?\s+(?:means|is defined as|refers to)\b", body))


def compose(package, question: str) -> str | None:
    """The definitions of every term the question asks about, verbatim and cited, or None."""
    from regulus.rag.generation import _verbatim, whole_sentences
    from regulus.retrieval.evidence import TIER_EXACT

    terms = asked_terms(question)
    if package is None or not terms:
        return None
    chosen = []
    for term in terms:
        # the requested provisions' own text only: a definition found in background evidence may belong to another
        # programme ("distinctive by design" in the drug rule 211.132 for a 700.25 question)
        hit = next((s for s in package.sources if not s.is_docket and s.corpus == "ecfr" and s.tier == TIER_EXACT
                    and s.requirement_ids and _defines(s.text, term)), None)
        if hit is None:
            return None            # every asked term defined, or the model answers
        if hit not in chosen:
            chosen.append(hit)
    lines = [LEAD_ONE if len(terms) == 1 else LEAD_MANY]
    for s in chosen:
        body = whole_sentences(s.text or "")
        if not body:
            return None
        lines.append(f"{s.citation} (eCFR, binding, verbatim): {_verbatim(body, s.index)}")
    return "\n".join(lines)
