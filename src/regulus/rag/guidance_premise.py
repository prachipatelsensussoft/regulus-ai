"""GP1 (5 Nov 2026): a question that asks which FDA guidance REQUIRES something rests on a false premise.

"Which FDA guidance requires a device manufacturer to qualify its suppliers?" (300Q PB-005, FAIL: the search went to
the guidance corpus, found nothing that "requires", and declined). Guidance documents create no requirement - 21 CFR
10.115(d)(1): "Guidance documents do not establish legally enforceable rights or responsibilities. They do not legally
bind the public or FDA." - so the correct answer corrects the premise and names the regulation that does require it.

The question is asked again with "guidance" read as "regulation" ("Which regulation requires a device manufacturer to
qualify its suppliers?"), which the pipeline answers like any other (5 Nov probe: drug stability -> 211.166, food safety
plan -> 117.126), and the delivered answer is led by the premise note, quoted from 10.115(d)(1) as the database holds it.
Only "which/what guidance ... requires / mandates / obligates / compels / imposes / makes ... mandatory / says ... must",
never a question asking which guidance COVERS or RECOMMENDS something (that asks for a guidance document, which exists).
"""
from __future__ import annotations

import re

# "which (FDA) guidance (document)" / "under / according to which (FDA) guidance"
_WHICH_GUIDANCE = re.compile(r"\b(?:which|what)\s+(?:FDA\s+|of\s+FDA'?s\s+)?guidance(?:\s+documents?)?\b", re.I)
# the guidance is said to create the duty
_DUTY = re.compile(r"\b(?:requires?|mandates?|obligates?|compels?|imposes?|forces?|makes?\b[^?]*\bmandatory|"
                   r"says?\b[^?]*\b(?:must|shall)|(?:are|is)\s+(?:\w+\s+){0,3}required|must)\b", re.I)
# the guidance phrase as it is replaced: "FDA guidance document(s)" -> "regulation"
_GUIDANCE_PHRASE = re.compile(r"\b(?:FDA\s+|of\s+FDA'?s\s+)?guidance(?:\s+documents?)?\b", re.I)
PREMISE_SECTION, PREMISE_PATH = "10.115", "(d)"
_NOT_BINDING = re.compile(r"\(1\)\s*No\.\s*(Guidance documents do not establish legally enforceable rights or "
                          r"responsibilities\.(?:\s*They do not legally bind the public or FDA\.)?)")
PREMISE_LEAD = ("No FDA guidance document creates this requirement: under 21 CFR 10.115(d)(1), \"{quote}\" "
                "The requirement comes from the regulation:")


def premise_rewrite(question: str) -> str | None:
    """The question with its guidance read as the regulation, or None when it does not say a guidance requires something."""
    q = question or ""
    m = _WHICH_GUIDANCE.search(q)
    if m is None or not _DUTY.search(q[m.end():] if not q[:m.start()].strip() else q):
        return None
    return _GUIDANCE_PHRASE.sub("regulation", q, count=1)


def premise_note(db) -> str | None:
    """The lead quoting 10.115(d)(1) from the database, or None when the database does not hold that sentence."""
    if db is None:
        return None
    try:
        rows = db.query("SELECT body_text FROM cfr_records WHERE section_number = %(s)s AND paragraph_path = %(p)s "
                        "AND removed_on IS NULL ORDER BY path_ordinal LIMIT 1", {"s": PREMISE_SECTION, "p": PREMISE_PATH})
    except Exception:  # noqa: BLE001 - the note is an aid: without it the answer is delivered as it is
        return None
    m = _NOT_BINDING.search((rows[0]["body_text"] if rows else "") or "")
    return PREMISE_LEAD.format(quote=m.group(1)) if m else None
