"""TF1 (4 Nov): the protected core of the evidence - the five most useful distinct provisions, admitted first.

"Top-5" is not a cap on the evidence (selection admits up to 48 paragraphs and 15,000 characters): it is the order in
which required evidence competes for that ceiling when it does not all fit. The core is chosen from distinct UNITS - a
section's top-level paragraph family ("111.70(e)", or a whole undivided section) - so five paragraphs of one list cannot
crowd out the other provisions the question needs:

  1. the question is split into its asks (question_clauses: question marks, and "what ... how ... and can ..." in its
     last sentence - the boundaries QD1 reads, without the model);
  2. each ask in turn takes its most similar unit not yet taken (Title 21 paragraph index similarity of the unit's best
     paragraph, + DUTY_BONUS when that paragraph states a duty, exception or definition in the question's terms), until
     CORE units are chosen - so a multi-part question gets a unit for every part before any part gets a second;
  3. expansion, up to MAX units, each with its reason: a question with more than CORE asks gets one per ask, and a core
     paragraph that points to another paragraph of its section ("except as provided in paragraph (d)") brings it;
  4. each core unit's lead row and best paragraph compete first, in core order; everything else keeps its place.
Selection still presents what it keeps in document order, and records every unit with its score and reason.
"""
from __future__ import annotations

import re
from typing import Any

CORE = 5
MAX = 8
DUTY_BONUS = 0.03
_ASK = re.compile(r"(?:^|[,;:]\s*(?:and\s+|or\s+)?|\b(?:and|or)\s+)(?=(?:how|what|when|which|who|where|whether|can|could|"
                  r"must|may|should|does|do|is|are)\b)", re.I)
_DUTY = re.compile(r"\b(?:must|shall|required|requires?|prohibited|may not|except|unless|exempt|does not apply|means)\b", re.I)
_WORD = re.compile(r"[a-z]{4,}")


def question_clauses(question: str) -> list[str]:
    """The asks of a question, each as text: its sentences ending in "?", and within the last one the asks that start
    after a comma or "and"/"or". One clause when it asks one thing."""
    q = " ".join((question or "").split())
    if not q:
        return []
    parts = [p.strip() for p in re.split(r"(?<=\?)\s+", q) if p.strip()]
    last = parts.pop() if parts else q
    starts = [m.end() for m in _ASK.finditer(last)]
    starts = sorted({0, *[s for s in starts if s > 0]})
    pieces = [re.sub(r"(?:[\s,;:]+(?:and|or))?[\s,;:]*$", "", last[a:b]).strip(" ,;:")   # no trailing ", and"
              for a, b in zip(starts, starts[1:] + [len(last)])]
    pieces = [p for p in pieces if len(re.findall(r"[a-z]{3,}", p.lower())) >= 2] or [last]   # "how do we pay": an ask
    return parts + pieces


def unit_of(section: str, path: str) -> str:
    """A section's top-level paragraph family: "111.70" + "(e)" for "(e)(2)(i)"; the section for an undivided row."""
    m = re.match(r"\([^)]+\)", path or "")
    return section + (m.group(0) if m else "")


def choose_core(rows: list[dict[str, Any]], clauses: list[str], clause_scores: list[dict[str, float]],
                question: str) -> tuple[list[str], list[dict[str, Any]]]:
    """`rows`: the required eCFR evidence in competition order, as {"id", "section", "path", "text"}; `clause_scores`:
    for each clause, evidence id -> similarity. Returns (evidence ids to admit first, in order; the core record)."""
    if not rows or not clause_scores:
        return [], []
    qwords = set(_WORD.findall((question or "").lower()))
    units: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        units.setdefault(unit_of(r["section"], r["path"]), []).append(r)

    def score(r, sims):
        bonus = DUTY_BONUS if _DUTY.search(r["text"] or "") and qwords & set(_WORD.findall((r["text"] or "").lower())) else 0.0
        return sims.get(r["id"], 0.0) + bonus

    best: dict[tuple[int, str], tuple[float, dict]] = {}
    for c, sims in enumerate(clause_scores):
        for u, members in units.items():
            top = max(members, key=lambda r: score(r, sims))
            best[(c, u)] = (score(top, sims), top)
    target = min(MAX, max(CORE, len(clauses)))
    chosen: list[dict[str, Any]] = []
    taken: set[str] = set()
    while len(chosen) < min(target, len(units)):
        progressed = False
        for c in range(len(clause_scores)):
            if len(chosen) >= min(target, len(units)):
                break
            left = [(best[(c, u)][0], u) for u in units if u not in taken]
            if not left:
                break
            s, u = max(left)
            taken.add(u)
            progressed = True
            reason = (f"ask {c + 1} of {len(clauses)}" if len(clause_scores) > 1 else "top score")
            if len(chosen) >= CORE:
                reason = f"expanded: the question has {len(clauses)} asks"
            chosen.append({"unit": u, "score": round(s, 4), "ask": c + 1, "best": best[(c, u)][1]["id"], "reason": reason})
        if not progressed:
            break
    # a core paragraph that points to another paragraph of its section brings that paragraph's unit
    for rec in list(chosen):
        if len(chosen) >= MAX:
            break
        text = next(r["text"] for r in rows if r["id"] == rec["best"]) or ""
        section = rec["unit"].split("(")[0]
        for ref in re.findall(r"(?:except as (?:provided|otherwise provided) in|as defined in|under|in accordance with)\s+"
                              r"paragraphs?\s+(\([a-z0-9]{1,4}\))", text, re.I):
            u = section + ref
            if u in units and u not in taken and len(chosen) < MAX:
                taken.add(u)
                top = max(units[u], key=lambda r: score(r, clause_scores[0]))
                chosen.append({"unit": u, "score": round(score(top, clause_scores[0]), 4), "ask": rec["ask"],
                               "best": top["id"], "reason": f"expanded: {rec['unit']} points to paragraph {ref}"})
    first: list[str] = []
    for rec in chosen:
        lead = next((r["id"] for r in units[rec["unit"]] if unit_of(r["section"], r["path"]) == rec["unit"]
                     and r["path"] == rec["unit"][len(rec["unit"].split("(")[0]):]), None)
        for eid in (lead, rec["best"]):
            if eid and eid not in first:
                first.append(eid)
    return first, chosen
