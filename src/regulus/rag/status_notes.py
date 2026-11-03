"""Status notes (fix S1, 23 Oct 2026): facts about a provision's current status that its codified text does not carry.

The keys of 710.3 (the voluntary registration program closed in March 2023), 740.18 (stayed since 1980) and the FD&C Act
606 questions (no GMP rule issued) need a fact no eCFR paragraph states, so no evidence or prompt change can supply it.
Each note in data/reference/status_notes_v1.json is dated and sourced; it is appended verbatim to a delivered answer
that CITES a matching source, after validation, and is never handed to the model. Switched by "S1" in
regulus.rag.validator_switches.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_NOTES_PATH = Path(__file__).resolve().parents[3] / "data" / "reference" / "status_notes_v1.json"
_MARK = re.compile(r"(?<![A-Za-z0-9])\[(\d{1,2})\]")
_CFR = re.compile(r"\b21\s*CFR\s*(\d{1,4})\.(\d{1,4}[a-z]?)")
_ACT = re.compile(r"(?:FD&C Act|Act)\s*§+\s*(\d{3}[A-Z]?)|21\s*U\.S\.C\.\s*(\d{3}[a-z]?)", re.I)
_USC_TO_ACT = {"364": "604", "364a": "605", "364b": "606", "364c": "607", "364d": "608", "364e": "609", "364f": "610",
               "364g": "611", "364h": "612", "364i": "613", "364j": "614"}


@lru_cache(maxsize=1)
def _notes() -> tuple[dict, ...]:
    return tuple(json.loads(_NOTES_PATH.read_text(encoding="utf-8"))["notes"])


def _cited_provisions(answer: str, sources) -> tuple[set[str], set[str], set[str]]:
    by_index = {s.index: s for s in sources}
    parts, sections, acts = set(), set(), set()
    for n in {int(m) for m in _MARK.findall(answer or "")}:
        src = by_index.get(n)
        if src is None:
            continue
        label = f"{src.citation or ''} {src.heading or ''}"
        for m in _CFR.finditer(label):
            parts.add(m.group(1))
            sections.add(f"{m.group(1)}.{m.group(2)}")
        if getattr(src, "corpus", None) == "usc":
            for m in _ACT.finditer(label):
                acts.add(m.group(1) or _USC_TO_ACT.get((m.group(2) or "").lower(), ""))
    return parts, sections, acts - {""}


def _qs1() -> bool:
    from regulus.rag.validator_switches import enabled

    return enabled("QS1")


# the switches a note may name; a note naming any other is never shown
_NOTE_SWITCHES = {"QS1": _qs1}


def status_notes(answer: str, sources, question: str | None = None) -> list[dict]:
    """The notes whose provision the delivered answer cites (by its citation markers), in file order. A note with a
    "switch" needs it on (regulus.rag.validator_switches); one with a "question" pattern only goes with a question it
    matches (QS1, 3 Nov: the QMSR effective date for "what changed ... and what is the compliance date?", not for every
    part 820 answer)."""
    parts, sections, acts = _cited_provisions(answer, sources)
    return [n for n in _notes()
            if (parts & set(n.get("cfr_parts", ())) or sections & set(n.get("cfr_sections", ()))
                or acts & set(n.get("act_sections", ())))
            and (not n.get("switch") or _NOTE_SWITCHES.get(n["switch"], lambda: False)())
            and (not n.get("question") or bool(question and re.search(n["question"], question, re.I)))]


def supplied_status_notes(sources) -> list[dict]:
    """ST2 (27 Oct 2026): the notes whose provision is among the supplied sources - shown to the model before it answers,
    so an answer about the stayed 740.18 warning does not state it as a current duty (stratified 50: the answer said the
    warning "must appear" above an appended note saying the section is stayed)."""
    every = " ".join(f"[{s.index}]" for s in sources)
    return status_notes(every, sources)
