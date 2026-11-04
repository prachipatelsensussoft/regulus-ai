"""Questions about the former Quality System regulation and about ISO 13485 clauses (fix QM1, 26 Oct 2026).

On 2 February 2026 the Quality Management System Regulation (QMSR) replaced the Quality System regulation: part 820 now
consists of 820.1, 820.3, 820.7, 820.10 (a quality management system complying with ISO 13485:2016, incorporated by
reference), 820.35 (records) and 820.45 (labeling and packaging); the other former sections - 820.30 design controls,
820.50 purchasing, 820.100 corrective and preventive action and the rest - are not in the current eCFR. Users and the
acceptance benchmark still ask for them ("What are the CAPA requirements of 21 CFR 820.100?"); the pipeline found no such
provision and declined, while the benchmark scores 0 for describing a removed section as in force and expects the ISO
13485 clause, by number and title, for part 820 questions.

What is said is what can be shown:
  * that the section is not in the current eCFR - from the database (no live text for it) and the former heading, from
    the eCFR as of 31 Jan 2026 (data/reference/qmsr_v1.json);
  * the current requirement - the live successor's text (820.35, 820.45) or 820.10's, quoted verbatim and cited; the
    question is re-planned onto those sections so their text is the evidence;
  * the ISO 13485:2016 clause on the same subject - number and title only, never the standard's text - with its basis:
    the regulation's own words, the final rule's preamble, or the clause title alone, in which case the answer says that
    FDA published no section-by-section mapping (89 FR 7496, Comment 18).
A clause number ISO 13485:2016 does not have ("Clause 9.2") is said not to exist. Switched by "QM1"
(regulus.rag.validator_switches).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from regulus.config import PROJECT_ROOT

REFERENCE = PROJECT_ROOT / "data" / "reference" / "qmsr_v1.json"
SUCCESSOR_DEFAULT = "820.10"
QUOTE_CHARS = 2200
# a clause of ISO 13485 named in a question: "ISO 13485 Clause 9.2", "Clause 7.3 of ISO 13485", "ISO 13485:2016 8.5.2"
_CLAUSE = re.compile(r"\bISO\s*13485(?::2016)?(?:\(E\))?\s*,?\s*(?:Clause|cl\.|section|§)?\s*(\d{1,2}(?:\.\d{1,2}){0,3})(?!\d)"
                     r"|\bClause\s+(\d{1,2}(?:\.\d{1,2}){0,3})\s+(?:of|in)\s+(?:the\s+)?ISO\s*13485", re.I)
# the answer's pipeline-written sentences (what the database and the reference file show; exempt from the claim and
# figure checks, like the status notes)
REMOVED_SENTENCE = "is not in the current eCFR: it belonged to the former Quality System regulation"
CLAUSE_SENTENCE = "In ISO 13485:2016 (incorporated by reference, 21 CFR 820.7)"
NO_CLAUSE_SENTENCE = "ISO 13485:2016 has no Clause"
# whole lines, matched on wording that survives the validator's removal of supplied citations ("21 CFR 820.7")
PIPELINE_SENTENCES_RE = re.compile(r"[^\n]*?is not in the current eCFR: it belonged to the former Quality System regulation[^\n]*"
                                   r"|No section of the current part 820 uses the terms[^\n]*"
                                   r"|In ISO 13485:2016 \(incorporated by reference[^\n]*|ISO 13485:2016 has no Clause[^\n]*")


@lru_cache(maxsize=1)
def reference() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def former_refs(plan, live_sections: set[str]) -> list[tuple[str, str]]:
    """(section, paragraph path) of every former part 820 section the question names that has no live text."""
    former = reference()["former_sections"]
    out: list[tuple[str, str]] = []
    for r in plan.requirements:
        ref = r.reference
        if ref.title == "21" and ref.part == "820" and ref.section in former and ref.section not in live_sections:
            if (ref.section, ref.paragraph_path or "") not in out:
                out.append((ref.section, ref.paragraph_path or ""))
    return out


# QM2 (27 Oct): the former regulation's record types, asked by name ("What is the difference between a Device Master
# Record, a Device History Record and a Design History File under the current Part 820?"), are its former sections
_FORMER_NAMES = (
    (re.compile(r"\bdevice\s+master\s+records?\b|\bDMRs?\b"), ("820.181", "")),
    (re.compile(r"\bdevice\s+history\s+records?\b|\bDHRs?\b"), ("820.184", "")),
    (re.compile(r"\bdesign\s+history\s+files?\b|\bDHFs?\b"), ("820.30", "(j)")),
    (re.compile(r"\bquality\s+system\s+records?\b"), ("820.186", "")),
)


TERMS_SENTENCE = "No section of the current part 820 uses the terms"


def record_terms(question: str) -> list[str]:
    """The former record types the question names, in its order, as the regulation wrote them (QM2)."""
    names = {"820.181": "device master record", "820.184": "device history record", "820.30": "design history file",
             "820.186": "quality system record"}
    hits = sorted((m.start(), ref[0]) for pat, ref in _FORMER_NAMES if (m := re.search(pat.pattern, question or "", re.I)))
    return list(dict.fromkeys(names[sec] for _, sec in hits))


def former_by_name(question: str, live_sections: set[str]) -> list[tuple[str, str]]:
    """(section, paragraph path) of every former part 820 record type the question names (QM2), in question order."""
    former = reference()["former_sections"]
    hits = sorted((m.start(), ref) for pat, ref in _FORMER_NAMES if (m := re.search(pat.pattern, question or "", re.I)))
    return list(dict.fromkeys(ref for _, ref in hits if ref[0] in former and ref[0] not in live_sections))


def unknown_clauses(question: str) -> list[str]:
    """ISO 13485 clause numbers the question names that ISO 13485:2016 does not have."""
    known = set(reference()["clauses"]) | set(reference()["parent_titles"])
    found = [a or b for a, b in _CLAUSE.findall(question or "")]
    return list(dict.fromkeys(c for c in found if c and c not in known and c != "13485"))


def successors(former: list[tuple[str, str]]) -> list[str]:
    """The live sections that carry the former sections' subject (820.10 by default)."""
    ref = reference()["former_sections"]
    out = [ref[s].get("successor", SUCCESSOR_DEFAULT) for s, _ in former]
    return list(dict.fromkeys(out)) or [SUCCESSOR_DEFAULT]


def clauses_for(section: str, path: str) -> tuple[list[str], str]:
    info = reference()["former_sections"][section]
    first = re.match(r"\([a-z0-9]+\)", path or "")
    by_par = (info.get("paragraphs") or {}).get(first.group(0)) if first else None
    return (by_par or info.get("clauses") or []), info.get("basis", "title")


def clause_label(c: str) -> str:
    ref = reference()
    title = ref["clauses"].get(c) or ref["parent_titles"].get(c)
    return f"Clause {c} ({title})" if title else f"Clause {c}"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def compose(package, question: str, former: list[tuple[str, str]], unknown: list[str] | None = None,
            *, limit: int = 4, unused_terms: list[str] | None = None) -> str | None:
    """The answer (see the module docstring), or None when the current text to quote is not in the evidence."""
    from regulus.rag.conversation import subject_words
    from regulus.rag.generation import _verbatim, whole_sentences
    from regulus.retrieval.evidence import TIER_EXACT

    if package is None or not (former or unknown):
        return None
    ref = reference()["former_sections"]
    succ = successors(former) if former else [SUCCESSOR_DEFAULT]
    exact = [s for s in package.sources if s.tier == TIER_EXACT and not s.is_docket
             and any(re.match(rf"21 CFR {re.escape(x)}(?![\d.])", s.citation or "") for x in succ)]
    if not exact:
        return None
    wanted = [c for s, p in former for c in clauses_for(s, p)[0]]
    asked = subject_words(question)

    def names_wanted(s) -> bool:
        body = s.text or ""
        return any(re.search(rf"Clauses?\s+(?:[\d.]+,?\s*(?:and\s+)?)*{re.escape(c.split('.')[0])}(?:\.\d+)*", body)
                   and re.search(rf"(?<![\d.]){re.escape(c)}(?![\d])", body) for c in wanted)

    def kind(s) -> int:
        cit = s.citation or ""
        if not re.search(r"\((?:[a-z0-9]{1,4}|[ivxl]{5,8})\)$", cit):
            return 0                                   # the section's lead-in sentence
        if cit.endswith("820.10(a)"):
            return 1                                   # the quality management system requirement itself
        return 2 if names_wanted(s) else 3

    # 820.10: its lead-in, (a) and the paragraphs naming the clause on the subject - nothing else (its UDI or
    # enforcement paragraphs say nothing about CAPA). A successor (820.35, 820.45): its most relevant paragraphs too.
    keep = [s for s in exact if kind(s) < 3 or not s.citation.startswith("21 CFR 820.10")]
    chosen = sorted(keep, key=lambda s: (kind(s), -len(asked & subject_words(s.text or "")), s.index))[:limit]
    # a successor paragraph naming the clause comes with its sub-paragraphs: 820.35(a) is the list (a)(1)-(7)
    heads = [s.citation for s in chosen if kind(s) == 2 and not s.citation.startswith("21 CFR 820.10")]
    used = sum(len(s.text or "") for s in chosen)
    for s in sorted(exact, key=lambda x: x.index):
        if s not in chosen and any((s.citation or "").startswith(h) for h in heads) and used + len(s.text or "") <= QUOTE_CHARS:
            chosen.append(s)
            used += len(s.text or "")
    lines: list[str] = []
    for section, path in former:
        lines.append(f"21 CFR {section}{path} (\"{ref[section]['heading']}\") {REMOVED_SENTENCE}, which the Quality "
                     f"Management System Regulation replaced on 2 February 2026.")
    if unused_terms:
        # QM2: checked against the database by the caller - no live part 820 text uses these terms
        lines.append(f"{TERMS_SENTENCE} {_join([f'“{t}”' for t in unused_terms])}: the Quality Management System "
                     f"Regulation does not define or require these records by those names.")
    for c in unknown or ():
        lines.append(f"{NO_CLAUSE_SENTENCE} {c}: its requirements are organized in Clauses 4 through 8.")
    lines.append(f"The current requirement is in {_join([f'21 CFR {x}' for x in succ])}, verbatim:")
    for s in sorted(chosen, key=lambda x: x.index):
        body = whole_sentences(s.text or "")
        if body:
            lines.append(f"{s.citation} (eCFR, binding, verbatim): {_verbatim(body, s.index)}")
    for section, path in former:
        cl, basis = clauses_for(section, path)
        if not cl:
            continue
        line = (f"{CLAUSE_SENTENCE} the subject of former 21 CFR {section}{path} is addressed by "
                f"{_join([clause_label(c) for c in cl])}.")
        if basis == "title":
            line += " FDA published no section-by-section mapping from the former regulation; the clause is named by subject."
        lines.append(line)
    if any(line.startswith(CLAUSE_SENTENCE) for line in lines):
        lines.append("The standard's text is not reproduced here.")   # once: repeated per section it read as degenerate
    return "\n".join(lines)


# QC1 (27 Oct): "Does Part 820 require ISO 13485 certification, and does a certificate exempt us from FDA inspection?"
# 820.10(a) requires a documented quality management system that complies with ISO 13485; no live part 820 text
# mentions certification (checked against the database for every answer, not assumed). The model, given 820.10, wrote
# conditions the text does not state and was withheld (device baseline M-041).
_CERTIFICATION = re.compile(r"\bcertif(?:y|ied|ication|icate)s?\b", re.I)
_ISO = re.compile(r"\bISO\s*13485\b", re.I)
CERTIFICATION_SENTENCE = "No section of the current part 820 mentions certification"


def asks_certification(question: str) -> bool:
    return bool(_CERTIFICATION.search(question or "") and _ISO.search(question or ""))


def compose_certification(package, question: str) -> str | None:
    """The QC1 answer, or None when 820.10(a) is not in the evidence. The caller has confirmed that no live part 820
    text mentions certification."""
    from regulus.rag.generation import _verbatim, whole_sentences

    if package is None or not asks_certification(question):
        return None
    by_cite = {s.citation: s for s in package.sources if not s.is_docket}
    lead, a = by_cite.get("21 CFR 820.10"), by_cite.get("21 CFR 820.10(a)")
    if a is None:
        return None
    lines = [f"Part 820 does not require ISO 13485 certification: 21 CFR 820.10(a) requires the manufacturer to document a "
             f"quality management system that complies with the applicable requirements of ISO 13485 [{a.index}].",
             f"{CERTIFICATION_SENTENCE}: it neither requires an ISO 13485 certificate nor exempts a certificate holder "
             f"from FDA inspection.",
             "21 CFR 820.10, verbatim:"]
    for s in (lead, a):
        if s is not None and whole_sentences(s.text or ""):
            lines.append(f"{s.citation} (eCFR, binding, verbatim): {_verbatim(whole_sentences(s.text or ''), s.index)}")
    return "\n".join(lines)


# SU1 (4 Nov 2026, off until measured): "What does 21 CFR 820.45 require that the incorporated standard does not?" - a
# Part 820 section that opens "In addition to the requirements of Clause 7.5.1 of ISO 13485 ..." IS the answer: its own
# text is what it adds. The model, given exactly 820.45, wrote "820.45 requires that the incorporated standard (ISO 13485)
# includes the elements ..." and "does not require that the incorporated standard exclude any element" (M-025, scored
# muddled). The same question about 820.35 is answered by LS1 with the section's text verbatim and scores PASS (M-020).
# Only a question naming ONE Part 820 section and asking what it adds to / requires beyond the standard; only when that
# section's supplied lead paragraph itself says "In addition to ... Clause(s) X ... ISO 13485".
_SUPPLEMENT_ASKED = re.compile(r"\b(?:add|adds|added|require|requires|impose|imposes|supplement|supplements|beyond|in addition)\b", re.I)
_STANDARD_NAMED = re.compile(r"\b(?:incorporated standard|ISO\s*13485|the standard)\b", re.I)
_PART820_SECTION = re.compile(r"(?<![\d.])(820\.\d{1,3})(?![\d.])")
_IN_ADDITION = re.compile(r"^\s*In addition to (?:the requirements of )?Clauses? (\d{1,2}(?:\.\d{1,2}){0,3}"
                          r"(?:(?:,\s*|,?\s+and\s+)\d{1,2}(?:\.\d{1,2}){0,3})*)\s+(?:of|in)\s+ISO 13485", re.I)
SUPPLEMENT_LEAD = "Under 21 CFR {sec}, which adds to ISO 13485 {clauses}, what it requires in addition to the standard, verbatim:"


def asks_supplement(question: str) -> str | None:
    """The one Part 820 section a question asks the FDA additions of, or None."""
    named = set(_PART820_SECTION.findall(question or ""))
    if len(named) != 1 or not _STANDARD_NAMED.search(question or "") or not _SUPPLEMENT_ASKED.search(question or ""):
        return None
    return named.pop()


def compose_supplement(package, question: str) -> str | None:
    """The SU1 answer (see above), or None."""
    from regulus.rag.generation import _verbatim, whole_sentences

    sec = asks_supplement(question)
    if package is None or sec is None:
        return None
    own = [s for s in package.sources if not s.is_docket and re.fullmatch(rf"21 CFR {re.escape(sec)}(?:\([^)\s]+\))*",
                                                                           (s.citation or "").strip())]
    lead = next((s for s in own if (s.citation or "").strip() == f"21 CFR {sec}"), None)
    m = _IN_ADDITION.match(lead.text or "") if lead is not None else None
    if m is None:
        return None
    clauses = re.split(r",\s*|,?\s+and\s+", m.group(1))
    named = ("Clause " + clauses[0]) if len(clauses) == 1 else ("Clauses " + ", ".join(clauses[:-1]) + " and " + clauses[-1])
    lines = [SUPPLEMENT_LEAD.format(sec=sec, clauses=named)]
    for s in sorted(own, key=lambda s: s.index):
        body = whole_sentences(s.text or "")
        if body:
            lines.append(f"{s.citation} (eCFR, binding, verbatim): {_verbatim(body, s.index)}")
    return "\n".join(lines) if len(lines) > 1 else None
