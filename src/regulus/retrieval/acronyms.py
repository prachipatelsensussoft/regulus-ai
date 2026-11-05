"""The acronyms 21 CFR defines in its own text (switch SY4, 30 Oct 2026).

A question writing "RACC" shares no word with 101.12 ("Reference amounts customarily consumed per eating occasion"), so
the section was never a candidate (L-008, full300d: 101.12 missing, PARTIALLY PASS). data/reference/acronyms_v1.json
(scripts/build_acronyms.py) holds every acronym the regulations spell out before it in parentheses ("reference amount
commonly consumed (RACC)"), with the senses questions do not use excluded (PDP is a product development protocol in
the corpus and a principal display panel on a label), and ALIASES the programme names the CFR never defines in
parentheses ("510(k)", "PMA", "FSVP"). The expansion is added to the search words only: the picker and the model read
the question as asked.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).resolve().parents[3] / "data" / "reference" / "acronyms_v1.json"
_ACRONYM = re.compile(r"(?<![\w&])([A-Z][A-Z&]{2,7})s?(?![\w&])")
# The programme names FDA and industry use that the CFR never spells out in parentheses, with the words its headings and
# text do use (P-003, 30 Oct: "When must a 510(k) be submitted ..." never reached 807.81 "When a premarket notification
# submission is required"). Each expansion is in at least one 21 CFR heading or in the defining section's text.
ALIASES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"(?<!\w)510\s*\(k\)"), "premarket notification"),                     # 807 Subpart E, 13 headings
    (re.compile(r"(?<![\w&])PMAs?(?![\w&])"), "premarket approval"),                      # Part 814, 19 headings
    (re.compile(r"(?<![\w&])MDRs?(?![\w&])"), "medical device reporting"),                # Part 803
    (re.compile(r"(?<![\w&])FSVPs?(?![\w&])"), "foreign supplier verification programs"), # Part 1 Subpart L
    (re.compile(r"(?<![\w&])PCQIs?(?![\w&])"), "preventive controls qualified individual"),  # 117.180, 507.53
    (re.compile(r"(?<![\w&])PMTAs?(?![\w&])"), "premarket tobacco product application"),     # Part 1114
    (re.compile(r"(?<![\w&])MRTPA?s?(?![\w&])"), "modified risk tobacco product"),        # Part 1114, FD&C Act 911
    (re.compile(r"(?<![\w&])CAPAs?(?![\w&])"), "corrective and preventive action"),      # 820 (ISO 13485 8.5.2)
)


# an expansion used in more sections than this is common, not specific: it dilutes the search. SY4 calibration on the
# 300Q (30 Oct): "food and drug administration" (906 sections), "current good manufacturing practice" (507) and "good
# manufacturing practice" (860) cost F-003 its rank-1 key 117.10 and G-006 201.17; "reference amount commonly consumed"
# (1), "investigational device exemption" (31) brought L-008's 101.12 and P-010's 812.43 / 812.46
MAX_SECTIONS = 50


@lru_cache(maxsize=1)
def _data() -> dict:
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def table() -> dict[str, str]:
    """The specific acronyms: defined in the corpus, expansion used in at most MAX_SECTIONS sections."""
    try:
        return {k: v["expansion"] for k, v in _data()["acronyms"].items() if v.get("sections", 0) <= MAX_SECTIONS}
    except (KeyError, TypeError, AttributeError):
        return {}


def _specific_alias(words: str) -> bool:
    return (_data().get("alias_sections") or {}).get(words, 0) <= MAX_SECTIONS


def expansions(question: str) -> list[str]:
    """The expansions of the defined acronyms the question writes in capitals and does not already spell out."""
    known, q = table(), (question or "").lower()
    out = []
    for acronym in dict.fromkeys(_ACRONYM.findall(question or "")):
        words = known.get(acronym)
        if words and words not in q and words not in out:
            out.append(words)
    for pattern, words in ALIASES:
        if pattern.search(question or "") and _specific_alias(words) and words not in q and words not in out:
            out.append(words)
    return out


# SY4H (5 Nov 2026): an alias whose expansion is too common in body text to search on (MAX_SECTIONS) may still be specific
# in the HEADINGS of its programme's own part - "premarket notification" is written in 1,158 sections (every device
# classification says "exempt from premarket notification procedures"), but in Part 807 only the headings of Subpart E
# carry it (807.81 "When a premarket notification submission is required", 807.87, 807.90, 807.100 ...). P-003 ("When
# must a 510(k) be submitted for a change ...?") never had 807.81 as a candidate. Field-weighted: the expansion is matched
# against the home part's section headings only.
# Only where the home headings use the expansion rather than the acronym: Part 814's headings say "PMA" itself (814.37,
# 814.39 ...), which the question's own word already matches; its one "premarket approval" heading is 814.9 (confidentiality)
ALIAS_HOMES: dict[str, tuple[str, ...]] = {
    "premarket notification": ("807",),
}


def heading_aliases(question: str) -> list[tuple[str, tuple[str, ...]]]:
    """(expansion, home parts) of each alias the question writes whose expansion SY4 leaves out as too common in body
    text, and that the question does not already spell out."""
    q = (question or "").lower()
    out = []
    for pattern, words in ALIASES:
        if words in ALIAS_HOMES and pattern.search(question or "") and not _specific_alias(words) and words not in q:
            out.append((words, ALIAS_HOMES[words]))
    return out
