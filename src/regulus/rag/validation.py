"""Deterministic post-generation validation (Stage 4).

Checks a generated answer against the GenerationPackage (requested
provisions, selected sources with tiers, coverage) and the generation
result. Everything here is decidable from identities and structure:

    requirement identity     is each requested provision addressed?
    citation validity        does every [n] point at a supplied source?
    evidence association     does a statement about R1 cite a source
                             related to R1, and never a DISCOVERY source?
    tier safety              is a provision with no EXACT source asserted
                             as if it existed?
    output integrity         truncation, empty output, degenerate repetition
    answer shape             Stage 12, when the package carries an AnswerPlan:
                             see regulus.rag.shape_validation

What it cannot decide is semantic entailment: that a paraphrase of a
source is faithful. Keyword overlap is not entailment and is not used as
such. The report says what was checked; nothing here claims the answer is
"correct".
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from regulus.evaluation.grounding_eval import REFUSAL, classify_answer, split_sentences
from regulus.rag.validator_switches import enabled as precision_fix
from regulus.inference.types import GenerationResult
from regulus.rag.generation import GenerationPackage, SHAPE_DOCKET, SHAPE_GENERAL
from regulus.rag.shape_validation import APPLICABILITY_DROPPED, ENUMERATION_INCOMPLETE, MATERIAL_GROUP_UNCOVERED, PROMPT_ECHO, check_shape
from regulus.retrieval.evidence import TIER_DISCOVERY, TIER_EXACT, TIER_STRUCTURAL, TIER_SUPPORTING
from regulus.retrieval.references import parse_cfr_references

STATUS_VALID = "valid"
STATUS_INCOMPLETE = "incomplete"
STATUS_UNSUPPORTED = "unsupported"
STATUS_CITATION_INVALID = "citation_invalid"
STATUS_EVIDENCE_MISMATCHED = "evidence_mismatched"
STATUS_DEGENERATE = "degenerate"
STATUS_EMPTY = "empty"
# most severe first; the report status is the first one present
_STATUS_ORDER = (STATUS_EMPTY, STATUS_DEGENERATE, STATUS_UNSUPPORTED, STATUS_CITATION_INVALID, STATUS_EVIDENCE_MISMATCHED, STATUS_INCOMPLETE)

# problem codes
EMPTY_OUTPUT = "empty_output"
DEGENERATE_OUTPUT = "degenerate_output"
TRUNCATED = "truncated"
REQUIREMENT_NOT_ADDRESSED = "requirement_not_addressed"
REFUSAL_WITH_EVIDENCE = "refusal_with_evidence"
ASSERTED_MISSING_PROVISION = "asserted_missing_provision"
INVALID_CITATION = "invalid_citation"
MALFORMED_CITATION = "malformed_citation"
CITATION_MISMATCH = "citation_mismatch"
DISCOVERY_CITED_FOR_REQUIREMENT = "discovery_cited_for_requirement"
UNCITED_STATEMENT = "uncited_statement"
UNSUPPORTED_FIGURE = "unsupported_figure"                 # a number/amount that appears in no supplied source
PROVISION_CITATION_MISMATCH = "provision_citation_mismatch"  # names provision X, cites a source that is not X
LOW_OVERLAP = "low_overlap_with_cited_source"             # warning only: few content words shared with the cited exact source

from regulus.inference.repetition_text import normalise_sentence  # noqa: E402
from regulus.rag.claim_validation import (  # noqa: E402
    CONDITION_NOT_IN_CITED,
    FALSE_DENIAL,
    FIGURE_MISATTRIBUTED,
    INVENTED_CONDITION,
    OBLIGATION_STRENGTHENED,
    PARAGRAPH_MISATTRIBUTION,
    UNSUPPORTED_CLAIM,
    UNSUPPORTED_PREMISE,
    check_claims,
)

_CODE_STATUS = {
    EMPTY_OUTPUT: STATUS_EMPTY,
    DEGENERATE_OUTPUT: STATUS_DEGENERATE,
    ASSERTED_MISSING_PROVISION: STATUS_UNSUPPORTED,
    DISCOVERY_CITED_FOR_REQUIREMENT: STATUS_UNSUPPORTED,
    INVALID_CITATION: STATUS_CITATION_INVALID,
    CITATION_MISMATCH: STATUS_EVIDENCE_MISMATCHED,
    PROVISION_CITATION_MISMATCH: STATUS_EVIDENCE_MISMATCHED,
    UNSUPPORTED_FIGURE: STATUS_UNSUPPORTED,
    REQUIREMENT_NOT_ADDRESSED: STATUS_INCOMPLETE,
    REFUSAL_WITH_EVIDENCE: STATUS_INCOMPLETE,
    TRUNCATED: STATUS_INCOMPLETE,
    ENUMERATION_INCOMPLETE: STATUS_INCOMPLETE,  # Stage 12: an explicitly requested item is missing
    MATERIAL_GROUP_UNCOVERED: STATUS_INCOMPLETE,  # Stage 12: a synthesis left out a whole requirement group
    PROMPT_ECHO: STATUS_DEGENERATE,             # Stage 12: the answer copies the prompt's instructions
    # Stage 7 claim-level checks (regulus.rag.claim_validation); a status only when enforced as errors
    PARAGRAPH_MISATTRIBUTION: STATUS_EVIDENCE_MISMATCHED,
    FIGURE_MISATTRIBUTED: STATUS_EVIDENCE_MISMATCHED,   # FC2: the right figure under the wrong paragraph
    FALSE_DENIAL: STATUS_UNSUPPORTED,                    # FD1: denies what a supplied paragraph states
    INVENTED_CONDITION: STATUS_EVIDENCE_MISMATCHED,   # a removable statement, repaired like a mis-cited one
    UNSUPPORTED_PREMISE: STATUS_UNSUPPORTED,
    UNSUPPORTED_CLAIM: STATUS_UNSUPPORTED,
    CONDITION_NOT_IN_CITED: STATUS_UNSUPPORTED,
    OBLIGATION_STRENGTHENED: STATUS_UNSUPPORTED,   # E1: a duty its source only requests or recommends
    APPLICABILITY_DROPPED: STATUS_UNSUPPORTED,  # Stage 12 shape check, an error since the 16 Oct 2026 severity experiment
}

MARK_RE = re.compile(r"(?<![A-Za-z0-9])\[(\d{1,2})\]")
MALFORMED_MARK_RE = re.compile(r"\[(?:source|src|sources)?\s*\d{1,2}\s*(?:[-–,]\s*\d{1,2}\s*)+\]|\[(?:source|src)\s*\d{1,2}\]", re.IGNORECASE)
_OBLIGATION_RE = re.compile(r"\b(require[sd]?|must|shall|need(?:s|ed)?|mandate[sd]?|prohibit[sd]?|document|notify|handled?)\b", re.IGNORECASE)
_NEGATION_CUES = (
    "not found", "does not contain", "do not contain", "not contain", "not among", "cannot", "could not", "can not",
    "no exact", "unavailable", "not available", "not retrieved", "not present", "no text", "not in the", "no live record",
    "not provided", "not included", "not exist", "does not exist", "no source", "not supplied", "not covered",
)
_DEGENERATE_MIN_TOKENS = 30
_DEGENERATE_NGRAM = 4
_DEGENERATE_REPEAT_RATIO = 0.5
_DEGENERATE_SENTENCE_REPEATS = 3
_FIGURE_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|(?<![\w.(\[])\d{2,}(?:[.,]\d+)*(?![\w(])")
# A regulatory period carries its own unit, so it is a figure however few digits it has:
# "_FIGURE_RE" needs two of them, which made "7 calendar days" invisible to the fabrication
# check. Asked to contrast the 15- and 7-calendar-day deadlines of 21 CFR 312.32 with (c)(2)
# missing from its evidence, the model asserted a 7-calendar-day deadline that no supplied source
# stated, and the check passed it.
_PERIOD_RE = re.compile(r"(?<![\w.])(\d{1,3})(?:\.\d+)?[\s-]*(?:calendar|business|working)?[\s-]*"
                        r"(?:days?|weeks?|months?|years?|hours?|minutes?)\b", re.I)
_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                 "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12"}
_WORD_PERIOD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")[\s-]+(?:calendar|business|working)?[\s-]*"
                             r"(?:days?|weeks?|months?|years?|hours?|minutes?)\b", re.I)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "must", "shall", "are", "was", "were", "have", "has", "not", "its",
    "any", "all", "such", "from", "into", "under", "per", "requires", "required", "require", "requirement", "requirements",
    "cfr", "section", "paragraph", "provision", "chapter", "part", "accordance", "applicable", "manufacturer", "manufacturers",
    "states", "state", "specifies", "specify", "provides", "provide", "which", "when", "where", "then", "than", "also",
    "ecfr", "binding", "regulation", "guidance", "clause", "clauses", "iso", "subclause",
}
LOW_OVERLAP_THRESHOLD = 0.4
# (?<![\d.,$]): the "500.00" inside "$2,500.00" is an amount, not section 500.00 (withheld 70.19 drafts, 26 Oct)
_CFR_CITE_RE = re.compile(r"(?:§§?\s*|\b21\s+CFR\s+(?:§\s*)?)?(?<![\d.,$])(\d{1,4}\.\d{1,5}[a-z]?)(?:\s*\([A-Za-z0-9]{1,6}\))*", re.I)
_ACT_CITE_RE = re.compile(r"(?:§§?\s*|\bsection\s+|\bsec\.\s*|\bU\.?S\.?C\.?\s*§*\s*)(\d{3}[a-z]{0,2})(?:\s*\([A-Za-z0-9]{1,6}\))*", re.I)


def _strip_supplied_citations(text: str, supplied_text: str) -> str:
    """Citations whose number the evidence itself carries, taken out of the text the figure check reads (fix A1).

    '§74.705(a)(1)' in a draft reads as the figure 74 (the figure pattern cannot take 74.705 before a parenthesis),
    while the source's '§ 74.705 (a)(1)' reads as 74.705, so a faithful draft was withheld for an 'unsupported figure
    74' (82.705, 82.1328 on the 22 Oct holdout). A section number the evidence does not carry is left in place, so an
    invented citation is still caught; amounts and currency are never touched. FD&C Act section numbers ('§ 605(e)')
    are still read as figures: on the blind A/B (23 Oct) the false figure 605 was the only finding on a misleading
    draft (FDCA605-Q1, an invented reporting route); they wait for the misstatement checks (failure analysis fix E)."""
    def drop(m: re.Match) -> str:
        return " " if re.search(rf"(?<![\d.]){re.escape(m.group(1))}(?![\d])", supplied_text) else m.group(0)

    return _CFR_CITE_RE.sub(drop, text)


_QUOTED = re.compile(r"[\"\u201c]([^\"\u201c\u201d]{20,})[\"\u201d]")


def _flat_text(text: str) -> str:
    # markers out (attach_trailing_markers moves "[2]" inside a closing quote), spaces before punctuation closed up
    return re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\s+", " ", MARK_RE.sub(" ", text or ""))).strip().lower()


def _without_cited_quotes(sentence: str, cited_texts: list[str]) -> str:
    """The sentence without the quotations that are its cited sources' own words (fix C82): a provision those words name
    ("... conforms to the requirements of § 82.5 ..." quoted from 82.50) is the source speaking, not a provision the
    sentence attributes to the wrong source. A paraphrase, or a quotation the cited text does not contain, is kept."""
    flat = [_flat_text(t) for t in cited_texts if t]

    def drop(m: re.Match) -> str:
        inner = _flat_text(m.group(1)).rstrip(" .")
        return " " if any(inner in f for f in flat) else m.group(0)

    return _QUOTED.sub(drop, sentence)


@dataclass(frozen=True)
class ValidationProblem:
    code: str
    severity: str                  # "error" | "warning"
    detail: str
    requirement_id: str | None = None
    citation: int | None = None
    sentence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity, "detail": self.detail, "requirement_id": self.requirement_id,
                "citation": self.citation, "sentence": self.sentence}


@dataclass(frozen=True)
class ValidationReport:
    status: str
    problems: tuple[ValidationProblem, ...]
    addressed: tuple[str, ...]          # satisfied requirements the answer addresses
    unaddressed: tuple[str, ...]        # satisfied requirements the answer leaves out
    cited: tuple[int, ...]              # citation markers found, in order of first appearance
    invalid_citations: tuple[int, ...]
    truncated: bool
    finished: bool | None
    generated_tokens: int | None
    max_new_tokens: int | None
    answer_class: str                   # refusal | hedged | answer (project classifier)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_VALID

    @property
    def errors(self) -> tuple[ValidationProblem, ...]:
        return tuple(p for p in self.problems if p.severity == "error")

    def codes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(p.code for p in self.problems))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.ok,
            "codes": list(self.codes()),
            "problems": [p.to_dict() for p in self.problems],
            "addressed": list(self.addressed),
            "unaddressed": list(self.unaddressed),
            "cited": list(self.cited),
            "invalid_citations": list(self.invalid_citations),
            "truncated": self.truncated,
            "finished": self.finished,
            "generated_tokens": self.generated_tokens,
            "max_new_tokens": self.max_new_tokens,
            "answer_class": self.answer_class,
            **self.diagnostics,
        }


# ------------------------------------------------------------ mentions


def _mention_patterns(package: GenerationPackage) -> dict[str, list[re.Pattern[str]]]:
    """Regexes that detect a mention of each requested provision in the answer.

    Full form always ("820.10(b)(2)", spaces tolerated). The short paragraph
    form ("(b)(2)") only when every requested provision is in the same
    section, so it cannot be ambiguous.
    """
    labels = package.requirement_labels
    sections: set[str] = set()
    parsed: dict[str, tuple[str, str, str]] = {}
    for rid, label in labels.items():
        m = re.match(r"(\d{1,2}) CFR (?:part (\d+)|(\d+\.\d+[a-z]?)((?:\([A-Za-z0-9]+\))*))$", label)
        if not m:
            continue
        if m.group(2):
            parsed[rid] = ("part", m.group(2), "")
        else:
            parsed[rid] = ("section", m.group(3), m.group(4) or "")
            sections.add(m.group(3))
    same_section = len(sections) == 1
    out: dict[str, list[re.Pattern[str]]] = {}
    for rid, (kind, number, para) in parsed.items():
        pats: list[re.Pattern[str]] = []
        if kind == "part":
            pats.append(re.compile(rf"\bpart\s+{re.escape(number)}\b", re.IGNORECASE))
        else:
            para_re = "".join(rf"\s*\(\s*{re.escape(t)}\s*\)" for t in re.findall(r"\(([A-Za-z0-9]+)\)", para))
            if para:
                pats.append(re.compile(rf"(?<![\d.]){re.escape(number)}{para_re}(?!\s*\()"))
                if same_section:
                    pats.append(re.compile(rf"(?<![\w)]){para_re.lstrip()}(?!\s*\()"))
            else:
                pats.append(re.compile(rf"(?<![\d.]){re.escape(number)}(?![\d.]|\s*\()"))
        out[rid] = pats
    return out


def _mentions(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(text) for p in patterns)


_TRAILING_MARKS_RE = re.compile(r"([.!?])([\"\u201d')]?)((?:\s*\[\d{1,2}\])+)")


def attach_trailing_markers(text: str) -> str:
    """Move markers written after a full stop ("... chapter. [1]") back inside the
    sentence so they attach to the statement they follow, not the next one."""
    return _TRAILING_MARKS_RE.sub(lambda m: " " + " ".join(re.findall(r"\[\d{1,2}\]", m.group(3))) + m.group(1) + m.group(2), text)


_EVIDENCE_DOES_NOT_GOVERN = re.compile(r"\b(?:do(?:es)? not|cannot|can not|is not|are not)\s+(?:\w+\s+){0,3}?(?:govern|cover|address|"
                                       r"apply|regulate|establish|reach|concern|deal with)\b|\bnot (?:governed|covered|addressed) by\b", re.I)
# A3 (regulus.rag.validator_switches): sentence findings a whole-sentence cut can answer
EXCISABLE_CODES = frozenset({"unsupported_claim", "invented_condition"})
REPEAT_OVERLAP = 0.8   # share of a flagged sentence's content words an earlier passing sentence already states
EXCISE_MAX = 2         # filler is one or two sentences; a draft with more unsupported sentences is off its evidence


def excise_unsupported_sentences(text: str, report: "ValidationReport", question: str | None = None) -> tuple[str, int] | None:
    """The draft without its unsupported filler, or None (fix A3).

    Trimmed only when EVERY error is an unsupported_claim or invented_condition finding on a whole sentence that is not
    the first (verdict) sentence and either carries no citation or repeats what a passing sentence says ("The use in
    body washes is not addressed in any provision, so it is not permitted" after a cited verdict that body wash is not
    permitted). Sentences go whole - never a clause inside one, the route that once made a conditional requirement
    unconditional (invented_conditions_removed). Guards, from the exact replay of 170 recorded answers (23 Oct): a
    draft whose unsupported sentences ARE its answer is a draft about the wrong section (FDCA601-Q2, UC-002, UC-009:
    "None of these provisions address contamination by rodent droppings"), and cutting them would deliver it as valid.
    So at most EXCISE_MAX sentences and at most a third of the substantive ones; every question word a removed
    sentence speaks to must still be spoken to by one that stays; what stays must cite a source and must not open by
    declining. The caller validates the rest again, completeness included, and uses it only if it validates."""
    errors = report.errors
    if not errors or any(p.code not in EXCISABLE_CODES or not p.sentence for p in errors):
        return None
    if report.truncated or report.finished is False:
        return None   # a cut-off draft is not trimmed into a valid one (701.13-G1: repetitive and truncated, blind A/B)
    attached = attach_trailing_markers(text)
    sentences = split_sentences(attached)
    if not sentences or _EVIDENCE_DOES_NOT_GOVERN.search(sentences[0]):
        # the verdict itself says the supplied provision does not govern the question: the evidence is the wrong
        # section, and the honest outcome is the withheld answer (UC-015: "cannot establish compliance under 700.27")
        return None
    flagged = list(dict.fromkeys(p.sentence for p in errors))
    words = lambda s: _content_words(MARK_RE.sub(" ", s))  # noqa: E731
    substantive = [s for s in sentences if len(words(s)) >= 4]
    if len(flagged) > EXCISE_MAX or 3 * len(flagged) > max(len(substantive), 1):
        return None
    kept = [s for s in sentences if s not in flagged]
    if not kept or not any(MARK_RE.search(s) for s in kept) or classify_answer(kept[0]) == REFUSAL:
        return None
    kept_words = set().union(*(words(s) for s in kept))
    asked = _content_words(question or "")
    for s in flagged:
        if s.strip() == sentences[0].strip() or s not in attached:
            return None
        w = words(s)
        repeats = bool(w) and any(len(w & words(q)) / len(w) >= REPEAT_OVERLAP for q in kept)
        if MARK_RE.search(s) and not repeats:
            return None   # a cited claim the evidence does not support is a real error, not filler
        if (w & asked) - kept_words:
            return None   # it is the only sentence that speaks to part of the question
    out = attached
    for s in flagged:
        out = out.replace(s, "", 1)
    out = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", out)).strip()
    return (out, len(flagged)) if out else None


def _is_negated(sentence: str) -> bool:
    s = sentence.lower()
    return any(cue in s for cue in _NEGATION_CUES)


def _normalise_figure(token: str) -> str:
    out = token.replace("$", "").replace(",", "").replace(" ", "").strip(".")
    if precision_fix("MF1"):
        # MF1: "$75" is the source's "$75.00" (70.19(l), 27 Oct: a correct deposit answer was withheld over it)
        out = re.sub(r"(?<=\d)\.0+$", "", out)
    return out


def _figures(text: str) -> set[str]:
    return {_normalise_figure(t) for t in _FIGURE_RE.findall(text)} - {""}


def _periods(text: str) -> set[str]:
    """Numbers that measure a period of time in `text` ("7 calendar days" -> "7")."""
    return {m.group(1).lstrip("0") or "0" for m in _PERIOD_RE.finditer(text or "")}


def _supplied_figures(text: str) -> set[str]:
    """Every number a source can be said to supply: the figures proper, the periods it states, and
    the periods it spells out, so an answer that writes "7 calendar days" for a source's "seven
    calendar days" is not accused of inventing it.

    Counting every digit run instead - paragraph numbering included - was measured over the 300
    recorded acceptance answers to suppress seven findings the old rule made, one of them on a row
    the rubric scores N=2; this form keeps all of them and adds two, both on N=0 rows."""
    out = (_figures(text) | _periods(text)
           | {_NUMBER_WORDS[m.group(1).lower()] for m in _WORD_PERIOD_RE.finditer(text or "")})
    if precision_fix("AF1"):
        # AF1 (27 Oct): a section the source names with its paragraph ("section 601(c) of the act") supplies its number;
        # the figure pattern skips a number followed by "(", so an answer's "§ 601" read as invented (700.27-Q1 withheld)
        out |= set(_SECTION_WITH_PARAGRAPH.findall(text or ""))
    return out


_SECTION_WITH_PARAGRAPH = re.compile(r"(?<![\w.(\[])(\d{2,})(?=\((?:[a-z]|\d{1,2})\))")

# AR1 (27 Oct): a figure the answer computes in plain sight - "2,000 pounds x $0.45 per pound = $900" (80.10-Q1, a
# correct fee withheld as an invented figure) - is supported when both operands are supplied and the arithmetic holds
_NUM = r"\$?\s?(\d[\d,]*(?:\.\d+)?)"
_UNITS = r"(?:\s+(?:per\s+)?[A-Za-z]+){0,3}"
_ARITHMETIC = re.compile(_NUM + _UNITS + r"\s*(×|x|\*|times|multiplied\s+by|\+|plus|−|-|minus|/|÷|divided\s+by)\s*" + _NUM
                         + _UNITS + r"\s*(?:=|equals|comes\s+to|totals?|gives)\s*" + _NUM, re.I)


def verified_arithmetic(text: str, supplied: set[str]) -> set[str]:
    """The results of the answer's explicit calculations whose operands are supplied figures and whose arithmetic is
    right (to half a percent, as a stated amount is rounded)."""
    out = set()
    for a, op, b, c in _ARITHMETIC.findall(text or ""):
        na, nb, nc = (_normalise_figure(x) for x in (a, b, c))
        if na not in supplied or nb not in supplied:
            continue
        try:
            x, y, z = float(na), float(nb), float(nc)
        except ValueError:
            continue
        op = op.lower()
        if op in ("×", "x", "*", "times") or op.startswith("multiplied"):
            r = x * y
        elif op in ("+", "plus"):
            r = x + y
        elif op in ("−", "-", "minus"):
            r = x - y
        elif y:
            r = x / y
        else:
            continue
        if abs(r - z) <= max(0.005 * abs(z), 0.01):
            out.add(nc)
    return out


_UNIT_AFTER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*-?\s*(%|percent|ppm|parts per million|mg|g|kg|ml|fl\.? oz|oz|ounces?|"
                         r"square inches|inches|business days?|calendar days?|working days?|days?|weeks?|months?|years?|"
                         r"hours?|minutes?)(?![A-Za-z])", re.I)


def _question_figures(question: str, source_texts) -> set[str]:
    """Figures of the question that an answer may repeat as facts: every figure of the question except one written
    with a unit that a source states with a different number."""
    src_units: dict[str, set[str]] = {}
    for text in source_texts:
        for num, unit in _UNIT_AFTER.findall(text or ""):
            src_units.setdefault(_unit_key(unit), set()).add(_normalise_figure(num))
    out = set()
    for f in _supplied_figures(question):
        units = {_unit_key(u) for n, u in _UNIT_AFTER.findall(question or "") if _normalise_figure(n) == f}
        if any(src_units.get(u) and f not in src_units[u] for u in units):
            continue
        out.add(f)
    return out


def _unit_key(unit: str) -> str:
    u = unit.lower().rstrip("s").replace(".", "")
    return {"%": "percent", "parts per million": "ppm", "ounce": "oz", "fl oz": "fl oz", "business day": "day",
            "calendar day": "day", "working day": "day", "square inche": "square inch", "inche": "inch"}.get(u, u)


_CL1_LABEL = re.compile(r"(?:21\s*CFR\s*)?(?:Part\s+\d+|§?\s*\d+\.\d+[a-z]?(?:\s*\([A-Za-z0-9]+\))*)|\[\d+\]", re.I)


# words that point at a citation without saying anything ("See 21 CFR 820.45 [8]")
_CL1_POINTERS = frozenset("see also and under per cfr part parts section sections paragraph paragraphs subpart".split())
CL1_MIN_WORDS = 2   # a line saying something keeps at least this many words once its labels are removed


def citations_only(text: str) -> bool:
    """CL1: two or more lines, none of which says anything once its citation labels and source markers are removed.
    A terse statement says something ("21 CFR 820.10(b)(4): advisory notices. [2]" - 30 Oct: the first threshold of four
    words read it as a bare label); M-042's "21 CFR 820.45(a)(1) [8]" lines keep no word at all."""
    lines = [l for l in (text or "").splitlines() if l.strip()]
    if len(lines) < 2:
        return False
    def says(line: str) -> bool:
        words = [w for w in re.findall(r"[A-Za-z]{3,}", _CL1_LABEL.sub(" ", line)) if w.lower() not in _CL1_POINTERS]
        return len(words) >= CL1_MIN_WORDS
    return not any(says(l) for l in lines)


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)} - _STOPWORDS


def _source_identity(source) -> tuple[str, str | None, str] | None:
    f = source.citation_fields or {}
    if source.corpus != "ecfr" or not f.get("section"):
        return None
    return (str(f.get("title") or "21"), f.get("section"), f.get("paragraph") or "")


def _related_identity(named: tuple[str, str | None, str], have: tuple[str, str | None, str]) -> bool:
    """Same provision, or its parent/child (same section, one path a prefix of the other)."""
    if named[0] != have[0] or named[1] != have[1]:
        return False
    a, b = named[2], have[2]
    return a == b or not a or not b or a.startswith(b + "(") or b.startswith(a + "(")


REPOINT_MIN_SUPPORT = 0.4   # A4: share of the sentence's content words its section's supplied text must state ...
REPOINT_MIN_OWN = 0.25      # ... and the named provision's own text (a one-line paragraph reads with its siblings:
                            # 71.30(b) "The fees specified in § 70.19 of this chapter shall be applicable")


def repoint_named_citations(text: str, package) -> tuple[str, int]:
    """(text, markers re-pointed): fix A4 (regulus.rag.validator_switches).

    A sentence that names one provision ("21 CFR 71.30(b) requires that the fees specified in § 70.19 ...") but cites a
    source that is neither it nor its parent/child - the model numbered its citation by a neighbouring source or by the
    paragraph label ("(a)(5)" -> [5]) - has its marker(s) replaced by the one supplied source that IS that provision,
    when exactly one is and the sentence's content words are stated by it (REPOINT_MIN_OWN) and by its section's
    supplied text (REPOINT_MIN_SUPPORT). Only the citation number changes; the words are the model's. Anything less
    certain is left to provision_citation_mismatch."""
    identities = {src.index: _source_identity(src) for src in package.sources}
    by_index = {s.index: s for s in package.sources}
    attached = attach_trailing_markers(text)
    out, n = attached, 0
    for sentence in split_sentences(attached):
        marks = [int(m) for m in MARK_RE.findall(sentence) if int(m) in by_index]
        if not marks:
            continue
        # the provision a sentence presents is the one it names first, at its start ("21 CFR 71.30(b) requires that the
        # fees specified in § 70.19 ...": 71.30(b), not 70.19) - the same reading as claim validation's _LEAD
        named = {(r.title, r.section, r.paragraph_path) for r in parse_cfr_references(MARK_RE.sub(" ", sentence)[:45])
                 if r.section is not None}
        if len(named) != 1:
            continue
        nm = next(iter(named))
        if any(identities.get(i) and _related_identity(nm, identities[i]) for i in marks):
            continue
        holders = [i for i, ident in identities.items() if ident == nm]
        if len(holders) != 1:
            continue
        words = _content_words(MARK_RE.sub(" ", sentence))
        section = " ".join(by_index[i].text for i, ident in identities.items() if ident and ident[:2] == nm[:2])
        if not words or len(words & _content_words(by_index[holders[0]].text)) / len(words) < REPOINT_MIN_OWN \
                or len(words & _content_words(section)) / len(words) < REPOINT_MIN_SUPPORT:
            continue
        first = MARK_RE.search(sentence)
        fixed = sentence[:first.start()] + f"[{holders[0]}]" + MARK_RE.sub("", sentence[first.end():])
        if fixed != sentence and sentence in out:
            out, n = out.replace(sentence, fixed, 1), n + 1
    return (out, n) if n else (text, 0)


def _degenerate(text: str) -> tuple[bool, str]:
    # sentences compared normalised (citation markers, case, spacing): a sentence repeated with an
    # incrementing citation number is still the same sentence (regulus.inference.repetition_text)
    sentences = [normalise_sentence(s) for s in split_sentences(text)]
    sentences = [s for s in sentences if len(s) >= 20]
    if sentences:
        top, n = Counter(sentences).most_common(1)[0]
        if n >= _DEGENERATE_SENTENCE_REPEATS:
            return True, f"sentence repeated {n} times: {top[:60]!r}"
    tokens = text.split()
    if len(tokens) >= _DEGENERATE_MIN_TOKENS:
        grams = [" ".join(tokens[i:i + _DEGENERATE_NGRAM]) for i in range(len(tokens) - _DEGENERATE_NGRAM + 1)]
        repeated = sum(1 for g, c in Counter(grams).items() if c > 1 for _ in range(c - 1))
        ratio = repeated / max(len(grams), 1)
        if ratio > _DEGENERATE_REPEAT_RATIO:
            return True, f"{ratio:.0%} of 4-grams are repeats"
    return False, ""


# ------------------------------------------------------------ validate


def _is_yes_no(question: str | None) -> bool:
    if not question:
        return False
    from regulus.planning import plan_query
    from regulus.planning.intent import analyze_question

    return analyze_question(plan_query(question)).is_yes_no


def validate_answer(answer: str, package: GenerationPackage, generation: GenerationResult | None = None) -> ValidationReport:
    """Deterministic checks of `answer` against the package it was generated from."""
    text = (answer or "").strip()
    problems: list[ValidationProblem] = []
    n_sources = len(package.sources)
    by_index = {s.index: s for s in package.sources}
    explicit = package.shape not in (SHAPE_GENERAL, SHAPE_DOCKET) and bool(package.requirements)

    # -- output integrity
    finished = generation.finished if generation is not None else None
    generated = generation.generated_tokens if generation is not None else None
    truncated = False
    if generation is not None:
        if finished is False:
            truncated = True
        elif finished is None and generated is not None and package.max_new_tokens and generated >= package.max_new_tokens:
            truncated = True
    if not text:
        problems.append(ValidationProblem(EMPTY_OUTPUT, "error", "the model produced no text"))
    else:
        degenerate, why = _degenerate(text)
        if degenerate:
            problems.append(ValidationProblem(DEGENERATE_OUTPUT, "error", why))
        elif precision_fix("CL1") and citations_only(text):
            # CL1 (29 Oct): M-042 delivered fifteen lines of "21 CFR 820.45(a)(1) [8]" and nothing else - every check
            # passed, since nothing was asserted
            problems.append(ValidationProblem(DEGENERATE_OUTPUT, "error", "the answer lists citations but states nothing"))
    if truncated:
        # Naming *which* limit was hit is the difference between an actionable
        # report and a shrug: an allowance below the ceiling can be raised for
        # this shape of question, whereas the ceiling itself is the pathology
        # guard and hitting it usually means the model was not stopping.
        at_ceiling = package.max_new_tokens >= package.budget.hard_max_tokens
        which = "the hard safety ceiling" if at_ceiling else "this question's allowance"
        problems.append(ValidationProblem(
            TRUNCATED, "error",
            f"generation stopped at {which} ({generated}/{package.max_new_tokens} tokens) "
            f"before the answer finished",
        ))

    # -- citations: every marker must point at a supplied source
    cited = tuple(dict.fromkeys(int(m) for m in MARK_RE.findall(text)))
    invalid = tuple(i for i in cited if i < 1 or i > n_sources)
    for i in invalid:
        problems.append(ValidationProblem(INVALID_CITATION, "error", f"[{i}] does not correspond to any supplied source (1..{n_sources})", citation=i))
    for m in MALFORMED_MARK_RE.findall(text):
        problems.append(ValidationProblem(MALFORMED_CITATION, "warning", f"citation syntax {m!r} is not the [n] form and was not counted"))

    # -- requirements
    answer_class = classify_answer(text) if text else REFUSAL
    patterns = _mention_patterns(package)
    addressed: list[str] = []
    unaddressed: list[str] = []
    sentences = split_sentences(attach_trailing_markers(text))
    if explicit:
        for rid in package.satisfied_ids:
            cites_exact = any(rid in by_index[i].requirement_ids and by_index[i].tier == TIER_EXACT for i in cited if i in by_index)
            if _mentions(text, patterns.get(rid, [])) or cites_exact:
                addressed.append(rid)
            else:
                unaddressed.append(rid)
                optional = package.requirement_labels[rid] in package.optional_labels
                problems.append(ValidationProblem(REQUIREMENT_NOT_ADDRESSED, "warning" if optional else "error", f"{package.requirement_labels[rid]} has an EXACT source but the answer does not address it", requirement_id=rid))
        if package.satisfied_ids and answer_class == REFUSAL and text:
            problems.append(ValidationProblem(REFUSAL_WITH_EVIDENCE, "error", "the answer declines although EXACT sources were supplied for requested provisions"))
        # a provision with no EXACT source may only be mentioned as not found
        missing_ids = [rid for rid, _ in package.requirements if rid not in package.satisfied_ids]
        for rid in missing_ids:
            for sentence in sentences:
                if _mentions(sentence, patterns.get(rid, [])) and not _is_negated(sentence):
                    problems.append(ValidationProblem(ASSERTED_MISSING_PROVISION, "error", f"{package.requirement_labels[rid]} has no EXACT source but is described as if it existed", requirement_id=rid, sentence=sentence))
                    break
        # a named provision must be supported by a source of that identity (or its parent/child)
        identities = {src.index: _source_identity(src) for src in package.sources}
        for sentence in sentences:
            marks = [int(m) for m in MARK_RE.findall(sentence) if int(m) in by_index]
            if not marks:
                continue
            named_text = (_without_cited_quotes(sentence, [by_index[i].text for i in marks]) if precision_fix("C82")
                          else sentence)
            for ref in parse_cfr_references(named_text):
                if ref.section is None:
                    continue
                named = (ref.title, ref.section, ref.paragraph_path)
                if not any(ident and _related_identity(named, ident) for ident in identities.values()):
                    continue  # the provision is not among the sources; handled by the missing-provision rule
                if precision_fix("CM1") and any(identities.get(i) and identities[i][1] != ref.section
                                                and re.search(rf"(?<![\d.]){re.escape(ref.section)}(?![\d])", by_index[i].text or "")
                                                for i in marks):
                    # CM1: the cited source names that provision itself ("The fee for listing a diluent under § 80.35
                    # ... shall be $250.00" in 70.19(j)): the sentence restates it, it does not mis-cite (27 Oct)
                    continue
                if not any(identities.get(i) and _related_identity(named, identities[i]) for i in marks):
                    # attribute the problem to the requested provision the sentence is about, when there is exactly one
                    about = [rid for rid in package.satisfied_ids if _mentions(sentence, patterns.get(rid, []))]
                    rid = about[0] if len(about) == 1 else None
                    problems.append(ValidationProblem(PROVISION_CITATION_MISMATCH, "error", f"the sentence names {ref.canonical} but cites sources that are not it or its parent/child", requirement_id=rid, citation=marks[0], sentence=sentence))
                    break
        # evidence association per sentence
        for sentence in sentences:
            marks = [int(m) for m in MARK_RE.findall(sentence)]
            mentioned = [rid for rid in package.satisfied_ids if _mentions(sentence, patterns.get(rid, []))]
            if len(mentioned) != 1:
                continue
            rid = mentioned[0]
            valid_marks = [i for i in marks if i in by_index]
            if not valid_marks:
                if _OBLIGATION_RE.search(sentence) and not _is_negated(sentence):
                    problems.append(ValidationProblem(UNCITED_STATEMENT, "warning", "statement about a requested provision carries no citation", requirement_id=rid, sentence=sentence))
                continue
            related = [i for i in valid_marks if rid in by_index[i].requirement_ids and by_index[i].tier in (TIER_EXACT, TIER_STRUCTURAL)]
            discovery = [i for i in valid_marks if by_index[i].tier == TIER_DISCOVERY]
            if discovery and not related:
                problems.append(ValidationProblem(DISCOVERY_CITED_FOR_REQUIREMENT, "error", f"a DISCOVERY source is cited as support for {package.requirement_labels[rid]}", requirement_id=rid, citation=discovery[0], sentence=sentence))
            elif not related:
                problems.append(ValidationProblem(CITATION_MISMATCH, "error", f"the sources cited for {package.requirement_labels[rid]} are not its EXACT or STRUCTURAL evidence", requirement_id=rid, citation=valid_marks[0], sentence=sentence))

    # figures (amounts, numbers of two or more digits) must come from a supplied source
    if text:
        supplied = set()
        for src in package.sources:
            supplied |= _supplied_figures(src.text) | _supplied_figures(src.citation)
            if precision_fix("A1"):
                supplied |= _supplied_figures(src.heading or "")   # a heading's numbers are supplied text too
        for rid, label in package.requirements:
            supplied |= _supplied_figures(label)
        figure_text = text
        if precision_fix("A1"):
            blob = "\n".join([s.text for s in package.sources] + [s.citation for s in package.sources]
                             + [s.heading or "" for s in package.sources] + [label for _, label in package.requirements]
                             + [package.question or ""])
            figure_text = _strip_supplied_citations(text, blob)
        if precision_fix("P1"):
            # the practice-silence answer's search sentence names the scope the pipeline searched ("21 CFR Parts
            # 700-740 and FD&C Act sections 601-614"), not a figure taken from the evidence (regulus.rag.practice_silence)
            from regulus.rag.practice_silence import SEARCH_SENTENCE_RE

            figure_text = SEARCH_SENTENCE_RE.sub(" ", figure_text)
        if precision_fix("FP1"):
            # the premise answer's first sentence quotes the QUESTION's figure to say the provision does not state it
            # (regulus.rag.premise_figures): not a figure taken from the evidence
            from regulus.rag.premise_figures import PREMISE_SENTENCE_RE

            figure_text = PREMISE_SENTENCE_RE.sub(" ", figure_text)
        if precision_fix("QM1"):
            # the QMSR answer's status and clause sentences (dates, clause numbers) come from the reference file
            from regulus.rag.qmsr import PIPELINE_SENTENCES_RE

            figure_text = PIPELINE_SENTENCES_RE.sub(" ", figure_text)
        if package.answer_style == "apply" and package.question:
            # the question's own figures ("section 721", "80 days", "12 fl oz") are facts it states - unless a source
            # states a different number in the same unit: "up to 10 percent" in the question is not supplied when
            # the source says "not more than 0.6 percent" (code review: that false number used to validate)
            supplied |= _question_figures(package.question, [s.text for s in package.sources]) - supplied
            if getattr(package, "earlier_question", None):   # a follow-up: the conversation's stated facts too
                supplied |= _question_figures(package.earlier_question, [s.text for s in package.sources]) - supplied
        if precision_fix("AR1"):
            supplied |= verified_arithmetic(figure_text, supplied)
        stray = sorted((_figures(figure_text) | _periods(figure_text)) - supplied
                       - {f for f in _figures(figure_text) | _periods(figure_text)
                          if any(f in sf or sf in f for sf in supplied if len(f) >= 3 and len(sf) >= 3)})
        if stray:
            problems.append(ValidationProblem(UNSUPPORTED_FIGURE, "error", f"figure(s) not found in any supplied source: {', '.join(stray[:6])}"))
        # low content-word overlap with the cited EXACT source: a warning, not proof of anything
        for sentence in sentences:
            marks = [int(m) for m in MARK_RE.findall(sentence) if int(m) in by_index]
            exact_cited = [by_index[i] for i in marks if by_index[i].tier == TIER_EXACT]
            if not exact_cited:
                continue
            words = _content_words(MARK_RE.sub("", sentence))
            if len(words) < 4:
                continue
            best = max(len(words & _content_words(src.text)) / len(words) for src in exact_cited)
            if best < LOW_OVERLAP_THRESHOLD:
                problems.append(ValidationProblem(LOW_OVERLAP, "warning", f"only {best:.0%} of the content words appear in the cited exact source", citation=marks[0], sentence=sentence))

    # claim level (Stage 7): does each statement's wording come from the evidence it presents?
    if text:
        for f in check_claims(sentences, package, question=package.question, is_yes_no=_is_yes_no(package.question),
                              identity=_source_identity, related=_related_identity):
            problems.append(ValidationProblem(f.code, f.severity, f.detail, requirement_id=f.requirement_id, citation=f.citation, sentence=f.sentence))

    # answer shape (Stage 12): warnings except a missing item of an explicit enumeration
    shape_diagnostics = None
    if text and package.answer_plan is not None:
        findings, shape_diagnostics = check_shape(attach_trailing_markers(text), package, package.answer_plan)
        problems.extend(ValidationProblem(f.code, f.severity, f.detail, citation=f.citation, sentence=f.sentence) for f in findings)

    status = STATUS_VALID
    present = {_CODE_STATUS.get(p.code) for p in problems if p.severity == "error"}
    for candidate in _STATUS_ORDER:
        if candidate in present:
            status = candidate
            break
    diagnostics = {
        "n_sources": n_sources,
        "explicit": explicit,
        "cited_tiers": {str(i): by_index[i].tier for i in cited if i in by_index},
        "warnings": sum(1 for p in problems if p.severity == "warning"),
        "errors": sum(1 for p in problems if p.severity == "error"),
    }
    if shape_diagnostics is not None:
        diagnostics["shape"] = shape_diagnostics
    return ValidationReport(
        status=status,
        problems=tuple(problems),
        addressed=tuple(addressed),
        unaddressed=tuple(unaddressed),
        cited=cited,
        invalid_citations=invalid,
        truncated=truncated,
        finished=finished,
        generated_tokens=generated,
        max_new_tokens=package.max_new_tokens,
        answer_class=answer_class,
        diagnostics=diagnostics,
    )


def strip_invalid_citations(answer: str, invalid: tuple[int, ...]) -> str:
    """Remove markers that point at no supplied source. Nothing is added."""
    if not invalid:
        return answer
    bad = set(invalid)
    out = MARK_RE.sub(lambda m: "" if int(m.group(1)) in bad else m.group(0), answer)
    return re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", out)).strip()
