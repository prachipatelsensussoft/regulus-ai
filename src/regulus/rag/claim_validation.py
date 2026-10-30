"""Claim-level validation (Stage 7): does each statement's content come from the evidence it names?

Existing validation checks that citations exist and point at the right provision. These checks
look at the WORDS of each statement:

  paragraph_misattribution  a statement presents provision X, but its wording is copied from a
                            different, unrelated supplied provision Y and not from X
                            (source "(a) A. (b) B." -> answer "(b): A.")
  invented_condition        a statement carries a condition ("if applicable", "as appropriate",
                            "unless", "except", "to the extent") that none of the sources it cites
                            contains (source "(b) B." -> answer "(b) B, except when X.")
  unsupported_premise       a yes/no answer affirms terms of the question that no supplied source
                            contains ("Does X require annual audits?" -> "X requires annual audits.")
  unsupported_claim         a statement whose content words mostly appear in NO supplied source

Dropped conditions are already detected by shape validation (`qualifier_dropped`); this module does
not duplicate that check. All checks are lexical and deterministic: they detect copying and absence,
not meaning, and every threshold is reported with the finding. Severities are decided by
`SEVERITY` below from Stage 7 measurements (docs/retrieval_validation_stage7.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from regulus.rag.answer_plan import QUALIFIER_EQUIVALENTS
from regulus.rag.validator_switches import enabled as precision_fix

PARAGRAPH_MISATTRIBUTION = "paragraph_misattribution"
INVENTED_CONDITION = "invented_condition"
CONDITION_NOT_IN_CITED = "condition_not_in_cited_source"
UNSUPPORTED_PREMISE = "unsupported_premise"
UNSUPPORTED_CLAIM = "unsupported_claim"
OBLIGATION_STRENGTHENED = "obligation_strengthened"
FIGURE_MISATTRIBUTED = "figure_misattributed"
FALSE_DENIAL = "false_denial"
CLAIM_CODES = (PARAGRAPH_MISATTRIBUTION, INVENTED_CONDITION, CONDITION_NOT_IN_CITED, UNSUPPORTED_PREMISE, UNSUPPORTED_CLAIM,
               OBLIGATION_STRENGTHENED, FIGURE_MISATTRIBUTED, FALSE_DENIAL)

# measured before being enforced (see module docstring); "warning" findings are reported, never repaired
# Stage 7 measurement on the 63 recorded Stage 6 answers (artifacts/validation_eval/):
#   paragraph_misattribution: fired only on answers already reviewed as wrong        -> error (repaired per row)
#   invented_condition (in NO supplied source): the 0.6B "If applicable, the requirement
#     limits the scope to ..." sentence (an echo of the prompt's own instruction)      -> error
#   unsupported_premise: one hit, a false premise affirmed                              -> error
#   condition_not_in_cited_source (the condition is real but cited to another source)  -> warning
#   unsupported_claim: few true positives, lexical overlap cannot judge paraphrase      -> warning (metric)
# Raised to error after the 16 Oct 2026 validator audit and paired offline severity experiment (300 recorded
# answers): as warnings, both fired only on delivered answers the rubric scored FAIL (D-015's fabricated
# citation among them) and on no PASS answer. As errors they withhold those answers; see validation._CODE_STATUS.
SEVERITY = {
    PARAGRAPH_MISATTRIBUTION: "error",
    INVENTED_CONDITION: "error",
    CONDITION_NOT_IN_CITED: "error",
    UNSUPPORTED_PREMISE: "error",
    UNSUPPORTED_CLAIM: "error",
    # E1 (23 Oct): on 228 recorded answers it fired on 6, none of them a PASS answer (G2); an error, so the draft goes
    # back to the model with the source's own word before any repair
    OBLIGATION_STRENGTHENED: "error",
    # FC2 (29 Oct): a figure stated under one paragraph's citation that only another supplied paragraph contains
    FIGURE_MISATTRIBUTED: "error",
    # FD1 (29 Oct): a denial a supplied paragraph contradicts in so many words
    FALSE_DENIAL: "error",
}

# FD1: denials of applicability or of a topic's presence. "No provision REQUIRES X" is left alone: it is how a correct
# false-premise answer reads (NC-004), and a source that merely mentions X does not contradict it.
_FD1_NOT_APPLY = re.compile(r"\b(?:do(?:es)?\s+not|doesn't|don't)\s+apply\s+to\s+(?P<x>[^.;:\n]{4,120})"
                            r"|(?P<y>[A-Za-z][^.;:\n]{3,100}?)\s+(?:is|are)\s+not\s+(?:subject\s+to|covered\s+by)\b", re.I)
_FD1_NO_TOPIC = re.compile(r"\b(?:do(?:es)?\s+not|doesn't|don't)\s+(?:provide|address|contain|include|cover|discuss|mention|extend)"
                           r"\b(?:\s+(?:any|specific|explicit))*(?:\s+(?:requirements?|provisions?|rules?)(?:\s+or\s+provisions?)?)?"
                           r"(?:\s+(?:regarding|about|on|for|concerning|that\s+address|addressing))?\s+(?P<x>[^.;:\n]{4,120})"
                           r"|\bno\s+provisions?\s+(?:regarding|about|on|for|concerning|address(?:es|ing)?|cover(?:s|ing)?)\s+(?P<y>[^.;:\n]{4,120})",
                           re.I)
# a negation that governs the applying verb ("does not apply", "are not subject", "nor ... apply"), not one elsewhere in the
# clause (820.1(a)(4) applies the part to HCT/Ps "subject to premarket review or notification, or exempt from notification")
_FD1_APPLY_NEGATED = re.compile(r"\b(?:not|no|nor|neither|never)\s+(?:\w+\s+){0,2}(?:appl|subject)|\bexcept\b", re.I)
_FD1_NEGATED = re.compile(r"\b(?:not|no|nor|neither|never)\b", re.I)
_FD1_CUT = re.compile(r",|\b(?:which|that\s+(?:is|are)|because|since|so|but|though|instead)\b", re.I)
# "an application that does not contain ..." is a relative clause, not a denial (K-004 offline)
_FD1_RELATIVE = re.compile(r"\b(?:that|which|who|whose|if|unless|whether|when)\s+(?:\w+\s+){0,1}$", re.I)
# a denial about one named provision is checked against that provision only (M-015: "21 CFR 820.10(b)(2) ... this does
# not address complaint reporting" is true of (b)(2), whatever (b)(3) says)
_FD1_SCOPED = re.compile(r"\b(?:this|that|the\s+(?:cited|quoted))\s+(?:provision|paragraph|section|subpart|definition|text)\b"
                         r"|^\s*(?:21\s*CFR\s*)?§?\s*\d+\.\d+[a-z]?(?:\([A-Za-z0-9]+\))*|\bthis\s+(?:does|do)\s+not\b", re.I)
_FD1_ANAPHOR = re.compile(r"^\s*(?:such|these|those|them|it|this|that)\b", re.I)
# a paragraph that itself excludes or denies cannot contradict a denial (M-003, R-007 offline)
_FD1_PARAGRAPH_DENIES = re.compile(r"\b(?:not|nor|neither)\s+(?:\w+\s+){0,2}(?:appl|subject|include|cover)|\bdoes\s+not\s+include\b"
                                   r"|\bexclud", re.I)
_FD1_STOP = set("""the this that these those part parts section sections provision provisions regulation regulations
requirement requirements specific any such with from into under their them they which what when where while about other
also only must shall should apply applies device devices""".split())
FD1_TOPIC_SHARE = 0.75   # share of the denied topic's key words one supplied paragraph must contain


def _fd1_keys(phrase: str) -> list[str]:
    phrase = _FD1_CUT.split(phrase or "", maxsplit=1)[0]
    words = re.findall(r"[a-z][a-z/-]{2,}", phrase.lower())
    return list(dict.fromkeys(w[:6] for w in words if w not in _FD1_STOP))


def _fd1_clauses(text: str) -> list[str]:
    return [c for c in re.split(r"(?<=[.;:])\s+|\n+", text or "") if c.strip()]


_FD1_MARKERS = re.compile(r"\s*\[\d{1,3}\]")
FD1_VERBATIM_MIN = 25   # characters: a shorter sentence is too generic to be told apart from a quotation


def _fd1_verbatim(sentence: str, texts) -> bool:
    """The sentence is a supplied paragraph's own wording (citation markers and spacing aside)."""
    core = " ".join(_FD1_MARKERS.sub("", sentence or "").split()).strip(" .;:").lower()
    return len(core) >= FD1_VERBATIM_MIN and any(core in " ".join((t or "").split()).lower() for t in texts)


def false_denial(sentence: str, sources, cited=None) -> tuple[int, str] | None:
    """FD1: (source index, its affirming clause) when `sentence` denies what a supplied paragraph affirms: "does not
    apply to X" against a clause that applies the part to X, or "does not provide/address/... X" against a paragraph
    holding nearly all of X's key words with no negation in that clause. None otherwise - and None for a sentence that
    is a supplied paragraph's own wording: the regulation's exclusion ("It does not include an adverse event ... that,
    had it occurred in a more severe form, might have caused death", 312.32(a)) is not a denial by the answer (30 Oct:
    a verbatim 312.32 answer was failed against (c)(1))."""
    if _fd1_verbatim(sentence, [getattr(s, "text", "") for s in list(sources) + list(cited or [])]):
        return None
    for m in (_FD1_NOT_APPLY.search(sentence or ""), _FD1_NO_TOPIC.search(sentence or "")):
        if m and _FD1_RELATIVE.search((sentence or "")[:m.start()]):
            return None
    head = (sentence or "")[:max([m.start() for m in (_FD1_NOT_APPLY.search(sentence or ""),
                                                        _FD1_NO_TOPIC.search(sentence or "")) if m] or [0])]
    if _FD1_SCOPED.search(head):
        sources = list(cited or [])   # about a named provision: its own text decides
    sources = [s for s in sources if not _FD1_PARAGRAPH_DENIES.search(s.text or "")]
    m = _FD1_NOT_APPLY.search(sentence or "")
    if m:
        phrase = m.group("x") or m.group("y")
        keys = [] if _FD1_ANAPHOR.match(phrase or "") else _fd1_keys(phrase)
        if len(keys) >= 2:
            for s in sources:
                for clause in _fd1_clauses(s.text):
                    low = clause.lower()
                    if re.search(r"\b(?:appl(?:y|ies)\s+to|subject\s+to)\b", low) and not _FD1_APPLY_NEGATED.search(low) \
                            and all(k in low for k in keys):
                        return s.index, clause.strip()
        return None
    m = _FD1_NO_TOPIC.search(sentence or "")
    if m:
        phrase = m.group("x") or m.group("y")
        keys = [] if _FD1_ANAPHOR.match(phrase or "") else _fd1_keys(phrase)
        if len(keys) >= 2:
            for s in sources:
                low = (s.text or "").lower()
                if sum(k in low for k in keys) / len(keys) >= FD1_TOPIC_SHARE:
                    clause = next((c for c in _fd1_clauses(s.text) if sum(k in c.lower() for k in keys) >= 2), "")
                    if clause and not _FD1_NEGATED.search(clause.lower()):
                        return s.index, clause.strip()
    return None

# FC2: the figures a sentence states - amounts, multi-digit numbers and dates, and periods ("7 working days") - after its
# CFR citations and paragraph designators are removed; the same, written out, for the numbers a source may spell
_FC2_REFS = re.compile(r"§+\s*|\b\d+\.\d+[a-z]?(?:\([A-Za-z0-9]+\))*|\([A-Za-z0-9]{1,4}\)"
                       r"|\b(?:parts?|subparts?|sections?|titles?|chapters?)\s+\d+(?:\s*(?:,|and|or|through|to|-)\s*\d+)*"
                       # a standard's identifier and its clauses are names (M-010, M-017: "ISO 13485:2016", "Clause 7.3")
                       r"|\b(?:ISO|IEC|ASTM|AAMI|ANSI|CLSI|ICH)\s*[\d.:/-]+(?:\(E\))?|\bclauses?\s+\d+(?:\.\d+)*", re.I)
_FC2_FIGURE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|(?<![\w.])\d{2,}(?:,\d{3})*(?:\.\d+)?(?![\w.])"
                         r"|(?<![\w.])\d+(?=\s+(?:calendar\s+|working\s+|business\s+)?(?:days?|weeks?|months?|years?|hours?|feet|"
                         r"inches|percent)\b)", re.I)
_FC2_WORDS = {str(n): w for n, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                              "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}


def _fc2_figures(text: str) -> set[str]:
    return {f.replace(",", "").replace("$", "").strip() for f in _FC2_FIGURE.findall(_FC2_REFS.sub(" ", text or ""))} - {""}


def _fc2_in(text: str, figure: str) -> bool:
    flat = (text or "").replace(",", "")
    if re.search(rf"(?<![\d.]){re.escape(figure)}(?![\d])", flat):
        return True
    word = _FC2_WORDS.get(figure)
    return bool(word and re.search(rf"\b{word}\b", flat, re.I))

MISATTRIBUTION_OWN_MAX = 0.25     # at most this share of the statement's word 3-grams in the named provision
MISATTRIBUTION_OTHER_MIN = 0.60   # and at least this share in one other, unrelated supplied provision
UNSUPPORTED_MIN_WORDS = 5
_OBLIGATION = re.compile(r"\b(?:must|shall|required?|requires|need(?:s)? to|has to|have to|prohibited|may not|obligat\w*)\b", re.I)
UNSUPPORTED_MAX_SUPPORT = 0.30    # share of content words found in any supplied source

_MARK = re.compile(r"(?<![A-Za-z0-9])\[(\d{1,2})\]")
_LABEL = re.compile(r"(?:21\s*CFR\s*)?(?:§+\s*)?\d{1,4}\.\d{1,4}[a-z]?(?:\([a-z0-9]{1,4}\))*\s*:?", re.I)
_TOKEN = re.compile(r"[a-z0-9]+")
_WORD = re.compile(r"[a-z][a-z'-]{3,}")
_STOP = frozenset(
    "that this these those with from have which their there where when what shall must will would could should such "
    "into upon under other than also each only more most very being been were does your they them then subject part "
    "section paragraph chapter title requires required require requirement requirements provision provisions states "
    "stated according following described specified means include includes including".split()
)
_NEGATED = re.compile(r"\b(?:not|no|never|neither|nor|does not|do not|is not|are not)\b", re.I)
_CONDITION_GROUPS = {
    "applicable": re.compile(r"\b(?:if|where|when|as)\s+applicable\b", re.I),
    "appropriate": re.compile(r"\b(?:as|where|when|if)\s+appropriate\b", re.I),
    "extent": re.compile(r"\bto the extent\b", re.I),
    "exception": re.compile(r"\b(?:unless|except(?:ing)?|other than|exempt(?:ed|ion)?)\b(?! as provided)", re.I),
}
# which group a source wording satisfies (a source "except" supports an answer "unless", and so on)
# a source supports a condition group only with a conditional PHRASE of that group, not the bare word
# ("other applicable requirements" is not "if applicable")
_SUPPORTED_BY = {
    "applicable": re.compile(r"\b(?:if|where|when|as)\s+applicable\b", re.I),
    "appropriate": re.compile(r"\b(?:as|where|when|if)\s+appropriate\b", re.I),
    "extent": re.compile(r"\bto the extent\b", re.I),
    "exception": re.compile(r"\b(?:unless|except|exempt|other than|exception)", re.I),
}
# EX1 (27 Oct): "certification procedure is not necessary" (71.27) is an exemption; the answer's "exempt it from
# certification" restates it
_EXEMPT_SUPPORT = re.compile(r"\b(?:unless|except|exempt|other than|exception)|\b(?:is|are|be)\s+not\s+(?:necessary|required)\b|"
                             r"\bneed\s+not\b", re.I)
# pipeline-authored and refusal sentences are not claims about the evidence
_NOT_A_CLAIM = re.compile(r"could not be validated|do(?:es)? not provide enough information|do(?:es)? not (?:cover|establish)|"
                          r"not found in the available|were not verified|no answer is given|verbatim\)?:|"
                          r"sources? do(?:es)? not provide|^\s*status note \(|^\s*a search of 21 CFR parts 700-740 and FD&C Act sections|"
                          r"^\s*state or local licensing requirements are not part of FDA.s regulations|"
                          # FP1 / RS1 (regulus.rag.premise_figures, regulus.rag.records_silence): the pipeline's own
                          # statement of what its check of the database text found
                          r"^\s*no\. the figures? .{1,120} (?:does|do) not appear in the text of .{1,200} supplied for this answer|"
                          r"sets? no recordkeeping requirement: no paragraph of (?:its|their) text in this system requires|"
                          # QM1 (regulus.rag.qmsr): a former part 820 section's status and the ISO 13485 clause on its subject
                          r"is not in the current eCFR: it belonged to the former Quality System regulation|"
                          r"^\s*in ISO 13485:2016 \(incorporated by reference|^\s*ISO 13485:2016 has no Clause|"
                          r"^\s*FDA published no section-by-section mapping from the former regulation|"
                          r"^\s*the standard.s text is not reproduced here|"
                          # QC1 (regulus.rag.qmsr.compose_certification): what the database check of part 820 found
                          r"^\s*no section of the current part 820 mentions certification|"
                          r"^\s*no section of the current part 820 uses the terms|"
                          # Q1 (regulus.rag.fallback.focused_quote): what the quotation is and is not
                          r"^\s*this is the provision.s own text, not an application of it|"
                          r"^\s*what the regulation defines th(?:is term|ese terms) as, verbatim", re.I)
_LEAD = 45  # a provision label within the first characters of a statement names what it presents

# A2 (regulus.rag.validator_switches): "These provisions set no limit on the amount of guanine" is a claim about the
# cited text, which lexical overlap cannot judge (9% of its content words appear in the source, so it read as
# unsupported and correct drafts were withheld: 73.2120, 82.1330, 82.1602 on the 22 Oct holdout). It is checked
# against the text instead: supported when it is scoped to the provisions themselves, names what kind of limit is
# absent, and the cited sources carry no limiting term of that kind. Unscoped denials ("there is no requirement",
# "FDA does not regulate") are never accepted here: they caused the false-denial regression (SFT v2).
_SCOPE_SUBJECT = (r"(?:(?:this|these|the(?:\s+(?:cited|supplied|quoted|listed))?)\s+(?:provisions?|sections?|paragraphs?|"
                  r"regulations?|listings?|rules?|texts?)|21\s*CFR\s*§?\s*\d{1,4}\.\d+[a-z]?(?:\s*\([a-z0-9]{1,4}\))*|"
                  r"§+\s*\d{1,4}\.\d+[a-z]?(?:\s*\([a-z0-9]{1,4}\))*)")
_ABSENCE_VERB = (r"(?:do(?:es)?\s+not\s+(?:specify|set|impose|state|establish|place|provide|prescribe|include|contain|"
                 r"restrict|limit)|(?:specif(?:y|ies)|sets?|imposes?|states?|establish(?:es)?|places?|provides?|prescribes?|"
                 r"contains?)\s+no)")
_SCOPED_ABSENCE = re.compile(rf"{_SCOPE_SUBJECT}[^.;:]{{0,80}}?\b{_ABSENCE_VERB}\b", re.I)
_NO_X_STATED = re.compile(r"\bno\s+(?:specific\s+|numeric(?:al)?\s+|maximum\s+|minimum\s+)?(?:(?:amount|use|concentration|time)\s+)?(?:limits?|restrictions?|amounts?|"
                          r"quantit(?:y|ies)|concentrations?|levels?|percentages?|frequenc(?:y|ies)|time\s+limits?|deadlines?)\s+"
                          r"(?:is|are)\s+(?:specified|stated|set|imposed|given|prescribed|established)\s+(?:in|by)\s+" + _SCOPE_SUBJECT, re.I)
# what kind of limit the statement says is absent -> the terms that would state one in a source
_ABSENT_KIND = {
    "amount": (re.compile(r"\b(?:amounts?|quantit(?:y|ies)|concentrations?|use levels?|levels?|percent(?:ages?)?|maximum|minimum|"
                          r"how much|numeric(?:al)?\s+limits?|ppm)\b", re.I),
               re.compile(r"\b(?:not (?:more|less) than|not to exceed|not exceeding|percent|per cent|ppm|parts per million|p/m|"
                          r"mg/kg|milligrams?|maximum|minimum)\b|%|\d+(?:\.\d+)?\s*(?:percent|ppm|mg)\b", re.I)),
    # no "time" kind: a scoped "these provisions set no timeframe" was true of the cited 812.40 and misleading, because
    # the deadline lives in 812.46, which was not supplied (P-010, blind A/B 23 Oct). Amount and use limits of a color
    # live in its own listing's uses paragraph; deadlines often live elsewhere.
    "use": (re.compile(r"\b(?:restrict(?:ion|ions|s|ed)?\s+(?:on\s+)?(?:its\s+|their\s+|the\s+)?use|use\s+restrictions?|"
                       r"limit(?:s|ations?)?\s+on\s+(?:its\s+|their\s+|the\s+)?use|which products|product types?)\b", re.I),
            re.compile(r"\b(?:only|restricted|restriction|may not|shall not|must not|not be used|prohibited|except|excluding|"
                       r"limited to|unless)\b", re.I)),
}


def scoped_absence(sentence: str) -> bool:
    return bool(_SCOPED_ABSENCE.search(sentence) or _NO_X_STATED.search(sentence))


# an amount of USE is limited in a provision's uses paragraph; a specifications paragraph's "not more than 100 parts per
# million" limits an impurity in the additive, not how much may be used (73.2120(b) vs (c))
_USE_CONTEXT = re.compile(r"\b(?:uses?|used|usage|added|in amounts?)\b", re.I)
# a uses paragraph that defers to other text cannot show by itself that no limit exists
_DEFERS = re.compile(r"\b(?:subject to|in accordance with|as provided|specified in|set forth in|paragraph\s+\([a-z0-9]+\)|§)", re.I)


def absence_supported(sentence: str, texts: list[str]) -> bool:
    """A scoped absence statement the texts bear out: it names at least one kind of limit, and no text states a limit
    of any kind it names (for an amount: no uses text, and none that defers elsewhere)."""
    kinds = [k for k, (said, _) in _ABSENT_KIND.items() if said.search(sentence)]
    if not (kinds and texts):
        return False
    for k in kinds:
        pool = texts
        if k == "amount":
            pool = [x for x in texts if _USE_CONTEXT.search(x or "")]
            if not pool or any(_DEFERS.search(x) for x in pool):
                return False
        if any(_ABSENT_KIND[k][1].search(x or "") for x in pool):
            return False
    return True


@dataclass(frozen=True)
class ClaimFinding:
    code: str
    severity: str
    detail: str
    citation: int | None = None
    sentence: str | None = None
    requirement_id: str | None = None


# E1 (regulus.rag.validator_switches, holdout failure fix E): a statement that makes a duty of what its source only
# requests, recommends or permits ("is required to file" for 720.1's "is requested to file"; "must conform" for 82.50's
# "may be certified ... if it conforms"). The source sentence it restates is the cited sentence sharing most of its
# content words; the finding needs that sentence to carry a weak modal and no duty word of its own.
_DUTY = re.compile(r"\b(?:must(?! not)|shall|(?:is|are)\s+(?:required|obligated|responsible\s+for)|required\s+to|requires?|"
                   r"mandatory|obligat(?:ory|ion|ed))\b", re.I)
_SOURCE_DUTY = re.compile(r"\b(?:must|shall|required|requires?|requirement|mandatory|prohibit\w*|may not|no person)\b", re.I)
# "may" is left out: "may be certified only if it conforms" / "may be used in amounts consistent with GMP" state the
# conditions of a permission, which ARE requirements (82.1050 restated as "requires" scored PASS on 82.1255, 82.1336);
# so is a bare "request", mostly a noun ("requests for a hearing", "De Novo Request"). Measured 23 Oct on 228 recorded
# answers: with them, 34 fires and 3 on PASS answers; without, fires only on the voluntary-program sections.
_WEAK = re.compile(r"\b(?:should|(?:is|are)\s+requested|voluntar\w*|encourag\w*|recommend\w*)\b", re.I)
# E2: the regulation asks the reader to act ("is requested to file", "should", "voluntary"); "if a sample is
# requested" (71.4) is FDA making a request, not a weaker duty
_REQUESTED_TO = re.compile(r"\b(?:is|are)\s+requested\s+to\b|\bshould\b|\bvoluntar\w*|\bencourag\w*|\brecommend\w*", re.I)
# E1S (27 Oct): a prescribed label statement's own words ("Caution—... a physician should be consulted", 740.12(b)) are the
# statement's wording, not the strength of the duty to bear it - 740.12-Q1's "the label must include the caution" was
# withheld as "'must' where the source says 'should'"
_PRESCRIBED_STATEMENT = re.compile(r"^[\(\w\)\s]{0,12}[\"“]?(?:Warning|Caution|WARNING|CAUTION)\s*[—–-]")
_SECTION_NUMBER = re.compile(r"(?<![\d.])(\d{1,4}\.\d{1,4}[a-z]?)(?![\d])")   # E2: "720.1", "21 CFR 720.1(a)"
_SOURCE_SENTENCE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


def requested_duty(sentence: str, texts: list[str]) -> str | None:
    """E2: the weak wording ("is requested") of a source sentence that only requests what `sentence` makes a duty of -
    the duty word's neighbouring words ("requires a filing", "filing is required") share a stem with that source
    sentence ("is requested to file Form FDA 2512") - or None."""
    body = _LABEL.sub(" ", _MARK.sub(" ", sentence))
    duty = _DUTY.search(body)
    if not duty or _NEGATED.search(body[:duty.start()][-12:]):
        return None
    tokens = re.findall(r"[a-z]+", body.lower())
    at = len(re.findall(r"[a-z]+", body[:duty.start()].lower()))
    def stem(t):
        return re.sub(r"(?:ing|ed|es|e|s)$", "", t)
    near = {stem(t) for t in tokens[max(0, at - 3):at + 4] if len(t) > 3 and not _DUTY.fullmatch(t)}
    for text in texts:
        for piece in _SOURCE_SENTENCE.split(text or ""):
            weak = _REQUESTED_TO.search(piece)
            if weak and not _SOURCE_DUTY.search(piece) and near & {stem(t) for t in re.findall(r"[a-z]+", piece.lower()) if len(t) > 3}:
                return weak.group(0)
    return None


def duty_stated(sentence: str, sources) -> bool:
    """Whether a supplied source sentence states the sentence's duty itself ("must", not "is encouraged") and shares at
    least 40% of its words - E1's escape, used by E2 under E2B. A paragraph under a lead-in that states the duty ("A
    manufacturer subject to this part ... must:" over 820.10(a) "Document a quality management system ...") states it
    too, as C2 reads a lead-in."""
    words = _words(_LABEL.sub(" ", _MARK.sub(" ", sentence)))
    if len(words) < UNSUPPORTED_MIN_WORDS:
        return False
    leads = [s.citation for s in sources if s.citation and (s.text or "").rstrip().endswith(":")
             and _SOURCE_DUTY.search(s.text or "") and not _WEAK.search(s.text or "")]
    for s in sources:
        governed = any(s.citation and s.citation != c and s.citation.startswith(c) for c in leads)
        for piece in _SOURCE_SENTENCE.split(s.text or ""):
            if (governed or _SOURCE_DUTY.search(piece)) and not _WEAK.search(piece) \
                    and len(words & _words(piece)) / len(words) >= 0.4:
                return True
    return False


def strengthened_obligation(sentence: str, cited_texts: list[str], supplied_texts: list[str] = ()) -> str | None:
    """The weak source wording a duty statement strengthens, or None (no duty word, nothing cited, the closest cited
    source sentence states a duty itself, or another supplied source states that duty: then the duty may be right and
    only mis-cited - FDCA613-Q2 cited the voluntary 710.1 "is requested to register" for MoCRA's mandatory 607
    registration, and correcting it to "requested" made a misleading answer, blind A/B 23 Oct)."""
    body = _LABEL.sub(" ", _MARK.sub(" ", sentence))
    duty = _DUTY.search(body)
    if not duty or not cited_texts or _NEGATED.search(body[:duty.start()][-12:]):
        return None
    words = _words(body)
    if len(words) < UNSUPPORTED_MIN_WORDS:
        return None
    best, score, home = None, 0.0, ""
    for text in cited_texts:
        for piece in _SOURCE_SENTENCE.split(text or ""):
            share = len(words & _words(piece)) / len(words)
            if share > score:
                best, score, home = piece, share, text or ""
    if best is not None and precision_fix("E1S") and _PRESCRIBED_STATEMENT.match(home.strip()):
        return None   # E1S: the source is a statement the label must bear ("Caution—... should be consulted")
    if best is None or score < 0.4 or _SOURCE_DUTY.search(best):
        return None
    for text in supplied_texts:
        for piece in _SOURCE_SENTENCE.split(text or ""):
            if _SOURCE_DUTY.search(piece) and not _WEAK.search(piece) and len(words & _words(piece)) / len(words) >= 0.4:
                return None
    weak = _WEAK.search(best)
    return f"'{duty.group(0)}' where the source says '{weak.group(0)}'" if weak else None


def _grams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    t = _TOKEN.findall((text or "").lower())
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def _coverage(segment: str, source: str) -> float:
    g = _grams(segment)
    return len(g & _grams(source)) / len(g) if g else 0.0


def _words(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower())) - _STOP


# PT1: where a product is used -> how a source states that use. Eye and lip use must be stated expressly (21 CFR
# 70.5(a): a color additive is used in the area of the eye only when its listing says so); "cosmetics generally" covers
# products applied to the external body
_CATEGORY_STATED = {"eye": re.compile(r"area of the eye|eye area|eye makeup", re.I),
                    "lips": re.compile(r"\blips?\b|lipsticks?|mucous membrane", re.I),
                    "ext": re.compile(r"cosmetics generally|externally applied cosmetics", re.I)}
_NOT_PERMITTED = re.compile(r"\bnot\b|\bexcept\b|\bexclud|\bprohibit|\bshall not\b|\bmay not\b", re.I)


def covered_product_words(question: str, texts: list[str]) -> set[str]:
    """PT1 (28 Oct): the words of a product the question names ("eye shadow") whose category a supplied source permits
    in a sentence that is not an exception - a member of a stated category is not a premise the source fails to state.
    "Is mica permitted as a color additive in eye shadow?" was declined: 73.2496(b) says "cosmetics generally,
    including cosmetics applied to the area of the eye" and no source contains "shadow" (API smoke test, 27 Oct)."""
    from regulus.rag.color_listing import _PRODUCT_RE

    sentences = [s for t in texts for s in re.split(r"(?<=[.;])\s+", t or "")]
    out: set[str] = set()
    for m in _PRODUCT_RE.finditer(question or ""):
        stated = _CATEGORY_STATED[m.lastgroup]
        if any(stated.search(s) and not _NOT_PERMITTED.search(s) for s in sentences):
            out |= _words(m.group(0))
    return out


def check_claims(sentences: list[str], package, *, question: str | None = None, is_yes_no: bool = False,
                 identity=None, related=None) -> list[ClaimFinding]:
    """`identity(source)` and `related(a, b)` come from regulus.rag.validation (provision identity and
    parent/child relation), passed in to avoid a circular import."""
    from regulus.retrieval.references import parse_cfr_references

    by_index = {s.index: s for s in package.sources}
    identities = {i: identity(s) for i, s in by_index.items()} if identity else {}
    findings: list[ClaimFinding] = []
    all_words = set().union(*(_words(s.text) for s in package.sources)) if package.sources else set()
    # answer_style "apply": the answer applies the provisions to the facts the question states ("a colorant in a lip
    # product caused tumors in rats"), so the question's own words are given, not claims the sources must support
    given = _words(question) if question and getattr(package, "answer_style", "restate") == "apply" else set()
    # a follow-up: the facts the user stated earlier in the conversation are given too ("our tinted body lotion"), not
    # claims a source must support (regulus.rag.conversation; withheld follow-up drafts, 26 Oct)
    earlier = getattr(package, "earlier_question", None)
    if earlier and given:
        given |= _words(earlier)

    # A2 applies only when the answer also says something positive: an answer made of nothing but "X does not cover Y"
    # statements is the signature of a wrong section pick (FDCA601-Q2, UC-002 on the 22 Oct holdout), kept withheld
    a2 = precision_fix("A2")
    if a2:
        substantive = [s for s in sentences if len(_words(_LABEL.sub(" ", _MARK.sub(" ", s)))) >= UNSUPPORTED_MIN_WORDS
                       and not _NOT_A_CLAIM.search(_MARK.sub(" ", s))]
        a2 = bool(substantive) and not all(scoped_absence(s) for s in substantive)
    exact_texts = [s.text for s in package.sources if getattr(s, "tier", None) == "exact"]

    for sentence in sentences:
        marks = [int(m) for m in _MARK.findall(sentence) if int(m) in by_index]
        body = _MARK.sub(" ", sentence)
        content = _LABEL.sub(" ", body)

        # -- paragraph attribution: the named provision's own text vs another supplied provision
        if _NOT_A_CLAIM.search(body):
            continue
        refs = [r for r in parse_cfr_references(body[:_LEAD]) if r.section is not None]
        if refs and identity and len(_grams(content)) >= 4:
            named = (refs[0].title, refs[0].section, refs[0].paragraph_path)
            own = [s for i, s in by_index.items() if identities.get(i) == named]
            if own:
                own_cov = max(_coverage(content, s.text) for s in own)
                others = [(i, _coverage(content, s.text)) for i, s in by_index.items()
                          if identities.get(i) and not related(named, identities[i])]
                best = max(others, key=lambda x: x[1], default=(None, 0.0))
                if own_cov <= MISATTRIBUTION_OWN_MAX and best[1] >= MISATTRIBUTION_OTHER_MIN:
                    findings.append(ClaimFinding(PARAGRAPH_MISATTRIBUTION, SEVERITY[PARAGRAPH_MISATTRIBUTION],
                                                 f"presented as {refs[0].canonical}, but {best[1]:.0%} of its wording is from "
                                                 f"{by_index[best[0]].citation} and {own_cov:.0%} from {refs[0].canonical}",
                                                 citation=own[0].index, sentence=sentence,  # the named provision's own source
                                                 requirement_id=next(iter(own[0].requirement_ids), None)))

        # -- FC2 (29 Oct): a figure stated under a cited paragraph must be in it, its parent or its child. 300Q stepA:
        # restating 117.201(c)(2) the model slid each deadline one label up past a lead-in-only paragraph (F-016 "by July
        # 31" cited to (c)(2)(i)(B), whose text has no 31; PASS lost) and put (a)(6)'s "7 working days" under 117.180(a)(5)
        # (F-015) - wording-overlap checks pass both, since the words are the sources'
        # each line on its own: list lines ending in ":" or "; or" are one "sentence" to the splitter, and four cited lines
        # together cite the parent whose children hold every figure (F-016 replay)
        for line in (re.split(r"\n+", sentence) if precision_fix("FC2") and identities and related else ()):
            lmarks = [int(m) for m in _MARK.findall(line) if int(m) in by_index]
            if not lmarks or not all(identities.get(i) for i in lmarks):
                continue
            figures = _fc2_figures(_LABEL.sub(" ", _MARK.sub(" ", line))) - _fc2_figures(question or "")
            if not figures:
                continue
            family = set(lmarks) | {j for i in lmarks for j, idj in identities.items() if idj and related(identities[i], idj)}
            family_text = " ".join(by_index[j].text or "" for j in family)
            elsewhere = {f: [j for j in by_index if j not in family and _fc2_in(by_index[j].text, f)]
                         for f in figures if not _fc2_in(family_text, f)}
            stray = {f: js for f, js in elsewhere.items() if js}
            if stray:
                js = sorted(stray.items())[0][1]
                findings.append(ClaimFinding(
                    FIGURE_MISATTRIBUTED, SEVERITY[FIGURE_MISATTRIBUTED],
                    f"{', '.join(sorted(stray))} is stated under {by_index[lmarks[0]].citation}, whose text does not contain "
                    f"it; {by_index[js[0]].citation} does", citation=lmarks[0], sentence=line,
                    requirement_id=next(iter(by_index[lmarks[0]].requirement_ids), None)))

        # -- FD1 (29 Oct): a denial a supplied EXACT paragraph contradicts - "the regulations do not apply to HCT/Ps that are
        # regulated as devices" beside 820.1(a)(4) "The provisions of this part apply to manufacturers of ... HCT/Ps" (M-001),
        # "Part 820 does not provide ... regarding foreign manufacturers that refuse to permit an FDA inspection" beside
        # 820.1(c) "Foreign manufacturers" (M-006)
        if precision_fix("FD1"):
            hit = false_denial(body, [s for s in package.sources if getattr(s, "tier", None) == "exact"],
                               cited=[by_index[i] for i in marks])
            if hit is not None:
                findings.append(ClaimFinding(FALSE_DENIAL, SEVERITY[FALSE_DENIAL],
                                             f"{by_index[hit[0]].citation} states: \"{hit[1][:220]}\"", citation=hit[0],
                                             sentence=sentence, requirement_id=next(iter(by_index[hit[0]].requirement_ids), None)))

        # -- conditions the cited sources do not contain
        cited_texts = [by_index[i].text for i in marks]
        all_texts = [s.text for s in package.sources]
        if precision_fix("C2") and identities and related:
            # C2 (26 Oct): a lead-in governs its sub-paragraphs - "FDA may withdraw a qualified facility exemption if:" in
            # 117.251(a) is the condition of 117.251(a)(1), so a sentence citing (a)(1) states it from its own provision.
            # The supplied parent or child of a cited paragraph counts as cited, the tolerance provision_citation_mismatch
            # already gives ("it or its parent/child")
            kin = {j for i in marks if identities.get(i) for j, idj in identities.items()
                   if j not in marks and idj and related(identities[i], idj)}
            if precision_fix("C3"):
                # C3 (27 Oct): a paragraph of the same section that names the cited one states its condition too -
                # 740.2(a) "unless an exemption pursuant to paragraph (b) ... is established" for a sentence citing (b)
                for i in marks:
                    idi = identities.get(i)
                    top = re.match(r"\([a-z0-9]+\)", (idi[2] or "")) if idi else None
                    if top is None:
                        continue
                    names = re.compile(rf"\bparagraphs?\s+(?:\([a-z0-9]+\)(?:,\s*|\s+(?:and|or|through)\s+))*"
                                       rf"{re.escape(top.group(0))}")
                    kin |= {j for j, idj in identities.items() if j not in marks and idj and idj[:2] == idi[:2]
                            and names.search(by_index[j].text or "")}
            cited_texts = cited_texts + [by_index[j].text for j in sorted(kin)]
        for group, pattern in _CONDITION_GROUPS.items():
            m = pattern.search(content)
            if not m:
                continue
            supported_by = (_EXEMPT_SUPPORT if group == "exception" and precision_fix("EX1") else _SUPPORTED_BY[group])
            if not any(supported_by.search(t or "") for t in all_texts):
                findings.append(ClaimFinding(INVENTED_CONDITION, SEVERITY[INVENTED_CONDITION],
                                             f"condition '{m.group(0)}' appears in no supplied source",
                                             citation=marks[0] if marks else None, sentence=sentence))
            elif cited_texts and not any(supported_by.search(t or "") for t in cited_texts):
                findings.append(ClaimFinding(CONDITION_NOT_IN_CITED, SEVERITY[CONDITION_NOT_IN_CITED],
                                             f"condition '{m.group(0)}' is in a supplied source, but not in the one(s) cited",
                                             citation=marks[0], sentence=sentence))

        # -- E1: a duty the cited source only requests, recommends or permits
        if precision_fix("E1") and cited_texts:
            weaker = strengthened_obligation(sentence, cited_texts, [t for i, t in ((s.index, s.text) for s in package.sources)
                                                                      if i not in marks])
            if weaker:
                findings.append(ClaimFinding(OBLIGATION_STRENGTHENED, SEVERITY[OBLIGATION_STRENGTHENED],
                                             f"states a duty ({weaker})", citation=marks[0], sentence=sentence))

        # -- E2: the same for a sentence with no citation marker that names the section ("Yes, 720.1 still requires a
        # filing" beside a cited "is requested to file [1]"; 26 Oct full run, 720.1 follow-ups F1/F4) - or, as the
        # answer's opening sentence, says it of the supplied provisions ("Yes, filing is required.")
        if precision_fix("E2") and not marks:
            named = set(_SECTION_NUMBER.findall(content))
            texts = [s.text for s in package.sources if (_SECTION_NUMBER.findall(s.citation or "")[:1] or [None])[0] in named]
            if not texts and sentence is sentences[0]:
                texts = exact_texts
            weaker = requested_duty(sentence, texts) if texts else None
            if weaker and precision_fix("E2B") and duty_stated(sentence, package.sources):
                # E2B (27 Oct): another supplied source states the duty itself - 820.1(a) "must establish and maintain a
                # quality management system" beside 820.1(a)(2)'s component makers who "are encouraged" (M-012 withheld)
                weaker = None
            if weaker:
                findings.append(ClaimFinding(OBLIGATION_STRENGTHENED, SEVERITY[OBLIGATION_STRENGTHENED],
                                             f"states a duty ({weaker})", sentence=sentence))

        # -- statements whose content no supplied source supports
        words = _words(content)
        # apply style: words only the question contains are facts the answer restates - they are taken out of the
        # count, not counted as support, so whatever the sentence adds must still come from a source (code review)
        own = words - (given - all_words)
        if not own and _OBLIGATION.search(content):
            own = words   # an obligation stated only in the question's words ("you must recall the batch") is still a claim
        if len(words) >= UNSUPPORTED_MIN_WORDS and all_words and own:
            support = len(own & all_words) / len(own)
            if support <= UNSUPPORTED_MAX_SUPPORT and a2 and scoped_absence(body) \
                    and absence_supported(body, [by_index[i].text for i in marks] or exact_texts):
                support = 1.0   # A2: a scoped absence the cited text bears out
            if support <= UNSUPPORTED_MAX_SUPPORT:
                findings.append(ClaimFinding(UNSUPPORTED_CLAIM, SEVERITY[UNSUPPORTED_CLAIM],
                                             f"only {support:.0%} of its content words appear in any supplied source",
                                             citation=marks[0] if marks else None, sentence=sentence))

    # -- yes/no questions: an affirmed premise that no source contains
    if is_yes_no and question:
        premise = _words(question) - all_words
        if precision_fix("PR2"):
            # PR2: ordinary regulatory English ("instead", "outer", "appear") is not a premise by itself
            from regulus.rag.premise_vocabulary import common

            premise = {w for w in premise if not common(w)}
        if precision_fix("PT1") and premise:
            premise -= covered_product_words(question, [s.text or "" for s in package.sources])
        for sentence in sentences:
            asserted = premise & _words(sentence)
            if asserted and not _NEGATED.search(sentence) and not _NOT_A_CLAIM.search(sentence):
                findings.append(ClaimFinding(UNSUPPORTED_PREMISE, SEVERITY[UNSUPPORTED_PREMISE],
                                             f"affirms {sorted(asserted)}, which no supplied source contains", sentence=sentence))
                break
    return findings
